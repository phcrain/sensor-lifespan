import ingest
import prep
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, fbeta_score, make_scorer
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from xgboost.sklearn import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib

# Set the prediction threshold. It is very low for this small dataset
threshold = 0.005

# List of files we want to ingest
data_fp = ['data/carelink_20241101_20241130.csv',
           'data/carelink_20241218_20250116.csv',
           'data/carelink_20241201_20241217.csv']

# Read in data, transform, clean
carelink_raw = ingest.CarelinkIngest(data_fp)  # convert csv files into combined tables
carelink_raw.display_meta()  # view metadata
carelink_prepped = prep.CarelinkPrep(carelink_raw.data)  # join relevant tables, flatten, clean, create addtl fields

# Remove rows with null values in Sensor Glucose
# Note: the beginning of the dataframe is full of None values (likely before a sensor was ever used)
# If any nulls still exist in the dataframe, smote sampling will fail
carelink_final = carelink_prepped.data.filter(pl.col('Sensor Glucose (mg/dL)').is_not_null())

# Split into target and predictor features
X = carelink_final.drop(
    'Datetime',  # Drop timeseries field
    'sensor_end_1h',  # Drop target field
    'sensor_end_2h',  # Drop target field
    'ISIG Value',  # Very strongly correlated (~0.97) with Sensor Glucose
    'ISIG Value_60m_delta',  # Very strongly correlated (~0.97) with Sensor Glucose deltas
    'Month'  # After getting model with Month included, noticed it was in the top 5 features by importance...
    # ... Removing month increasing recall for the minor outcome. Month is likely not truly meaningful/generalizable
    # (e.g., sensor started at end of month and didn't end until new month started)
)
y = carelink_final.select('sensor_end_2h').rename({'sensor_end_2h': 'Sensor_End'})  # Select only target field
#  Can choose above between 1h and 2h predictions for the model

# Split into train and test sets
X_train_imb, X_test, y_train_imb, y_test = train_test_split(X, y, shuffle=False)

# Decrease target class imbalance via smote sampling
sm = SMOTE(sampling_strategy='auto', random_state=123)
X_train, y_train = sm.fit_resample(X_train_imb.to_pandas(), y_train_imb.to_pandas())
# Note: X train and Y train had to be converted to Pandas dfs to work with SMOTEs np expectations

# Convert back to polars df
X_train = pl.DataFrame(X_train)
y_train = pl.DataFrame(y_train)

# get y pos and neg
y_counts = y_train['Sensor_End'].value_counts(sort=True)
y_pos = y_counts.filter(pl.col('Sensor_End') == 1).select('count').item()
y_neg = y_counts.filter(pl.col('Sensor_End') == 0).select('count').item()
pos_weight = y_neg/y_pos

# Define an fbeta scorer
fbeta_scorer = make_scorer(fbeta_score, beta=4, response_method='predict')

# Init a time series cross-validator
tss = TimeSeriesSplit(n_splits=2)

# Init the classifier
model = XGBClassifier(random_state=123,
                      eval_metric='aucpr',
                      n_estimators=1000,
                      reg_alpha=0.25)

# Define the pipeline
grid_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the data
    ('feature_selection', SelectFromModel(model)),  # Feature selection
    ('model', model)  # Build xgb model
])

# Define hyperparameter grid
param_grid = {
    'model__learning_rate': [0.01, 0.1],
    'model__max_depth': [10, 15],  # Tree depth
    'model__scale_pos_weight': [pos_weight, pos_weight * 1.25, pos_weight * 0.75]  # Scaling for handling imbalance
}

# Init and fit grid search
grid_search = GridSearchCV(grid_pipeline, param_grid, cv=tss, scoring=fbeta_scorer, verbose=1, n_jobs=-1)
grid_search.fit(X, y)

# Print best hyperparameters
print('Hyperparams:', grid_search.best_params_)

best_pipeline = grid_search.best_estimator_  # Get the best model
best_pipeline.fit(X_train, y_train)  # Retrain on full training set
y_pred = best_pipeline.predict_proba(X_test)[:, 1] > threshold  # Evaluate performance with really low prob threshold
class_rep = classification_report(y_test, y_pred)
print(class_rep)
roc_auc = roc_auc_score(y_test, y_pred)
print('AUC Score:', roc_auc)

# Extract model from pipeline
best_model = best_pipeline.named_steps['model']

# Get feature importance from model
importance_values = best_model.feature_importances_

# Extract feature selection from pipeline
selector = best_pipeline.named_steps['feature_selection']

# Get features names from X
importance_names = np.array(X.columns)[selector.get_support()]

# Map importance ratios to feature names
importance_df = (
    # Get feature names and importance values
    pl.DataFrame({'Feature': importance_names, 'Importance': importance_values})
    # Remove features that had 0 importance
    .filter(pl.col('Importance') > 0)
    # Sort feature importance from highest to lowest
    .sort('Importance', descending=True)
)

# Print feature importance
print(importance_df)

# Check correlations of selected features:
correlation_matrix = X.select(importance_df['Feature'].to_list()).corr()

col_names = correlation_matrix.columns
correlation_df = (
    pl.DataFrame({
        'Col1': [col1 for col1 in col_names for col2 in col_names],
        'Col2': [col2 for col1 in col_names for col2 in col_names],
        'Correlation': np.abs(correlation_matrix.to_numpy().flatten())
    })
    .sort('Correlation', descending=True)
    .filter(pl.col('Col1').is_in(importance_df['Feature'])  # Keep only rows where both cols are in the selected feats
            & pl.col('Col2').is_in(importance_df['Feature'])  # Keep only rows where both cols are in the selected feats
            & (pl.col('Col1') != pl.col('Col2')))  # Remove self correlation
)
print(correlation_df)

# Create list of correlation dicts
corr_list = []
for field1 in importance_df['Feature'].to_list():
    corr_dict = {}
    for field2 in importance_df['Feature'].to_list():
        if field1 == field2:
            corr = 1.0
        else:
            corr = correlation_df.filter((pl.col('Col1') == field1) & (pl.col('Col2') == field2))['Correlation'][0]
        corr_dict[field2] = corr
    corr_list.append(corr_dict)

# Create metadata dict
metadata = {
    'classification_report': class_rep,
    'roc_auc_score': roc_auc,
    'feature_names': importance_df['Feature'].to_list(),
    'feature_importance': importance_df['Importance'].to_list(),
    'feature_correlation': corr_list,
    'prediction_threshold': threshold
}

# Save model pipeline and metadata
joblib.dump({'pipeline': best_pipeline,
             'metadata': metadata},
            'model/model.joblib')

# Example using the saved pipeline -----------------------------------

# Load the saved pipeline
saved_data = joblib.load('model/model.joblib')

# Get the pipeline
loaded_pipeline = saved_data['pipeline']

# Get the metadata
metadata = saved_data['metadata']

# Print metadata
print('Classification Report', metadata['classification_report'])
print('ROC AUC Score:', metadata['roc_auc_score'])
print('Feature Names:', metadata['feature_names'])
print('Feature Importance:', metadata['feature_importance'])
print('Feature Correlation:', pl.DataFrame(metadata['feature_correlation']))

# Get the prediction threshold
threshold = metadata['prediction_threshold']

# Predict on new data
# Where `X` is new data
predictions = loaded_pipeline.predict_proba(X)[:, 1] > threshold
