import polars as pl
import random
import warnings

class CarelinkPrep:

    def __init__(self, data):
        """
        Initialize the carelink_prep class and join sensor and pump data into one pl.Lazyframe
        :param data: list of 3 Polars dataframes returned from carelink ingest
        """

        pump_data = self.remove_null_cols(
            data[0]
            .filter(pl.col('Alert').is_null() & pl.col('User Cleared Alerts').is_null())
        )
        sensor_data = self.remove_null_cols(
            data[2].filter(
                pl.col('Event Marker').is_in(['Start of the day', 'End of the day']).lt(1)
                | pl.col('Event Marker').is_null()
            )
        )
        # flatten data and 1hot categories:
        flat = self.__clean(sensor_data, pump_data)
        # engineer target feature and other features:
        final = self.__feat_engineer(flat)
        # Collect LazyFrame as object:
        self.data = final.collect()

    # Filter out columns with all null values
    def remove_null_cols(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Remove columns that contain all null values
        :param df: Polars eager DataFrame
        """
        non_null_cols = [col for col in df.columns if df[col].null_count() < len(df)]
        return df.select(non_null_cols)

    def __clean(self, sensor_df: pl.DataFrame, pump_df: pl.DataFrame) -> pl.LazyFrame:
        """
        Join the sensor and pump data, one hot encode category cols, and flatten into one row per 5min Duration
        :param sensor_df: polars DataFrame; sensor data from carelink ingest
        :param pump_df: polars DataFrame; pump data from carelink ingest
        :return: pl.LazyFrame
        """
        carelink_data = (
            sensor_df
            .join(pump_df, on=['Date', 'Time'], how='full', suffix='_2', coalesce=True)
            .drop(['Index', 'Index_2'])
        )
        # Replace empty strings with null values in all string columns
        carelink_data = carelink_data.lazy().with_columns([
            pl.when(pl.col(col) == "")
            .then(None)
            .otherwise(pl.col(col))
            .alias(col) for col in carelink_data.columns
        ])

        carelink_data = (
            carelink_data
            .with_columns(
                pl.col('Sensor Glucose (mg/dL)').cast(pl.UInt16),  # cast as int
                pl.col('ISIG Value').cast(pl.Float32),  # cast as float
                pl.coalesce(
                    pl.col('Date').str.strptime(pl.Date, '%m/%d/%Y', strict=False),  # convert MM/DD/YYYY str to Date
                    pl.col('Date').str.strptime(pl.Date, '%Y/%m/%d', strict=False)  # convert YYYY/MM/DD str to Date
                ),
                pl.col('Time').str.strptime(pl.Time, "%H:%M:%S")  # convert str to Time
            )
            # combine Date and Time str into Datetime:
            .with_columns((pl.col('Date').cast(pl.Datetime) + pl.col('Time').cast(pl.Duration)).alias('Datetime'))
        )

        carelink_data = (
            carelink_data
            .sort('Datetime')  # order by datetime
            .drop(  # drop unneeded fields
                'Date',  # Not needed (data is contained in datetime)
                'Time',  # Not needed (will recreate this from truncated time)
                'Prime Volume Delivered (U)',  # Not relevant
                'Estimated Reservoir Volume after Fill (U)',  # Not relevant
                'Bolus Type',  # Assume constant for me
                'BWZ Target High BG (mg/dL)',  # Assume constant for me
                'BWZ Target Low BG (mg/dL)',  # Assume constant for me
                'BWZ Carb Ratio (g/U)',  # Assume constant for me
                'BWZ Insulin Sensitivity (mg/dL/U)',  # Assume constant for me
                'Bolus Number',  # Not relevant: bolus id
                'Scroll Step Size',  # Not relevant
                'BLE Network Device',  # Not relevant
                'BG Reading (mg/dL)',  # Very sparse
                'BWZ BG/SG Input (mg/dL)',  # Very sparse
                'BWZ Active Insulin (U)',  # Very sparse
                'Sensor Calibration BG (mg/dL)',  # Very sparse
                'BWZ Unabsorbed Insulin Total (U)',  # Very sparse; likely not relevant
                'New Device Time', # Sparse and not relevant
                'Insulin Action Curve Time',  # Sparse and not relevant
                strict=False
            )
        )
        # Identify any questionable datetimes and drop them plus throw warning
        timestamps = carelink_data.select(pl.col("Datetime").cast(pl.Int64)).collect()
        mean_date = timestamps.mean()
        std_date = timestamps.std()
        # Set valid range to +- 3 std from mean
        valid_start = pl.from_epoch((mean_date - 3 * std_date) // 1e6, time_unit='s').item()
        valid_end = pl.from_epoch((mean_date + 3 * std_date) // 1e6, time_unit='s').item()
        # Get rows with out-of-range dates
        oor_dates = carelink_data.filter(~pl.col("Datetime").is_between(valid_start, valid_end)).collect()
        # Drop rows and throw warning, if they exist
        if oor_dates.shape[0] != 0:
            carelink_data = carelink_data.filter(pl.col("Datetime").is_between(valid_start, valid_end))
            oor_date_list = oor_dates['Datetime'].unique().to_list()
            if len(oor_date_list) > 4:
                oor_date_list[4] = "..."
                oor_date_list = oor_date_list[:5]
            warnings.warn(
                f"Removing {oor_dates.shape[0]} rows with out-of-range dates.\n" +
                f"Calculated valid range: {valid_start} to {valid_end}\n" +
                f"Out-of-range dates:\n{'\n'.join([str(x) for x in oor_date_list])}",
                category=RuntimeWarning
            )
        # Resample the data to have one row every 5 minutes
        carelink_resampled = (
            carelink_data
            .with_columns(pl.col('Datetime').dt.truncate('5m'))  # break time into 5-min intervals
            .group_by('Datetime')
            .agg([
                pl.when(pl.col(col).drop_nulls().n_unique() < 2)
                .then(pl.col(col).drop_nulls().first())
                .otherwise(pl.col(col))
                .alias(col)
                for col in carelink_data.collect_schema().names() if col != 'Datetime'
            ])
            # Explode list columns:
            .explode([col for col in carelink_data.collect_schema().names() if col != 'Datetime'])
            .unique()  # dedup df on all cols
        )
        # One hot encode category columns
        # list cat cols
        cat_cols = ['Sensor Exception',
                    'BG Source',
                    'Basal Rate (U/h)',
                    'Prime Type',
                    'SmartGuard Correction Bolus Feature',
                    'Suspend',
                    'Rewind',
                    'BWZ Status',
                    'Bolus Cancellation Reason',
                    'Sensor Calibration Rejected Reason',
                    'Bolus Source']
        # one hot encode the cat cols
        carelink_onehot = carelink_resampled.collect().to_dummies(cat_cols).lazy()
        # drop null one hot fields
        carelink_onehot = carelink_onehot.drop(
            [col for col in carelink_onehot.collect_schema().names() if col[-4:] == 'null']
        )
        # standardize one hot values within groups and flatten to one row per Datetime
        onehot_cols = [col for col in carelink_onehot.collect_schema().names()
                       if col not in carelink_resampled.collect_schema().names()]  # get names of generated cols
        sum_cols = ['Bolus Volume Selected (U)',
                    'Bolus Volume Delivered (U)',
                    'BWZ Estimate (U)',
                    'BWZ Carb Input (grams)',
                    'BWZ Correction Estimate (U)',
                    'BWZ Food Estimate (U)',
                    'Final Bolus Estimate']  # cols we should sum when combining
        avg_cols = ['BWZ BG/SG Input (mg/dL)',
                    'BWZ Active Insulin (U)',
                    'Sensor Calibration BG (mg/dL)',
                    'BWZ Unabsorbed Insulin Total (U)']  # cols to average when combined
        carelink_onehot = (
            carelink_onehot
            .group_by('Datetime')
            .agg([
                pl.when(col in onehot_cols)
                .then(pl.col(col).max())  # keep 1's within groups for one hot cols
                .when(col in sum_cols)
                .then(pl.col(col).cast(pl.Float32).sum())  # sum values within groups for sum cols
                .when(col in avg_cols)
                .then(pl.col(col).cast(pl.Float32).mean())  # average values within groups for avg cols
                .otherwise(pl.col(col))  # keep same values in list for all other cols
                for col in carelink_onehot.collect_schema().names() if col != 'Datetime'])
            .explode([col for col in carelink_onehot.collect_schema().names() if col != 'Datetime'])
            .unique()
            .with_columns(
                # pl.col('Datetime').dt.time().alias('Time'),  # Set new Time col
                pl.col('Datetime').dt.hour().alias('Hour'),
                pl.col('Datetime').dt.minute().alias('Minute'),
                pl.col('Datetime').dt.month().alias('Month')  # Create month col
                # ^^^ (might be able to use this after yrs of data collection)
            )
            # convert 1hot cols back into ints
            .with_columns([pl.col(col).cast(pl.UInt8) for col in onehot_cols])
            # convert summed cols from strings into floats
            .with_columns([pl.col(col).cast(pl.Float32) for col in sum_cols])
        )

        return carelink_onehot

    def __feat_engineer(self, ldf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Engineer target feat and new features for carelink data such as time since start, G/ISIG ratio, glucsose deltas
        :param ldf: polars LazyFrame; sensor data from carelink ingest
        :return: pl.LazyFrame
        """
        # Add artificial rows at beginning so that we can fill delta values later on
        # Get first 12 non-null SG values
        # (note this could be problematic if lots of null at beginning of df or mixed nulls
        artificial_dt = ldf.sort("Datetime").filter(pl.col("Sensor Glucose (mg/dL)").is_not_null()).collect()[0:12]
        artificial_rows = (
            pl.DataFrame({
                "Datetime": artificial_dt["Datetime"].dt.offset_by("-60m"),
                **{col: [None] * 12 for col in artificial_dt.columns if col != "Datetime"}
            })
            # Fill first SG and first ISIG
            .with_columns([
                # add linear predicted value followed by null values:
                pl.Series(col, [intercept + 12 * slope] + [None] * 11)
                .cast(pl.Float32)  # cast to Float32 instead of f64 (issues joining to ISIG value in append if f64)
                # intercept and slope derived below:
                for col in ["Sensor Glucose (mg/dL)", "ISIG Value"]  # column of interest
                # set intercept == 1st val, we will predict the 12th previous val:
                for intercept in [artificial_dt[col][0]]
                # calc avg slope:
                for slope in [(sum([(artificial_dt[col][i] - artificial_dt[col][i+5]) / 6 for i in range(6)])/6
                              * random.uniform(0.5, 1.5))]  # add random variation to slope
            ])
        )

        # Append new rows to df and sort
        carelink_artificial = pl.concat([ldf.collect(), artificial_rows], how="vertical").sort("Datetime")

        # Fill null values
        # First remove rows where there are no NA and they won't factor into slope/intercept calcs:
        carelink_fillna = (
            carelink_artificial
            .lazy()
            .sort("Datetime")
            .with_columns([
                pl.col(col).interpolate()
                for col in ["Sensor Glucose (mg/dL)", "ISIG Value"]
                ])
        )

        # Feat eng: Create Glucose/ISIG ratio (GISIG_Ration),
        # then features containing deltas for SG, ISIG, and GISIG over time
        carelink_features = (
            # Create glucose/ISIG ratio
            carelink_fillna.with_columns(
                (pl.col('Sensor Glucose (mg/dL)') / pl.col('ISIG Value')).alias('GISIG_Ratio')
            )
            # Add features to look at deltas in Sensor Glucose, ISIG, and GISIG ratio over time
            .sort('Datetime')
            .with_columns([
                pl.col(col).shift(n=steps).alias(f"{col}_{steps*5}m")
                for col in ['Sensor Glucose (mg/dL)', 'ISIG Value', 'GISIG_Ratio']
                for steps in [3, 6, 12]
            ])
            .with_columns([
                (pl.col(col) - pl.col(f"{col}_{steps*5}m")).alias(f"{col}_{steps*5}m_delta")
                for col in ['Sensor Glucose (mg/dL)', 'ISIG Value', 'GISIG_Ratio']
                for steps in [3, 6, 12]
            ])
            .drop([
                f"{col}_{steps*5}m"
                for col in ['Sensor Glucose (mg/dL)', 'ISIG Value', 'GISIG_Ratio']
                for steps in [3, 6, 12]
            ])
            .filter(pl.col("Datetime").is_in(artificial_rows["Datetime"]).lt(1))  # remove the artificial rows
        )
        # Feat eng: create time since start
        # Create groups based on when 1's in start field where encountered
        carelink_features = (
            carelink_features
            .with_columns(
                # Create numbered groups for every group of 1s:
                (pl.col("Sensor Exception_SENSOR_INIT_CODE").diff().gt(0).cum_sum()
                 * pl.col("Sensor Exception_SENSOR_INIT_CODE"))  # Change 0's to group 0
                .replace(0, None)  # Replace 0s with null, so they can be filled with group number
                .fill_null(strategy="forward")  # Fill nulls with group number
                .fill_null(0)  # fill the starting rows w/0
                # we don't know when ^^^ the sensor for earliest rows started, so we'll have to fill later
                .alias("group")
            )
        )
        # Impute Group 0 Start Time:
        # get the start times for each group
        start_times = (
            carelink_features
            .group_by('group')
            .agg(pl.col("Datetime").first().alias("Start_Time"))
            .collect()
        )
        # find the average time of sensor duration:
        avg_timedelta = (
            start_times
            .sort("group")
            .with_columns(pl.col("Start_Time").diff())
            .select("Start_Time").mean().item()
        )
        # find a possible start time for row 0 by imputing based off average timedelta
        start_time_0 = start_times.filter(pl.col("group") == 1)["Start_Time"].item(0) - avg_timedelta
        # set the group 0 start time to be the average timedelta, unless it is earlier
        start_times = start_times.lazy().with_columns(
            pl.when(pl.col("group") != 0)
            .then(pl.col("Start_Time"))
            .otherwise(
                pl.when(pl.col("Start_Time") > pl.lit(start_time_0))
                .then(start_time_0)
                .otherwise(pl.col("Start_Time"))
            )
        )
        # Join start time to each row, find time difference since start time, and convert to minutes (int):
        carelink_features = (
            carelink_features
            .join(start_times, on="group", how="left")
            .with_columns((pl.col("Datetime") - pl.col("Start_Time"))
                          .dt.total_minutes().alias("Start_Time_Delta"))  # convert microseconds to minutes
            .drop("group", "Start_Time")
        )

        # Create target var
        # Combine sensor end flags for the purposes of creating target
        carelink_features = (
            carelink_features
            .sort('Datetime')
            .with_columns(
                ((pl.col('Sensor Exception_SENSOR_CHANGE_SENSOR_ERROR') == 1)
                 | (pl.col('Sensor Exception_SENSOR_END_OF_LIFE') == 1)
                 ).alias('sensor_end')
            )
        )
        # Create and two options for a target: sensor ending within 2 hours or within 1 hour
        # Create 1h target
        target_1h = (
            carelink_features
            .group_by_dynamic("Datetime", every="5m", period="1h", closed="right")
            .agg(
                pl.col("sensor_end").max().alias("sensor_end_1h"),
            )
        )
        # Create 2h target
        target_2h = (
            carelink_features
            .group_by_dynamic("Datetime", every="5m", period="2h", closed="right")
            .agg(
                pl.col("sensor_end").max().alias("sensor_end_2h"),
            )
        )
        # Join 1/2h targets
        carelink_target = (
            carelink_features
            .join(target_1h, on="Datetime", how="left")  # join 1h target field
            .join(target_2h, on="Datetime", how="left")  # join 2h target field
            .with_columns(
                (pl.when(pl.col("sensor_end") == 1)
                 .then(0)  # change val to 0 if the sensor has already ended
                 .otherwise(pl.col("sensor_end_1h"))
                 .fill_null(strategy="forward")  # fill nulls from breaks in time series with prev val
                 .alias("sensor_end_1h")),
                (pl.when(pl.col("sensor_end") == 1)
                 .then(0)  # change val to 0 if the sensor has already ended
                 .otherwise(pl.col("sensor_end_2h"))
                 .fill_null(strategy="forward")  # fill nulls from breaks in time series with prev val
                 .alias("sensor_end_2h"))
            )
            .drop("sensor_end")  # drop composite feature
        )

        return carelink_target
