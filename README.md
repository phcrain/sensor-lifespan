# sensor-lifespan
Build a ML pipeline for predicting when a sensor is close to failing and will need to be replaced

## Overview
This repository contains a machine learning pipeline designed to predict when a sensor is nearing failure and requires replacement. The pipeline leverages personal CareLink data, including pump and sensor readings, to build predictive models that enhance sensor lifespan management.

## Installation
To set up the project, clone this repository and install dependencies using `uv`:
```sh
git clone https://github.com/phcrain/sensor-lifespan.git
cd sensor-lifespan
uv venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
uv sync
```

## Usage
### Data Ingestion
Run the `ingest.py` script to load and process sensor data:
```sh
python ingest.py
```

### Data Preprocessing
Prepare the data for modeling by running:
```sh
python prep.py
```

### Model Training & Evaluation
Execute the main script to train and evaluate the model:
```sh
python main.py
```
The trained model will be saved under `model/model.joblib`.

## Dependencies
Dependencies are managed using `uv`. See `uv.lock` for the complete list.

## License
This project is open-source and available under the MIT License.



For questions or contributions, feel free to open an issue or submit a pull request!
