# Turbofan - predicting RUL

#### Predicting Remaining Useful Life (RUL) of a unit based on sensor data. 

This is a showcase for applying Machine Learning methods to the Turbofan dataset.

### Dataset
Turbofan dataset is chosen to reflect the common properties on the real data

* input data are dirty: some of the sensors do not work
* input data are noisy and redundant (highly-correlated)
* different sensors are on completely different scales

* it is not clear in advance what value to predict; it should be discussed with business-uers
* the value to be predicted it not given, but should be extracted from input data

## Development
###  Exploratory analysis 

    RUL/explore_with_notebooks/exploratory.ipynb

Many visualizations. 
Questions to discuss with business-users and with fellow ML-professionals.
Linear Regression as baseline.

### Data Drift Check 

    RUL/explore_with_notebooks/DataDriftCheck.ipynb 

Various  approaches to check if the training and input data come from the same distribution.

### Simplest model with GRU (Gated Recurrent Unit) Neutral Network

    RUL/explore_with_notebooks/SimplestGRU.ipynb .

Keras 2-layer NN with GRU units. 
Questions to discuss with business-users and fellow ML-professionals.


## Staging
python scripts (with argparse abd logging) to be executed from command line

    * python3 RUL/train_model.py
    * python3 RUL/predict_on_input.py 

use the helper functions in

    * RUL/utils/utils.py


## Production
is not present here. 
The simplest way - by scheduling scripts with Apache Airflow.
Many other MLOps pipelines are possible. 














 







