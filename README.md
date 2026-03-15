# Forest Cover Type Prediction

Multiclass classification problem predicting forest cover type from cartographic
variables. Based on the UCI / Kaggle Forest Cover Type dataset.

## Problem Statement

Given elevation, slope, soil type, wilderness area, and distance measurements
for 30x30 meter patches of Roosevelt National Forest in Colorado, predict one
of 7 forest cover types.

## Project Structure

├── preprocessors.py   # Helper functions (slope_dir, fire_chances, moisture_level)
│                        and custom sklearn transformers (NonLinearPreprocessor,
│                        LinearPreprocessor)
├── pipelines.py       # Pipeline builder functions for imputation, feature
│                        engineering, encoding, and model pipelines
├── train.py           # End-to-end training script — loads data, builds pipelines,
│                        trains stacking classifier, evaluates and saves model
├── predict.py         # Loads saved model, runs inference on test.csv, writes
│                        Predictions_v1.csv
├── eda.py             # Exploratory analysis — distributions, class balance,
│                        mutual information scores
└── requirements.txt

## Setup

Tested on Python 3.10+.

pip install -r requirements.txt

## Usage

Place train.csv and test.csv in the project root, then:

# Train
python train.py

# Predict
python predict.py

Outputs:
- stacking_model.pkl   — saved stacking classifier
- label_encoder.pkl    — saved label encoder
- Predictions_v1.csv  — final predictions on test set

## Feature Engineering

Raw features are transformed inside the sklearn pipeline before training:

- Soil_Type_Class       : 40 one-hot soil columns collapsed into a single ordinal
                          column. Rare types with <=200 samples are grouped as -1.
- Wilderness_Area_Class : 4 one-hot wilderness columns collapsed into a single
                          ordinal column.
- Slope_Direction       : Aspect (degrees) bucketed into North / East / South / West.
- Chances_of_Fire       : Horizontal distance to fire points bucketed into 3 risk
                          levels (1 = low, 3 = high).
- Centered_Distance     : Euclidean distance to hydrology combining both horizontal
                          and vertical distances.
- Moisture_Level        : Centered distance bucketed into 3 moisture levels.

Missing values are handled via ColumnTransformer:
- Mean imputation for continuous numeric columns
- Mode imputation for binary Soil and Wilderness columns

## Modeling

A stacking ensemble is used as the final model:

Base learners:
- RandomForestClassifier   (n_estimators=1200, max_depth=25, balanced_subsample)
- XGBClassifier            (n_estimators=1200, max_depth=9, learning_rate=0.03)
- LGBMClassifier           (n_estimators=1500, num_leaves=63, learning_rate=0.03)

Meta learner:
- LogisticRegression       (multinomial, lbfgs, max_iter=2000)

All hyperparameters were found via HalvingGridSearchCV with 5-fold cross validation.
Each base learner has the full preprocessing pipeline embedded inside it, so
predict.py requires no manual preprocessing.

## Evaluation

Metrics on 20% stratified holdout:
- Accuracy
- Macro F1 score
- Confusion Matrix

## Notes

- XGBoost requires LabelEncoded targets (0-indexed), handled in train.py.
  Predictions are inverse transformed back to original class labels before saving.
- The linear pipeline (LinearPreprocessor) is also available in pipelines.py for
  use with Logistic Regression or SVM, which were explored but not included in
  the final ensemble.
