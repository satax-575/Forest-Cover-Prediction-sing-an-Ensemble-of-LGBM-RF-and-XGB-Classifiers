import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def slope_dir(i):
    if i <=45 or i>315: return 'North'
    elif i in range(46,136): return 'East'
    elif i in range(136,226): return 'South'
    elif i in range(226,316): return 'West'
    else: return 'Error'


def fire_chances(i):
    if i<=1500 : return 3
    elif i<=3000 and i>1500: return 2
    else: return 1


def vert_water_dist_class(i):
    if i in range(-10,50): return 2  #higher the number , more moist the environment
    elif i>50: return 1
    else: return 3


def moisture_level(i):
    if i < 200: return 3
    elif 200 <= i <= 500: return 2
    # elif i in range(200,500): return 2 #[WONT WORK FOR FLOATS][Changed tp integer later on]
    else: return 1


class NonLinearPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.input_columns_ = X.columns
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.input_columns_) if not hasattr(X, 'columns') else X.copy()

        # Soil
        soil_cols = [c for c in X.columns if c.startswith('mode_impute__Soil')]  #compensate for the imputed name
        if len(soil_cols) > 0:
            X['Soil_Type_Class'] = np.argmax(X[soil_cols].values, axis=1) + 1
            counts = X['Soil_Type_Class'].value_counts()
            rare = counts[counts <= 200].index
            X['Soil_Type_Class'] = X['Soil_Type_Class'].replace(rare, -1)
            X.drop(columns=soil_cols, inplace=True)

        # Wilderness
        wild_cols = [c for c in X.columns if c.startswith('mode_impute__Wilderness')]
        if len(wild_cols) > 0:
            X['Wilderness_Area_Class'] = np.argmax(X[wild_cols].values, axis=1) + 1
            X.drop(columns=wild_cols, inplace=True)

        # Other numeric columns
        aspect = [c for c in X.columns if 'Aspect' in c][0]
        hdh = [c for c in X.columns if 'Horizontal_Distance_To_Hydrology' in c][0]
        vdh = [c for c in X.columns if 'Vertical_Distance_To_Hydrology' in c][0]
        hdf = [c for c in X.columns if 'Horizontal_Distance_To_Fire_Points' in c][0]

        X['Slope_Direction'] = X[aspect].apply(slope_dir)
        X['Chances_of_Fire'] = X[hdf].apply(fire_chances)
        X['Centered_Distance'] = np.sqrt(X[hdh]**2 + X[vdh]**2).astype(int)
        X['Moisture_Level'] = X['Centered_Distance'].apply(moisture_level)

        return X


class LinearPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.input_columns_ = X.columns
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.input_columns_) if not hasattr(X, 'columns') else X.copy()

        # Other numeric columns
        aspect = [c for c in X.columns if 'Aspect' in c][0]
        hdh = [c for c in X.columns if 'Horizontal_Distance_To_Hydrology' in c][0]
        vdh = [c for c in X.columns if 'Vertical_Distance_To_Hydrology' in c][0]
        hdf = [c for c in X.columns if 'Horizontal_Distance_To_Fire_Points' in c][0]

        # X['Slope_Direction'] = X[aspect].apply(slope_dir)
        X['Chances_of_Fire'] = X[hdf].apply(fire_chances)
        X['Centered_Distance'] = np.sqrt(X[hdh]**2 + X[vdh]**2).astype(int)
        X['Moisture_Level'] = X['Centered_Distance'].apply(moisture_level)

        return X
