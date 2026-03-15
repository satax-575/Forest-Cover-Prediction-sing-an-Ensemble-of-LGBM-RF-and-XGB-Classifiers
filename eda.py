import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif

from preprocessors import slope_dir, fire_chances, moisture_level

pd.set_option('display.max_columns', None)

df = pd.read_csv("train.csv")
df.head()

features = df.drop(columns = ['Cover_Type','Id'])
target = df['Cover_Type']

df_soil = df[[col for col in features.columns if col.startswith('Soil')]]
features['Soil_Type_Class'] = (np.argmax(df_soil.values, axis = 1) + 1).astype(int) ##argmax starts from 0 ,but type starts from 1
features.drop(columns = df_soil.columns, inplace = True)
features['Soil_Type_Class'].head()

df_wilderness = features[[col for col in features.columns if col.startswith('Wilderness')]]
features['Wilderness_Area_Class'] = (np.argmax(df_wilderness.values, axis = 1) + 1).astype(int) #argmax starts from 0 ,but type starts from 1
features.drop(columns = df_wilderness.columns, inplace = True)
features['Wilderness_Area_Class']

features.head()

features.duplicated().sum()

target.isnull().sum()

features.describe()

df['Cover_Type'].value_counts()

sns.countplot(x = 'Cover_Type', data = df, palette='husl')
plt.title('Target Distribution')
plt.show()

sns.countplot(x = 'Wilderness_Area_Class', data = features, palette = 'husl')
plt.show()

sns.countplot(x = 'Soil_Type_Class', data = features, palette='husl')
plt.show()

count_soil_type = features['Soil_Type_Class'].value_counts()
rare_soil_type = count_soil_type[count_soil_type <= 200].index
features['Soil_Type_Class'] = (features['Soil_Type_Class'].replace(rare_soil_type, -1))  #-1 for other  (done for xgboost ,change later)

sns.countplot(x = 'Soil_Type_Class', data = features, palette='husl')
plt.show()

sns.histplot(features['Elevation'], palette = 'husl')
plt.show()

sns.histplot(features['Horizontal_Distance_To_Hydrology'], palette = 'husl')
plt.show()

sns.histplot(features['Vertical_Distance_To_Hydrology'], palette = 'husl')
plt.show()

sns.histplot(features['Horizontal_Distance_To_Fire_Points'], palette = 'husl')
plt.show()

features.isnull().sum()

#1
features['Slope_Direction'] = features['Aspect'].apply(slope_dir)

#2
features['Chances of Fire'] = features['Horizontal_Distance_To_Fire_Points'].apply(fire_chances)

#Combine Vertical and Horizontal distances to Hydrology[Water Bodies]
features['Centered_Distance'] = (np.sqrt(features['Horizontal_Distance_To_Hydrology']**2 + features['Vertical_Distance_To_Hydrology']**2)).astype(int)

sns.histplot(features['Centered_Distance'], bins = 40, kde = True, palette = 'husl')
plt.show()

features['Moisture Level'] = features['Centered_Distance'].apply(moisture_level)

sns.histplot(features['Moisture Level'])
plt.show()

features['Moisture Level'].value_counts()

features.head()

x_train1, x_test1, y_train1, y_test1 = train_test_split(features, target, test_size = 0.2, random_state = 11)

x_train1

# copy train set
x_train_encoded = x_train1.copy()

# automatically detect categorical columns
cat_cols = x_train_encoded.select_dtypes(include='object').columns

# encode categorical columns
enc = OrdinalEncoder()
x_train_encoded[cat_cols] = enc.fit_transform(x_train_encoded[cat_cols])

# create boolean mask for discrete features
discrete_mask = x_train_encoded.dtypes == 'float64'
discrete_mask[cat_cols] = True  # mark categorical columns as discrete

# compute MI
mi = mutual_info_classif(
    x_train_encoded,
    y_train1,
    discrete_features=discrete_mask.values,
    random_state=0
)

# make table
mi_table = pd.DataFrame({
    "feature": x_train_encoded.columns,
    "MI": mi
}).sort_values("MI", ascending=False).reset_index(drop=True)

print(mi_table)
