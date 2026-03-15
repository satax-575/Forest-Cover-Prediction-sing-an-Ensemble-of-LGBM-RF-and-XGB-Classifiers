import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from pipelines import build_combined_pipeline, build_model_pipelines, build_stacking_classifier

pd.set_option('display.max_columns', None)

df11 = pd.read_csv("train.csv")
features11 = df11.drop(columns = ['Cover_Type','Id'])
target11 = df11['Cover_Type'] #xgboost cant predict

mode_cols = [col for col in features11.columns if col.startswith('Soil')] + [col for col in features11.columns if col.startswith('Wild')]
mean_cols = [col for col in features11.columns if col not in mode_cols]

target11.unique()

combined_pline = build_combined_pipeline(mean_cols, mode_cols)

x_train, x_test, y_train, y_test = train_test_split(features11, target11, test_size = 0.2, stratify = target11, random_state = 11)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

x_train2 = combined_pline.fit_transform(x_train)
x_train2.head()

rf_pline, xg_pline, lg_pline = build_model_pipelines(combined_pline)

base_estimators = [('rf', rf_pline), ('xg', xg_pline), ('lg', lg_pline)]

sc = build_stacking_classifier(base_estimators)

sc.fit(x_train, y_train)
y_pred = sc.predict(x_test)

print('Accuracy : ', accuracy_score(y_test, y_pred))
print('Confusion Matrix', confusion_matrix(y_test, y_pred))
macro_f1 = f1_score(y_test, y_pred, average='macro')
print('Macro F1', macro_f1)

original_pred = le.inverse_transform(y_pred)
original_test_pred = le.inverse_transform(y_test)

joblib.dump(sc, 'stacking_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
