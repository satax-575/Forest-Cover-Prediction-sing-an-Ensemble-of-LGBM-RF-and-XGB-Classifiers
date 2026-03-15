import pandas as pd
import joblib

sc = joblib.load('stacking_model.pkl')
le = joblib.load('label_encoder.pkl')

t_df = pd.read_csv("test.csv")

preds = sc.predict(t_df)   #preprocessing done automatically inside cause its integrated into stacking pipeline

preds = le.inverse_transform(preds)

preds

pred_df = pd.DataFrame({'Id' : t_df['Id'], 'Cover_Type' : preds})
pred_df.to_csv("Predictions_v1.csv", index = False)
