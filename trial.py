from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

file_path = r'datasets\Fraud Analytics Dataset.xlsx'
data = pd.read_excel(file_path)

payment_data = data.copy() # make a copy of the original data file
features = [] #the features to be included in the model


# Isolation Forest to detect anomalies:
contamination_rate = 0.1 # Anticipating 1% fraudlent transactions
model = IsolationForest(contamination= contamination_rate)

model.fit(payment_data[features])

anomalies = model.predict(payment_data[features])
payment_data['is_anomaly'] = anomalies

#display fraudulent transactions data:
fraudulent_txn = payment_data[payment_data['is_anomaly'] == -1]
print(fraudulent_txn)