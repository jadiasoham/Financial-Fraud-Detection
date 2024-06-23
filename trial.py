import pandas as pd
import numpy as np
from fraud_data_analytics import Preprocessing, Visualization
from predictive_modeling import PrepareFeatures

filepath = r'datasets/Fraud Analytics Dataset.xlsx'
fts = ['time_of_day',
 'cred_type',
 'error_code',
 'payee_requested_amount',
 'payee_settlement_amount',
 'difference_amount',
 'targets']
processor = Preprocessing(file_path= filepath)
df = processor.data_with_targets()
print(type(df))

feature_preparer = PrepareFeatures(df= df, features= fts)