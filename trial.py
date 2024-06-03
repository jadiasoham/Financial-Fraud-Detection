from typing import Sequence, Optional, List
import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme('notebook')

# Constants:
FRAUD_TAG = 'Fraudulent Transaction' # Identification tag for fraudulent transactions
COL_WITH_FRAUD_TAG = 'txn_subtype' # Name of the column containing the `FRAUD_TAG`.


class PreProcessing:
    def __init__(self, file_path: str) -> None:
        self.file = file_path
        self.data = self.open_file()
        self.fraud_data = self.generate_fraud_data()
        self.labelled_data = self.data_with_targets()

    def open_file(self) -> pd.DataFrame:
        file = self.file
        extension = pathlib.Path(file).suffix
        if extension == '.xlsx' or '.xls':
            return pd.read_excel(file)
        elif extension == '.csv':
            return pd.read_csv(file)
        else:
            raise ValueError("Invalid File Format")
    
    @staticmethod
    def segment_day(hour) -> str:
        if 0 <= hour < 3:
            return 'LateNight'
        elif 3 <= hour < 6:
            return 'EarlyMorning'
        elif 6 <= hour < 9:
            return 'Morning'
        elif 9 <= hour < 12:
            return 'LateMorning'
        elif 12 <= hour < 15:
            return 'Afternoon'
        elif 15 <= hour < 18:
            return 'LateAfternoon'
        elif 18 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
        
    def create_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        df['time_of_day'] = df['hour'].apply(self.segment_day)
        return df
    
    def get_difference(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add column named difference_amount:
        df['difference_amount'] = df['payee_requested_amount'] - df['payee_settlement_amount']
        return df
    
    def create_dt_feat(self, df: pd.DataFrame) -> pd.DataFrame:
        # Engineer features related to datetime:
        # Convert dt_txn_comp to pandas DateTime format:
        df['dt_txn_comp'] = pd.to_datetime(df.dt_txn_comp)
        df['txn_comp_time'] = pd.to_datetime(df['txn_comp_time'], format="%H:%M:%S")
        # Extract year value from dt_txn_comp column:
        df['year'] = df.dt_txn_comp.dt.year
        # Extract month value from dt_txn_comp column:
        df['month'] = df.dt_txn_comp.dt.month
        # Extract hour of the day value from txn_comp_time:
        df['hour'] = df.txn_comp_time.dt.hour
        df['txn_comp_time'] = df['txn_comp_time'].dt.time
        # Rearrange the columns:
        df = df[['txn_id', 'dt_txn_comp', 'year', 'month', 'txn_comp_time', 'hour', 'txn_type',
                'txn_subtype', 'initiating_channel_id', 'txn_status', 'error_code',
                'payer_psp', 'payee_psp', 'remitter_bank', 'beneficiary_bank',
                'payer_handle', 'payer_app', 'payee_handle', 'payee_app',
                'payee_requested_amount', 'payee_settlement_amount',
                'difference_amount', 'payer_location', 'payer_city', 'payer_state',
                'payee_location', 'payee_city', 'payee_state', 'payer_os_type',
                'payee_os_type', 'beneficiary_mcc_code', 'remitter_mcc_code',
                'custref_transaction_ref', 'cred_type', 'cred_subtype',
                'payer_app_id', 'payee_app_id', 'initiation_mode',
                'dt_time_txn_compl', 'time_of_day']]
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Handle missing value:
        df.fillna(0, inplace= True)
        
        # Engineer various features:
        df = self.get_difference(df)
        df = self.create_dt_feat(df)
        df = self.create_segments(df)

        return df

    @staticmethod
    def targets(txn_subtype: str) -> int:
        return 1 if txn_subtype == FRAUD_TAG else 0
    
    def generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        df['targets'] = df[COL_WITH_FRAUD_TAG].apply(self.targets)
        return df
    
    def generate_fraud_data(self) -> pd.DataFrame:
        df = self.data.copy()
        df = self.preprocess(df= df)
        fraud_df = df[df.COL_WITH_FRAUD_TAG == FRAUD_TAG]
        fraud_df = fraud_df.drop(columns= COL_WITH_FRAUD_TAG)
        return fraud_df
    
    def data_with_targets(self) -> pd.DataFrame:
        df = self.data.copy()
        df = self.preprocess(df= df)
        df_with_targets = self.generate_targets(df= df)
        df_with_targets = df_with_targets.drop(columns= COL_WITH_FRAUD_TAG)
        return df_with_targets
