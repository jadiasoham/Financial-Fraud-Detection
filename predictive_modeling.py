from typing import List, Sequence, Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import logging

logging.basicConfig(level= logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')


class PrepareFeatures:
    def __init__(self, df: pd.DataFrame, features: List[str]) -> None:
        """
        Initializes the PrepareFeatures class with a DataFrame and a list of features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing the data.
        features : List[str]
            A list of feature names including the target as the last column.
        """
        logging.info("Initializing PrepareFeatures class...")
        self.data = self.get_data(df, features)
        self.inputs = self.data.iloc[:, :-1] # Inputs (X) for the machine learning model.
        self.targets = self.data.iloc[:, -1] # Targets (y) for the machine learning model.
        self.cat_features, self.num_features = self.separate_features()
        self.cat_data = self.data[self.cat_features]
        self.num_data = self.data[self.num_features]
        self.scaled_data = self.custom_scaler()
        self.encoded_data = self.cat_encoder()
    
    def get_data(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        if not isinstance(features, list):
            raise ValueError("featres shoud be list of strs...")
        return df[list(features)]

    def separate_features(self) -> Tuple[List[str], List[str]]:
        """
        Separate categorical and numerical features.
        
        Returns:
            cat_features (List[str]): List of categorical feature names.
            num_features (List[str]): List of numerical feature names.
        """
        logging.info("Identifying categorical features...")
        cat_features = self.inputs.select_dtypes(include= ['object', 'category']).columns.tolist()
        logging.info("Identifying numerical features...")
        num_features = self.inputs.select_dtypes(include= ['number']).columns.tolist()
        return cat_features, num_features

    def custom_scaler(self) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Returns:
            scaled_df (pd.DataFrame): DataFrame with scaled numerical features.
        """
        logging.info("Gathering numerical data...")
        data = self.num_data
        logging.info("Initializing StandardScaler...")
        scaler = StandardScaler()
        logging.info("Performing standardization on data...")
        scaled_data = scaler.fit_transform(data)
        logging.info("Converting scaled numpy series object to pandas DataFrame...")
        scaled_df = pd.DataFrame(scaled_data, columns= self.num_features)
        return scaled_df
    
    def cat_encoder(self, method: str= 'onehot', **kwargs) -> pd.DataFrame:
        """
        Encodes categorical features using the specified method.
        
        Parameters:
        -----------
        method : str, optional
            The method for encoding categorical features. Can be 'onehot' or 'ordinal'. Default is 'onehot'.
        **kwargs : dict, optional
            drop (str, default=None): Specifies which categories to drop in OneHotEncoder.
            Can be 'first', 'if_binary' or array-like of shape (n_featuers,).
            Only to be passed if method= 'onehot'
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with encoded categorical features.
        
        Raises:
        -------
        ValueError
            If the provided method is not 'onehot' or 'ordinal'.
        """
        logging.info("Gathering categorical data...")
        data = self.cat_data
        drop = kwargs.get('drop', None)
        if method.lower() == 'onehot':
            logging.info("Initializing OneHotEncoder...")
            encdoer = OneHotEncoder(sparse_output= False, drop= drop)
            logging.info(f"Encoding data with OneHotEncoding and drop= {drop}...")
            encoded_cat_data = encdoer.fit_transform(data)
            encoded_cat_df = pd.DataFrame(encoded_cat_data, columns= encdoer.get_feature_names_out(self.cat_features), index= self.data.index)
        elif method.lower() == 'ordinal':
            logging.info("Initializing OrdinalEncoder...")
            encoder = OrdinalEncoder()
            logging.info("Encoding categorical data with OrdinalEncoder...")
            encoded_cat_data = encoder.fit_transform(data)
            encoded_cat_df = pd.DataFrame(encoded_cat_data, columns= self.cat_features, index= self.data.index)
        else:
            logging.error(f"The method name should be one from 'onehot' or 'ordinal' but '{method}' passed.")
            raise ValueError(f"The method name should be one from 'onehot' or 'ordinal' but '{method}' passed.")
        return encoded_cat_df
    
    def transform(self, cat_encoding_method: str = 'onehot', **kwargs):
        """
        Transforms the data by scaling numerical features and encoding categorical features.
        
        Parameters:
        -----------
        cat_encoding_method : str, optional
            The method for encoding categorical features. Can be 'onehot' or 'ordinal'. Default is 'onehot'.
        **kwargs : dict, optional
            drop (str, default=None): Specifies which categories to drop in OneHotEncoder.
            Can be 'first', 'if_binary' or array-like of shape (n_featuers,).
            Only to be passed if method= 'onehot'
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with transformed features.
        """
        drop = kwargs.get('drop', None)
        self.encoded_data = self.cat_encoder(cat_encoding_method, drop= drop)
        logging.info("Combining standardized numerical and encoded categorical data into a single DataFrame...")
        transformed_df = pd.concat([self.scaled_data, self.encoded_data], axis= 1)
        logging.info(f"DataFrame of shape: {transformed_df.shape} returned...")
        return transformed_df