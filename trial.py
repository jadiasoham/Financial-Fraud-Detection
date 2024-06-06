"""
trial.py: A script to test the functionality of fraud_data_analytics.py

This script aims to test the functionality of the fraud_data_analytics.py module, which contains classes for preprocessing fraud data and visualizing fraud-related insights.

Usage:
1. Import all classes from the fraud_data_analytics module using:
    ```
    from fraud_data_analytics import *
    ```

2. Consecutively, write test codes to assess the functionality of the Preprocessing and Visualization classes. These test codes can include:
    - Loading a sample dataset
    - Instantiating the Preprocessing and Visualization classes
    - Calling methods from these classes to preprocess data and generate visualizations
    - Displaying or saving the generated visualizations
    - Analyzing the results to ensure they align with expectations

Purpose:
The purpose of trial.py is to verify that the Preprocessing and Visualization classes within the fraud_data_analytics module are functioning correctly. By running this script, users can ensure that the data preprocessing and visualization techniques implemented in fraud_data_analytics.py produce the expected outputs and insights when applied to real or simulated fraud transaction data.

Expected Output:
Upon execution, trial.py should:
- Successfully import classes from the fraud_data_analytics module
- Execute test codes without errors, demonstrating the functionality of the Preprocessing and Visualization classes
- Produce visualizations and/or output that can be analyzed to verify the correctness of the implemented techniques
"""

from fraud_data_analytics import Preprocessing, Visualization

def test_preprocessing(dataset_path):
    # Instantiate the preprocessing path:
    preprocessor = Preprocessing(file_path= dataset_path)

    # Perform preprocessing on the dataset:
    df = preprocessor.data
    preprocessed_data = preprocessor.preprocess(df)
    # display for confirmation:
    preprocessed_data.head()

def test_visualization(dataset_path):
    # Instantiate the preprocessing:
    preprocessor = Preprocessing(file_path= dataset_path)

    # Obtain fraudulent_transaction data:
    fraud_df = preprocessor.generate_fraud_data()
    # display for confirmation:
    fraud_df.head()

    # Instantiate the visualization class:
    visualizer = Visualization(df= fraud_df)

    # Create all the plots
    visualizer.create_plots()
    # Save all the plots to an html file
    visualizer.save_plots(filename= "Visualization.html")

# Run the tests:
if __name__ == '__main__':
    df = r'datasets\Fraud Analytics Dataset.xlsx'
    test_preprocessing(dataset_path= df)
    test_visualization(dataset_path= df)
    print("Tests completed successfully!")
