# Introduction to the Fraud Analytics Dataset
 The dataset used in this analysis is sourced from an Excel file named `Fraud Analytics Dataset.xlsx`. The dataset comprise 55672 rows and 36 columns.
 This dataset contains transaction data relevant to fraud detection and analytics. Each row represents a unique transaction and includes various attributes related to the transaction details, parties involved, and their respective information. Below is a brief overview of the key columns in the dataset:

- `txn_id`: Unique identifier for each transaction.
- `dt_txn_comp`: Date when the transaction was completed.
- `txn_comp_time`: Time when the transaction was completed.
- `txn_type`: Type of transaction (e.g., payment, transfer).
- `txn_subtype`: Subtype of the transaction (e.g., Fraudulent Transaction, Normal Transaction).
- `initiating_channel_id`: Identifier for the channel initiating the transaction.
- `txn_status`: Status of the transaction (e.g., completed, pending).
- `error_code`: Error code associated with the transaction, if any.
- `payer_psp`: Payment Service Provider of the payer.
- `payee_psp`: Payment Service Provider of the payee.
- `remitter_bank`: Bank of the remitter.
- `beneficiary_bank`: Bank of the beneficiary.
- `payer_handle`: Unique handle of the payer.
- `payer_app`: Application used by the payer.
- `payee_handle`: Unique handle of the payee.
- `payee_app`: Application used by the payee.
- `payee_requested_amount`: Amount requested by the payee.
- `payee_settlement_amount`: Amount settled by the payee.
- `payer_location`: Location of the payer.
- `payer_city`: City of the payer.
- `payer_state`: State of the payer.
- `payee_location`: Location of the payee.
- `payee_city`: City of the payee.
- `payee_state`: State of the payee.
- `payer_os_type`: Operating system used by the payer.
- `payee_os_type`: Operating system used by the payee.
- `beneficiary_mcc_code`: Merchant Category Code for the beneficiary.
- `remitter_mcc_code`: Merchant Category Code for the remitter.
- `custref_transaction_ref`: Customer reference transaction identifier.
- `cred_type`: Type of credential used.
- `cred_subtype`: Subtype of the credential used.
- `payer_app_id`: Identifier of the payer's application.
- `payee_app_id`: Identifier of the payee's application.
- `initiation_mode`: Mode of transaction initiation.
- `dt_time_txn_compl`: Date and time of transaction completion.
- `time_of_day`: Time of day when the transaction was completed.
---
### The objective of this analysis is to clean and preprocess the data, engineer relevant features, and prepare it for further analysis and predictive modeling, particularly focusing on identifying fraudulent transactions.
---
## Usage Instructions

- **Environment Setup**:
  ```sh
  pip install -r requirements.txt
---
## Acknowledgments

- **Dataset Source**: The Excel file containing the dataset was sourced from [Super AI](https://www.getsuper.ai/post/online-fraud-analytics-python-use-case).
