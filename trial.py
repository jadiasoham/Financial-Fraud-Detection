from typing import Sequence, Optional, List, Dict, Union
import pathlib
import calendar
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from plotly import express as px
from plotly import offline as pyo
from plotly import graph_objects as go
from plotly import io
import logging

logging.basicConfig(level= logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')

# Constants:
FRAUD_TAG = 'Fraudulent Transaction' # Identification tag for fraudulent transactions
COL_WITH_FRAUD_TAG = 'txn_subtype' # Name of the column containing the `FRAUD_TAG`.


class Preprocessing:
    def __init__(self, file_path: str) -> None:
        self.file = file_path
        self.data = self.open_file()
        self.fraud_data = self.generate_fraud_data()
        self.labelled_data = self.data_with_targets()

    def open_file(self) -> pd.DataFrame:
        file = self.file
        extension = pathlib.Path(file).suffix
        try:
            if extension in ['.xlsx', 'xls']:
                df = pd.read_excel(file)
            elif extension == '.csv':
                df = pd.read_excel(file)
        except Exception as e:
            logging.error(f"Error occured while opening file: {e}.")
            print(f"Execption occured while opening file: {e}.")
    
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
    

class Visualization:
    def __init__(self, df: pd.DataFrame) -> None:
        self.data = df
        self.df_viz_dict = self.prepare_df()
        self.plots = self.create_plots()

    @staticmethod
    def line_graph(
        df: pd.DataFrame, x: str, y: str,
        title: Optional[str] = None,
        xlabel: Optional[str] = None, ylabel: Optional[str] = None,
        xticks: Optional[Dict] = None, yticks: Optional[Dict] = None,
        color: Optional[str] = None, marker: bool = True
    ) -> None:
        if not color:
            fig = px.line(data_frame= df, x= x, y= y, title= title, markers= marker)
        fig = px.line(data_frame=df, x= x, y= y, title= title, color= color, markers= marker)
        if xlabel:
            fig.update_layout(xaxis_title= xlabel)
        if ylabel:
            fig.update_layout(yaxis_title= ylabel)
        if xticks:
            tickvals = xticks.get('tickvals')
            labels = xticks.get('labels', tickvals) if tickvals else None
            if tickvals:
                fig.update_xaxes(tickvals= tickvals, ticktext= labels)
            if 'rotation' in xticks:
                fig.update_xaxes(tickangle= xticks['rotation'])

        if yticks:
            tickvals = yticks.get('tickvals')
            labels = yticks.get('labels', tickvals) if tickvals else None
            if tickvals:
                fig.update_yaxes(tickvals= tickvals, ticktext= labels)
            if 'rotation' in yticks:
                fig.update_yaxes(tickangle= yticks['rotation'])

        fig.show()

    @staticmethod
    def bar_graph(
        dataset: pd.DataFrame, x: str, y: str,
        title: Optional[str] = None,
        xlabel: Optional[str] = None, ylabel: Optional[str] = None,
        xticks: Optional[Dict] = None, yticks: Optional[Dict] = None,
        color: Optional[str] = None
    ) -> None:
        if not color:
            fig = px.bar(data_frame=dataset, x=x, y=y, title=title)
        fig = px.bar(data_frame=dataset, x=x, y=y, title=title, color=color)
        if xlabel:
            fig.update_layout(xaxis_title=xlabel)
        if ylabel:
            fig.update_layout(yaxis_title=ylabel)

        if xticks:
            tickvals = xticks.get('tickvals')
            labels = xticks.get('labels', tickvals) if tickvals else None
            if tickvals:
                fig.update_xaxes(tickvals=tickvals, ticktext=labels)
            if 'rotation' in xticks:
                fig.update_xaxes(tickangle=xticks['rotation'])

        if yticks:
            tickvals = yticks.get('tickvals')
            labels = yticks.get('labels', tickvals) if tickvals else None
            if tickvals:
                fig.update_yaxes(tickvals=tickvals, ticktext=labels)
            if 'rotation' in yticks:
                fig.update_yaxes(tickangle=yticks['rotation'])

        fig.show()

    def prepare_df(self) -> Dict[str, pd.DataFrame]:
        fraudulent_data = self.data
        segments = ['EarlyMorning', 'Morning', 'LateMorning',
                    'Afternoon', 'LateAfternoon', 'Evening',
                    'Night', 'LateNight']
        # (loss_amt vs. payee_state)/ 2 plots
        # df1:
        df1 = fraudulent_data.groupby('payee_state')['payee_settlement_amount'].sum().reset_index()
        df1['loss_amount_(in_lakhs)'] = np.round(df1.payee_settlement_amount / 1e5, 2)
        df1.drop(columns='payee_settlement_amount', inplace=True)
        df1.sort_values(by='loss_amount_(in_lakhs)', ascending=False, inplace=True)
        # (loss amount monthly trends separately and cumulatively)/ 4 plots
        # df2:
        df2 = fraudulent_data.groupby(['year', 'month'])['payee_settlement_amount'].sum().reset_index()
        df2['loss_amt_(in_lakhs)'] = np.round(df2['payee_settlement_amount'] / 1e5, 2)
        df2.drop(columns='payee_settlement_amount', inplace=True)
        df2.sort_values(by='loss_amt_(in_lakhs)', ascending=False)
        # df3:
        df3 = fraudulent_data.groupby('month')['payee_settlement_amount'].sum().reset_index()
        df3['loss_amt_(in_lakhs)'] = np.round(df3['payee_settlement_amount'] / 1e5, 2)
        df3.drop(columns='payee_settlement_amount', inplace=True)
        df3.sort_values(by='loss_amt_(in_lakhs)', ascending=False)
        # df4:
        df4 = fraudulent_data.groupby(['year', 'month']).size().reset_index(name='fraud_counts')
        # df5:
        df5 = fraudulent_data.groupby('month').size().reset_index(name='fraud_counts')
        # (loss amount by credit account type)/ 1 plot
        # df6:
        df6 = fraudulent_data.groupby('cred_type')['payee_settlement_amount'].sum().reset_index()
        df6['loss_amt_(in_lakhs)'] = np.round(df6.payee_settlement_amount / 1e5, 2)
        df6.drop(columns='payee_settlement_amount', inplace=True)
        # (loss amount by time of day)/ 3 plots
        # df7:
        df7 = fraudulent_data.groupby('time_of_day')['payee_settlement_amount'].sum().reset_index()
        df7['loss_amt_(in_lakhs)'] = np.round(df7.payee_settlement_amount / 1e5, 2)
        df7.drop(columns='payee_settlement_amount', inplace=True)
        df7['time_of_day'] = pd.Categorical(df7['time_of_day'], categories=segments, ordered=True)
        df7.sort_values('time_of_day', inplace=True)
        # df8:
        df8 = fraudulent_data.groupby('time_of_day')['difference_amount'].sum().reset_index()
        df8['diff_amt_(in_lakhs)'] = np.round(df8['difference_amount'] / 1e5, 2)
        df8.drop(columns='difference_amount', inplace=True)
        df8['time_of_day'] = pd.Categorical(df8['time_of_day'], categories=segments, ordered=True)
        df8.sort_values('time_of_day', inplace=True)
        # df9:
        underpayments = fraudulent_data[fraudulent_data.difference_amount > 0].groupby('time_of_day')[
            'difference_amount'].sum().reset_index()
        underpayments['difference_amount_(in_lakhs)'] = np.round(underpayments.difference_amount / 1e5, 3)
        underpayments.drop(columns=['difference_amount'], inplace=True)

        overpayments = fraudulent_data[fraudulent_data.difference_amount < 0].groupby('time_of_day')[
            'difference_amount'].sum().reset_index()
        overpayments['difference_amount_(in_lakhs)'] = np.round(overpayments.difference_amount / 1e5, 3)
        overpayments.drop(columns=['difference_amount'], inplace=True)

        underpayments.rename(columns={'difference_amount_(in_lakhs)': 'underpayments'}, inplace=True)
        overpayments.rename(columns={'difference_amount_(in_lakhs)': 'overpayments'}, inplace=True)

        df9 = pd.merge(underpayments, overpayments, on='time_of_day', how='outer').fillna(0)

        df9['time_of_day'] = pd.Categorical(df9['time_of_day'], categories=segments, ordered=True)
        df9.sort_values('time_of_day', inplace=True)
        viz_dict = {'df1': df1, 'df2': df2, 'df3': df3,
                    'df4': df4, 'df5': df5, 'df6': df6,
                    'df7': df7, 'df8': df8, 'df9': df9,
                    'segments': segments}
        return viz_dict
        
    def create_plots(self) -> List[go.Figure]:
        logging.info("Extracting relevant dataframes from master dataframe.")
        df_dict = self.df_viz_dict
        df1 = df_dict["df1"]
        df2 = df_dict["df2"]
        df3 = df_dict["df3"]
        df4 = df_dict["df4"]
        df5 = df_dict["df5"]
        df6 = df_dict["df6"]
        df7 = df_dict["df7"]
        df8 = df_dict["df8"]
        df9 = df_dict["df9"]
        segments = df_dict["segments"]

        logging.info("Initializing a list to store all the plots.")
        figures = []
        # plot 1:
        logging.info("Creating plot 1")
        fig1 = px.bar(data_frame=df1,
                      x='loss_amount_(in_lakhs)',
                      y='payee_state',
                      color='loss_amount_(in_lakhs)',
                      title='Loss Amount By States')
        fig1.update_layout(xaxis_title='Loss Amount (Lakh Rs.)', yaxis_title='Payee State')
        figures.append(fig1)
        # Plot 2
        logging.info("Creating plot 2")
        df1_top10 = df1.nlargest(n=10, columns='loss_amount_(in_lakhs)').sort_values(by='loss_amount_(in_lakhs)',
                                                                                     ascending=False)
        fig2 = px.bar(data_frame=df1_top10,
                      x='loss_amount_(in_lakhs)',
                      y='payee_state',
                      color='loss_amount_(in_lakhs)',
                      title='Top 10 States By Loss Amount')
        fig2.update_layout(xaxis_title='Loss Amount (Lakh Rs.)', yaxis_title='Payee State')
        figures.append(fig2)
        # Plot 3
        logging.info("Creating plot 3")
        fig3 = px.line(data_frame=df2,
                       x='month', y='loss_amt_(in_lakhs)',
                       title='Loss Amount Monthly Trend (for each Year)',
                       color='year', markers=True)
        fig3.update_layout(xaxis_title='Month', yaxis_title='Loss Amount (Lakh Rs.)')
        fig3.update_xaxes(tickvals=df2['month'], ticktext=calendar.month_abbr[1:13])
        figures.append(fig3)
        # Plot 4
        logging.info("Creating plot 4")
        fig4 = px.line(data_frame=df3,
                       x='month', y='loss_amt_(in_lakhs)',
                       title='Loss Amount Monthly Trend (Cumulative)',
                       markers=True)
        fig4.update_layout(xaxis_title='Month', yaxis_title='Loss Amount (Lakh Rs.)')
        fig4.update_xaxes(tickvals=df3['month'], ticktext=calendar.month_abbr[1:13])
        figures.append(fig4)
        # Plot 5
        logging.info("Creating plot 5")
        fig5 = px.line(data_frame=df4,
                       x='month', y='fraud_counts',
                       title='Fraud Incidents Monthly Trend (for each Year)',
                       color='year', markers=True)
        fig5.update_layout(xaxis_title='Months', yaxis_title='Fraud Incidents')
        fig5.update_xaxes(tickvals=df4['month'], ticktext=calendar.month_abbr[1:13])
        figures.append(fig5)
        # Plot 6
        logging.info("Creating plot 6")
        fig6 = px.line(data_frame=df5,
                       x='month', y='fraud_counts',
                       title='Fraud Incidents Monthly Trend (Cumulative)',
                       markers=True)
        fig6.update_layout(xaxis_title='Month', yaxis_title='Fraud Incidents')
        fig6.update_xaxes(tickvals=df5['month'], ticktext=calendar.month_abbr[1:13])
        figures.append(fig6)
        # Plot 7
        logging.info("Creating plot 7")
        fig7 = px.bar(data_frame=df6,
                      x='cred_type', y='loss_amt_(in_lakhs)',
                      title='Loss Amount by Credit type',
                      color='loss_amt_(in_lakhs)')
        fig7.update_layout(xaxis_title='Credit Account Type', yaxis_title='Loss Amount (Lakh Rs.)')
        figures.append(fig7)
        # Plot 8
        logging.info("Creating plot 8")
        fig8 = px.bar(data_frame=df7,
                      x='time_of_day', y='loss_amt_(in_lakhs)',
                      title='Fraudulent Transactions by Time of Day',
                      color='loss_amt_(in_lakhs)')
        fig8.update_layout(xaxis_title='Time Of Day', yaxis_title='Loss Amount (Lakh Rs.)')
        fig8.update_xaxes(tickvals=segments, ticktext=segments)
        figures.append(fig8)
        # Plot 9
        logging.info("Creating plot 9")
        fig9 = px.bar(data_frame=df8,
                      x='time_of_day', y='diff_amt_(in_lakhs)',
                      title='Difference Amount by Time of Day',
                      color='diff_amt_(in_lakhs)')
        fig9.update_layout(xaxis_title='Time Of Day', yaxis_title='Difference Amount (Lakh Rs.)')
        fig9.update_xaxes(tickvals=segments, ticktext=segments)
        figures.append(fig9)
        # Plot 10
        logging.info("Creating plot 10")
        trace1 = go.Bar(
            x=df9['time_of_day'],
            y=df9['underpayments'],
            name='Underpayments in Lakhs',
            marker=dict(color='red')
        )
        trace2 = go.Bar(
            x=df9['time_of_day'],
            y=df9['overpayments'],
            name='Overpayments in Lakhs',
            marker=dict(color='blue')
        )
        layout = go.Layout(
            title='Underpayments and Overpayments by Time of Day',
            xaxis=dict(title='Time of Day'),
            yaxis=dict(title='Amount (in Lakhs)'),
            barmode='group'
        )
        fig10 = go.Figure(data=[trace1, trace2], layout=layout)
        figures.append(fig10)
        logging.info("All plots created and saved to list.")
        return figures
    
    def save_plots(self, filename: str) -> None:
        res_html = """
        <html>
        <head>
            <title>Multiple Plots</title>
        </head>
        <body>
        """
        for fig in self.plots:
            fig_html = io.to_html(fig, full_html= False)
            res_html += fig_html

        res_html += """
        </body>
        </html>
        """

        with open(filename, encoding= 'utf-8') as file:
            file.write(res_html)
        print(f"All plots saved in {filename}.")
        logging.info(f"All plots saved in {filename}.")


class PrepareFeatures:
    def __init__(self) -> None:
        pass