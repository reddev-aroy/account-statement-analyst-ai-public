import streamlit as st
from st_aggrid import AgGrid
import pandas as pd
# from streamlit_gsheets import GSheetsConnection
from gspread_dataframe import set_with_dataframe, get_as_dataframe
import gspread
from streamlit_option_menu import option_menu

import os
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
import pandas as pd
from pandasai import SmartDataframe
import streamlit as st
from st_aggrid import AgGrid
# from streamlit_gsheets import GSheetsConnection
import gspread
import sqlite3
from sqlite3 import Error
import re
import numpy as np
import glob
from datetime import datetime, timedelta
import time
from time import sleep
import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders
from dotenv import load_dotenv
from googledr_package.googledr import get_final_dataframe, upload_to_drive, downloadFile, getCredentials, findFolderIdByName, findFileIdByName, upload_account_statement_db
from googleapiclient.discovery import build
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import decimal
from streamlit_extras.colored_header import colored_header
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from google.oauth2.service_account import Credentials

load_dotenv()
llm = ChatGroq(model='llama3-70b-8192', temperature=0.9, api_key=os.getenv('GROQ_API_KEY'))


st.set_page_config(layout="wide")
# Set pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

# conn = st.connection("gsheets", type=GSheetsConnection)

# data = conn.read(worksheet="FinalDataframe", usecols=list(range(5)))
# st.dataframe(data)

# gc = gspread.oauth(credentials_filename='config/service_account.json')
# sh = gc.open("Account_Statements")


scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

credentials = Credentials.from_service_account_file(
    'config/service_account.json',
    scopes=scopes
)

gc = gspread.authorize(credentials)
sh = gc.open("Account_Statements")

class EmailSender:
    def __init__(self, SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_SENDER):
        self.SMTP_SERVER = SMTP_SERVER
        self.SMTP_PORT = SMTP_PORT
        self.SMTP_USERNAME = SMTP_USERNAME
        self.SMTP_PASSWORD = SMTP_PASSWORD
        self.SMTP_SENDER = SMTP_SENDER

    def send_email_with_attachment(self, recipients, file_name, email_subject, email_body, attachment_bytes=None):
        for recipient in recipients:
            msg = MIMEMultipart()
            msg['From'] = self.SMTP_SENDER
            msg['To'] = recipient
            msg['Subject'] = email_subject

            # Attach the body text
            body = email_body
            
            msg.attach(MIMEText(body, 'plain'))

            if attachment_bytes:
                # Create a MIMEBase object for the attachment
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment_bytes)
                encoders.encode_base64(part)
                # Specify the filename for the attachment
                part.add_header("Content-Disposition", f"attachment; filename={file_name}")
                msg.attach(part)


            retry_count = 0
            max_retries = 3
            sleep_time = 10 # 5 seconds between each retry

            while retry_count < max_retries:
                try:
                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL(self.SMTP_SERVER, self.SMTP_PORT, context=context) as server:
                        server.login(self.SMTP_USERNAME, self.SMTP_PASSWORD)
                        server.sendmail(self.SMTP_SENDER, recipient.split(','), msg.as_string())
                        break
                except Exception as e:
                    print(f"Failed to send email to {recipient}: {e}")  # Change this to logging or any other error handling mechanism if needed
                    logging.error(f"Failed to send email to {recipient}: {e}")
                    sleep(sleep_time)
                    retry_count += 1

            if retry_count == max_retries:
                print(f"Max retries reached. Failed to send email to {recipient}")
                logging.error(f"Max retries reached. Failed to send email to {recipient}")

    def send_email(self, recipients, file_name, subject, message, attachment_bytes=None):
        try:
            self.send_email_with_attachment(recipients, file_name, subject, message, attachment_bytes)
        except Exception as e:
            print(f"Error sending email: {e}")
            # raise e

def categorize_transactions(transaction_names, llm):
    print(f"Categorizing transactions: {transaction_names}")
    # print(f"Categorizing transactions: {transaction_names}")
    response = llm.invoke(f"""Can you provide output in the below required format for the provided TRANSACTIONS. I will provide you with the transactions and you need to provide me with the formatted output. The transaction scenario examples are as follows:
    
    Scenario 1: if you see something like this: BY TRANSFER/NEFT MINI MICRO HYDEL      SBIN124093020577
    you should output BY TRANSFER/NEFT MINI MICRO HYDEL      SBIN124093020577 -- NEFT -- CREDIT -- MINI MICRO HYDEL -- SBIN124093020577
    This is because BY TRANSFER means it is a debit transaction and also it states it is NEFT and followed by the company name and account number. But your output should be BY TRANSFER/NEFT MINI MICRO HYDEL -- NEFT -- DEBIT -- MINI MICRO HYDEL -- SBIN124093020577. Follow the exact format for this type of transactions.
    
    Scenario 2: If you see something like this: TO TRANSFER/NEFT Sabyasachi Ghosh      CBINI24100591824
    In the above case, TO TRANSFER means it is CREDIT transaction, followed by type of transaction that is NEFT followed by Name of the account holder and their account number. In this case you should output TO TRANSFER/NEFT Sabyasachi Ghosh      CBINI24100591824 -- NEFT -- DEBIT -- Sabyasachi Ghosh -- CBINI24100591824
    
    Another scenario - CASH CHEQUE/Paid to SIBU PRASAD PASI
    If you see anything like this in this case you should output: CASH CHEQUE/Paid to SIBU PRASAD PASI -- CASH CHEQUE -- DEBIT -- SIBU PRASAD PASI -- CHQ
    
    Another scenario - CAS PRES CHQ/063178STAR SYSTEMS        H D F C BANK LTD
    If you see anything like this then the output should be: CAS PRES CHQ/063178STAR SYSTEMS        H D F C BANK LTD -- CASH CHEQUE -- DEBIT -- STAR SYSTEMS -- 06317
    
    Another scenario - BY CLEARING / CHEQUE/ZETADEL TECHNOLOGIES PVT  AXIS BANK LTD.(AXS)/MITRA                                   0000000377
    In this case output should be: BY CLEARING / CHEQUE/ZETADEL TECHNOLOGIES PVT  AXIS BANK LTD.(AXS)/MITRA                                   0000000377 -- CHEQUE -- CREDIT -- ZETADEL TECHNOLOGIES PVT  AXIS BANK LTD -- 0000000377
    
    Another scenario: TO TRANSFER/308 11 45 53/BILLPAYMENT/WBSEDCL
    The output of something like above should be: TO TRANSFER/308 11 45 53/BILLPAYMENT/WBSEDCL -- Online -- DEBIT -- BILLPAYMENT WBSEDCL -- 308114553
    
    Another Scenario: TO TRANSFER/GSTN-24041900194267
    The output of above should be: TO TRANSFER/GSTN-24041900194267 -- GSTN -- DEBIT -- GST -- GSTN-24041900194267
    
    Another Scenario: TO TRANSFER/312375991/PAYU/NA
    The output of above should be: TO TRANSFER/312375991/PAYU/NA -- Online -- DEBIT -- PAYU -- 312375991
    
    If you see anything else, you can analyze and decide what the output should be but there should always be four "-" for each transaction.
    
    The Rule of thumb - Firstly if the transaction starts with "BY TRANSFER" or "BY CLEARING", it means it is a CREDIT transaction. So if you consider 1 -- 2 -- 3 -- 4 -- 5 then it will be CREDIT in place of 2 for example: 1 -- 2 -- CREDIT -- 4 -- 5 always in the output. Also if transactions start with "TO TRANSFER" then it means it is a DEBIT transaction and will be DEBIT in place of 3 for example: 1 -- 2 -- DEBIT -- 4 -- 5.
    Also the general output format should be: Original_Input_Transaction -- Transfer_Type -- Transaction_Type -- Account_Name -- Reference_Number. This is just a guideline or a rule of thumb for you. Rember it is vital that the first place should always contain the original input transaction that you are taking for formatting in place of Original_Input_Transaction
    
    You will only generate the output without saying anything else. Also do not provide any numbering to your output like 1., 2.,... and it should be in each line by line.
    
    TRANSACTIONS: """ + transaction_names).content
    # response = response.split('\n')
    print(response.split('\n'))

    #
    transaction_response = response.strip().split('\n')
    # Function to process each transaction
    def process_transaction(transaction):
        # Splitting the transaction into components
        components = transaction.split('--')
        
        if len(components) < 5:
            # Handle the case where the transaction doesn't match the expected format
            # For example, you could return a default value or log a warning
            print(f"Warning: Transaction '{transaction}' does not match the expected format.")
            return [None, None, None, None, None]
        
        # Assuming the structure is always the same, adjust as necessary
        transaction_description = components[0]
        transfer_type = components[1]
        transaction_type = components[2]
        account_name = components[3]
        reference_number = components[4]
        # Further processing can be done here if needed
        return [transaction_description, transfer_type, transaction_type, account_name, reference_number]
    # Processing each transaction
    processed_transactions = [process_transaction(transaction) for transaction in transaction_response]

    # Creating a DataFrame from the processed transactions
    df = pd.DataFrame(processed_transactions, columns=['Transaction Description', 'Transfer Type', 'Transaction Type', 'Account Name', 'Reference Number'])
    # Drop rows with NaN values
    df_cleaned = df.dropna()
    # Displaying the DataFrame
    print(df_cleaned)
    
    #
    
    # Put in dataframe
    # categories_df = pd.DataFrame({'Transaction vs category': response.split('\n')})
    # categories_df = categories_df.apply(lambda x: x.apply(strip_newlines))
    # categories_df.dropna(how='all', axis=0, inplace=True)
    # categories_df[['Transaction Description', 'Transfer Type', 'Transaction Type', 'Account Name', 'Reference Number']] = categories_df['Transaction vs category'].str.split(' -- ', expand=True)
    print(f"\n\ncategories DF: \n{df_cleaned}")
    return df_cleaned

# Get index list
#https://stackoverflow.com/questions/47518609/for-loop-range-and-interval-how-to-include-last-step
def hop(start, stop, step):
    for i in range(start, stop, step):
        yield i
    yield stop


@st.cache_data
def process_transactions_with_cache(final_df):
    index_list = list(hop(0, len(final_df), 30))
    categorize_transactions_cache = {}
    categories_df_all = pd.DataFrame()
    for i in range(len(index_list) - 1):
        print(f"Processing transactions from {index_list[i]} to {index_list[i+1]}...")
        # print(f"Processing transactions from {index_list[i]} to {index_list[i+1]}...")
        transaction_names = ','.join(final_df['Account Description'].iloc[index_list[i]:index_list[i+1]])
        categories_df = categorize_transactions(transaction_names, llm)   
        categories_df_all = pd.concat([categories_df_all, categories_df], ignore_index=True)
    
    return categories_df_all

def calculate_grand_df(final_df, categories_df_all):
    # Your existing logic to calculate grand_df goes here
    # For example:
    selected_columns_df1 = final_df[['Post Date', 'Branch Code', 'Cheque Number', 'Debit', 'Credit', 'Balance']]
    selected_columns_df2 = categories_df_all[['Transaction Description', 'Transfer Type', 'Transaction Type', 'Account Name', 'Reference Number']]

    grand_df = pd.concat([selected_columns_df1, selected_columns_df2], axis=1)
    new_column_order = ['Post Date', 'Branch Code', 'Cheque Number', 'Transaction Description', 'Account Name', 'Transfer Type', 'Transaction Type', 'Debit', 'Credit', 'Balance', 'Reference Number']
    grand_df = grand_df.reindex(columns=new_column_order)
    
    return grand_df


def trim_all_columns(df):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x
    return df.applymap(trim_strings)

# Function to handle blank strings and convert to float, rounding to 2 decimal places
def handle_blank_and_convert(value):
    if pd.isna(value) or value.strip() == '':
        return None # or 0.0 if you prefer to fill with zeros
    else:
        return round(float(value), 2)

def format_date(date):
    formatted_date = f"{date.day}th {date.strftime('%B')} {date.year}"

    # Correct the day suffix for the first day
    if date.day in [1, 21, 31]:
        formatted_date = formatted_date.replace("th", "st")
    if date.day in [2, 22]:
        formatted_date = formatted_date.replace("th", "nd")
    if date.day in [3, 23]:
        formatted_date = formatted_date.replace("th", "rd")
    
    return formatted_date

def visualize():
    
    # st.markdown(f"## Visualize")
    st.title("Dashboard")
    
    worksheet = sh.worksheet('Accounts')
    gs_df = pd.DataFrame(worksheet.get_all_records())
    
    if gs_df.empty:
        st.info("No data in Databse. Please upload your Bank Statements.")
    
    else:
        # Assuming 'df' is your DataFrame loaded with bank statements
        # gs_df['Balance'] = gs_df['Balance'].str.replace(' CR', '').astype(float)  # 2024-Jun-15 (error)
        gs_df['Balance'] = pd.to_numeric(gs_df['Balance'].str.replace(' CR', ''), errors='coerce').fillna(0).astype(float)  # Convert to float
        
        # Get the current date
        current_date = datetime.now().date()

        desc_order = gs_df.sort_values(by='Post Date', ascending=False)
        # print(f"FIRST RECORD BALANCE - \n\n{desc_order}")
        first_record_balance = desc_order.iloc[0]['Balance']
        # Format the balance in Indian Rupee format
        formatted_balance = currency_in_indian_format(first_record_balance)
        
        
        # first_record_post_date = gs_df.sort_values(by='Post Date', ascending=False)
        first_record_post_date_str = desc_order.iloc[0]['Post Date']
        first_record_post_date = pd.to_datetime(first_record_post_date_str)
        formatted_post_date = format_date(first_record_post_date)
        # Display total balance
        # st.divider()
        st.markdown(f'<h3 style="text-align: left; font-weight: normal;">Total Balance as of {formatted_post_date} - <span style="color: MediumSeaGreen; font-weight: bold;"> Rs. {formatted_balance}/- </span></h3>', unsafe_allow_html=True)
        # st.divider()
        # st.markdown(f"#### Total Balance as of {formatted_post_date} - **Rs. {formatted_balance}/-**")
        # st.dataframe(filtered_df)
        # AgGrid(filtered_df)
        
        # Create date picker widgets for start and end dates
        # col1, col2 = st.columns([1,1])
        # with col1:
        st.sidebar.markdown(f"### *Select a Start & End Date*")
        default_start_date = current_date - timedelta(days=30)
        start_date = st.sidebar.date_input("Start Date", value=default_start_date, max_value=current_date)
        # with col2:
        end_date = st.sidebar.date_input("End Date", max_value=current_date)
        
        st.sidebar.divider()
        
        if start_date > end_date:
            warn = st.warning("Start date cannot be greater than the end date. Please correct the dates.")
            st.markdown("## 👈🏻 Select a Date Range")
            time.sleep(5) # Wait for 3 seconds
            warn.empty()
        else:
            # Convert 'Post Date' to datetime and then to date format to match start_date and end_date
            gs_df['Post Date'] = pd.to_datetime(gs_df['Post Date']).dt.date
            # print(f"gs df - \n{gs_df.head(5)}")
            # print(f"gs df - \n{gs_df.dtypes}")
            
            # Filter the DataFrame based on the selected date range
            filtered_df = gs_df[(gs_df['Post Date'] >= start_date) & (gs_df['Post Date'] <= end_date)]
            # print(filtered_df.head())
            oldest_date = None
            latest_date = None
            # Convert 'Post Date' to datetime format
            date_df = filtered_df['Post Date']
            date_df = pd.to_datetime(date_df)
            
            # Calculate oldest_date and latest_date
            oldest_date = date_df.min()
            latest_date = date_df.max()
            default_columns = ["Post Date", "Account Name", "Transfer Type", "Transaction Type", "Credit", "Debit", "Balance"]
            
            default_columns = [col for col in default_columns if col in filtered_df.columns]
            
            if pd.isna(oldest_date) or pd.isna(latest_date):
                    alert = st.warning("No data available for the selected date range.")
                    time.sleep(5)
                    alert.empty()
            else:
                if oldest_date and latest_date:
                    
                    formatted_oldest_date = format_date(oldest_date)
                    formatted_latest_date = format_date(latest_date)

                    st.markdown(f"##### *{formatted_oldest_date} - {formatted_latest_date}*", unsafe_allow_html=True)
                    
                selected_columns = st.multiselect("Select columns to display", options=gs_df.columns, default=default_columns)
                if 'Post Date' in selected_columns:
                    # Move 'Post Date' to the beginning of the list
                    selected_columns.remove('Post Date')
                    selected_columns.insert(0, 'Post Date')
                filtered_df = filtered_df[selected_columns]
                if filtered_df.empty:
                    
                    alert = st.info(f"No data available for the selected date range.") # Display the alert
                    time.sleep(3) # Wait for 3 seconds
                    alert.empty()
                else:
                    filtered_df = filtered_df.sort_values(by='Post Date', ascending=False)
                    # Display the filtered DataFrame
                    st.dataframe(filtered_df)
                    # AgGrid(filtered_df)
                    # After loading and processing the DataFrame
                    # Convert 'Debit' and 'Credit' to numeric types
                    filtered_df['Debit'] = filtered_df['Debit'].astype(str)
                    filtered_df['Credit'] = filtered_df['Credit'].astype(str)

                    # Replace empty strings with '0'
                    filtered_df['Debit'] = filtered_df['Debit'].replace('', '0')
                    filtered_df['Credit'] = filtered_df['Credit'].replace('', '0')

                    # Convert to numeric, coercing errors to NaN and filling NaN with 0
                    filtered_df['Debit'] = pd.to_numeric(gs_df['Debit'], errors='coerce').fillna(0)
                    filtered_df['Credit'] = pd.to_numeric(gs_df['Credit'], errors='coerce').fillna(0)
                    
                    # gs_df['Debit'] = gs_df['Debit'].str.replace(r'[^\d.]', '').astype(float)
                    # gs_df['Credit'] = gs_df['Credit'].str.replace(r'[^\d.]', '').astype(float)

                    # Group by Transaction Type and sum Debit and Credit
                    transaction_types = filtered_df.groupby('Transaction Type')['Debit', 'Credit'].sum().reset_index()

                    # Plot
                    col1, col2 = st.columns([1,1])
                    with col1:
                        fig = px.bar(transaction_types, x='Transaction Type', y=['Debit', 'Credit'],
                        labels={'Debit': 'Debit Amount', 'Credit': 'Credit Amount'},
                        title='Debit vs Credit by Transaction Type')
                        # Update y-axis to better fit the data
                        fig.update_yaxes(range=[0, max(max(transaction_types['Debit']), max(transaction_types['Credit']))])
                        # Ensure x-axis labels are treated as categorical data
                        fig.update_xaxes(type='category')
                        # Display the chart in Streamlit
                    
                        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                    
                    with col2:
                        transfer_counts = filtered_df['Transfer Type'].value_counts().reset_index()
                        transfer_counts.columns = ['Transfer Type', 'Count']
                        # Now, use 'Count' for the values parameter
                        fig = px.pie(transfer_counts, values='Count', names='Transfer Type', title='Transfer Types Distribution')
                        col2 = st.plotly_chart(fig, use_container_width=True)
                    
                    
                    col1, col2 = st.columns([1,1])
                    
                    with col1:
                        fig = px.line(filtered_df, x='Post Date', y='Balance', title='Balance Over Time')
                        col1 = st.plotly_chart(fig, use_container_width=True)
                    
                    
                    with col2:
                        filtered_df['Post Date'] = pd.to_datetime(filtered_df['Post Date'])
                        
                        # Filter the DataFrame to include only transactions from the current year
                        current_year = datetime.now().year
                        filtered_df = filtered_df[filtered_df['Post Date'].dt.year == current_year]
                        
                        # Aggregate data to count transactions per day
                        daily_transactions = filtered_df.groupby('Post Date').size().reset_index(name='transaction_count')
                        
                        # Create a heatmap
                        fig = go.Figure(go.Heatmap(z=daily_transactions['transaction_count'], x=daily_transactions['Post Date']))
                        
                        # Adjust the layout to ensure the chart starts from the current year and is zoomed in
                        fig.update_layout(
                            title='Transaction Frequency Over Time',
                            xaxis_title='Post Date',
                            yaxis_title='Transaction Count',
                            xaxis=dict(
                                range=[datetime(current_year, 1, 1), datetime(current_year, 12, 31)]  # Adjust the range as needed
                            )
                        )
                        
                        col2 = st.plotly_chart(fig, use_container_width=True)


def currency_in_indian_format(n):
    """ Convert a number (int / float) into Indian formatting style """
    d = decimal.Decimal(str(n))

    if d.as_tuple().exponent < -2:
        s = str(n)
    else:
        s = '{0:.2f}'.format(n)

    l = len(s)
    i = l - 1

    res, flag, k = '', 0, 0
    while i >= 0:
        if flag == 0:
            res += s[i]
            if s[i] == '.':
                flag = 1
        elif flag == 1:
            k += 1
            res += s[i]
            if k == 3 and i - 1 >= 0:
                res += ','
                flag = 2
                k = 0
        else:
            k += 1
            res += s[i]
            if k == 2 and i - 1 >= 0:
                res += ','
                flag = 2
                k = 0
        i -= 1

    return res[::-1]


def bank_statements(authenticator, name):

    
    # Streamlit file uploader
    uploaded_files = st.sidebar.file_uploader("Upload Bank Account Statements", type=['csv','xls', 'xlsx'], accept_multiple_files=True)
    st.sidebar.divider()
    if uploaded_files:
        st.cache_data.clear()
        # Initialize an empty list to hold dataframes
        dfs = []

        # Process each uploaded file
        for uploaded_file in uploaded_files:
            # Read the Excel file into a DataFrame
            df = pd.read_excel(uploaded_file, usecols='A:H', skiprows=20) 
            
            # Drop the first row
            df.drop(df.index[0], inplace=True)
            
            # Drop rows with all NaN values
            df.dropna(how='all', axis=0, inplace=True)
            
            # Remove specific phrases from the 'Post Date' column
            phrases_to_remove = [
                "Statement Downloaded By",
                "Unless a constituent notifies the Bank",
                "him in this statement",
                "END OF STATEMENT - from Internet Banking"
            ]
            for phrase in phrases_to_remove:
                df = df[~df['Post Date'].str.contains(phrase, na=False, case=False, regex=True)]
            
            # Append the processed DataFrame to the list
            dfs.append(df)

        if dfs:
            # Concatenate all dataframes into one
            final_df = pd.concat(dfs, ignore_index=True)

            # Drop duplicates based on specific columns
            final_df.drop_duplicates(subset=['Post Date', 'Credit', 'Debit', 'Balance'], inplace=True)

            # Convert 'Post Date' to datetime and then back to string in the desired format
            # final_df['Post Date'] = pd.to_datetime(final_df['Post Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
            
            # final_df['Branch Code'] = final_df['Branch Code'].astype('int64') # 2024-Jun-15 (error)
            final_df['Branch Code'] = pd.to_numeric(final_df['Branch Code'], errors='coerce').fillna(0).astype('int64')
            # Apply the function to 'Credit' and 'Debit' columns in final_df
            final_df['Credit'] = final_df['Credit'].apply(handle_blank_and_convert)
            final_df['Debit'] = final_df['Debit'].apply(handle_blank_and_convert)
            
            final_df.fillna('', inplace=True)

            # Convert all columns in final_df to string format
            final_df = final_df.astype(str)
            
            # Sort the DataFrame by 'Post Date' in descending order
            # final_df = final_df.sort_values(by='Post Date', ascending=False)
            final_df = trim_all_columns(final_df)
            # Display the final DataFrame
            # st.dataframe(final_df)

            data_to_append = final_df.values.tolist()
            
            # Update worksheet
            worksheet = sh.worksheet('FinalDataframe')

            gs_df = pd.DataFrame(worksheet.get_all_records())
            # Check if gs_df is empty
            if gs_df.empty:
                # print(gs_df.columns) 
                # If gs_df is empty, insert all records from final_df
                data_to_append = [final_df.columns.tolist()] + data_to_append
                worksheet.append_rows(data_to_append)
                # print("gs_df is empty")
            else:
                gs_df.fillna('', inplace=True)
                gs_df = gs_df.astype(str)
                gs_df['Credit'] = gs_df['Credit'].apply(handle_blank_and_convert)
                gs_df['Debit'] = gs_df['Debit'].apply(handle_blank_and_convert)
                gs_df.fillna('', inplace=True)
                gs_df = gs_df.astype(str)
                
                # print(f"final DF - \n\n{final_df}")
                # print(f"GS DF - \n\n{gs_df}")
                merged_df = pd.merge(final_df, gs_df, how='outer', indicator=True,
                                    on = ['Post Date', 'Debit', 'Credit', 'Balance'])

                
                # print(f"merged_df : \n{merged_df}")
                new_records = merged_df[merged_df['_merge'] == 'left_only']
                
                # Drop the '_merge' column as it's no longer needed
                new_records.drop(columns=['_merge'], inplace=True)
                # print(f"new_records - \n{new_records}") 
                
                # Drop the '_y' columns
                new_records = new_records.drop(columns=[col for col in new_records.columns if col.endswith('_y')])

                # Rename the '_x' columns to their original names
                new_records.columns = [col.replace('_x', '') for col in new_records.columns]
                # Now, 'new_records' should only contain the columns you're interested in
                # print(f"Cleaned new_records - \n\n{new_records}")

                
                # Convert the new records DataFrame to a list of lists
                data_to_append = new_records.values.tolist()

                # set_with_dataframe(worksheet, new_records)
                # Append the new records to the Google Sheet
                worksheet.append_rows(data_to_append)
                
                st.success("Database Updated 🤓")
            
            
        else:
            print("")
            # st.info("Upload your Account Statements here.") 
            
        oldest_date = None
        latest_date = None
        default_columns = ["Post Date", "Account Name", "Transfer Type", "Transaction Type", "Credit", "Debit", "Balance"]

        worksheet = sh.worksheet('FinalDataframe')
        final_df = pd.DataFrame(worksheet.get_all_records())

        categories_df_all = process_transactions_with_cache(final_df)
        grand_df = calculate_grand_df(final_df, categories_df_all)
        
        grand_df['Post Date'] = pd.to_datetime(grand_df['Post Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
        grand_df = grand_df.sort_values(by='Post Date', ascending=False)
        # print(f"grand_df:- {grand_df.head(100)}")
        
        # Update or Insert grand_df
        grand_df.dropna(subset=['Post Date', 'Debit', 'Credit', 'Balance'], inplace=True)
        grand_df = grand_df.astype(str)
        
        # print(grand_df.dtypes)
        data_to_append = grand_df.values.tolist()
        worksheet_gdf = sh.worksheet('Accounts')
        gs_df = pd.DataFrame(worksheet_gdf.get_all_records())
        # Check if gs_df is empty
        if gs_df.empty:
            # print(gs_df.columns) 
            # If gs_df is empty, insert all records from grand_df
            data_to_append = [grand_df.columns.tolist()] + data_to_append
            worksheet_gdf.append_rows(data_to_append)
            # print("gs_df is empty")
        else:
            gs_df.fillna('', inplace=True)
            gs_df = gs_df.astype(str)
            gs_df['Credit'] = gs_df['Credit'].apply(handle_blank_and_convert)
            gs_df['Debit'] = gs_df['Debit'].apply(handle_blank_and_convert)
            gs_df.fillna('', inplace=True)
            gs_df = gs_df.astype(str)
            
            # print(f"Grand DF - \n\n{grand_df}")
            # print(f"GS DF - \n\n{gs_df}")
            merged_df = pd.merge(grand_df, gs_df, how='outer', indicator=True,
                                on = ['Post Date', 'Debit', 'Credit', 'Balance'])

            
            # print(f"merged_df : \n{merged_df}")
            new_records = merged_df[merged_df['_merge'] == 'left_only']
            
            # Drop the '_merge' column as it's no longer needed
            new_records.drop(columns=['_merge'], inplace=True)
            # print(f"new_records - \n{new_records}") 
            
            # Drop the '_y' columns
            new_records = new_records.drop(columns=[col for col in new_records.columns if col.endswith('_y')])

            # Rename the '_x' columns to their original names
            new_records.columns = [col.replace('_x', '') for col in new_records.columns]
            # Now, 'new_records' should only contain the columns you're interested in
            # print(f"Cleaned new_records - \n\n{new_records}")

            
            # Convert the new records DataFrame to a list of lists
            data_to_append = new_records.values.tolist()

            # set_with_dataframe(worksheet, new_records)
            # Append the new records to the Google Sheet
            worksheet_gdf.append_rows(data_to_append)
            
            # st.success("Worksheet Updated 🤓")
    

    st.title(f"Account Statement")
    worksheet = sh.worksheet('Accounts')
    gs_df = pd.DataFrame(worksheet.get_all_records())
    
    if gs_df.empty:
        st.info("No data in Databse. Please upload your Bank Statements.")
    
    else:
        gs_df['Credit'] = gs_df['Credit'].astype(str)
        gs_df['Debit'] = gs_df['Debit'].astype(str)
        
        gs_df.fillna('', inplace=True)
        gs_df.astype(str)
        
        oldest_date = None
        latest_date = None
        # Convert 'Post Date' to datetime format
        date_df = gs_df['Post Date']
        date_df = pd.to_datetime(date_df)
        
        # Calculate oldest_date and latest_date
        oldest_date = date_df.min()
        latest_date = date_df.max()
        default_columns = ["Post Date", "Account Name", "Transfer Type", "Transaction Type", "Credit", "Debit", "Balance"]
        
        default_columns = [col for col in default_columns if col in gs_df.columns]
            
        if oldest_date and latest_date:
            formatted_oldest_date = format_date(oldest_date)
            formatted_latest_date = format_date(latest_date)

            st.markdown(f"##### *{formatted_oldest_date} - {formatted_latest_date}*", unsafe_allow_html=True)
            
        selected_columns = st.multiselect("Select columns to display", options=gs_df.columns, default=default_columns)
        if 'Post Date' in selected_columns:
            # Move 'Post Date' to the beginning of the list
            selected_columns.remove('Post Date')
            selected_columns.insert(0, 'Post Date')
        filtered_df = gs_df[selected_columns]
        filtered_df = filtered_df.sort_values(by='Post Date', ascending=False)
        # st.dataframe(filtered_df)
        AgGrid(filtered_df)
        
        # Convert the DataFrame to CSV format
        data = filtered_df.to_csv(index=False)
        file_name = f"Account_Statement_{datetime.now().strftime('%Y-%m-%d__%H-%M')}.csv"
        mime = "text/csv"
        st.download_button(
                label="Download DataFrame as CSV",
                data=data,
                file_name=file_name,
                mime=mime
            )
            
        st.divider()
        
        df2 = SmartDataframe(gs_df, config={"llm": llm})
        st.markdown(f"### Chat with your Data")
        prompt2  = st.text_area("Enter your prompt:", placeholder="When was the GST transaction done?")

        if st.button("Generate"):
            if prompt2:
                with st.spinner("Generating response.."): 
                    st. write(df2.chat(prompt2))
            else:
                st.warning("Please enter a prompt!")
        
        
        st.divider()
        st.markdown(f"### Send Email")
        # Create a text input for recipients
        recipients_input = st.text_input("Enter recipients (comma separated):",placeholder=f"example1@google.com, example2.gmail.com")
        # Create a button for sending the email
        if st.button("Send Email"):
            if recipients_input:
                recipients = recipients_input.split(',')
                subject = f"Account Statement for {formatted_oldest_date} to {formatted_latest_date}"
                message = f"This is your Bank Account Statement for the dates ranging from {formatted_oldest_date} to {formatted_latest_date}"
                
                # Convert the DataFrame to a CSV string
                csv_string = df.to_csv(index=False)
                
                # Encode the CSV string to bytes
                attachment_bytes = csv_string.encode('utf-8')
                # Create an instance of the EmailSender class
                email_sender = EmailSender(
                    SMTP_SERVER=os.getenv('SMTP_SERVER'),
                    SMTP_PORT=int(os.getenv('SMTP_PORT')),
                    SMTP_USERNAME=os.getenv('SMTP_USERNAME'),
                    SMTP_PASSWORD=os.getenv('SMTP_PASSWORD'),
                    SMTP_SENDER=os.getenv('SMTP_SENDER')
                )
                try:
                    # Assuming email_sender is an instance of your EmailSender class
                    email_sender.send_email(recipients, file_name, subject, message, attachment_bytes)
                    st.success("Email sent successfully!")
                except Exception as e:
                    st.error(f"Error sending email: {e}")
            else:
                st.warning("Please enter recipients!")
        
        
        # st.sidebar.divider()

        
                
# def logmeout(authenticator):
#     print("here")
#     # authenticator.logout("Logout", location="sidebar", key="logout button")
#     authenticator.logout(location="unrendered")
        
def main():
    
    colored_header(
        label="Bank Statements",
        description="© Precission Engineering Safety Enterprise",
        color_name="green-70",
    )
    
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Statements"],
        icons=["headset-vr", "bank2"],
        menu_icon="bank",
        default_index=0,
        orientation="horizontal",
    )
    
    THIS_DIR = Path(__file__).parent.parent
    CSS_FILE = THIS_DIR / "style" / "style.css"
    
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
    # Load the configuration file
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path) as file:
        config = yaml.load(file, Loader=SafeLoader)

    # Create an authenticator object
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['pre-authorized']
    )
        
    name, authentication_status, username = authenticator.login(fields=["username", "password"])
    
    if authentication_status == False:
        st.error("Username/Password is Incorrect")
        
    if authentication_status == None:
        st.warning("Please enter username and password")
        
    if not authentication_status:
        st.markdown(f"## Please Login to view this page")
        
    if authentication_status:
        # col1, col2, col3 = st.columns([1,1,1])
        # with col2:
            # st.title("Bank Statements")
        
        
        if selected:
            if selected == "Statements":
                bank_statements(authenticator, name)
            
            if selected == "Dashboard":
                visualize()
            
            if st.session_state["authentication_status"]:
                with st.sidebar:
                    # st.markdown("<h2 style='text-align: center;'>Sidebar Header 1</h2>", unsafe_allow_html=True)
                    st.markdown(f'<p style="text-align: center; font-weight: bold;">Welcome <span style="color: gray;">{st.session_state["name"]}</span></p>', unsafe_allow_html=True)
                    # st.markdown("<p style='text-align: center;'><button>Button 1</button></p>", unsafe_allow_html=True)

                    col1, col2, col3 = st.sidebar.columns(3)
                    
                    # button = col2.button("logOut", use_container_width=True)
                    if col2.button("Log Out"):
                        authenticator.logout(location="unrendered")
                    
            
        # st.divider()
     
            
        # llm = Ollama(model="crewai-llama3:8b-instruct-q6_K")
        # llm = Ollama(model="herald/phi3-128k")
        llm = ChatGroq(model='llama3-70b-8192', temperature=0.9, api_key=os.getenv('GROQ_API_KEY'))
        
        PATTERN = 'data/AccountStatement_*.xls'
        OUTPUT_FOLDER = 'output'
        EXCEL_FILE_PATH = os.path.join(OUTPUT_FOLDER, f'Account_Statement_{datetime.now().strftime("%Y-%m-%d__%H-%M")}.xlsx')
        CSV_FILE_PATH = os.path.join(OUTPUT_FOLDER, f'Account_Statement_{datetime.now().strftime("%Y-%m-%d__%H-%M")}.csv')

        
        
    
if __name__ == "__main__":
    main()
