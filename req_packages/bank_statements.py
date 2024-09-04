import os
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
import pandas as pd
from pandasai import SmartDataframe
import streamlit as st
from st_aggrid import AgGrid
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

load_dotenv()
llm = ChatGroq(model='llama3-70b-8192', temperature=0.9, api_key=os.getenv('GROQ_API_KEY'))
# Set the root logger level
logging.basicConfig(level=logging.INFO)

# Create a logger with its own name
logger = logging.getLogger(__name__)

# Set pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

st.set_page_config(
    page_title="Pese-AI",
    page_icon="💵",
    layout="wide"
)


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
    logger.debug(f"Categorizing transactions: {transaction_names}")
    logger.info(f"Categorizing transactions: {transaction_names}")
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
    logger.info(f"\n\ncategories DF: \n{df_cleaned}")
    return df_cleaned

# Get index list
#https://stackoverflow.com/questions/47518609/for-loop-range-and-interval-how-to-include-last-step
def hop(start, stop, step):
    for i in range(start, stop, step):
        yield i
    yield stop

def strip_newlines(s):
    if s is None:
        return None  # or return '' if you prefer to have empty strings instead of None
    return s.replace('\n', '')

def create_table(conn, table_schema):
    """Create a table in the SQLite database specified by conn."""
    try:
        c = conn.cursor()
        c.execute(table_schema)
    except Error as e:
        print(e)

def create_connection(db_file):
    """Create a database connection to the SQLite database specified by db_file."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return conn

def insert_data(conn, table_name, columns, data):
    """Insert data into the specified table, replacing rows that would cause a unique constraint violation."""
    sql = f'''INSERT OR REPLACE INTO {table_name}({', '.join(columns)}) VALUES({', '.join(['?' for _ in columns])})'''
    cur = conn.cursor()
    cur.executemany(sql, data)
    conn.commit()

def create_ai_statement_table(conn):
    """Create the ai_statement table in the SQLite database specified by conn."""
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS ai_statement (
                Post_Date TEXT,
                Branch_Code INTEGER,
                Cheque_Number INTEGER,
                Transaction_Description TEXT,
                Account_Name TEXT,
                Transfer_Type TEXT,
                Transaction_Type TEXT,
                Debit REAL,
                Credit REAL,
                Balance TEXT,
                Reference_Number TEXT,
                PRIMARY KEY (Post_Date, Debit, Credit, Balance)
            );
        """)
    except Error as e:
        print(e)

def insert_ai_statement_data(conn, data):
    """Insert data into the ai_statement table."""
    sql = """
        INSERT OR REPLACE INTO ai_statement(
            Post_Date, Branch_Code, Cheque_Number, Transaction_Description, Account_Name, 
            Transfer_Type, Transaction_Type, Debit, Credit, Balance, Reference_Number
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cur = conn.cursor()
    cur.executemany(sql, data)
    conn.commit()

def display_dataframe(df):
    # Your existing logic for displaying the dataframe with AgGrid
    AgGrid(df)

def fetch_data_from_cloud():
    database = "account_statement.db"
    conn = create_connection(database)
    # Your existing logic for fetching data from the cloud
    folder_name = "pandasai"
    final_df = fetch_final_dataframe(folder_name)
    print(f"\n\nFinal DF :- \n{final_df}")
    logger.info(f"\n\nFinal DF :- \n{final_df}")
    # print(final_df.dtypes)
    if conn is not None:
        # Define the table schema
        table_schema = """
        CREATE TABLE IF NOT EXISTS bank_statement (
            Post_Date TEXT,
            Value_Date TEXT,
            Branch_Code INTEGER,
            Cheque_Number INTEGER,
            Account_Description TEXT,
            Debit REAL,
            Credit REAL,
            Balance TEXT,
            PRIMARY KEY (Post_Date, Debit, Credit, Balance)
        );
        """

        # Create the table
        create_table(conn, table_schema)
        # Convert columns to desired data types
        # final_df['Branch Code'] = final_df['Branch Code'].astype(str).apply(lambda x: re.sub(r'\D', '0', x))
        # final_df['Branch Code'] = final_df['Branch Code'].fillna(0).astype(int)
        # final_df['Cheque Number'] = final_df['Cheque Number'].astype(str).apply(lambda x: re.sub(r'\D', '0', x))
        # final_df['Cheque Number'] = final_df['Cheque Number'].fillna(0).astype(int)
        # Replace empty strings or strings with only whitespace with NaN
        final_df['Credit'] = final_df['Credit'].replace(r'^\s*$', np.nan, regex=True)
        final_df['Credit'] = final_df['Credit'].fillna(0).astype(float).round(2)
        final_df['Debit'] = final_df['Debit'].replace(r'^\s*$', np.nan, regex=True)
        final_df['Debit'] = final_df['Debit'].fillna(0).astype(float).round(2)
        final_df['Credit'] = final_df['Credit'].apply(lambda x: f"{x:.2f}")
        final_df['Debit'] = final_df['Debit'].apply(lambda x: f"{x:.2f}")
        
        # Assuming final_df is your DataFrame
        # Convert the DataFrame to a list of tuples
        data_to_insert = final_df.to_records(index=False).tolist()

        # Define the table name and columns
        table_name = "bank_statement"
        columns = ['Post_Date', 'Value_Date', 'Branch_Code', 'Cheque_Number', 'Account_Description', 'Debit', 'Credit', 'Balance']

        # Insert the data
        insert_data(conn, table_name, columns, data_to_insert)
    
        categories_df_all = process_transactions_with_cache(final_df)
            
        grand_df = calculate_grand_df(final_df, categories_df_all)
        
        create_ai_statement_table(conn)
        data_to_insert_ai_statement = grand_df.to_records(index=False).tolist()
        insert_ai_statement_data(conn, data_to_insert_ai_statement)
        
    else:
        print("Error! Cannot create the database connection.")

    conn.close()
    return grand_df

def bkp_fetch_data_from_database(rerun_counter):
    # Your existing logic for fetching data from the database
    conn = create_connection("account_statement.db")
    if conn is not None:
        cur = conn.cursor()
        cur.execute("SELECT * FROM ai_statement")
        data = cur.fetchall()
        columns = [description[0] for description in cur.description]
        df_from_db = pd.DataFrame(data, columns=columns)
        return df_from_db
    else:
        st.error("Error! Cannot create the database connection.")
        return None
    
@st.cache_data
def fetch_data_from_database(rerun_counter):
    # Assuming you have a function to get the Google Drive service
    service = build("drive", "v3", credentials=getCredentials())
    
    # Find the folder ID for 'pandasai'
    folder_id = findFolderIdByName('pandasai')
    
    # Assuming you have a function to find the file ID of 'account_statement.db' in the 'pandasai' folder
    file_id = findFileIdByName(folder_id, 'account_statement.db')
    
    # Download the database file from Google Drive
    downloadFile(service, file_id, 'account_statement.db')
    
    # Read data from the SQLite database
    conn = create_connection('account_statement.db')
    df = pd.read_sql_query("SELECT * FROM ai_statement", conn)
    conn.close()
    
    return df

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

@st.cache_data() # Use allow_output_mutation=True for DataFrames
def fetch_final_dataframe(folder_name):
    """
    Fetch the final DataFrame from the specified folder.
    
    Parameters:
    - folder_name: The name of the folder containing the data.
    
    Returns:
    - A DataFrame with the final data.
    """
    return get_final_dataframe(folder_name)

@st.cache_data(ttl=3600) # This line is crucial for caching
def calculate_grand_df(final_df, categories_df_all):
    # Your existing logic to calculate grand_df goes here
    # For example:
    selected_columns_df1 = final_df[['Post Date', 'Branch Code', 'Cheque Number', 'Debit', 'Credit', 'Balance']]
    selected_columns_df2 = categories_df_all[['Transaction Description', 'Transfer Type', 'Transaction Type', 'Account Name', 'Reference Number']]

    grand_df = pd.concat([selected_columns_df1, selected_columns_df2], axis=1)
    new_column_order = ['Post Date', 'Branch Code', 'Cheque Number', 'Transaction Description', 'Account Name', 'Transfer Type', 'Transaction Type', 'Debit', 'Credit', 'Balance', 'Reference Number']
    grand_df = grand_df.reindex(columns=new_column_order)
    
    return grand_df

@st.cache_data(ttl=3600)
def process_transactions_with_cache(final_df):
    index_list = list(hop(0, len(final_df), 30))
    categorize_transactions_cache = {}
    categories_df_all = pd.DataFrame()
    for i in range(len(index_list) - 1):
        logger.debug(f"Processing transactions from {index_list[i]} to {index_list[i+1]}...")
        logger.info(f"Processing transactions from {index_list[i]} to {index_list[i+1]}...")
        transaction_names = ','.join(final_df['Account Description'].iloc[index_list[i]:index_list[i+1]])
        categories_df = categorize_transactions(transaction_names, llm)   
        categories_df_all = pd.concat([categories_df_all, categories_df], ignore_index=True)
    
    return categories_df_all

def homepage():
    st.title("Chat with your data")
    uploader_file = st.file_uploader("Upload a CSV or Excel file and chat with it!", type=['csv', 'xlsx'])
    if uploader_file is not None:
        file_extension = uploader_file.name.split('.')[-1]
        if file_extension == 'csv':
            data = pd.read_csv(uploader_file)
        elif file_extension == 'xlsx':
            data = pd.read_excel(uploader_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
        
        st.dataframe(data)
        df = SmartDataframe(data, config={"llm": llm})
        prompt = st.text_area("Enter your prompt:")
        
        if st.button("Generate "):
            if prompt:
                with st.spinner("Generating response.."):
                    st.write(df.chat(prompt))
            else:
                st.warning("Please enter a prompt!")


               
DATABASE_PATH = 'account_statement.db'             
def download_database():
    if os.path.exists(DATABASE_PATH):
        with open(DATABASE_PATH, 'rb') as f:
            data = f.read()
        return data
    else:
        return None

def upload_database(uploaded_file):
    if uploaded_file is not None:
        with open(DATABASE_PATH, 'wb') as f:
            f.write(uploaded_file.getvalue())
        st.success("Database updated successfully.")
    else:
        st.warning("No file uploaded.")
        
def bank_statements(authenticator, name):
    
    st.title("Bank Statements")
    st.divider()
    
    data_source = st.sidebar.selectbox("Choose data source:", ["Pull from Cloud", "Pull from Database"], index=1)
    st.markdown(f"## Account Statement")
    oldest_date = None
    latest_date = None
    default_columns = ["Post Date", "Account Name", "Transfer Type", "Transaction Type", "Credit", "Debit", "Balance"]
    if data_source == "Pull from Cloud":
        df = fetch_data_from_cloud()
        # Upload the updated database file to Google Drive
        creds = getCredentials()
        service = build("drive", "v3", credentials=creds)
        folder_id = findFolderIdByName('pandasai')
        # upload_to_drive(service, 'account_statement.db', 'account_statement.db', folder_id)
        upload_account_statement_db(service, 'account_statement.db', folder_id)
        
        oldest_date = pd.to_datetime(df['Post Date'].min(), format='%d/%m/%Y')
        latest_date = pd.to_datetime(df['Post Date'].max(), format='%d/%m/%Y')
        default_columns = ["Post Date", "Account Name", "Transfer Type", "Transaction Type", "Credit", "Debit", "Balance"]
        
    elif data_source == "Pull from Database":
        df = fetch_data_from_database(st.session_state['rerun_counter']) 
        oldest_date = pd.to_datetime(df['Post_Date'].min(), format='%d/%m/%Y')
        latest_date = pd.to_datetime(df['Post_Date'].max(), format='%d/%m/%Y')
        default_columns = ["Post_Date", "Account_Name", "Transfer_Type", "Transaction_Type", "Credit", "Debit", "Balance"]
    
    if df is not None:
        if oldest_date and latest_date:
            formatted_oldest_date = format_date(oldest_date)
            formatted_latest_date = format_date(latest_date)

            st.markdown(f"##### *{formatted_oldest_date} - {formatted_latest_date}*", unsafe_allow_html=True)
            
        selected_columns = st.multiselect("Select columns to display", options=df.columns, default=default_columns)
        if 'Post Date' in selected_columns:
            # Move 'Post Date' to the beginning of the list
            selected_columns.remove('Post Date')
            selected_columns.insert(0, 'Post Date')
        filtered_df = df[selected_columns]
        display_dataframe(filtered_df)
        
        # Create two columns side by side
        col1, col2 = st.columns([0.1,0.9])
        with col1:
            # Existing code for reloading data and downloading DataFrame as CSV
            if st.button("Reload Data"):
                # Trigger data fetching logic again
                # df = fetch_data_from_database()
                df = fetch_data_from_cloud()
                st.session_state['rerun_counter'] += 1    
            # Convert the DataFrame to CSV format
            data = df.to_csv(index=False)
            file_name = f"Account_Statement_{datetime.now().strftime('%Y-%m-%d__%H-%M')}.csv"
            mime = "text/csv"

        with col2:
            # Create a download button for the CSV format
            st.download_button(
                label="Download DataFrame as CSV",
                data=data,
                file_name=file_name,
                mime=mime
            )

        
        
        df2 = SmartDataframe(df, config={"llm": llm})
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
    
    st.sidebar.divider()
    if st.session_state["authentication_status"]:
        st.sidebar.write(f'Welcome *{st.session_state["name"]}*')
        authenticator.logout("Logout", location="sidebar", key="logout button")
        
        
def main():
    
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
        st.warning("Please enter your username and password")
        
    if not authentication_status:
        st.markdown(f"## Please login to view this page")
        
    if authentication_status:
        
        
        if 'rerun_counter' not in st.session_state:
            st.session_state['rerun_counter'] = 0
        if 'reload_cloud_data' not in st.session_state:
            st.session_state['reload_cloud_data'] = False        
        
        # llm = Ollama(model="crewai-llama3:8b-instruct-q6_K")
        # llm = Ollama(model="herald/phi3-128k")
        llm = ChatGroq(model='llama3-70b-8192', temperature=0.9, api_key=os.getenv('GROQ_API_KEY'))
        
        PATTERN = 'data/AccountStatement_*.xls'
        OUTPUT_FOLDER = 'output'
        EXCEL_FILE_PATH = os.path.join(OUTPUT_FOLDER, f'Account_Statement_{datetime.now().strftime("%Y-%m-%d__%H-%M")}.xlsx')
        CSV_FILE_PATH = os.path.join(OUTPUT_FOLDER, f'Account_Statement_{datetime.now().strftime("%Y-%m-%d__%H-%M")}.csv')

        bank_statements(authenticator, name)

    
# if __name__ == "__main__":
file_handler = None

if not os.path.exists('logs'):
    os.makedirs('logs')

# Set the file handler to write log messages to files
file_handler = logging.FileHandler(f"logs/log-{datetime.now().strftime('%Y-%m-%d__%H-%M')}.txt")   
try:
    
    # Set the file handler to write log messages to files
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    # Add the file handler to the logger
    logger.addHandler(file_handler)
            
    main()

finally:
    if file_handler:
        file_handler.close()