import os
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
import pandas as pd
from pandasai import SmartDataframe
import streamlit as st
from streamlit_option_menu import option_menu
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
from googledr_package.googledr import get_final_dataframe
import importlib
from pathlib import Path
import gspread
from google.oauth2.service_account import Credentials
# from googledr_package.bank_statements import main as bank

scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

credentials = Credentials.from_service_account_file(
    'config/service_account.json',
    scopes=scopes
)

gc = gspread.authorize(credentials)
# gc = gspread.oauth(credentials_filename='config/service_account.json')
sh = gc.open("Account_Statements")


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
            
def data_chat():
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

def homepage():
    st.markdown(
    """
    # Welcome to 🚀 DataChat! 🚀

    DataChat is your ultimate companion for data exploration and analysis. With an open-source AI chat functionality, you can now chat with your data directly from CSV or Excel files. Simply upload your documents and start chatting right away! 📊💬

    ## 📊 DataChat: Your Personal Data Assistant

    - **Upload and Chat**: Easily upload your CSV or Excel files and start chatting with your data. No need for complex queries or manual data analysis.
    - **AI-Powered Insights**: Leverage the power of AI to gain insights from your data. Ask questions, and let DataChat provide you with the answers.
    - **Visualize and Download**: Analyze your data visually and download the insights in your preferred format.

    ## 📁 Bank Statements: A New Level of Financial Insight

    DataChat also offers a special section for Bank Statements. Connect your Google Drive and set up a folder for your personal data. Start chatting with your financial data directly from the web app. The underlying database and data processing are handled in the cloud, ensuring fast and secure access to your information.

    - **Cloud-Based Processing**: Your data is securely processed in the cloud, providing fast and efficient analysis.
    - **Visual Data Analysis**: Analyze your bank statements visually and gain insights directly from the web app.
    - **Downloadable Insights**: Choose to download your visual data analysis for further review or sharing.

    **Note**: The Bank Statements functionality is currently under development and requires authentication. Stay tuned for updates on when this feature becomes available to all users! 🔐🚧

    ## Get Started with DataChat Today!

    Ready to transform your data analysis experience? Start by uploading your first file and let DataChat guide you through the insights. 🌟
    """,
    unsafe_allow_html=True)
    
def contact():
    st.markdown(
        """
        # 📞 Get in Touch with Us!

        We're here to help you make the most of DataChat. Whether you have questions, feedback, or just want to say hi, we're all ears! 👂

        ## 📧 Contact Details

        - **Name**: Mr. Abhishek Roy
        - **Organization**: Precision Engineering Safety Enterprise (PESE)
        - **Email**: [pese.server@gmail.com](mailto:pese.server@gmail.com)
        - **Contact**: +91 7044454933

        ## 📝 Feedback and Support

        If you have any questions, feedback, or need assistance, feel free to reach out to us. We're here to help you navigate the world of data analysis and make your experience with DataChat as smooth as possible. 🚀

        ## 💬 Chat with Us

        You can also chat with us directly through this app. Just type your message in the chat box below, and we'll get back to you as soon as possible. 💬

        """,
        unsafe_allow_html=True
    )
    
    # Collect user information
    user_name = st.text_input("Enter your name:")
    user_email = st.text_input("Enter your email:")
    user_message = st.text_area("Type your message here:")
    
    if user_email:
        # Validate email format
        email_regex = r"[^@]+@[^@]+\.[^@]+"
        if not re.match(email_regex, user_email):
            st.warning("Please enter a valid email address.")
            return
        
    # Send email
    if st.button("Send"):
        if user_message:
            with st.spinner("Sending your message..."):
                worksheet = sh.worksheet('Contact')
                gs_df = pd.DataFrame(worksheet.get_all_records())
                if gs_df.empty:
                    print("here")
                    column_headers = ['User Name', 'Email', 'Message']
                    worksheet.append_row(column_headers)
                new_row = [user_name, user_email, user_message]
                worksheet.append_row(new_row)
                if send_email(user_name, user_email, user_message):
                    st.success("Your message has been sent We'll get back to you shortly.")
                else:
                    st.error("Failed to send your message. Please try again later.")
        else:
            st.warning("Please enter a message before sending.")
            

            
def send_email(user_name, user_email, user_message):
    # Set up the SMTP server
    email_sender = EmailSender(
                SMTP_SERVER=os.getenv('SMTP_SERVER'),
                SMTP_PORT=int(os.getenv('SMTP_PORT')),
                SMTP_USERNAME=os.getenv('SMTP_USERNAME'),
                SMTP_PASSWORD=os.getenv('SMTP_PASSWORD'),
                SMTP_SENDER=os.getenv('SMTP_SENDER')
            )

    recipients = ["abhishek93.roy@gmail.com"]
    subject = f"Feedback from {user_name} via Data-Chat"
    message = f"Name: {user_name}\nEmail: {user_email}\nMessage:\n{user_message}"
    
    try:
        email_sender.send_email_with_attachment(recipients, None, subject, message, attachment_bytes=None)
        return True
    except Exception as e:
        print(f"Failed to send Feedback! - {e}")

def main():
    
    THIS_DIR = Path(__file__).parent
    CSS_FILE = THIS_DIR / "style" / "style.css"
    
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "DataChat", "Contact"],
        icons=["house", "chat-square-text", "envelope"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    
    if selected == "Home":
        homepage()
    
    if selected =="DataChat":
        data_chat()
        # Dynamically import the module
        # bank()
        # st.switch_page("pages/2_📑_Bank Statements.py")
        
    if selected =="Contact":
        contact()

if __name__ == "__main__":    
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