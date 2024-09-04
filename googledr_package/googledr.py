import os.path
import os
import io
import tempfile
import pandas as pd
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/drive"]

def getCredentials():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds

def findFolderIdByName(folder_name):
    try:
        service = build("drive", "v3", credentials=getCredentials())
        query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}'"
        results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
        items = results.get("files", [])
        if not items:
            print(f"No folder named '{folder_name}' found.")
            return None
        else:
            # Assuming the first item is the correct folder
            return items[0]['id']
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None
    
def findFileIdByName(folder_id, file_name):
    try:
        service = build("drive", "v3", credentials=getCredentials())
        query = f"'{folder_id}' in parents and name='{file_name}'"
        results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
        items = results.get("files", [])
        if not items:
            print(f"No file named '{file_name}' found in the folder.")
            return None
        else:
            # Assuming the first item is the correct file
            return items[0]['id']
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

def listFilesInFolder(folder_id):
    try:
        service = build("drive", "v3", credentials=getCredentials())
        results = service.files().list(q=f"'{folder_id}' in parents",
                                       fields="nextPageToken, files(id, name)").execute()
        items = results.get("files", [])
        if not items:
            print("No files found.")
        else:
            print("Files:")
            for item in items:
                print(f"{item['name']} ({item['id']})")
    except HttpError as error:
        print(f"An error occurred: {error}")

def upload_to_drive(service, file_path, file_name, folder_id):
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, mimetype='application/octet-stream')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'File ID: "{file.get("id")}".')


def downloadFile(service, file_id, file_name):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}.")
    fh.seek(0)
    with open(file_name, 'wb') as f:
        f.write(fh.read())
    print(f"File {file_name} downloaded.")

def ensure_backup_folder_exists(service, folder_id):
    backup_folder_name = "bkp"
    query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and name='{backup_folder_name}'"
    results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
    items = results.get("files", [])
    if not items:
        # Backup folder does not exist, create it
        folder_metadata = {
            'name': backup_folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [folder_id]
        }
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        print(f"Backup folder created with ID: {folder.get('id')}")
        return folder.get('id')
    else:
        # Backup folder already exists
        print(f"Backup folder already exists with ID: {items[0]['id']}")
        return items[0]['id']

def move_file_to_backup(service, file_id, backup_folder_id):
    # Get the current parents of the file
    file = service.files().get(fileId=file_id, fields='parents').execute()
    current_parents = ",".join(file.get('parents'))
    
    # Prepare the request to move the file to the backup folder
    file_metadata = {
        'addParents': backup_folder_id,
        'removeParents': current_parents
    }
    
    # Perform the move operation
    service.files().update(fileId=file_id, body=file_metadata).execute()
    print(f"Moved file to backup folder.")

def copy_file_to_backup_and_delete(service, file_id, backup_folder_id):
    # Copy the file to the backup folder
    file_metadata = {
        'parents': [backup_folder_id]
    }
    copied_file = service.files().copy(fileId=file_id, body=file_metadata, fields='id').execute()
    print(f"Copied file to backup folder with ID: {copied_file.get('id')}")
    
    # Delete the original file
    service.files().delete(fileId=file_id).execute()
    print(f"Deleted original file with ID: {file_id}")
    
def upload_account_statement_db(service, file_path, folder_id):
    # Ensure the backup folder exists
    backup_folder_id = ensure_backup_folder_exists(service, folder_id)
    
    # Find the existing account_statement.db file
    file_id = findFileIdByName(folder_id, 'account_statement.db')
    
    if file_id:
        # Attempt to move the existing file to the backup folder
        try:
            # move_file_to_backup(service, file_id, backup_folder_id)
            copy_file_to_backup_and_delete(service, file_id, backup_folder_id)
        except Exception as e:
            print(f"Failed to move file to backup folder: {e}")
            # If moving fails, copy the file to the backup folder and delete it from the original folder
            copy_file_to_backup_and_delete(service, file_id, backup_folder_id)
    
    # Upload the new file to the pandasai folder
    upload_new_file(service, file_path, 'account_statement.db', folder_id)

def upload_new_file(service, file_path, file_name, folder_id):
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, mimetype='application/octet-stream')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'New file ID: "{file.get("id")}".')
    
def processExcelFiles(folder_id):
    service = build("drive", "v3", credentials=getCredentials())
    query = f"'{folder_id}' in parents and mimeType='application/vnd.ms-excel'"
    results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
    items = results.get("files", [])
    if not items:
        print("No Excel files found.")
        return

    phrases_to_remove = [
        "Statement Downloaded By",
        "Unless a constituent notifies the Bank",
        "him in this statement",
        "END OF STATEMENT - from Internet Banking"
    ]

    dfs = []
    for item in items:
        file_id = item['id']
        file_name = item['name']
        temp_file_path = os.path.join(tempfile.gettempdir(), file_name)
        downloadFile(service, file_id, temp_file_path)
        df = pd.read_excel(temp_file_path, usecols='A:H', skiprows=20)
        df.drop(df.index[0], inplace=True)
        df.dropna(how='all', axis=0, inplace=True)
        for phrase in phrases_to_remove:
            df = df[~df['Post Date'].str.contains(phrase, na=False, case=False, regex=True)]
        dfs.append(df)
        os.remove(temp_file_path) # Clean up temporary file

    final_df = pd.concat(dfs, ignore_index=True)
    
    # Drop duplicates based on specific columns
    final_df.drop_duplicates(subset=['Post Date', 'Credit', 'Debit', 'Balance'], inplace=True)
    
    # Convert columns to specific data types
    # Assuming 'Post Date' is in a format that pandas can recognize as a date
    final_df['Post Date'] = pd.to_datetime(final_df['Post Date'], format='%d/%m/%Y').dt.strftime('%d/%m/%Y')
    # final_df['Branch Code'] = final_df['Branch Code'].astype(int)
    # final_df['Cheque Number'] = final_df['Cheque Number'].str.replace(' ', '')
    # final_df['Cheque Number'] = final_df['Cheque Number'].astype(int)
    # final_df['Debit'] = final_df['Debit'].apply(lambda x: 0 if not str(x).replace('.', '', 1).isdigit() else x)
    # final_df['Debit'] = final_df['Debit'].astype(float).round(2)
    # final_df['Credit'] = final_df['Credit'].apply(lambda x: 0 if not str(x).replace('.', '', 1).isdigit() else x)
    # final_df['Credit'] = final_df['Credit'].astype(float).round(2)
    
    # Convert all other columns to string format
    # for col in final_df.columns:
    #     if col not in ['Post Date', 'Branch Code', 'Debit', 'Credit']:
    #         final_df[col] = final_df[col].astype(str)
    
    # print(f"\n\nFinal DF :- \n{final_df}")
    
    return final_df

def get_final_dataframe(folder_name):
    folder_id = findFolderIdByName(folder_name)
    if folder_id:
        return processExcelFiles(folder_id)
    else:
        return pd.DataFrame()
    
def main():
    folder_name = "pandasai"
    folder_id = findFolderIdByName(folder_name)
    if folder_id:
        processExcelFiles(folder_id)

if __name__ == "__main__":
    main()
