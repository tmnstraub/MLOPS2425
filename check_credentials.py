import yaml
import os

try:
    cred_path = 'conf/local/credentials.yml'
    print(f"Attempting to load credentials from {os.path.abspath(cred_path)}")
    
    with open(cred_path, 'r') as file:
        credentials = yaml.safe_load(file)
    
    print('Credentials loaded successfully:', list(credentials.keys()))
    
    if 'feature_store' in credentials:
        fs_creds = credentials['feature_store']
        print('Feature store section:', list(fs_creds.keys()))
        print('API Key exists:', 'FS_API_KEY' in fs_creds)
        print('Project name exists:', 'FS_PROJECT_NAME' in fs_creds)
    else:
        print('Feature store section not found in credentials')
except Exception as e:
    print(f"Error loading credentials: {e}")
