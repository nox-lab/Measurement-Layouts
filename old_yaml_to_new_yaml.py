import re
import os 
import sys

# Replace old parameters with new ones in the configuration file.
# This loops through every subfolder in the Variations folder and its subfolders such taht every file in the main folder is touched.

# Replace old parameters with new ones in the configuration file.
def replace_yaml_keys(file_path, replacements):
    try:
        # Reads the original YAML file
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Replaces keys based on the replacements dictionary
        for old_key, new_key in replacements.items():
            content = content.replace(f'{old_key}:', f'{new_key}:')
        
        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.write(content)
        
        print(f'YAML parameters replaced successfully in {file_path}!')
    except FileNotFoundError:
        print(f'File not found: {file_path}')
    except Exception as e:
        print(f'An error occurred while processing {file_path}: {str(e)}')

# Define the folder path containing the YAML files
for i in os.walk(r"C:\Users\talha\Documents\iib_projects\Variations\Variations"):
    folder_path = i[0]

    replacements = {
        't': 'timeLimit',
        'pass_mark': 'passMark'
    }

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f'Folder not found: {folder_path}')
    else:
        # Iterate over each file in the folder
        yaml_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.yaml')]
        
        if len(yaml_files) == 0:
            print(f'No YAML files found in {folder_path}')
        else:
            for filename in yaml_files:
                file_path = os.path.join(folder_path, filename)
                
                # Call the function to implement the changes for each file
                replace_yaml_keys(file_path, replacements)