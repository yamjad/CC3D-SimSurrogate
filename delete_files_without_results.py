# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:46:46 2024

@author: pdalc
"""

import os
import shutil

def delete_folders_without_file(root_folder, required_filename):
    # Loop through all items in the root folder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # Check if the required file exists in the subfolder
            if required_filename not in os.listdir(subfolder_path):
                # Delete the subfolder if the required file is not found
                shutil.rmtree(subfolder_path)
                print(f"Deleted folder: {subfolder_path}")

# Example usage
root_folder = 'Sims'  # Specify the folder path
required_filename = 'result.dat'  # Specify the required file's name
delete_folders_without_file(root_folder, required_filename)