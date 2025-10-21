# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:46:46 2024

@author: pdalc
"""

import os
import shutil
from pathlib import Path

def find_files_with_name(root_dir, target_name):
    matched_files = []

    def scan_dir(directory):
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir() and entry.name == target_name:
                    matched_files.append(entry.path)
                elif entry.is_dir():
                    print(entry)
                    scan_dir(entry.path)  # Recursively scan subdirectories

    scan_dir(root_dir)
    return matched_files

# Example usage
# List of file paths

# Count files with fewer than 100 lines

#print(f"Number of files with fewer than 100 lines: {count}")

root_folder = 'Sims'  # Specify the folder path
required_filename = '__pycache__'  # Specify the required file's name
list_of_files = find_files_with_name(root_folder, required_filename)
#count = sum(1 for file in list_of_files if sum(1 for _ in open(file)) < 100)
print(len(list_of_files))
