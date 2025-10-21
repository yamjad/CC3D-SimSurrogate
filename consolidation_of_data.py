# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:48:50 2024

@author: pdalc
"""
import os
import shutil
import random
import numpy as np
import subprocess
import time
import glob
import pandas as pd
from pathlib import Path
 
def find_files_with_name(root_dir, target_name):
    matched_files = []

    def scan_dir(directory):
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir() and entry.name == target_name:
                    matched_files.append(os.path.dirname(entry.path))
                elif entry.is_dir():
                    #print(entry)
                    scan_dir(entry.path)  # Recursively scan subdirectories

    scan_dir(root_dir)
    return matched_files

# Example usage
# List of file paths

# Count files with fewer than 100 lines

#print(f"Number of files with fewer than 100 lines: {count}")

# root_folder = 'Sims'  # Specify the folder path
# required_filename = '__pycache__'  # Specify the required file's name
# list_of_files = find_files_with_name(root_folder, required_filename)
#count = sum(1 for file in list_of_files if sum(1 for _ in open(file)) < 100)
# print(len(list_of_files))


def calculate_average(data_files, output_file):
    output_data = []
    
    # Iterate over each file in the list
    data_file_name = 'output.txt'
    param_file_name = 'parameter.py'
    for file_path in data_files:
        #print(file_path)
        # Read the file into a DataFrame
        try:
            df = pd.read_csv(f'{file_path}/{data_file_name}', header=None, delimiter=' ')  # Assuming no headers
            if df.empty:
                mean_list = [None]
            else:
                mean_list = df.select_dtypes(include='number').mean().tolist()
        except pd.errors.EmptyDataError:
            mean_list = [None]
        
        dg = pd.read_csv(f'{file_path}/{param_file_name}', header=None, delimiter=' ')
        param = dg.iloc[:, 2].tolist()
        #print(param)
        # Append the result (average, flag) to the output data
        output_data.append((*param, *mean_list))
    
    # Write the output data to the output file
    with open(output_file, 'w') as f:
        for items in output_data:
            f.write("\t".join(map(str, items))+"\n")

# data_files = glob.glob(os.path.join('Sims',"*"),recursive=True)
root_folder = 'Sims'  # Specify the folder path
required_filename = '__pycache__'  # Specify the required file's name
data_files = find_files_with_name(root_folder, required_filename)
# print(data_files)
# param_files = glob.glob(os.path.join('Sims',"**", "parameter_set.py"),recursive=True)
# print(param_files)
output_file = 'averages_output.dat'
calculate_average(data_files, output_file)