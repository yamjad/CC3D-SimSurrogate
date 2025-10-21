# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:48:50 2024

@author: pdalc
"""
 
import os
import shutil
import random
import numpy as np
import subprocess
import time
import glob

MAX_JOBS = 495
WAIT_TIME = 5 # seconds
N = 2 # Number of folders you want to create

# Define the list of parameters with their lower and upper limits
parameters = {
    'fluctuation_amplitude': (10, 10),  # parameter name and its range (lower, upper)
    'target_volume': (25, 150),
    'target_surface': (0.5, 2), # this is fractional of the 4*sqrt(target volume), i.e., from 4sqrtTV/2 to 4sqrtTV*2
    'lambda_volume': (5, 15),
    'lambda_surface': (0, 15),
    'Medium_cell_CE': (0, 15)
    # Add more parameters as needed
}

def generate_sbatch_string(cc3d_file, node=1, job_name='DiffMigAI', out_file='simulation_out_BIGRED',
                            err_file='err_file_BIGRED'):
    sbatch_string = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={node}
#SBATCH -J {job_name}
#SBATCH --time=10:00:00  # Set a max walltime to 24 hours
#SBATCH --mail-user=pecenci@iu.edu
#SBATCH -A r00128
#SBATCH -o /N/scratch/pecenci/NNM_CC3D/out.txt
#SBATCH -e /N/scratch/pecenci/NNM_CC3D/error.err
python /N/scratch/pecenci/NNM_CC3D/{cc3d_file}

    """
    return sbatch_string

def get_total_slurm_job_count():
    result = subprocess.run('squeue -u pecenci', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    num_lines = len(result.stdout.decode().splitlines())
    return(num_lines-1)
    #try:
    #    return int(output) - 1  # Subtract one for the header line
    #except ValueError:
    #    print("Error parsing job count: ", output)
    #    return 0
# Function to generate a single set of sampled parameters

def generate_set(parameters):
    sample_set = {}
    for param, (low, high) in parameters.items():
        sample_set[param] = float("{:.4g}".format(random.uniform(low, high)))
        if param == "target_volume":
            sample_set[param] = round(sample_set[param])
    opt_sur = 4*np.sqrt(sample_set["target_volume"])
    sample_set["target_surface"] = float("{:.4g}".format(opt_sur*random.uniform(parameters["target_surface"][0], parameters["target_surface"][1])))
    return sample_set
# Generate N sets
#sets = [generate_set(parameters) for _ in range(N)]

Sims_file = 'Sims'
if os.path.exists(Sims_file):
    list_files = os.listdir(Sims_file)
    list_files = [int(_) for _ in list_files]
    if list_files:
        MAX = max(list_files)+1
    if not list_files:
        MAX=0
else:
    MAX = 0
print(MAX)

files_toCall = []
file_to_copy = 'simulation.py'  # Replace with the name of the file to copy
# Check if the file exists in the current directory
if not os.path.isfile(file_to_copy):
    print(f"The file '{file_to_copy}' does not exist in the current directory.")
else:
    # Loop to create folders and copy the file
    for i in range(MAX, MAX+N):
        folder_name = f"Sims/{i}"
        # Create the folder
        os.makedirs(folder_name, exist_ok=True)
        # Copy the file into the folder
        shutil.copy(file_to_copy, folder_name)
        
        Set = generate_set(parameters)
        with open(f'{folder_name}/parameter.py', mode='w') as file:
            #for sample_set in sets:
            for param in Set:
                file.write(f"{param} = {Set[param]}\n")
        
        cc3d_file = f"{folder_name}/{file_to_copy}"
        with open(f'{folder_name}/output.txt', mode='a') as file:
            pass
        with open(f'{folder_name}/BatchCall.sh', mode='w') as f:
            f.write(generate_sbatch_string(cc3d_file, node=1,
                                            job_name='simulation',
                                              out_file='simulation_out',
                                                err_file='err_file'))
        files_toCall.append(f'{folder_name}/BatchCall.sh')
        #print(f"Copied '{file_to_copy}' into '{folder_name}'.")
        #os.system(f'python {folder_name}/diffusion_param_scan.py')
        #subprocess.run(['python', f'{folder_name}/diffusion_param_scan.py'])
        #print(f'path to sim = {folder_name}/diffusion_param_scan.py')

print("Folders created and files copied successfully.")

#  Call batch	
# files_toCall = glob.glob(os.path.join('Sims',"**", "BatchCall.sh"),recursive=True)
#print(files_toCall)
#filtered_files = [
#    file for file in files_toCall
#    if not os.path.exists(os.path.join(os.path.dirname(file), "result.dat"))
#]
#print(filtered_files)
#processes = []

for file in files_toCall:
    while get_total_slurm_job_count() >= MAX_JOBS:
        print(f"Waiting for available slots in queue... Current job count: {get_total_slurm_job_count()}, Max: {MAX_JOBS}")
        time.sleep(WAIT_TIME)
    
    os.chmod(file, 0o777)
    directory = os.path.dirname(file)
    #p = subprocess.Popen(['sbatch', file], cwd=directory)
    os.system(f'sbatch {file}')
    #os.system('python simulation.py')
    print(file)
    #processes.append(p)