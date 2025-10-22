# CC3D-SimSurrogate

## Objective

This project utilizes Machine Learning (ML) to infer parameter values for multi-scale, agent-based biological models built in CompuCell3D. One of the most important and time-consuming parts of the CompuCell3D simulation is the process of solving Partial Differential Equations, which describe behavior of certain simulation objects. Here we present a model that predict parameter values given a range of initial conditions, which provides a proof-of-concept for an ML model that is able to solve this diffusion problem. Through the realization of this goal, time-consuming CompuCell3D simulations can be replaced with  instant neural network predictions while maintaining accuracy within biological noise levels.

For more information on CompuCell3D and terminology used, please reference the CompuCell3D reference manual: https://compucell3dreferencemanual.readthedocs.io

This project delivers simulation scripts, a complete dataset, and a trained ML model to support rapid and accurate parameter inference. The model being presented is able to accurately predict the values for the mean volume and mean surface given varying values for total volume, total surface, lambda volume, lambda surface, and contact energy. The project workflow proceeded via the following three stages: (1) developing Python scripts that incorporate CompuCell3D simulation logic and allow automated variation of input parameters, (2) running these simulations repeatedly to generate a structured dataset of inputs and corresponding outcomes, and (3) using the dataset to train and evaluate a neural network model that learns the mapping between parameter values and simulation results. View the "Workflow" section for detailed information about each of these stages.

## HPC Access

##### VPN Connection

Before accessing BigRed200, you must have the IU VPN configured and connected. This is required for all off-campus access to university computing resources. Download and install the IU VPN client from the university IT services website.

##### SSH Access

**For Mac/Linux Users**: Open Terminal and connect using:

```bash
ssh [YOUR-IU-USERNAME]@bigred200.uits.iu.edu
```

**For Windows Users**: Use PuTTY to establish an SSH connection:

1. Download and install PuTTY
2. Enter hostname: `bigred200.uits.iu.edu`
3. Use your IU username and passphrase to authenticate

##### File Transfer Setup

Transferring files between your local machine and BigRed200 requires an FTP/SFTP client:

**For Mac Users**:
- Commander One (also known as FTP-Mac)

**For Windows Users**:
- WinSCP (recommended)

##### Directory Access

When accessing home directories, you may encounter permission issues with generic paths like `/geode2/home/u070/`. Instead, access your directory directly using the full path:

```
/geode2/home/u070/[YOUR-IU-USERNAME]
```

This bypasses directory listing permissions while still allowing you to access your own folder.

##### Environment Setup

**Installing Miniconda**

Once connected to BigRed200, install Miniconda to manage Python environments:

1. Download the Miniconda installer:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2. Run the installer:

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

3. Follow the prompts and initialize conda for your shell

**Creating a CompuCell3D Environment**

1. Create a new conda environment:

```bash
conda create -n cc3d python=3.9
```

2. Activate the environment:

```bash
conda activate cc3d
```

3. Install CompuCell3D following the instructions at the official CC3D website: [https://compucell3d.org/](https://compucell3d.org/)

##### Account and Storage Configuration

**Account Partition Access**

To submit jobs to BigRed200, you must be added to an account allocation. Common issues include:

**Problem**: Error message "Invalid account or account/partition combination specified"

**Solution**: Your Principal Investigator (PI) must add you to their BigRed200 allocation. Contact your PI or the project lead with the following information:

- Your IU username
- The account number (e.g., r00128)
- Request to be added to the BigRed200 allocation

After being added, it may take up to one hour for your username to be added to the account on the system.

**Verifying Account Access**

Check which accounts you have access to:

```bash
sacctmgr show user [YOUR-IU-USERNAME]
```

##### Storage Locations

**Scratch Storage** (Recommended for active simulations):

```
/N/scratch/[YOUR-IU-USERNAME]
```

**Important Limitations**:

- **Time limit**: Files in scratch storage are subject to automatic deletion after a certain period (typically 30 days)
- **Storage capacity**: Limited quota per user
- **Performance**: Optimized for high-speed I/O during simulations

Use scratch storage for running simulations and periodically move important results to long-term storage.

## Data Generation

In order to train the model, data from CompuCell3D simulations is required. In order to generate a large dataset, the files in the `data_generation`folder can be run to complete this task. For project-specific questions, contact your research supervisor or project lead. For technical HPC issues, contact IU Research Technologies support.

**IU HPC Support**: hpc@iu.edu

**BigRed200 Documentation**: [https://kb.iu.edu/d/aolp](https://kb.iu.edu/d/aolp)

**CompuCell3D Documentation**: [https://compucell3d.org/](https://compucell3d.org/)

**SLURM Documentation**: [https://slurm.schedmd.com/](https://slurm.schedmd.com/)

##### File Organization

Organize your simulation files in your scratch directory:

```bash
cd /N/scratch/[YOUR-IU-USERNAME]
mkdir NNM_CC3D
cd NNM_CC3D
```

**Uploading Project Files**

Use your FTP client to upload:

- `script.py`
- `simulation.py`
- `parameter.py`
- Any other required Python files

**Editing Paths**

Before running simulations, edit `script.py` to update paths:

- Change `/N/scratch/[YOUR-IU-USERNAME]/NNM_CC3D/` to `/N/scratch/[YOUR-IU-USERNAME]/NNM_CC3D/`
- Update email addresses in the SLURM batch script
- Verify account allocation number matches your access

##### Submitting Jobs

1. Activate your conda environment:

```bash
conda activate cc3d
```

2. Run the main script:

```bash
python script.py
```

This will automatically generate simulation directories and submit jobs to the SLURM queue.

##### Troubleshooting

**Cannot Access Home Directory**

**Symptom**: Permission denied when accessing `/geode2/home/u070/`

**Solution**: Use the direct path `/geode2/home/u070/[YOUR-IU-USERNAME]` instead

**Account Access Error**

**Symptom**: "Invalid account or account/partition combination specified"

**Solution**:

1. Verify you've been added to the account allocation by your PI
2. Wait up to 1 hour after being added for the system to update
3. Contact HPC support if issues persist: hpc@iu.edu

**Missing BatchCall.sh Files**

**Symptom**: Script complains about missing batch files

**Solution**: The `script.py` should generate these automatically. Verify:

- You're running the correct version of `script.py`
- You have write permissions in the target directory
- No errors occurred during directory creation

**Output Files Not Written**

**Symptom**: `output.txt` files remain empty after simulation

**Solution**:
1. Check SLURM error files for Python errors
2. Verify CompuCell3D is properly installed in your conda environment
3. Ensure file paths in `simulation.py` are correct
4. Check that you have write permissions in the simulation directories

**Jobs Not Submitting**

**Symptom**: Jobs remain in pending state or don't submit

**Solution**:
1. Check queue status: `squeue -u [YOUR-IU-USERNAME]`
2. Verify you haven't exceeded job limits (MAX_JOBS in script)
3. Check account allocation status
4. Review SLURM output for errors
##### Project Structure

```
.
├── script.py                           # Main orchestration script
├── simulation.py                       # CC3D simulation implementation
├── parameter.py                        # Parameter configuration template
├── consolidation_of_data.py            # Results aggregation script
├── count_files_without_results.py      # Diagnostic utility
├── delete_files_without_results.py     # Cleanup utility
├── delete_files_above_threshold.py     # Cleanup utility
└── Sims/                              # Generated simulation directories
    ├── 0/
    │   ├── simulation.py
    │   ├── parameter.py
    │   ├── BatchCall.sh
    │   └── output.txt
    ├── 1/
    └── ...
```

##### Core Files

##### `script.py`

**Purpose**: Main orchestration script for parameter scanning and job submission.

**Key Features**:

- Generates N simulation directories with randomized parameters
- Creates SLURM batch scripts for each simulation
- Manages job queue to respect SLURM limits
- Handles parameter sampling within defined ranges

**Configuration Variables**:

```python
MAX_JOBS = 495          # Maximum concurrent SLURM jobs
WAIT_TIME = 5           # Seconds between queue checks
N = 2                   # Number of simulations to generate
```

**Parameter Ranges**:

- `fluctuation_amplitude`: (10, 10) - Fixed value
- `target_volume`: (25, 150) - Cell target volume
- `target_surface`: (0.5, 2) - Fraction of optimal surface (4√V)
- `lambda_volume`: (5, 15) - Volume constraint strength
- `lambda_surface`: (0, 15) - Surface constraint strength
- `Medium_cell_CE`: (0, 15) - Medium-cell contact energy

**Usage**:

```bash
python script.py
```

##### `simulation.py`

**Purpose**: CompuCell3D simulation implementation using the Cellular Potts Model.

**Simulation Parameters**:

- Grid size: 100×100 with periodic boundaries
- Neighbor order: 2 (for energy calculations)
- Skip time: 100 MCS (Monte Carlo Steps)
- Total simulation time: 1000 MCS

**Plugins Used**:

- `PottsCore`: Core Cellular Potts Model
- `CenterOfMassPlugin`: Tracks cell centers
- `PixelTrackerPlugin`: Monitors pixel-level changes
- `VolumePlugin`: Volume constraints
- `SurfacePlugin`: Surface constraints
- `ContactPlugin`: Cell-medium contact energy

**Output**: Writes to `output.txt` for each cell:

```
average_volume std_volume average_surface std_surface
```

**Multiple Simulation Runs**: The script runs N=10 independent simulations sequentially to gather statistics.

##### `parameter.py`

**Purpose**: Template configuration file for simulation parameters.

**Default Parameters**:

```python
fluctuation_amplitude = 10
target_volume = 10
target_surface = 10
lambda_volume = 1
lambda_surface = 1
Medium_cell_CE = 0
```

**Note**: This file is overwritten in each simulation directory with sampled parameters.

##### Data Processing Files

##### `consolidation_of_data.py`

**Purpose**: Aggregates results from all simulation runs into a single output file.

**Process**:

1. Searches for directories containing `__pycache__` (indicator of completed Python execution)
2. Reads `output.txt` (simulation results) from each directory
3. Reads `parameter.py` to extract parameter values
4. Calculates mean values across all cells in each simulation
5. Writes consolidated data to `averages_output.dat`

**Output Format**:

```
param1  param2  param3  param4  param5  param6  avg_vol  avg_vol_std  avg_surf  avg_surf_std
```

**Usage**:

```bash
python consolidation_of_data.py
```

**Output File**: `averages_output.dat`

##### Utility Files

##### `count_files_without_results.py`

**Purpose**: Diagnostic tool to count simulation directories.

**Function**: Counts directories containing `__pycache__` in the `Sims` folder.

**Usage**:

```bash
python count_files_without_results.py
```

##### `delete_files_without_results.py`

**Purpose**: Cleanup tool to remove incomplete simulations.

**Function**: Deletes subdirectories in `Sims` that don't contain `result.dat`.

**Usage**:

```bash
python delete_files_without_results.py
```

**Warning**: This permanently deletes directories. Use with caution.

##### `delete_files_above_threshold.py`

**Purpose**: Cleanup tool to remove simulation directories above a numeric threshold.

**Configuration**:

```python
base_path = "/u/[YOUR-IU-USERNAME]/[FILE PATH]/Sims"
threshold = 683154 - 101
```

**Safety Feature**: Requires user confirmation before deletion.

**Usage**:

```bash
python delete_files_above_threshold.py
# Type 'yes' when prompted to confirm
```

##### Workflow

 **Generate and Submit Simulations**

```bash
# Edit script.py to set N (number of simulations)
python script.py
```

This will:

- Create N directories in `Sims/`
- Generate random parameters for each
- Submit SLURM jobs (respecting MAX_JOBS limit)

 **Monitor Job Progress**

```bash
squeue -u [YOUR-IU-USERNAME]
```

**Check Completion**

```bash
python count_files_without_results.py
```

**Consolidate Results**

```bash
python consolidation_of_data.py
```

This generates `averages_output.dat` with all results.

**Clean Up**

```bash
# Remove incomplete runs
python delete_files_without_results.py

# Remove runs above threshold
python delete_files_above_threshold.py
```

#### SLURM Configuration

**Batch Script Template** (auto-generated):

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -J simulation
#SBATCH --time=10:00:00
#SBATCH --mail-user=[YOUR-IU-USERNAME]@iu.edu
#SBATCH -A r00128
#SBATCH -o /N/scratch/[YOUR-IU-USERNAME]/NNM_CC3D/out.txt
#SBATCH -e /N/scratch/[YOUR-IU-USERNAME]/NNM_CC3D/error.err
```

**Customization**: Edit the `generate_sbatch_string()` function in `script.py` to modify:

- Job time limits
- Email notifications
- Account allocation
- Output/error file paths

## Output Data Structure

### Simulation Output (`output.txt`)

Each simulation produces one line per cell:

```
avg_volume  std_volume  avg_surface  std_surface
```

Example:

```
98.234 5.123 124.567 8.901
97.856 4.987 123.234 9.012
```

### Consolidated Output (`averages_output.dat`)

Tab-separated file with columns:

```
fluctuation_amplitude  target_volume  target_surface  lambda_volume  lambda_surface  Medium_cell_CE  mean_avg_vol  mean_std_vol  mean_avg_surf  mean_std_surf
```

## Parameter Descriptions

### Simulation Parameters

**fluctuation_amplitude**: Membrane fluctuation amplitude in the Potts model. Higher values allow more random cell shape changes.

**target_volume**: Target volume for cells. Cells are penalized for deviating from this value.

**target_surface**: Target surface area for cells. Calculated as a fraction of the optimal surface (4√V).

**lambda_volume**: Strength of volume constraint. Higher values enforce stricter volume maintenance.

**lambda_surface**: Strength of surface constraint. Higher values enforce stricter surface area maintenance.

**Medium_cell_CE**: Contact energy between cells and the medium. Affects cell adhesion behavior.

##### Troubleshooting

**Issue: Jobs not submitting**

**Solution**: Check that `get_total_slurm_job_count() < MAX_JOBS`. Increase WAIT_TIME if needed.

**Issue: No `__pycache__` directories found**

**Solution**: Simulations may not have completed or Python didn't create cache. Check SLURM output files.

**Issue: Empty `output.txt` files**

**Solution**: Simulation may have crashed. Check error logs in SLURM error files.

**Issue: Consolidation script fails**

**Solution**: Ensure all simulations have completed and `parameter.py` files are properly formatted.



## Neural Network Configuration

The following section includes directions on how to interpret and handle only the sections of code that are most crucial for training the model. The creation of this neural network was guided via the tutorial linked below, and should be used as a reference and provide rationale for the "best practices" followed. For additional information about other components of the file, please refer to the documentation for PyTorch and Pandas here:

**PyTorch Documentation:** https://docs.pytorch.org/docs/stable/index.html 

**Pandas Documentation:** https://pandas.pydata.org/docs/

**PyTorch Tutorial:** https://www.youtube.com/watch?v=V_xro1bcAuA

##### **Device Configuration**

```python
device = torch.device("cuda:0")
```

The device is the computational unit where tensors are stored and computations are performed. Device configuration allows the user to select either a GPU or a CPU for model training. If a GPU is unavailable, replace`cuda:0`with`cpu`understanding that training the model will take longer on average. 

##### **Data Loading**

```python
data = np.loadtxt("averages_output.dat")
torch_tensor = torch.from_numpy(data)

X_data = torch_tensor[:, 1:6]
y_data = torch_tensor[:, 6:]
```

Here, the data is loaded: the`np.loadtxt`function loads the data from the`.dat`file. As established in earlier steps, this data should not include any other information other than the raw data. This means the file should not have any headings or header text that describes the information contained in each column.

Include the correct file name to ensure appropriate data loading. For PyTorch to be able to handle the data, it must be transformed in to a tensor via the`torch.from_numpy()`function. The data is then split into x and y features based on the included data. Adjust the range of values parsed by`torch_tensor`as needed for the input data.

##### **Train-Test Split**

```python
train_split = int(0.8 * len(X_combined))
X_train, X_test = X_combined[:train_split], X_combined[train_split:]
y_train, y_test = y_combined[:train_split], y_combined[train_split:]
```

The data must be partitioned such that it can be used for training the model and evaluating it. An 80/20 split was utilized for the final simulation, which can be adjusted by changing the value of 0.8 to any other desired fractional split. For the dataset used, this split resulted in 800,000 datapoints designated for training and 200,000 datapoints designated for testing. 

##### **Data Labeling**

```python
headers = ["T", "TV", "TS", "LV", "LS", "CE", "MV", "STDV", "MS", "STDS"]

## In order: "Temperature", "Total Volume", "Total Surface", "Lambda Volume", "Lambda Surface", "Contact Energy", "Mean Volume", "Standard Deviation of Volume", "Mean Surface", and "Standard Deviation of Surface"

df = pd.read_csv('averages_output.dat', delimiter='\t', header=None, names=headers) 
```

Headings for each of the columns of data in the loaded`.dat`file, or referred to as "headers" in the code, are added at this step. Headers aid in data organization and ease data access. This stage is crucial for ensuring that the correct data has been accessed by the x and y tensors. See the comment in the code block above for the meanings of the abbreviations. After the list of headers is declared, the dataset is loaded using the`read_csv()`function and headers are assigned. 

##### **Model Architecture**

```python
class NonLinearModule(nn.Module):
    def __init__(self):
	    super().__init__()
    
        self.layer_1 = nn.Linear(5, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 4)
        self.relu = nn.ReLU()
        
        def forward(self, x):
	        return (self.layer_4(self.relu(self.layer_3(self.relu
	        (self.layer_2(self.relu(self.layer_1(x))))))))
```

The model structure is defined at this step. The NonLinearModule utilizes four layers, with `ReLU` functions in between each successive layer of the forward() function. The `ReLU` function is used to introduce non-linearity into the model, which allows the model to learn the complex patterns associated with the dataset. For more information, reference the tutorial mentioned at the beginning of this section.

The number of layers can be increased (or decreased) depending on the complexity of the relationship between the input and output data. This can be done by declaring an additional self.layer (e.g. `self.layer_5`), designating the desired number of in and out features, and updating the return statement in the`forward()`function accordingly. 

The number of features within layers can also be adjusted. The number of in features in the first layer and the number of out features in the final layer must correspond to the amount of input and output parameters. Otherwise, these values can be changed as desired to obtain a precise and accurate model. 

##### **Loss Function and Optimizer**

```python
loss = nn.L1Loss()
optimizer = torch.optim.Adam(params=model_3.parameters(), lr=0.1)
```

The loss function and optimizer utilized are assigned at this step. `L1Loss` is utilized for its robustness to outliers, which can strongly impact the accuracy of model predictions. `Adam` is utilized for its faster convergence and ease of use in model tuning compared to other optimizers. See PyTorch documentation for additional options and use cases for both. The learning rate is also assigned via the `lr` variable within the optimizer function.

The learning rate must be adjusted manually. Begin with a learning rate of 0.1 for the first 50-100 epochs. After running the training loop, decrease the learning rate by a power of 10 (i.e. 0.01). Ensure that you re-run the code block containing the loss and optimizer function after adjusting the learning rate before running the training loop. Repeat this process until the train and test loss values plateau and do not change significantly after learning rate adjustments.

##### **Batch Sizes**

```python
batch_size = 8192
```

Specify the appropriate batch size and adjust according to dataset size. Larger batch sizes can increase training speed but may provide less accurate results. Smaller batch sizes may be more computationally expensive, have greater stochasticity, and train with more noise. However, small batches can lead to better generalization. When training the model, a moderately large batch size was selected due to available GPU processing power and the need for faster training times. Utilize PyTorch documentation, the PyTorch tutorial, and other relevant sources to re-assess batch size if needed.

##### **Training Loop**

```python
for epoch in range(epochs):
    model_3.train()
    for batch_X, batch_y in train_dataloader:
        y_pred = model_3(batch_X.float())       # Forward pass
        loss_val = loss(y_pred, batch_y)        # Calculate loss
        optimizer.zero_grad()                   # Clear gradients
        loss_val.backward()                     # Backpropagation
        optimizer.step()                        # Update weights
    
    # ...
    
	if epoch % 10 == 0:
		epoch_count.append(epoch)
		loss_values.append(avg_train_loss)
		test_loss_values.append(avg_test_loss)
		print(f"Epoch: {epoch} | Train Loss: {avg_train_loss:.6f}
		| Test Loss: {avg_test_loss:.6f}")
```

The training loop follows the standard structure for a machine learning model. The following steps are designated by the comments in the code block above: a forward pass is made over the training data; a loss calculation is performed; the gradients are cleared; backpropagation is performed; the weights are updated. Additional information is included in the PyTorch documentation and tutorial.

The statement at the end of the code block allows for active tracking of the train and test loss during model training. Every 10 epochs, the current epoch is printed in addition to the average train and test loss for the model. This information is crucial for understanding the overall progress of the model in the training process and provides insight into when the learning rate must be adjusted. 

##### **Prediction Accuracy Assessment**

```python
output = []

TV = 134
TS = 25.67
LV = 13.87
LS = 14.88
CE = 0.115

input_data = torch.tensor([TV, TS, LV, LS, CE], dtype=torch.float32).to(device)
model = model_3
with torch.no_grad():
outputm = model(input_data)
print(outputm)
```

To assess the accuracy of the model's prediction, the following rudimentary yet effective system shown in the above code segment is utilized. Values are accessed from the labeled dataset, and the input variables of a given simulation are assigned. The data is fed into the model as input, which then generates output data based on the training. This output is printed and can be compared to the values of the variables of interest for the same simulation. 

Different values can be assigned in the variable declaration section. Other input variables of interest can also be designated or altered by changing the list associated with the torch.tensor() call for the input_data variable. 

##### **Model Saving**

```python
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "CC3D_final.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(model_3.state_dict(), f=MODEL_SAVE_PATH)
```

Here, the model's state is saved. To avoid re-training the model each time you load it, this code block can be executed to save the model and reload it at another time. A `models`folder is created in the working directory, and within it, the `.pth` file stores the `state_dict` of a PyTorch model, which is a Python dictionary containing the learned parameters of the model's layers. The name of the model can be changed in the `MODEL_NAME`variable.

##### **Model Loading**

```python
model_3 = NonLinearModule()

model_3.load_state_dict(torch.load('models/CC3D_final.pth'))

model_3.eval()

model_3.to(device)
```

The above code block allows for the loading of a model from a saved state. The `NonLinearModule()`is reinstantiated, and `torch.load`utilizes the path of the .pth file generated from the `torch.save`function. To load the items, first initialize the model and optimizer, then load the dictionary locally using `torch.load()`.
