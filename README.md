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
