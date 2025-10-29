from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import uvicorn
from torch import nn

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NonLinearModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(5, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 4)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return (self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))))))

# Initialize model architecture
model = NonLinearModule()

model.load_state_dict(torch.load('CC3D_final.pth', map_location=device))

# Set to evaluation mode and move to device
model.eval()
model.to(device)

print(f"✓ Model loaded successfully on {device}")

class InputData(BaseModel):
    TV: float
    TS: float
    LV: float
    LS: float
    CE: float

class OutputData(BaseModel):
    output1: float
    output2: float
    output3: float
    output4: float

@app.post("/predict", response_model=OutputData)
async def predict(data: InputData):
    try:
        # Prepare input tensor
        input_tensor = torch.tensor(
            [data.TV, data.TS, data.LV, data.LS, data.CE], 
            dtype=torch.float32
        ).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Convert output to list
        output_values = output.cpu().numpy().tolist()
        
        # Return structured response (model outputs 4 values, not 5)
        return OutputData(
            output1=output_values[0],
            output2=output_values[1],
            output3=output_values[2],
            output4=output_values[3]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
