# PyTorch Model Web Deployment

A simple, sleek web interface for your PyTorch model that takes 5 inputs and returns 5 outputs.

## Setup Instructions

### 1. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Replace the demo model in app.py with your actual model
# Look for the TODO comment and replace with:
model = torch.load('your_model.pth')
model.eval()
model.to(device)

# Run the server
python app.py
```

Visit `http://localhost:8000` in your browser.

### 2. Update Model Loading

In `app.py`, replace the demo model section with your actual model:

```python
# Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('your_model.pth', map_location=device)
model.eval()
model.to(device)
```

Or if you saved with `torch.save(model.state_dict(), 'model.pth')`:

```python
from your_model_file import YourModelClass

model = YourModelClass()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()
model.to(device)
```

### 3. Customize Output Labels (Optional)

Edit `static/index.html` to change the output labels from "Output 1-5" to meaningful names:

```html
<span class="output-label">Your Label Name:</span>
```

## Deployment Options

### Option A: Render (Recommended - Free Tier Available)

1. Push your code to GitHub
2. Go to [render.com](https://render.com)
3. Create a new "Web Service"
4. Connect your GitHub repo
5. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free tier works fine for light usage

### Option B: Railway

1. Push code to GitHub
2. Go to [railway.app](https://railway.app)
3. Click "New Project" → "Deploy from GitHub"
4. Railway auto-detects the Python app and deploys

### Option C: Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose "Docker" as the SDK
3. Add a Dockerfile:

```dockerfile
FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Option D: Local Network

Just run `python app.py` and access from any device on your network using your computer's IP address.

## Project Structure

```
.
├── app.py              # FastAPI backend
├── static/
│   └── index.html      # Frontend interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## API Endpoint

**POST** `/predict`

Request body:
```json
{
  "TV": 134.0,
  "TS": 25.67,
  "LV": 13.87,
  "LS": 14.88,
  "CE": 0.115
}
```

Response:
```json
{
  "output1": 0.1234,
  "output2": 0.5678,
  "output3": 0.9012,
  "output4": 0.3456,
  "output5": 0.7890
}
```

## Troubleshooting

**Model too large?** If your model is >100MB, consider:
- Using Git LFS for version control
- Hosting the model separately (S3, Hugging Face Hub)
- Loading it from a URL at startup

**Slow predictions?** 
- The free tiers use CPU. For GPU, upgrade to paid tiers on Render/Railway
- Consider batching if you have multiple predictions

**CORS errors?**
- Already configured in `app.py`, should work out of the box

## Security Notes

For production deployment:
- Add input validation
- Implement rate limiting
- Add authentication if needed
- Use environment variables for sensitive config
