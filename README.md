# Fraud Detection Dashboard - Spearhead

A real-time fraud detection pipeline demonstration with a live dashboard, showcasing Pure Storage FlashBlade and NVIDIA GPU acceleration.

## 🚀 Features

- **Real-time Dashboard** - Live metrics updating every second
- **4-Pod Architecture** - Data generation, feature engineering, model training, and inference
- **High Performance** - Processes 13.4M transactions in ~15 seconds
- **REST API Backend** - FastAPI server orchestrating pipeline execution
- **Interactive UI** - TailwindCSS-powered dashboard with progress tracking

## 📋 Prerequisites

- Python 3.11+
- 16GB+ RAM recommended
- Virtual environment (recommended)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anuj591/fraud_detection_spearhead.git
cd fraud_detection_spearhead
```

### 2. Create Virtual Environment

```bash
python3 -m venv myenv
source myenv/bin/activate  # On Linux/Mac
# OR
myenv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn pydantic polars pyarrow xgboost faker psutil
```

## 🏃 How to Run

### Option 1: Run with Dashboard (Recommended)

**Terminal 1 - Start Backend Server:**
```bash
cd fraud_detection_spearhead
source myenv/bin/activate
python3 backend_server.py
```

You should see:
```
======================================================================
Dashboard Backend v4 - Starting Server
======================================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 - Open Dashboard:**
```bash
# Open in browser
xdg-open dashboard-v4-preview.html
# Or navigate to: file:///path/to/dashboard-v4-preview.html
```

**Using the Dashboard:**
1. Click **"▶ Start"** to run the pipeline
2. Watch real-time metrics update:
   - Transaction count increasing
   - Business metrics (fraud prevented, transactions scored)
   - Pipeline progress bars
   - Timer incrementing
3. Click **"⏹ Stop"** to halt execution
4. Click **"↺ Reset"** to clear metrics

### Option 2: Run Pipeline Directly (No UI)

```bash
source myenv/bin/activate
python3 run_pipeline.py
```

Monitor output in terminal - you'll see telemetry logs like:
```
[TELEMETRY] stage=Ingest | status=Running | rows=1700000 | throughput=848664...
```

### Option 3: Test Backend API

```bash
# Get current dashboard data
curl http://localhost:8000/api/dashboard | python3 -m json.tool

# Start pipeline
curl -X POST http://localhost:8000/api/control/start

# Stop pipeline
curl -X POST http://localhost:8000/api/control/stop

# Reset metrics
curl -X POST http://localhost:8000/api/control/reset
```

## 📁 Project Structure

```
fraud_detection_spearhead/
├── backend_server.py           # FastAPI backend orchestrator
├── dashboard-v4-preview.html   # Live dashboard UI
├── run_pipeline.py             # Direct pipeline runner
├── pods/                       # 4 processing pods
│   ├── data-gather/           # Pod 1: Transaction generation
│   ├── data-prep/             # Pod 2: Feature engineering
│   ├── model-build/           # Pod 3: XGBoost training
│   └── inference/             # Pod 4: Triton inference client
├── run_data_output/           # Generated transaction data
├── run_features_output/       # Processed features
└── run_models_output/         # Trained XGBoost model
```

## 🔄 Pipeline Stages

### Stage 1: Data Generation (Pod 1)
- Generates 13.4M synthetic credit card transactions
- Uses 16 parallel workers
- Output: ~1.5GB Parquet files
- **Throughput:** ~1.1M rows/second

### Stage 2: Feature Engineering (Pod 2)
- Transforms raw data into ML features
- Creates time-based, geographic, and amount features
- Uses Polars for high-speed processing
- **Throughput:** ~3.9M rows/second

### Stage 3: Model Training (Pod 3)
- Trains XGBoost fraud detection model
- CPU-optimized training
- **Accuracy:** ~63%
- **Training time:** ~6 minutes for 13.4M samples

### Stage 4: Inference (Pod 4)
- Requires Triton Inference Server (optional)
- Validates model deployment
- ⚠️ Skipped if Triton not available

## 📊 Dashboard Metrics

The live dashboard displays:
- **Pipeline Progress** - Real-time row counts for each stage
- **Business Impact** - Fraud prevented ($), transactions scored, fraud blocked
- **Resource Utilization** - CPU, GPU, FlashBlade metrics
- **Throughput** - Transactions per second
- **Fraud Distribution** - Low/Medium/High risk breakdown

## 🛠️ API Endpoints

```
GET  /                         Health check
GET  /api/dashboard            Complete dashboard data (JSON)
POST /api/control/start        Start pipeline execution
POST /api/control/stop         Stop running pipeline
POST /api/control/reset        Reset all metrics
POST /api/control/scale        Update pod scaling (simulated)
GET  /api/control/scale        Get current scaling config
```

## 🧪 Testing

```bash
# Test backend endpoints
python3 -c "
import requests
r = requests.get('http://localhost:8000/api/dashboard')
print(r.json())
"
```

## 📝 Output Files

After running the pipeline, check these directories:

**Generated Data:**
```bash
ls -lh run_data_output/run_*/
# ~134 parquet files, 1.5GB total
```

**Processed Features:**
```bash
ls -lh run_features_output/
# features_run_*.parquet
```

**Trained Model:**
```bash
ls -lh run_models_output/
# fraud_xgboost (XGBoost model file)
```

## 🎯 Performance Benchmarks

| Stage | Rows Processed | Time | Throughput |
|-------|---------------|------|------------|
| Data Generation | 13.4M | 12s | 1.1M rows/s |
| Feature Engineering | 13.4M | 3.5s | 3.9M rows/s |
| Model Training | 13.4M | 356s | 37K rows/s |
| **Total Pipeline** | 13.4M | ~6min | End-to-end |

## 🐛 Troubleshooting

**Backend won't start:**
```bash
# Check if port 8000 is in use
lsof -i :8000
# Kill process if needed
kill -9 <PID>
```

**Dashboard not updating:**
- Open browser DevTools (F12) → Console tab
- Check for API connection errors
- Verify backend is running on http://localhost:8000

**Import errors:**
```bash
# Ensure all dependencies installed
pip install -r requirements.txt  # If available
# Or install manually
pip install fastapi uvicorn pydantic polars pyarrow xgboost faker psutil
```

**Inference stage fails:**
- This is expected - Triton server not configured
- Pipeline completes first 3 stages successfully
- Model is saved and ready for deployment

## 🔐 .gitignore

The following are excluded from git:
- `run_data_output/` - Generated transaction data (large)
- `run_features_output/` - Processed features (large)
- `run_models_output/` - Trained models (large)
- `myenv/` - Virtual environment
- `__pycache__/` - Python cache

## 📄 License

This is a demonstration project for Pure Storage fraud detection pipeline.

## 👥 Contributing

This is a demo project. For production use, consider:
- Adding actual GPU acceleration
- Implementing Kubernetes pod orchestration
- Integrating real Triton Inference Server
- Adding authentication to API
- Implementing WebSocket for real-time updates

## 🙏 Acknowledgments

- **Pure Storage** - FlashBlade integration concepts
- **NVIDIA** - GPU acceleration framework
- **XGBoost** - Machine learning model
- **Polars** - High-performance data processing
