# ⚡ WeldSense

> **EV Battery Cell Weld Inspection System — Digital Prototype**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

WeldSense is a cutting-edge digital prototype designed for real-time quality assurance of EV battery cell welds. By leveraging multi-modal sensor fusion (visual, acoustic, and thermal data) and advanced machine learning algorithms, WeldSense can instantly detect anomalies and classify specific weld defects with high precision.

---

## 🚀 Key Features

*   **Multi-Modal Sensor Fusion**: Combines acoustic (audio), thermal, and visual (images) data for comprehensive weld inspection.
*   **Intelligent Anomaly Detection**: Identifies subtle irregularities in the welding process that might be missed by human inspection.
*   **Defect Classification**: Accurately classifies specific types of defects using trained AI models.
*   **Interactive Dashboard**: A powerful, real-time Streamlit dashboard (`dashboard/app.py`) for monitoring, visualization, and analytics.
*   **One-Click Pipeline**: Seamlessly generate synthetic data, train models, and launch the dashboard with a single command.

## 🛠️ Technology Stack

*   **Core**: Python, NumPy, Pandas, SciPy
*   **Machine Learning**: Scikit-Learn, Joblib
*   **Computer Vision & Audio**: OpenCV, Pillow, Librosa
*   **Visualization & UI**: Streamlit, Plotly, Matplotlib

## 📂 Project Structure

```text
WeldSense/
├── dashboard/               # Streamlit application for real-time monitoring
│   └── app.py               # Main dashboard script
├── data/                    # Data storage and generation
│   ├── images/              # Visual inspection data
│   ├── thermal/             # Thermal imaging data
│   ├── audio/               # Acoustic emission data
│   └── generate_synthetic.py# Synthetic data generator
├── models/                  # Saved machine learning models (.pkl)
├── pipeline/                # Core processing and training pipeline
│   ├── pipeline_core.py     # Main pipeline logic
│   ├── sensors.py           # Sensor data processing
│   ├── models.py            # AI model architectures
│   ├── database.py          # Data persistence layer
│   └── train.py             # Model training script
├── import_real_data.py      # Utility to ingest real-world data
├── requirements.txt         # Project dependencies
└── run.py                   # One-click launcher
```

## 🏁 Quick Start

### 1. Prerequisites

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
python -m venv .venv
# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

WeldSense comes with a convenient one-click launcher that handles data generation, model training, and dashboard deployment automatically.

```bash
python run.py
```

**Workflow:**
1. Installs/verifies Python dependencies.
2. Generates synthetic data for visual, thermal, and acoustic sensors (if not already present).
3. Trains anomaly detection and defect classification AI models.
4. Launches the interactive Streamlit dashboard at `http://localhost:8501`.

**Optional Flags (for faster iteration):**
*   `--skip-data`: Skips the data generation step.
*   `--skip-train`: Skips the model training step.
*   *Example*: `python run.py --skip-data --skip-train` (Jumps straight to the dashboard)

---

## 🧠 AI Models

WeldSense employs a dual-model architecture:
1.  **Anomaly Detector**: A high-sensitivity model designed to flag any deviation from normal welding parameters, ensuring no potential issue is overlooked.
2.  **Defect Classifier**: A robust classifier that categorizes the flagged anomalies into specific defect types (e.g., porosity, lack of fusion, burn-through) based on the fused sensor data.

## 📈 Future Enhancements

*   **Deep Learning Integration**: Upgrade to PyTorch/TensorFlow for more complex visual and acoustic pattern recognition.
*   **XGBoost Risk Model**: Implement advanced gradient boosting for predictive maintenance and risk assessment.
*   **Cloud Deployment**: Dockerize the application for scalable cloud-based inspection.

---
*Developed for next-generation EV manufacturing quality control.*
