AI Warranty Validation System
This repository contains an educational prototype for automated warranty validation using computer vision and machine learning. The project demonstrates how to combine deep learning, optical character recognition (OCR), and database verification to assess the authenticity of warranty cards.

Note:
This is a small-scale, academic project originally developed as an internship assignment. It is not designed for commercial or production use, and is provided for learning, inspiration, and demonstration purposes only.

✨ Features
End-to-End Pipeline:
Image input → Preprocessing → Deep-learning classification → OCR text extraction → Database field matching → Result verdict.

Deep Learning Model:
Binary classification (authentic/fraudulent) using a ResNet18 CNN (PyTorch).

Smart OCR:
Multi-strategy OCR (EasyOCR) with custom postprocessing to correct common character errors (S↔5, Z↔2, etc).

Database Cross-Verification:
SQL database integration for warranty number and expiry date validation.

REST API:
FastAPI-based backend with clear endpoints, robust error handling, and configurable setup.

Desktop GUI:
Tkinter-based interface for quick local testing and demonstration.

Configurable & Portable:
All paths, models, and database URLs are user-configurable for easy setup on any system.

📦 Directory Structure
text
.
├── warranty_check.py           # API backend (FastAPI + ML + OCR)
├── warranty_validator_gui.py   # GUI client (Tkinter)
├── train_pytorch.py            # Model training script
├── predict_pytorch.py          # Prediction/inference script
├── requirements.txt
├── README.md
└── ...
🚀 Getting Started
Clone this repository

Install requirements

text
pip install -r requirements.txt
Edit paths/configs in the scripts to match your environment

Start the backend API

text
python warranty_check.py
# or, preferred: uvicorn warranty_check:app --reload
Run the GUI (optional)

text
python warranty_validator_gui.py
🛠️ Tech Stack
Python 3.8+

PyTorch

EasyOCR

FastAPI

SQLAlchemy + SQLite (or your choice of DB)

OpenCV

Tkinter

PIL

⚠️ Limitations
This project is an internship/academic prototype and is not hardened for production, security, concurrency, or large-scale deployment.

The CNN model requires further data, optimization, and evaluation for real-world use.

Database and GUI are built for demonstration and educational clarity—not for business usage.

🙏 Acknowledgments
Open-source projects: PyTorch, EasyOCR, FastAPI, SQLAlchemy, OpenCV

Community forums and documentation that supported debugging and integration

📖 License
Released under the MIT License.

Feel free to fork, adapt, and improve for your own learning or research! If you build improvements or extensions, contributions are welcome.
