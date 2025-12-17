# Differentially Private Synthetic Tabular Data Generation

This project generates synthetic tabular data using deep learning models with Differential Privacy.

---

## Models
- DP-CTGAN
- DP-DCF Diffusion

---

## Project Structure
agents/  
backend/  
frontend/  
models/  
training/  
data/  
results/  
requirements.txt  

---

## Requirements
- Python 3.9+
- PyTorch
- FastAPI
- Uvicorn
- NumPy
- Pandas
- Scikit-learn
- Opacus

---

## Setup
python -m venv .venv  
.venv\Scripts\activate  (Windows)  
source .venv/bin/activate  (Linux/Mac)  
pip install -r requirements.txt  

---

## Run Backend
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000  
Open: http://127.0.0.1:8000  

---

## Run Frontend
Open frontend/index.html in browser  

---

## Train Models
python training/dp_ctgan_train.py  
python training/dp_dcf_diffusion_train.py  

---

## Generate Data
Use frontend or API  
Download CSV output  

---

## Evaluate
python training/evaluation.py  

---

## Author
Deval Jadav
