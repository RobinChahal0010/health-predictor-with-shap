# 🩺 AI Health Risk Predictor

AI-powered **Health Risk Prediction Web App** built with **Flask + Machine Learning**.  
This project predicts health risks, explains model decisions with **SHAP values**, and provides **personalized recommendations**.  
It also supports **multilingual translation**, generates **PDF reports**, sends them via **Email**, and includes a **smart chatbot** for health FAQs.

---

## 🚀 Features

✅ **AI-Powered Predictions** – ML model trained on lifestyle & health dataset.  
✅ **Explainability (SHAP)** – See which factors increase/decrease your risk.  
✅ **BMI Calculation** – Auto-calculated based on height & weight.  
✅ **Downloadable PDF Report** – One-click professional health report.  
✅ **Email Integration** – Get your report directly in your inbox.  
✅ **Chatbot (FAQ)** – Smart chatbot with fuzzy matching for health queries.  
✅ **Multilingual Support** – Translate interface & tips easily.  
✅ **Secure Config** – Uses `.env` for credentials.  
✅ **User Authentication Ready** – SQLAlchemy + Flask-Login setup for future dashboards.

---

## 🖥️ Tech Stack

- **Frontend:** HTML, CSS, Jinja2 (Flask Templates)  
- **Backend:** Flask (Python)  
- **Database:** SQLite (can be upgraded to PostgreSQL/MySQL)  
- **ML/Explainability:** Scikit-Learn, SHAP  
- **Utils:** Pandas, NumPy, Joblib  
- **Reports/Email:** ReportLab, smtplib, dotenv  
- **Chatbot:** RapidFuzz (fuzzy keyword matching)  

---

## 📂 Project Structure
.
├── app.py # Main Flask app
├── models/health_model.pkl # Trained ML model
├── data/realistic_health_lifestyle_dataset.csv
├── templates/ # HTML templates (index, result, chatbot)
├── static/ # CSS/JS assets
├── utils/
│ ├── preprocess.py # Data preprocessing
│ └── translate.py # Multilingual translations
├── qa.json # Chatbot Q&A data
├── requirements.txt # Dependencies
└── README.md 

yaml
Copy code

---

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/health-risk-predictor.git
   cd health-risk-predictor
Create Virtual Environment

bash
Copy code
python -m venv venv
source venv/bin/activate     # (Linux/Mac)
venv\Scripts\activate        # (Windows)
Install Dependencies

bash
Copy code
pip install -r requirements.txt
Setup Environment Variables

Create a .env file in root:

ini
Copy code
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_app_password
Run the App

bash
Copy code
python app.py
Open browser → http://127.0.0.1:5000/
