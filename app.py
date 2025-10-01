from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import joblib
import shap
import json
import numpy as np
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv
from utils.preprocess import load_and_preprocess
from utils.translate import t
from rapidfuzz import fuzz
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

app = Flask(__name__)

model = joblib.load("models/health_model.pkl")

X, y, le_gender = load_and_preprocess("data/realistic_health_lifestyle_dataset.csv")

explainer = shap.TreeExplainer(model)


with open("qa.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

@app.route("/", methods=["GET","POST"])
def index():
    lang = request.args.get("lang", "en")
    if request.method == "POST":
        try:
            age = int(request.form["age"])
            gender = request.form["gender"]
            height_cm = float(request.form["height_cm"])
            weight_kg = float(request.form["weight_kg"])
            daily_steps = int(request.form["daily_steps"])
            sleep_hours = float(request.form["sleep_hours"])
            water_intake_liters = float(request.form["water_intake_liters"])
            stress_level = int(request.form["stress_level"])
            smoking = int(request.form["smoking"])
            alcohol = int(request.form["alcohol"])
            diet_score = int(request.form["diet_score"])
        except:
            return "Invalid Input!"

        bmi = round(weight_kg / ((height_cm/100)**2),1)
        gender_enc = le_gender.transform([gender])[0]

        input_df = pd.DataFrame([[age, gender_enc, height_cm, weight_kg, bmi,
                                  daily_steps, sleep_hours, water_intake_liters,
                                  stress_level, smoking, alcohol, diet_score]],
                                columns=['age','gender','height_cm','weight_kg','bmi',
                                         'daily_steps','sleep_hours','water_intake_liters',
                                         'stress_level','smoking','alcohol','diet_score'])

        pred = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df).max()

        shap_values_all = explainer.shap_values(input_df)
        shap_values = shap_values_all[1] if isinstance(shap_values_all, list) else shap_values_all

        shap_values_list = []
        tips = []
        feature_names = input_df.columns.tolist()

        for i, feature in enumerate(feature_names):
            shap_val = shap_values[0][i]
            if isinstance(shap_val, (np.ndarray, list)):
                shap_val = np.array(shap_val).flatten()[0]
            shap_val = float(shap_val)
            shap_values_list.append(shap_val)

            value = input_df.iloc[0, i]
            if shap_val > 0:
                tips.append(f"{feature.replace('_',' ').title()} = {value} is increasing your predicted health risk.")
            elif shap_val < 0:
                tips.append(f"{feature.replace('_',' ').title()} = {value} is contributing positively to your health score.")

        return render_template("result.html",
                               prediction=pred,
                               probability=round(pred_proba*100,2),
                               shap_values=json.dumps(shap_values_list),
                               feature_names=json.dumps(feature_names),
                               tips=tips,
                               lang=lang,
                               t=t)
    return render_template("index.html", lang=lang, t=t)


@app.route("/download", methods=["POST"])
def download_report():
    prediction = request.form.get("prediction")
    probability = request.form.get("probability")
    tips = json.loads(request.form.get("tips"))

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Health Risk Report")

    c.setFont("Helvetica", 14)
    c.drawString(50, height - 80, f"Prediction: {prediction}")
    c.drawString(50, height - 100, f"Confidence: {probability}%")
    c.drawString(50, height - 130, "Recommendations:")

    y = height - 150
    for tip in tips:
        c.drawString(60, y, f"- {tip}")
        y -= 20
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 14)
            y = height - 50

    c.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="health_report.pdf", mimetype="application/pdf")

@app.route("/email", methods=["POST"])
def email_result():
    email_address = request.form.get("email")
    prediction = request.form.get("prediction")
    probability = request.form.get("probability")
    tips = json.loads(request.form.get("tips"))

    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = email_address
    msg['Subject'] = "Your Health Risk Report"

    body = f"Prediction: {prediction}\nConfidence: {probability}%\n\nRecommendations:\n"
    for tip in tips:
        body += f"- {tip}\n"

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        return "Email sent successfully!"
    except Exception as e:
        print(e)
        return "Failed to send email."


@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_msg = request.json.get("message", "").lower()
    response = "Sorry, I didn't understand that. 🤔"
    max_score = 0

    for item in qa_data:
        for keyword in item["keywords"]:
            score = fuzz.partial_ratio(user_msg, keyword.lower())
            if score > 70 and score > max_score:  
                response = item["answer"]
                max_score = score

    return jsonify({"response": response})


@app.route("/chatbot_page")
def chatbot_page():
    return render_template("chatbot.html")  


if __name__ == "__main__":
    app.run(debug=True)
