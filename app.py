from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)


model = joblib.load('model.pkl')

def convert_to_numeric(form):
    return {
        'age': float(form['age']),
        'anaemia': 1 if form['anaemia'] == 'Yes' else 0,
        'creatinine_phosphokinase': float(form['creatinine_phosphokinase']),
        'diabetes': 1 if form['diabetes'] == 'Yes' else 0,
        'ejection_fraction': float(form['ejection_fraction']),
        'high_blood_pressure': float(form['high_blood_pressure']),
        'platelets': float(form['platelets']),
        'serum_creatinine': float(form['serum_creatinine']),
        'serum_sodium': float(form['serum_sodium']),
        'sex': 1 if form['sex'] == 'Male' else 0,
        'smoking': 1 if form['smoking'] == 'Yes' else 0,
        'time': float(form['time'])
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = convert_to_numeric(request.form)
    df = pd.DataFrame([user_input])
    prediction = model.predict(df)[0]
    result = "Likely to Survive" if prediction == 0 else "High Risk of Death"
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
