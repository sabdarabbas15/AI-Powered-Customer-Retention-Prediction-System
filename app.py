from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('Normalizer.pkl', 'rb'))

# Final columns (VERY IMPORTANT)
final_cols = [
    'SeniorCitizen', 'tenure_mad', 'MonthlyCharges_mad',
    'TotalCharges_mode_mad', 'gender_Male', 'Partner_Yes',
    'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check',
    'Networks_BSNL', 'Networks_Idea', 'Networks_Jio',
    'Contract_od'
]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        df = pd.DataFrame([{
            'SeniorCitizen': int(data['SeniorCitizen']),
            'tenure_mad': float(data['tenure']),
            'MonthlyCharges_mad': float(data['MonthlyCharges']),
            'TotalCharges_mode_mad': float(data['TotalCharges'])
        }])

        def set_col(x): df[x] = 1

        # Encoding
        if data['gender'] == 'Male': set_col('gender_Male')
        if data['Partner'] == 'Yes': set_col('Partner_Yes')
        if data['Dependents'] == 'Yes': set_col('Dependents_Yes')
        if data['PhoneService'] == 'Yes': set_col('PhoneService_Yes')

        if data['MultipleLines'] == 'No phone service':
            set_col('MultipleLines_No phone service')
        elif data['MultipleLines'] == 'Yes':
            set_col('MultipleLines_Yes')

        if data['InternetService'] == 'Fiber optic':
            set_col('InternetService_Fiber optic')
        elif data['InternetService'] == 'No':
            set_col('InternetService_No')

        for col in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']:
            if data[col] == 'No internet service':
                set_col(col + '_No internet service')
            elif data[col] == 'Yes':
                set_col(col + '_Yes')

        if data['PaperlessBilling'] == 'Yes':
            set_col('PaperlessBilling_Yes')

        if data['PaymentMethod'] == 'Credit card (automatic)':
            set_col('PaymentMethod_Credit card (automatic)')
        elif data['PaymentMethod'] == 'Electronic check':
            set_col('PaymentMethod_Electronic check')
        elif data['PaymentMethod'] == 'Mailed check':
            set_col('PaymentMethod_Mailed check')

        if data['Networks'] == 'BSNL': set_col('Networks_BSNL')
        elif data['Networks'] == 'Idea': set_col('Networks_Idea')
        elif data['Networks'] == 'Jio': set_col('Networks_Jio')

        if data['Contract'] != 'Month-to-month':
            set_col('Contract_od')

        # Fill missing columns
        for col in final_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[final_cols]

        # Scale
        scaled = scaler.transform(df)

        # Predict
        pred = model.predict(scaled)[0]

        result = "⚠️ CHURN Customer Will LEAVE " if pred == 1 else "✅ CHURN Customer Will STAY"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)