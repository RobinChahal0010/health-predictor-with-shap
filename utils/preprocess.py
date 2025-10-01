import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])
    
    X = df[['age','gender','height_cm','weight_kg','bmi',
            'daily_steps','sleep_hours','water_intake_liters',
            'stress_level','smoking','alcohol','diet_score']]
    y = df['health_risk_category']
    
    return X, y, le_gender
