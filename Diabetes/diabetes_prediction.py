"""
Train the diabetes prediction model and save diabetes_model.pkl + scaler.pkl.
Run this script once: python diabetes_prediction.py
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def assign_diet(row):
    g, b, i = row['Glucose'], row['BMI'], row['Insulin']
    if g > 140:   return 0  # Carbohydrate Rich
    if b < 25 and g <= 110: return 1  # Protein Rich
    if b >= 33 and g <= 140: return 2  # Fat Rich
    return 3                            # Balanced

def train():
    df = pd.read_csv('diabetes.csv')
    df['DietType'] = df.apply(assign_diet, axis=1)

    X = df.drop(columns='Outcome')
    y = df['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {acc*100:.2f}%")

    joblib.dump(model,  'diabetes_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Saved diabetes_model.pkl and scaler.pkl")

if __name__ == '__main__':
    train()
