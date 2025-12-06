import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Generate data dummy untuk training
np.random.seed(42)
n_samples = 1000

# Generate data dengan distribusi yang realistis
data = {
    'Academic Pressure': np.random.randint(0, 6, n_samples),  # 0-5 (integer)
    'CGPA': np.round(np.random.uniform(2.0, 10.0, n_samples), 2),  # 2.0-10.0 (decimal)
    'Study Satisfaction': np.random.randint(0, 6, n_samples)  # 0-5 (integer)
}

df = pd.DataFrame(data)

# Buat target variabel (contoh: tingkat stress/risk)
# Logika: Academic Pressure tinggi + CGPA rendah + Satisfaction rendah = Risk tinggi
def calculate_risk(row):
    risk_score = (
        (5 - row['Study Satisfaction']) * 0.4 +  # Satisfaction rendah = risk tinggi
        (5 - row['Academic Pressure']) * 0.2 +    # Pressure tinggi = risk tinggi (inverted)
        (10 - row['CGPA']) / 10 * 0.4            # CGPA rendah = risk tinggi
    )
    
    if risk_score < 0.3:
        return 0  # Low Risk
    elif risk_score < 0.6:
        return 1  # Medium Risk
    else:
        return 2  # High Risk

df['Risk_Level'] = df.apply(calculate_risk, axis=1)

# Pisahkan fitur dan target
X = df[['Academic Pressure', 'CGPA', 'Study Satisfaction']]
y = df['Risk_Level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model Random Forest
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Evaluasi model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")

# Simpan model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✅ Model berhasil disimpan sebagai 'model.pkl'")
print(f"\nFitur yang digunakan: {list(X.columns)}")
print(f"Target classes: {sorted(y.unique())}")

# Tampilkan contoh prediksi
print("\n📊 Contoh Prediksi:")
sample_data = pd.DataFrame({
    'Academic Pressure': [2, 4, 1],
    'CGPA': [8.5, 3.5, 9.0],
    'Study Satisfaction': [4, 1, 5]
})
predictions = model.predict(sample_data)
probabilities = model.predict_proba(sample_data)

for i, (idx, row) in enumerate(sample_data.iterrows()):
    print(f"\nSample {i+1}:")
    print(f"  Academic Pressure: {row['Academic Pressure']}")
    print(f"  CGPA: {row['CGPA']}")
    print(f"  Study Satisfaction: {row['Study Satisfaction']}")
    print(f"  Prediksi: Risk Level {predictions[i]}")
    print(f"  Probabilitas: {dict(zip([f'Class {j}' for j in range(3)], probabilities[i]))}")
