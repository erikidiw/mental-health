"""
Script untuk membuat model.pkl
Jalankan dengan: python create_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

print("=" * 50)
print("Membuat Model Machine Learning")
print("=" * 50)

# Generate data dummy untuk training
np.random.seed(42)
n_samples = 1000

print(f"\n📊 Membuat {n_samples} data dummy...")

# Generate data dengan distribusi yang realistis
data = {
    'Academic Pressure': np.random.randint(0, 6, n_samples),  # 0-5 (integer)
    'CGPA': np.round(np.random.uniform(2.0, 10.0, n_samples), 2),  # 2.0-10.0 (decimal)
    'Study Satisfaction': np.random.randint(0, 6, n_samples)  # 0-5 (integer)
}

df = pd.DataFrame(data)

# Buat target variabel (Risk Level: 0=Low, 1=Medium, 2=High)
# Logika: Academic Pressure tinggi + CGPA rendah + Satisfaction rendah = Risk tinggi
def calculate_risk(row):
    risk_score = (
        (5 - row['Study Satisfaction']) * 0.4 +  # Satisfaction rendah = risk tinggi
        row['Academic Pressure'] * 0.2 +          # Pressure tinggi = risk tinggi
        (10 - row['CGPA']) / 10 * 0.4             # CGPA rendah = risk tinggi
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

print(f"✅ Data berhasil dibuat")
print(f"   - Jumlah fitur: {X.shape[1]}")
print(f"   - Jumlah samples: {X.shape[0]}")
print(f"   - Target classes: {sorted(y.unique())}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🔄 Melatih model Random Forest...")

# Latih model Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5
)
model.fit(X_train, y_train)

# Evaluasi model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"✅ Model berhasil dilatih")
print(f"   - Training Accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
print(f"   - Test Accuracy: {test_score:.4f} ({test_score*100:.2f}%)")

# Simpan model
output_file = 'model.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(model, f)

print(f"\n💾 Model disimpan sebagai '{output_file}'")
print(f"   - Ukuran file: {os.path.getsize(output_file) / 1024:.2f} KB")

# Tampilkan contoh prediksi
print("\n" + "=" * 50)
print("📊 Contoh Prediksi:")
print("=" * 50)

sample_data = pd.DataFrame({
    'Academic Pressure': [2, 4, 1, 5, 0],
    'CGPA': [8.5, 3.5, 9.0, 2.5, 9.5],
    'Study Satisfaction': [4, 1, 5, 0, 5]
})

predictions = model.predict(sample_data)
probabilities = model.predict_proba(sample_data)

risk_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

for i, (idx, row) in enumerate(sample_data.iterrows()):
    print(f"\nSample {i+1}:")
    print(f"  Academic Pressure: {int(row['Academic Pressure'])}")
    print(f"  CGPA: {row['CGPA']:.1f}")
    print(f"  Study Satisfaction: {int(row['Study Satisfaction'])}")
    print(f"  ➜ Prediksi: {risk_labels[predictions[i]]} (Class {predictions[i]})")
    probs = {risk_labels[j]: f"{probabilities[i][j]:.2%}" for j in range(3)}
    print(f"  Probabilitas: {probs}")

print("\n" + "=" * 50)
print("✅ Selesai! File model.pkl siap digunakan di Streamlit app.")
print("=" * 50)
