"""
Script sederhana untuk generate model.pkl
Jalankan: python generate_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Membuat model.pkl...")

# Data training
np.random.seed(42)
n = 1000

X = pd.DataFrame({
    'Academic Pressure': np.random.randint(0, 6, n),
    'CGPA': np.round(np.random.uniform(2.0, 10.0, n), 2),
    'Study Satisfaction': np.random.randint(0, 6, n)
})

# Target: Risk Level (0=Low, 1=Medium, 2=High)
y = np.where(
    (X['CGPA'] < 5) | (X['Study Satisfaction'] < 2) | (X['Academic Pressure'] > 4),
    np.where((X['CGPA'] < 3) | (X['Study Satisfaction'] == 0), 2, 1),
    0
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ model.pkl berhasil dibuat!")
print(f"   Ukuran: {len(pickle.dumps(model)) / 1024:.1f} KB")


