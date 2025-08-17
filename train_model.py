# train_model.py
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---- Generate a better synthetic dataset (demo-friendly) ----
# Features: [temperature_C, humidity_pct, wind_speed_mps]
# Labels: 'Sunny', 'Cloudy', 'Rainy'
rng = np.random.default_rng(42)
N = 1200

# Ranges typical for many places
temps = rng.uniform(low=-5, high=45, size=N)       # °C
humid = rng.uniform(low=10, high=100, size=N)      # %
wind  = rng.uniform(low=0, high=20, size=N)        # m/s

X = np.column_stack([temps, humid, wind])

def label_rule(t, h, w):
    # Heuristic labeling for demo:
    # - High humidity or higher wind tends to push to Rainy.
    # - Moderate humidity or cooler temps tend to Cloudy.
    # - Warm + dry + low wind tends to Sunny.
    score_rain = 0.0
    score_cloud = 0.0
    score_sun = 0.0

    # Humidity influence
    if h >= 80: score_rain += 2.5
    elif h >= 60: score_cloud += 1.5
    else: score_sun += 1.0

    # Wind influence
    if w >= 12: score_rain += 1.5
    elif w >= 6: score_cloud += 1.0
    else: score_sun += 0.5

    # Temperature influence
    if t >= 30: score_sun += 2.0
    elif t <= 10: score_cloud += 1.5
    else: score_cloud += 0.8; score_sun += 0.4

    # Small randomness so the classes aren't perfectly separable
    jitter = rng.normal(0, 0.3)
    score_rain += jitter
    score_cloud += jitter / 2

    if score_rain >= max(score_cloud, score_sun):
        return 'Rainy'
    if score_sun >= max(score_cloud, score_rain):
        return 'Sunny'
    return 'Cloudy'

y = np.array([label_rule(t, h, w) for t, h, w in X])

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

# ---- RandomForest model ----
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=2,
    random_state=7,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ---- Simple evaluation (printed to console) ----
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Validation accuracy: {acc:.3f}")
print(classification_report(y_test, y_pred))

# ---- Save model ----
os.makedirs('model', exist_ok=True)
joblib.dump(model, os.path.join('model', 'weather_model.pkl'))
print("✅ New RandomForest model trained and saved to model/weather_model.pkl")
