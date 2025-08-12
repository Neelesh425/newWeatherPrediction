from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Dummy dataset for demo purposes
# Replace this with your actual training data
X = [
    [20, 60, 5],   # temperature, humidity, wind_speed
    [30, 40, 10],
    [25, 70, 3]
]
y = ['Sunny', 'Rainy', 'Cloudy']  # corresponding weather labels

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model
os.makedirs('model', exist_ok=True)
joblib.dump(model, os.path.join('model', 'weather_model.pkl'))

print("✅ New model trained and saved.")