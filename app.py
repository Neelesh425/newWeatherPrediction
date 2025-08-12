from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load the trained weather prediction model
MODEL_PATH = os.path.join('model', 'weather_model.pkl')
model = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form['location']
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])

        input_data = [[temperature, humidity, wind_speed]]
        prediction = model.predict(input_data)[0]

        return render_template(
            'index.html',
            location=location,
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            prediction_result=prediction
        )
    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
