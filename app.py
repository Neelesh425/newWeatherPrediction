# app.py
from flask import Flask, render_template, request, session
import joblib
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# Load model (robustly)
MODEL_PATH = os.path.join('model', 'weather_model.pkl')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file not found. Run `python train_model.py` first to create model/weather_model.pkl"
    )
model = joblib.load(MODEL_PATH)

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

def feels_like_c(temp_c, humidity_pct, wind_mps):
    """
    Rough 'feels like' calculation:
    - Heat index for warm conditions (≥27°C)
    - Wind chill for cold conditions (≤10°C and wind > 1.3 m/s)
    Otherwise returns ambient temperature.
    Formulas simplified for demo purposes.
    """
    t = temp_c
    h = humidity_pct
    w = wind_mps

    # Heat Index (approx) in C (converted from a simplified HI in F)
    if t >= 27:
        # Convert to Fahrenheit
        t_f = t * 9/5 + 32
        hi_f = -42.379 + 2.04901523*t_f + 10.14333127*h - 0.22475541*t_f*h \
               - 0.00683783*(t_f**2) - 0.05481717*(h**2) + 0.00122874*(t_f**2)*h \
               + 0.00085282*t_f*(h**2) - 0.00000199*(t_f**2)*(h**2)
        hi_c = (hi_f - 32) * 5/9
        return hi_c

    # Wind Chill (only meaningful when cold + some wind)
    if t <= 10 and w > 1.3:
        v = w * 3.6  # convert m/s to km/h for wind chill formula
        wc = 13.12 + 0.6215*t - 11.37*(v**0.16) + 0.3965*t*(v**0.16)
        return wc

    return t

@app.route('/', methods=['GET'])
def index():
    history = session.get('history', [])
    return render_template('index.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get('location', '').strip() or 'Unknown'
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])

        # Basic validation & clamping to sensible ranges
        temperature = clamp(temperature, -50, 60)
        humidity = clamp(humidity, 0, 100)
        wind_speed = clamp(wind_speed, 0, 60)

        input_data = [[temperature, humidity, wind_speed]]
        pred_label = model.predict(input_data)[0]

        # Probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            classes = model.classes_
            probs = sorted(
                [(cls, float(p) * 100.0) for cls, p in zip(classes, proba)],
                key=lambda x: x[1],
                reverse=True
            )
            top_conf = probs[0][1]
        else:
            probs = [(pred_label, 100.0)]
            top_conf = 100.0

        # Feels-like
        feels_like = feels_like_c(temperature, humidity, wind_speed)

        # Save to history (last 5)
        entry = {
            'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'location': location,
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'wind_speed': round(wind_speed, 1),
            'prediction': pred_label,
            'confidence': round(top_conf, 1),
        }
        history = session.get('history', [])
        history.insert(0, entry)
        session['history'] = history[:5]

        return render_template(
            'index.html',
            location=location,
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            prediction_result=pred_label,
            probs=probs,
            feels_like=feels_like,
            history=session['history']
        )
    except Exception as e:
        return render_template('index.html', error=str(e), history=session.get('history', []))

if __name__ == '__main__':
    # Run: python app.py
    app.run(debug=True)
