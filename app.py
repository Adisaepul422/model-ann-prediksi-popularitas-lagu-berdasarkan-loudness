# app.py - Aplikasi Web Flask (hanya loudness)

from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model yang sudah dilatih
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input loudness dari form
        loudness = float(request.form.get('loudness', 0))
        
        # Prediksi popularitas
        prediction = model.predict([[loudness]])[0]
        
        # Batasi range 0-100
        prediction = max(0, min(100, prediction))
        
        # Interpretasi hasil berdasarkan loudness
        if loudness >= -4:
            level = "🔊 SANGAT KERAS"
        elif loudness >= -7:
            level = "🔉 KERAS"
        elif loudness >= -10:
            level = "🔈 NORMAL"
        elif loudness >= -13:
            level = "🔇 PELAN"
        else:
            level = "🤫 SANGAT PELAN"
        
        # Interpretasi popularitas
        if prediction >= 80:
            popular_text = "🔥 SANGAT POPULER! (Top Chart)"
            color = "success"
        elif prediction >= 60:
            popular_text = "📈 POPULER"
            color = "primary"
        elif prediction >= 40:
            popular_text = "👍 CUKUP POPULER"
            color = "warning"
        else:
            popular_text = "🌱 KURANG POPULER"
            color = "secondary"
        
        return render_template('index.html', 
                             loudness=loudness,
                             prediction=round(prediction, 1),
                             level=level,
                             popular_text=popular_text,
                             color=color)
    
    except Exception as e:
        return render_template('index.html', 
                             error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)