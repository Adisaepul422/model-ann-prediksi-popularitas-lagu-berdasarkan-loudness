from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('ann_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        loudness = float(request.form.get('loudness', 0))
        
        input_scaled = scaler.transform([[loudness, 0]])[:, 0].reshape(1, -1)
        
        prediction_scaled = model.predict(input_scaled, verbose=0)[0][0]
        
        prediction = scaler.inverse_transform([[0, prediction_scaled]])[0][1]
        prediction = max(0, min(100, prediction))
        
        if prediction >= 80:
            popular_text = "🔥 SANGAT POPULER!"
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
                             popular_text=popular_text,
                             color=color)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)