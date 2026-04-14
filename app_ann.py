from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('ann_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        loudness = float(request.form.get('loudness', 0))
        
        input_scaled = scaler.transform([[loudness, 0]])[:, 0].reshape(1, -1)
        
        prediction_scaled = model.predict(input_scaled)[0]
        
        prediction = scaler.inverse_transform([[0, prediction_scaled]])[0][1]
        prediction = max(0, min(100, prediction))
        
        if prediction >= 80:
            popular_text = "🔥 SANGAT POPULER!"
            color = "success"
            emoji = "🏆"
        elif prediction >= 60:
            popular_text = "📈 POPULER"
            color = "primary"
            emoji = "🎧"
        elif prediction >= 40:
            popular_text = "👍 CUKUP POPULER"
            color = "warning"
            emoji = "👍"
        else:
            popular_text = "🌱 KURANG POPULER"
            color = "secondary"
            emoji = "🌱"
        
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
        
        return render_template('index.html', 
                             loudness=loudness,
                             prediction=round(prediction, 1),
                             popular_text=popular_text,
                             color=color,
                             emoji=emoji,
                             level=level)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)