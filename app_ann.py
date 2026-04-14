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
        
        input_2d = np.array([[loudness, 0]])
        input_scaled = scaler.transform(input_2d)
        X_input = input_scaled[:, 0].reshape(-1, 1)
        
        prediction_scaled = model.predict(X_input)[0]
        
        pred_array = np.array([[0, prediction_scaled]])
        prediction = scaler.inverse_transform(pred_array)[0][1]
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
                             level=level)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    print("\n" + "="*50)
    print("WEB APP BERJALAN!")
    print("Buka browser: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True)