import numpy as np
from flask import Flask, request, jsonify,render_template
import pickle


app = Flask(__name__)

# model = pickle.load(open("cropmodel.pkl", "rb"))

with open("cropmodel.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    if len(float_features) != 7:
            return "Error: Please provide all 7 input values"
    features = [np.array(float_features)]
    prediction = model.predict(features)
    print("Predicted crop:", prediction)
    return render_template('index.html', prediction_text='Recommended Crop is: {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True )