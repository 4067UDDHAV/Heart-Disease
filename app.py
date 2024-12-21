from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained stacking model
with open('model/heart_disease_model.pkl', 'rb') as file:
    stacking_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('home.html')  # Landing page with "Get Started"

@app.route('/form')
def form():
    return render_template('index.html')  # Form page

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form
    features = [float(x) for x in request.form.values()]  # Convert inputs to floats
    final_features = np.array([features])  # Convert to 2D array for the model
    
    # Use the stacking model to predict probability
    prediction_prob = stacking_model.predict_proba(final_features)  
    output_prob = prediction_prob[0][1]  # Extract probability of the positive class (heart disease)
    
    # Convert probability to percentage
    output_percentage = round(output_prob * 100, 2)
    
    # Display the probability in percentage
    return render_template(
        'index.html',
        prediction_text=f'Heart Disease Probability: {output_percentage}%'
    )

if __name__ == "__main__":
    app.run(debug=True)
