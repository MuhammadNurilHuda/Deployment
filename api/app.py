from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('api/model.pkl', 'rb'))  # Load the trained model (pickle file)
scaler = pickle.load(open('api/minmax_scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
    # return jsonify({'Message':'Hello, World!'})

@app.route( '/predict', methods=['POST'])
def predict():
    try:
        feature = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh','exng', 'oldpeak', 'slp', 'caa', 'thall']
        data = [float(request.form[f]) for f in feature]
        data_array = np.array([data]).reshape(1, -1)
        data_normalized = scaler.fit_transform(data_array)
        prediction = model.predict(data_normalized)[0]

        if prediction == 1:
            msg = 'You have more chance of heart attack'
        else:
            msg = 'No heart attack expected'

        # response = {
        #     'prediction':msg
        # }
        
        return render_template('index.html', prediction = msg)
        # return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == "__main__":
    app.run(debug=True)