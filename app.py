import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('diabetesPredictionModel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [str(x) for x in request.form.values()]

    for i in range(len(features)):
      if features[i] == 'Yes':
        features[i] = 1
      elif features[i] == 'Female':
        features[i] = 1
      elif features[i] == 'No':
        features[i] = 0
      elif features[i] == 'Male':
        features[i] = 0
      else:
      	features[i] = int(features[i])


    final_features = [np.array(features)]
    prediction = model.predict(final_features)


    if prediction[0] == 1:
      output = "You are at high risk of devloping diabete"
    else:
      output = "You are at low risk of devloping diabete"

    
    return render_template('index.html', prediction_text='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
