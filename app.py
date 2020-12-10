import numpy as np
# import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open('breast-cancer-model.pkl', 'rb'))


@app.route('/')
def hom():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/predict', methods=['POST'])
def predict():
   
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
        
    output = round(prediction[0], 1)

    if output == 2:
        res_val = "breast cancer"
    else:
        res_val = "no breast cancer"

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))
    
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
   
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
