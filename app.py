from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

pipe = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        name = request.form.get('name')
        company = request.form.get('company')
        year = int(request.form.get('year'))
        kms_driven = int(request.form.get('kms_driven'))
        fuel_type = request.form.get('fuel_type')

        prediction = pipe.predict(pd.DataFrame([[name, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))[0]

        return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)