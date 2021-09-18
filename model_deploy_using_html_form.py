from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
		return render_template('index.html')
		
		
@app.route('/predict' , methods = ['POST'])
def tumor_predict():

    model = pickle.load(open('model.pkl' , 'rb'))
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
   
    tumor_pred = model.predict(final_features)
    if tumor_pred == 1:
       return render_template('index.html' , prediction_text = "tumor is benign , it will not spread")
    else:
       return render_template('index.html' , prediction_text = "tumor is malignant , it will spread")
        
if __name__ =='__main__':
    app.run(port = 5555 , debug = True)