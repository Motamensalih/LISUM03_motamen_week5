from flask import Flask, jsonify, request
import pickle
import pandas as pd

app = Flask(__name__)


@app.route('/' , methods = ['GET' , 'POST'])
def home():
	if(request.method =='GET'):
		data = "Hello World"
		return jsonify({'data':data})
		
		
@app.route('/predict/')
def tumor_predict():
    model = pickle.load(open('model.pkl' , 'rb'))
    mean_radius = request.args.get('mean_radius')
    mean_texture = request.args.get('mean_texture')
    mean_perimeter = request.args.get('mean_perimeter')
    mean_area = request.args.get('mean_area')
    mean_smoothness = request.args.get('mean_smoothness')

    

    test_df = pd.DataFrame({'mean_radius':[mean_radius] , 'mean_texture':[mean_texture] ,
    'mean_perimeter':[mean_perimeter] , 'mean_area':[mean_area] , 'mean_smoothness':[mean_smoothness]})

    tumor_pred = model.predict(test_df)
    if tumor_pred == 1:
       return jsonify({'result': 'tumor is benign , it will not spread'})
    else:
       return jsonify({'result': 'tumor is malignant , it will spread'})
        
if __name__ =='__main__':
    app.run(port = 5000 , debug = True)