from flask import Flask,request,jsonify
import pickle
from model import Sentiment_Analysis



app = Flask(__name__)


def load():
    with open("model.pkl","rb") as f:
        model = pickle.load(f)

    return model


model = load()


@app.route("/predict",methods = ['POST','GET'])
def predict():
    try:
        if request.method == "GET":
            input_data = request.args.get('input','')
        else:
            data = request.get_json()
            input_data = data['input']

        if not isinstance(input_data, str) or not input_data:
            return jsonify({'error': 'Input must be a non-empty string'}), 400

        prediction = model.Predict(input_data)

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({"erroe": str(e)}),400

if __name__ == "__main__":
    app.run()



