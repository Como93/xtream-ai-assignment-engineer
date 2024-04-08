import flask
import pickle
import pandas as pd
from flask import request, render_template
import numpy as np

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods = ['GET'])
def main():
    if request.method == 'GET':
        return(render_template('main.html'))

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_form_dict = request.form.to_dict()
        to_form_dict["cut"] = int(to_form_dict["cut"])
        to_form_dict["color"] = int(to_form_dict["color"])
        to_form_dict["clarity"] = int(to_form_dict["clarity"])
        to_form_dict["carat"] = float(to_form_dict["carat"])
        to_form_dict["x"] = float(to_form_dict["x"])
        to_form_dict["y"] = float(to_form_dict["y"])
        to_form_dict["z"] = float(to_form_dict["z"])
        to_form_dict["table"] = float(to_form_dict["table"])
        to_form_dict["table"] = round(to_form_dict["table"],1)
        to_form_dict['depth'] = ((2 * to_form_dict['z']) / (to_form_dict['x'] + to_form_dict['y'])) * 100
        to_form_dict["depth"] = round(to_form_dict["depth"],1)

        predict_dict = {
             'carat': to_form_dict["carat"],
             'cut' : to_form_dict["cut"],
             'color' : to_form_dict["color"],
             'clarity' : to_form_dict["clarity"],
             'depth' : to_form_dict['depth'],
             'table' : to_form_dict['table'],
             'x' : to_form_dict['x'],
             'y' : to_form_dict['y'],
             'z' : to_form_dict['z']
        }

        predict_df = pd.DataFrame(predict_dict,index=[0])

        loaded_model = pickle.load(open("../../Challenge 2/best_model_regression.pkl", "rb"))
        price = loaded_model.predict(predict_df)
        return render_template("result.html", price = round(price[0],2))
if __name__ == '__main__':
    app.run()