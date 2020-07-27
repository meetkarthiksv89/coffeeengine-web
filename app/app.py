from flask import Flask, session, redirect, render_template, request, jsonify, flash
from flask_session import Session
import json
import webbrowser
import coremltools


app = Flask(__name__)
app.config["SECRET_KEY"] = "new key"

# Configure session to use filesystem
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.route("/")
def index():
    """ Show search box """

    return render_template("question.html")

@app.route('/test', methods=['GET', 'POST'])
def test():
    data = request.get_json(force=True)
    dataset=format(data)
    # Load the model
    model = coremltools.models.MLModel("CoffeeClassifier.mlmodel")

    predictions = model.predict({'text': dataset})

    print(predictions)
    print(predictions["label"])
    labelling = predictions["label"]
    if labelling == "Aroma Gold":
        return webbrowser.open_new_tab('https://pandurangacoffee.com/collections/frontpage/products/aroma-gold')
    elif labelling == "Brown Gold":
        return webbrowser.open_new_tab('https://pandurangacoffee.com/collections/frontpage/products/brown-gold')
    elif labelling == "French Blend":
        return webbrowser.open_new_tab('https://pandurangacoffee.com/collections/frontpage/products/french-blend')

