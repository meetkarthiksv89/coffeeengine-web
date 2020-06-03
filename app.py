from flask import Flask, session, redirect, render_template, request, jsonify, flash
from flask_session import Session


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
