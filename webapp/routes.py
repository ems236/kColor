from flask import abort, request, jsonify, make_response, send_file, render_template
from webapp import app

@app.route("/hello")
def hello_world():
    return "Hello World"

@app.route("/")
def index():
    return render_template("main.html")

@app.route("/submit")
def submit():
    return abort(404)
