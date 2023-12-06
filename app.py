from flask import Flask, jsonify
from flask import render_template

app = Flask(__name__)


@app.route("/")
def hello_world():
    return jsonify({ 'msg': "Hello world" })
