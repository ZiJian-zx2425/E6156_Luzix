from flask import Flask
from flask import request, redirect, session, url_for, render_template, request
import time


app = Flask(__name__)


@app.route("/patient")
def patients():
    time.sleep(5)
    return "Patient"


if __name__ == "__main__":
    app.run(port="8001")
