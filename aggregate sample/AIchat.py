from flask import Flask
from flask import request, redirect, session, url_for, render_template, request
import time


app = Flask(__name__)


@app.route("/aichat")
def patients():
    time.sleep(7)
    return "aichat"


if __name__ == "__main__":
    app.run(port="8002")
