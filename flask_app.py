
from flask import Flask
import sample as smp

app = Flask(__name__)


@app.route("/")
def hello():
    return smp.main()
