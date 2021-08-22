from flask import Flask

app = Flask(__name__)

from kColor.webapp import routes 