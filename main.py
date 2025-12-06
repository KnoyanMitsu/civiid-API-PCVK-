import os
from flask import Flask
from flask_cors import CORS
from routes.route import route


app = Flask(__name__)
CORS(app)
app.register_blueprint(route)

if __name__ == '__main__':
    app.run(debug=True)