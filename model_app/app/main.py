"""Core script that runs the container with a model"""
import logging
import os

from flask import Flask, request

from app.utils.cloud_handler import get_model
from app.utils.model import FlairNER, FlairClassifier

MODEL = None
ENV = os.environ["ENV"]
MODEL_TYPE = os.environ["MODEL_TYPE"]
MODEL_BUCKET = os.environ['MODEL_BUCKET']
MODEL_FILENAME = os.environ['MODEL_FILENAME']
MODEL_LANGUAGE = os.environ["MODEL_LANGUAGE"]
# return None if there is no such variable
SENTENCE_LENGTH = os.getenv("SENTENCE_LENGTH")
app = Flask(__name__)


@app.before_first_request
def _load_model():
    os.makedirs("/tmp", exist_ok=True)
    global MODEL

    model_filename = get_model(MODEL_BUCKET, MODEL_FILENAME, ENV)
    if MODEL_TYPE == "ner":
        MODEL = FlairNER(model=model_filename,
                         lang=MODEL_LANGUAGE,
                         max_length=SENTENCE_LENGTH)
    elif MODEL_TYPE == 'classifier':
        MODEL = FlairClassifier(model=model_filename,
                                lang=MODEL_LANGUAGE,
                                max_length=SENTENCE_LENGTH)
    else:
        raise ValueError("Wrong type of a model was chosen.")


@app.route('/', methods=['GET'])
def index():
    return str(MODEL), 200


@app.route('/predict', methods=['POST'])
def predict():
    return MODEL.response(request)


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
