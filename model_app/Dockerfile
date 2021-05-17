FROM gcr.io/google-appengine/python
RUN virtualenv /env -p python3.7

ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

ADD . /model_app
WORKDIR /model_app
ENV PYTHONPATH /model_app
EXPOSE 8080

ENV MODEL_BUCKET gcp_bucket_name

ARG language
ENV MODEL_LANGUAGE=$language

ARG model
ENV MODEL_FILENAME=$model

ARG length
ENV SENTENCE_LENGTH=$length

ARG cloud_env
ENV ENV=$cloud_env

ARG model_type
ENV MODEL_TYPE=$model_type

CMD gunicorn -b :$PORT app.main:app --timeout=300
