FROM gcr.io/google-appengine/python
RUN virtualenv /env -p python3.7

ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

ADD . /pipeliner
WORKDIR /pipeliner
ENV PYTHONPATH /pipeliner
EXPOSE 8080

CMD python app.py