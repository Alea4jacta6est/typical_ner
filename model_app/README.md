# Sampo model
This server is used to get predictions for chunk of text. These chunks are received from 
[task manager](https://bitbucket.enhesa.com/projects/SAMPO/repos/sampo-task-manager/browse). <br>
For getting prediction NLP flair models are used. This service allows to create multiple services with different models
by passing arguments in moment of creating docker container.
 
## Setting up

To start service via docker you should specify language, model, model type, max sentence length as args in 
`docker build` command.

- Language: two-letter abbreviation of language (e.x. en, fr, zh, es)
- Model: name of the model in pickle format (e.x. en_glove.pt)
- Model type: `ner` or `classifier`
- Max sentence length: is used fr BERT models

<b>Example<b/><br>
`docker build . -t ner_delete --build-arg language=en --build-arg model=en_glove.pt --build-arg length=None --build-arg model_type=ner
`

        