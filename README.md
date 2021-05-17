# NER handler
## Application architecture:
The application consists of 2 services communicating via REST API:
1. Main backend or pipeliner that describes the logic of sending requests 
from one model to another / to the model of the language of choice
3. NLP service with a model that contains everything for a model to be deployed locally / in GCP

### Supported languages:
- English

### Possible input formats:
Text data as a json
```
{
    "text_or_file_object": "input text or file object",
    "language": "en",
    "filename": "name of a file"
}
```

## Usage
Each service is packed into a docker container and all of them are managed by docker compose. <br>
To run the application in GCP you must have these secret files and put them as described below in folders structure
- `creds.json` as google credentials

### Project structure
Your project structure must look like this: <br>
```
typical_ner
├── docker-compose_template.yaml
├── creds.json
├── models (if local run ahead)
│   ├──  classification_subheading_m.pt
│   ├──  classification_subheading_n.pt
├── pipeliner
├── model_app
```

### Run application with docker-compose
1. Install docker-compose (e.g. `pip install docker-compose`)
2. Place latest versions of models as described below in models folder

To run docker-compose use:

`sudo docker-compose up --build`