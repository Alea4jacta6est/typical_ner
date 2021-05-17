import logging
import sys
from logging.config import dictConfig

FORMAT = "%(message)s"

dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': FORMAT,
        }},
    'handlers': {
        'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})
logger = logging.getLogger()
