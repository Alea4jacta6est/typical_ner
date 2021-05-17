import json
from time import time

from flair.data import Sentence
from flair.models import SequenceTagger, TextClassifier
from numpy import array_split
from segtok.segmenter import split_single

from app.utils.logger import logger
from app.utils.tokenizers_ import tokenizers


class FlairNER:
    """Wrapper for flair's SequenceTagger.

        Args:
            model (str, optional): Path to the model. Defaults to None.
            tokenizer (bool or callable, optional): flair-like tokenizer.
                Should return flair's Tokens.
                True stands for segtok_tokenizer, False for whitespace.
                Defaults to True.
            max_length (int, optional): Max length of input sentences.
                Defaults to None.
        """

    def __init__(self, model=None, lang=True, max_length=None):
        self.tokenizer = tokenizers[lang]
        self.model = SequenceTagger.load(model)
        self.max_length = int(max_length) if max_length else None
        logger.info(f"NER model was loaded. {model}")

    def _split_long_sentences(self, sentences):
        """Split long sentences.

        Args:
            sentences (list): list of flair's Sentences

        Returns:
            list:
        """
        extended = sentences.copy()
        tokenizer = self.model.embeddings.tokenizer
        offset = 0
        for i, sentence in enumerate(sentences):
            len_bpe = len(tokenizer.tokenize(sentence.to_tokenized_string()))
            if len_bpe > self.max_length:
                extended.pop(i + offset)
                num_pieces = len_bpe // self.max_length + 1
                for piece in array_split(sentence, num_pieces):
                    char_offset = piece[0].start_pos
                    sentence_piece = Sentence()
                    for token in piece:
                        token.start_pos -= char_offset
                        token.end_pos -= char_offset
                        sentence_piece.add_token(token)
                    piece[-1].whitespace_after = False
                    extended.insert(i + offset, sentence_piece)
                    offset += 1
                # we pop original sentence, so we should decrease offset by one
                offset -= 1
        logger.debug(f'Lengths before split: {[len(x) for x in sentences]}')
        logger.debug(f'Lengths after split: {[len(x) for x in extended]}')
        return extended

    def predict_paragraph(self, paragraph) -> dict:
        """Predict for paragraph.

        Args:
            paragraph (str): Input paragraph.

        Returns:
            spacy formatted dict with entities
        """
        # mismatch if there are empty sentences
        sentences = [
            Sentence(x, use_tokenizer=self.tokenizer) for x in split_single(paragraph)
        ]
        # move to separate function
        if self.max_length:
            sentences = self._split_long_sentences(sentences)

        self.model.predict(sentences)

        json_sentences = []
        for sent in sentences:
            spacy_format_ner = flair_to_spacy(sent)
            json_sentences.append(spacy_format_ner)

        ner = concat_json_sentences(json_sentences)
        return ner

    def response(self, request):
        X = request.get_json()['X']
        start = time()
        result = self.predict_paragraph(X)
        end = time()
        logger.debug("Prediction was made.")
        logger.debug(f'time - {end - start}s')
        logger.debug(result)
        return json.dumps({'y': result, 'time': end - start}), 200


def preprocess_label(ent_dict):
    try:
        label = ent_dict["type"].replace("_", " ")
    except KeyError:
        # Camembert embeddings exotic
        label = ent_dict["labels"][0].to_dict()["value"].replace("_", " ")

    return label


def flair_to_spacy(sent) -> dict:
    """Transform flair Sentence into spacy dict."""
    sent_dict = sent.to_dict(tag_type="ner")

    return {
        "text":
            sent_dict["text"],
        "ents": [{
            "start": ent["start_pos"],
            "end": ent["end_pos"],
            "label": preprocess_label(ent)
        } for ent in sent_dict["entities"]],
    }


def concat_json_sentences(sentences: list):
    result = {"text": " ".join([sentence["text"] for sentence in sentences]),
              "ents": []}
    # 'i' here is a number of previous sentences, as number of added spaces
    # whole_text_len is a shift depends of previous sentences length
    whole_text_len = 0
    for i, sentence in enumerate(sentences):
        for ent in sentence["ents"]:
            result["ents"].append({
                "start": ent["start"] + whole_text_len + i,
                "end": ent["end"] + whole_text_len + i,
                "label": ent["label"]
            })
        whole_text_len += len(sentence["text"])
    return result


class FlairClassifier:
    """Wrapper for flair's TextClassifier.

    Args:
        model (str, optional): Path to the model. Defaults to None.
        tokenizer (bool or callable, optional): flair-like tokenizer.
            Should return flair's Tokens.
            True stands for segtok_tokenizer, False for whitespace.
            Defaults to True.
        max_length (int, optional): Max length of input sentences.
            Defaults to None.

    """

    def __init__(self, model=None, lang=True, max_length=None):
        self.model = TextClassifier.load(model)
        self.tokenizer = tokenizers[lang]
        logger.debug(f"Classifier was loaded. {model}")

    def predict_paragraph(self, paragraph):
        """Predict for paragraph.

        Args:
            paragraph (str): Input paragraph.

        Returns:
            str: html with highlighted predictions.
        """
        sentence = Sentence(paragraph, use_tokenizer=self.tokenizer)
        self.model.predict(sentence)
        prediction = sentence.labels[0]

        return prediction.value, prediction.score

    def response(self, request):
        X = request.get_json()['text']
        start = time()
        result, score = self.predict_paragraph(X)
        end = time()
        result = result.replace('_', ' ')
        logger.debug(f'time - {end - start}s')
        logger.debug(result)
        return json.dumps({
            'label': result,
            'confidence': score,
            'heading_num': result.split(' ')[0]
        }), 200
