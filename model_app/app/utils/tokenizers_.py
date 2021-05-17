"""Tokenization functions for different languages"""
from flair.data import Token, segtok_tokenizer
from nltk import RegexpTokenizer
from razdel import sentenize, tokenize
from segtok.segmenter import split_newline, split_single
from segtok.tokenizer import split_contractions


def zh_tokenizer(text: str) -> list:
    """
    Tokenizes texts in Chinese
    Args:
        text (str): input text

    Returns: 
        flair Token objects
    """
    sentences = text.split("。")
    words, tokens = [], []
    for sentence in sentences:
        chars = list(sentence)
        words.extend(chars)
    prev_start_position = 0

    for word in words:
        start_position = text[prev_start_position:].index(word)
        token = Token(text=word,
                      start_position=prev_start_position + start_position,
                      whitespace_after=False)
        tokens.append(token)
        prev_start_position = start_position + prev_start_position + len(word)
    return tokens


def fr_tokenizer(text: str) -> list:
    """
    Tokenizes texts in French
    Args:
        text (str): input text

    Returns: 
        flair Token objects
    """
    tokens = []
    tokenizer = RegexpTokenizer(r"""\w'|\w’|\w`|\w\w+'\w+|[^\w\s]|\w+""")
    words = []
    sentences = split_single(text)
    for sentence in sentences:
        contractions = split_contractions(tokenizer.tokenize(sentence))
        words.extend(contractions)

    # determine offsets for whitespace_after field
    index = text.index
    current_offset = 0
    previous_word_offset = -1
    previous_token = None
    for word in words:
        try:
            word_offset = index(word, current_offset)
            start_position = word_offset
        except ValueError:
            word_offset = previous_word_offset + 1
            start_position = (current_offset +
                              1 if current_offset > 0 else current_offset)

        if word:
            token = Token(text=word,
                          start_position=start_position,
                          whitespace_after=True)
            tokens.append(token)

        if (previous_token is not None) and word_offset - 1 == previous_word_offset:
            previous_token.whitespace_after = False

        current_offset = word_offset + len(word)
        previous_word_offset = current_offset - 1
        previous_token = token

    return tokens


def ru_tokenizer(text: str) -> list:
    """
    Tokenizes texts in Russian
    Args:
        text (str): input text

    Returns: 
        flair Token objects
    """
    all_sentences = []
    for paragraph in split_newline(text):
        sentences = [x.text for x in list(sentenize(paragraph))]
        all_sentences.extend(sentences)
    words = []
    for sentence in all_sentences:
        sentence_tokens = [x.text for x in list(tokenize(sentence))]
        words.extend(sentence_tokens)
    prev_start_position = 0
    tokens = []
    for word in words:
        start_position = text[prev_start_position:].index(word)
        token = Token(text=word,
                      start_position=prev_start_position + start_position,
                      whitespace_after=False)
        tokens.append(token)
        prev_start_position = start_position + prev_start_position + len(word)
    return tokens


tokenizers = {
    "en": segtok_tokenizer,
    "fr": fr_tokenizer,
    "zh": zh_tokenizer,
    "es": segtok_tokenizer,
    "pt": segtok_tokenizer,
    "de": segtok_tokenizer,
    "ru": ru_tokenizer
}
