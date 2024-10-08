import re
from unidecode import unidecode
from phonemizer import phonemize

"""
from https://github.com/keithito/tacotron

Cleaners are transformations that run over the input text
at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner
names as the "cleaners" hyperparameter.
Some cleaners are English-specific. You'll typically want to use:
1.  "english_cleaners" for English text
2.  "transliteration_cleaners" for non-English text that can be
    transliterated to ASCII using the Unidecode library
    (https://pypi.python.org/pypi/Unidecode)
3.  "basic_cleaners" if you do not want to transliterate
    (in this case, you should also update the symbols in symbols.py
    to match your data).
"""


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

def lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)

def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  #text = lowercase(text)
  text = collapse_whitespace(text)
  text = replace_quote(text)
  text = remove_special_characters(text)
  text = remove_hyphen_at_start(text)
  return text

def replace_quote(text):
    return text.replace('’', "'")

def remove_special_characters(text):
    # Define the characters to remove
    characters_to_remove = ['«', '»']
    # Remove the characters from the text
    for char in characters_to_remove:
        text = text.replace(char, '')
    return text

def remove_hyphen_at_start(text):
    # Check if the text starts with '-'
    if text.startswith('-'):
        # Remove the hyphen at the start
        text = text[1:].lstrip()
    return text
