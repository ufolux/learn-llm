import re

class SimpleTokenizerV1:
  def __init__(self, vocab) -> None:
    self.str_to_int = vocab
    self.int_to_str = {v: k for k, v in vocab.items()}

  def encode(self, text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    ids = [self.str_to_int[s] for s in preprocessed]
    return ids

  def decode(self, tokens):
    text = " ".join([self.int_to_str[i] for i in tokens])
    text = re.sub(r'\s([,.:;?_!"()\'])', r'\1', text)
    return text
