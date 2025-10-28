import re

import Path

from tokenizer import SimpleTokenizerV1


def preprocess_file(file_path):
    if not file_path:
        raise ValueError('file_path must be provided')
    with Path.open(file_path, encoding='utf-8') as file:
        raw_text = file.read()
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item for item in preprocessed if item.strip()]
        print(len(preprocessed))
        print(preprocessed[:100])
        all_tokens = sorted(set(preprocessed))
        all_tokens.extend(['<|endoftext|>', '<|unk|>'])
        vocab_size = len(all_tokens)
        print(f'Vocab size: {vocab_size}')
        vocab = {word: idx for idx, word in enumerate(all_tokens)}
        for i, item in enumerate(list(vocab.items())[-5:]):
            print(item)
            if i >= 50:
                break
        return vocab


vocab = preprocess_file('the_verdict.txt')
tokenizer = SimpleTokenizerV1(vocab)
text = """
Hello world, my dear man!
<|endoftext|>
This is a test of the tokenizer. Let's see how it works.
"""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
