import re
from tokenizer import SimpleTokenizerV1

with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item for item in preprocessed if item.strip()]
    print(len(preprocessed))
    print(preprocessed[:100])
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(f"Vocab size: {vocab_size}")
    vocab = {word: idx for idx, word in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
      print(item)
      if i >= 50:
        break

    tokenizer = SimpleTokenizerV1(vocab)
    text = """
    The height of his glory"--that was what the women called it. I can hear Mrs. Gideon Thwing--his last Chicago sitter--deploring his unaccountable abdication. "Of course it's going to send the value of my picture 'way up; but I don't think of that, Mr. Rickham--the loss to Arrt is all I think of
    """
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))
