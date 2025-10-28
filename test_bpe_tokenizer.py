import tiktoken


tokenizer = tiktoken.get_encoding('gpt2')
text = (
    'Hello, do you like tea? <|endoftext|> In the sunlit terraces'
    'of someunknownPlace gctvhjvbjvkvc.'
)
integers = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
print(integers)
strings = tokenizer.decode(integers)
print(strings)

text2 = 'Akwirw ier'
ids = tokenizer.encode(text2, allowed_special={'<|endoftext|>'})
tokens = [tokenizer.decode([id]) for id in ids]
print('len_ids', len(ids), 'len_tokens', len(tokens))
for zip_item in zip(ids, tokens):
    print(zip_item)
print(tokens)
