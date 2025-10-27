import torch
from torch.utils.data import Dataset

# tokenizer = tiktoken.get_encoding("gpt2")
# # with open("the_verdict.txt", "r", encoding="utf-8") as file:
# #     raw_text = file.read()
# #     enc_text = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
# #     enc_sample = enc_text[50:]
# #     context_size = 4
# #     x = enc_sample[:context_size]
# #     y = enc_sample[1 : context_size + 1]
# #     print(f"x: {x}")
# #     print(f"y:      {y}")

# #     for i in range(1, context_size + 1):
# #         context = enc_sample[:i]
# #         desired = enc_sample[i]
# #         print(f"{tokenizer.decode(context)} ----> {tokenizer.decode([desired])}")

class GPTDatasetV1(Dataset):
  def __init__(self, txt, tokenizer, max_length, stride):
    self.input_ids = []
    self.target_ids = []

    token_ids = tokenizer.encode(txt)
    for i in range(0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i:i + max_length]
      target_chunk = token_ids[i + 1: i + max_length + 1]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]
