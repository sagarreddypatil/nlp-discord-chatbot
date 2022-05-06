from transformers import AutoTokenizer, GPTJForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
gpt = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit")

device = "cuda" if torch.cuda.is_available() else "cpu"
gpt.to(device)

prompt = tokenizer("A cat sat on a mat", return_tensors="pt")
prompt = {key: value.to(device) for key, value in prompt.items()}
out = gpt.generate(**prompt, min_length=128, max_length=128, do_sample=True)
tokenizer.decode(out[0])
