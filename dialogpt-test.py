import os
os.environ["TRANSFORMERS_CACHE"] = "cache/"

# This entire file is from https://huggingface.co/microsoft/DialoGPT-medium

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch


tokenizer = GPT2TokenizerFast.from_pretrained("microsoft/DialoGPT-medium")
model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")

# Let's chat for 5 lines
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(
        input(">> User:") + tokenizer.eos_token, return_tensors="pt"
    )

    # append the new user input tokens to the chat history
    bot_input_ids = (
        torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        if step > 0
        else new_user_input_ids
    )

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(
        bot_input_ids,
        do_sample=True,
        max_length=1000,
        num_beams=1,
        temperature=1,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    # pretty print last ouput tokens from bot
    print(
        "DialoGPT: {}".format(
            tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                skip_special_tokens=True,
            )
        )
    )
