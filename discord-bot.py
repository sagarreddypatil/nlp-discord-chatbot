import os
import discord
from transformers import (
    ConversationalPipeline,
    Conversation,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)

model = BlenderbotForConditionalGeneration.from_pretrained(
    "facebook/blenderbot-400M-distill"
)
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
pipeline = ConversationalPipeline(
    model=model, tokenizer=tokenizer, min_length_for_response=0, framework="pt"
)
generation_kwargs = {"num_beams": 3, "min_length": 0, "temperature": 1.5}

print("Loaded Model")

conversations = {}

client = discord.Client()


def select_or_create_convo(author: str):
    current_convo = None

    if author in conversations:
        current_convo = conversations[author]
    else:
        current_convo = Conversation(f"Hello! I am {message.author.display_name}")
        current_convo.mark_processed()
        current_convo.append_response(f"Hello! My name is Jane")
        conversations[author] = current_convo

    return current_convo


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.lower().startswith("jane "):
        utterance = message.content[5:]
        author = str(message.author)
        current_convo = select_or_create_convo(author)
        current_convo.add_user_input(utterance)

        pipeline(current_convo, **generation_kwargs)
        await message.reply(current_convo.generated_responses[-1][1:])


if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_KEY")
    client.run(TOKEN)
