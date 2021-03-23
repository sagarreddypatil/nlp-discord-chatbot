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

TOKEN = os.getenv("DISCORD_KEY")
client = discord.Client()

people = []
conversations = []


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.lower().startswith("jane "):
        utterance = message.content[5:]
        author = message.author.name
        current_convo = None

        if author in people:
            current_convo = conversations[people.index(author)]
            current_convo.add_user_input(utterance)
        else:
            people.append(author)
            current_convo = Conversation(utterance)
            conversations.append(current_convo)

        pipeline(current_convo, **generation_kwargs)
        await message.reply(current_convo.generated_responses[-1][1:])


client.run(TOKEN)
