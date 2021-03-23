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


def init_convo(author: str, author_display: str):
    new_convo = Conversation(f"Hello! I am {author_display}")
    new_convo.mark_processed()
    new_convo.append_response("Hello! My name is Jane")
    conversations[author] = new_convo

    return new_convo


def select_or_create_convo(author: str, author_display: str):
    current_convo = None

    if author in conversations:
        current_convo = conversations[author]
    else:
        current_convo = init_convo(author, author_display)

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
        current_convo = select_or_create_convo(author, message.author.display_name)

        if utterance == "-r" or "--reset":
            current_convo = init_convo(author, message.author.display_name)

            embed = discord.Embed(
                title="Reset",
                description="Your message history with Jane has been reset",
                color=discord.Color.blue(),
            )
            embed.set_author(
                name=message.author.display_name, icon_url=message.author.avatar_url
            )

            await message.send(embed=embed)
            return

        current_convo.add_user_input(utterance)
        pipeline(current_convo, **generation_kwargs)
        await message.reply(current_convo.generated_responses[-1][1:])


if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_KEY")
    client.run(TOKEN)
