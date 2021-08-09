from models import Blenderbot, DialoGPT
import os
import pickle
import discord
from transformers import (
    Conversation,
)

bot_name = "Jane"
bot_gender = "woman"

model = DialoGPT()

print("Loaded Model")

conversations = {}
if os.path.exists(
    "conversations.pkl"
):  # Loading cached conversations to preserve state between restarts
    with open("conversations.pkl", "rb") as file:
        conversations = pickle.load(file)

client = discord.Client()


def init_convo(
    author: str, author_display: str
):  # helper function to initialize all new conversations
    new_convo = Conversation(f"Hello! My name is {author_display}")
    new_convo.mark_processed()
    new_convo.append_response(f" Hello! I am a {bot_gender} named {bot_name}")
    conversations[author] = new_convo

    return new_convo


def select_or_create_convo(
    author: str, author_display: str
):  # Helper function, selects conversation by author if it exists and creates one if it doesn't.
    current_convo = None

    if author in conversations:
        current_convo = conversations[author]
    else:
        current_convo = init_convo(author, author_display)

    return current_convo


def create_embed(
    author, title: str, description: str, footer=None
):  # helper function to create Discord embed
    embed = discord.Embed(
        title=title, description=description, footer=footer, color=discord.Color.blue()
    )
    if author:
        embed.set_author(name=author.display_name, icon_url=author.avatar_url)
    if footer:
        embed.set_footer(text=footer)
    return embed


def generate_history(author_display, current_convo):  # pretty prints the conversation
    output = ""
    for is_user, text in list(current_convo.iter_texts()):
        name = author_display if is_user else bot_name
        output += "{} >> {}\n".format(name, text)
    return output


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.lower().startswith(
        f"{bot_name.lower()} "
    ) or message.content.lower().startswith(f"{bot_name.lower()},"):
        utterance = message.content[len(bot_name) + 1 :].strip()
        author = f"{message.guild.id}:{message.author}"
        current_convo = select_or_create_convo(author, message.author.display_name)

        if utterance == "-r" or utterance == "--reset":
            current_convo = init_convo(author, message.author.display_name)

            embed = create_embed(
                message.author,
                title="Reset",
                description=f"Your message history with {bot_name} has been reset",
            )
            await message.channel.send(embed=embed)
            return

        if utterance == "-h" or utterance == "--history":
            output = generate_history(message.author.display_name, current_convo)

            if len(output) == 0:
                output = "No history"

            embed = create_embed(
                message.author,
                title="Message History",
                description=output,
                footer=f"{bot_name} can only remember the last {model.pipeline.tokenizer.model_max_length} syllables in the conversation",
            )
            await message.channel.send(embed=embed)
            return

        if utterance.startswith("-a") or utterance.startswith("--amend"):
            history = generate_history(message.author.display_name, current_convo)
            if len(history) == 0:
                embed = create_embed(
                    message.author,
                    title="Amended Message History",
                    description="No history to amend",
                )
                await message.channel.send(embed=embed)
                return

            if utterance.startswith("-a"):
                utterance = utterance[3:]
            else:
                utterance = utterance[8:]

            current_convo.generated_responses[-1] = utterance
            embed = create_embed(
                message.author,
                title="Amended Message History",
                description=generate_history(
                    message.author.display_name, current_convo
                ),
            )
            await message.channel.send(embed=embed)
            return

        async with message.channel.typing():
            current_convo.add_user_input(utterance)
            model(current_convo)

            response = current_convo.generated_responses[-1].strip()

        if response != "":
            await message.reply(response)
        else:
            # embed = create_embed(author=None, title="No Response", description="")
            # await message.reply(embed=embed)
            await message.reply("*`No response`*")


if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_KEY")
    bot_name = os.getenv("NAME") if os.getenv("NAME") else bot_name
    bot_gender = os.getenv("GENDER") if os.getenv("GENDER") else bot_gender
    try:
        client.run(TOKEN)
    except KeyboardInterrupt:
        pass

with open(
    "conversations.pkl", "wb"
) as file:  # Cacheing conversations to preserve state between restarts
    pickle.dump(conversations, file, protocol=pickle.HIGHEST_PROTOCOL)
