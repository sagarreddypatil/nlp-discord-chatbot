import os
import pickle
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
if os.path.exists("conversations.pkl"):
    with open("conversations.pkl", "rb") as file:
        conversations = pickle.load(file)

client = discord.Client()


def init_convo(author: str, author_display: str):
    new_convo = Conversation(f"Hello! I am {author_display}")
    new_convo.mark_processed()
    new_convo.append_response(" Hello! My name is Jane")
    conversations[author] = new_convo

    return new_convo


def select_or_create_convo(author: str, author_display: str):
    current_convo = None

    if author in conversations:
        current_convo = conversations[author]
    else:
        current_convo = init_convo(author, author_display)

    return current_convo


def create_embed(author, title: str, description: str, footer=None):
    embed = discord.Embed(
        title=title, description=description, footer=footer, color=discord.Color.blue()
    )
    embed.set_author(name=author.display_name, icon_url=author.avatar_url)
    if footer:
        embed.set_footer(text=footer)
    return embed

def generate_history(author_display, current_convo)
    output = ""
    for is_user, text in list(current_convo.iter_texts())[2:][-14:]:
        name = author_display if is_user else "Jane"
        output += "{} >> {} \n".format(name, text)
    return output

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.lower().startswith(
        "jane "
    ) or message.content.lower().startswith("jane,"):
        utterance = message.content[5:].strip()
        author = f"{message.guild.id}:{message.author}"
        current_convo = select_or_create_convo(author, message.author.display_name)

        if utterance == "-r" or utterance == "--reset":
            current_convo = init_convo(author, message.author.display_name)

            embed = create_embed(
                message.author,
                title="Reset",
                description="Your message history with Jane has been reset",
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
                footer="Jane can only remember the last 128 words in the conversation",
            )
            await message.channel.send(embed=embed)
            return

        if utterance.startswith("-a") or utterance.startswith("--amend"):
            history = generate_history(message.author.display_name, current_convo)
            if len(history) == 0:
                embed = create_embed(
                    message.author,
                    title="Amended Message History",
                    description="No history to amend"
                )
                await message.channel.send(embed=embed)
                return
            
            if utterance.startswith("-a"): utterance = utterance[3:]
            else: utterance = utterance[8:]

            current_convo.generated_responses[-1] = utterance
            embed = create_embed(
                message.author,
                title="Amended Message History",
                description=generate_history(message.author.display_name, current_convo)
            )
            await message.channel.send(embed=embed)
            return


        current_convo.add_user_input(utterance)
        pipeline(current_convo, **generation_kwargs)
        await message.reply(current_convo.generated_responses[-1][1:])


if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_KEY")
    try:
        client.run(TOKEN)
    except KeyboardInterrupt:
        pass

with open("conversations.pkl", "wb") as file:
    pickle.dump(conversations, file, protocol=pickle.HIGHEST_PROTOCOL)