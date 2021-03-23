from transformers import (
    ConversationalPipeline,
    Conversation,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)


if __name__ == "__main__":
    import logging

    logger = logging.getLogger("transformers").setLevel(logging.CRITICAL)

    model = BlenderbotForConditionalGeneration.from_pretrained(
        "facebook/blenderbot-400M-distill"
    )
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

    convo = ConversationalPipeline(
        model=model, tokenizer=tokenizer, min_length_for_response=0, framework="pt"
    )

    ipt = input(">>> ")
    dialogue = Conversation(ipt)
    while True:
        try:
            convo(dialogue, num_beams=3, min_length=0, temperature=1.5)
            print(dialogue.generated_responses[-1][1:])
            dialogue.add_user_input(input(">>> "))
        except KeyboardInterrupt:
            break

    print("\n------Dialogue Summary------")
    print(dialogue)