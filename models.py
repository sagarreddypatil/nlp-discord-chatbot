from transformers import (
    ConversationalPipeline,
    Conversation,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)

import torch


class Blenderbot:
    def __init__(self):
        model = BlenderbotForConditionalGeneration.from_pretrained(
            "facebook/blenderbot-400M-distill"
        )
        tokenizer = BlenderbotTokenizer.from_pretrained(
            "facebook/blenderbot-400M-distill"
        )
        self.pipeline = ConversationalPipeline(
            model=model,
            tokenizer=tokenizer,
            min_length_for_response=0,
            framework="pt",
            device=0 if torch.cuda.is_available() else -1,
        )

        self.generation_kwargs = {
            "num_beams": 3,
            "min_length": 0,
            "temperature": 1.5,
            "top_p": 0.9,
        }

    def _build_conversation_input_ids_modified(
        self,
        conversation: Conversation,
    ):  # <- partially copied from BlenderbotTokenizer._build_conversation_input_ids
        inputs = []
        for is_user, text in conversation.iter_texts():
            if is_user:
                inputs.append(" " + text)
            else:
                inputs.append(text)

        full_string = "  ".join(inputs)
        input_ids = self.pipeline.tokenizer.encode(full_string)
        return input_ids

    def _truncate_convo_to_token_limit(self, conversation: Conversation):
        while (
            len(self._build_conversation_input_ids_modified(conversation))
            > self.pipeline.tokenizer.model_max_length
        ):
            if (
                len(conversation.past_user_inputs) > 0
                and len(conversation.generated_responses) > 0
            ):
                conversation.past_user_inputs.pop(0)
                conversation.generated_responses.pop(0)

    def __call__(self, conversation: Conversation):
        self._truncate_convo_to_token_limit(conversation)
        self.pipeline(conversation, **self.generation_kwargs)
