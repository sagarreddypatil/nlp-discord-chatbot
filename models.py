from transformers import (
    ConversationalPipeline,
    Conversation,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)

from transformers.models.gpt2 import GPT2LMHeadModel, GPT2TokenizerFast

import torch


class ModelInterface:
    def __init__(self):
        """Load the NLP model to memory"""
        pass

    def __call__(self, conversation: Conversation):
        """Run the model on the conversation and append the output to the conversation"""
        pass


class Blenderbot(ModelInterface):
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


class DialoGPT(ModelInterface):
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("microsoft/DialoGPT-medium")

        self.generation_kwargs = {
            "num_beams": 1,
            "temperature": 1,
            "top_k": 50,
            "top_p": 0.95,
        }

        class MockPipeline:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

        self.pipeline = MockPipeline(self.tokenizer)

    def _conv_to_model_input(self, conversation: Conversation):
        model_input = ""
        for _, text in conversation.iter_texts():
            model_input += text + self.tokenizer.eos_token

        return torch.tensor([self.tokenizer.encode(model_input)])

    def _truncate_convo_to_token_limit(self, conversation: Conversation):
        while (
            len(self._conv_to_model_input(conversation)[0])
            > self.tokenizer.model_max_length
        ):
            if (
                len(conversation.past_user_inputs) > 0
                and len(conversation.generated_responses) > 0
            ):
                conversation.past_user_inputs.pop(0)
                conversation.generated_responses.pop(0)

    def __call__(self, conversation: Conversation):
        self._truncate_convo_to_token_limit(conversation)

        model_input = self._conv_to_model_input(conversation)
        model_output = self.model.generate(
            model_input,
            do_sample=True,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            **self.generation_kwargs
        )

        model_output_text = self.tokenizer.decode(
            model_output[:, model_input.shape[-1] :][0], skip_special_tokens=True
        )

        conversation.mark_processed()
        conversation.append_response(model_output_text)
