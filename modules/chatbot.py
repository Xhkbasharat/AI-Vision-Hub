
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from config import CHATBOT_MODEL_NAME


class AIChatbot:

    def __init__(self):

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        print("Loading AI chatbot...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            CHATBOT_MODEL_NAME
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            CHATBOT_MODEL_NAME
        ).to(self.device)

        self.model.eval()

        self.max_history_tokens = 700
        self.max_new_tokens = 100

        print("AI chatbot loaded successfully!")

    def reset_chat(self):

        return []

    def chat(self, message, history=None):

        if not isinstance(message, str) or not message.strip():
            raise ValueError("Please enter a message.")

        if history is None:
            history = []

        if not isinstance(history, list):
            raise ValueError("Invalid conversation history.")

        message = message.strip()[:1000]

        # Keep only valid previous messages
        history = [
            str(item)[:1000]
            for item in history[-10:]
            if isinstance(item, str) and item.strip()
        ]

        # Add the current user message
        conversation = history + [message]

        # Convert conversation into model tokens
        token_ids = []

        for turn in conversation:

            tokens = self.tokenizer.encode(
                turn + self.tokenizer.eos_token,
                add_special_tokens=False
            )

            token_ids.extend(tokens)

        # Limit conversation length
        token_ids = token_ids[-self.max_history_tokens:]

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long
        ).to(self.device)

        attention_mask = torch.ones_like(input_ids)

        # Generate response
        with torch.no_grad():

            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.92,
                repetition_penalty=1.1
            )

        # Extract newly generated tokens
        response_ids = output_ids[
            :, input_ids.shape[-1]:
        ]

        response = self.tokenizer.decode(
            response_ids[0],
            skip_special_tokens=True
        ).strip()

        if not response:
            response = "I could not generate a response. Please try again."

        # Add assistant response to conversation
        updated_history = conversation + [response]

        return {
            "user_message": message,
            "bot_response": response,
            "history": updated_history
        }
