
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from config import CHATBOT_MODEL_NAME


class AIChatbot:

    def __init__(self):

        print("Loading AI chatbot...")

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            CHATBOT_MODEL_NAME
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            CHATBOT_MODEL_NAME
        ).to(self.device)

        self.model.eval()

        self.chat_history_ids = None

        self.max_history_tokens = 700
        self.max_new_tokens = 100

        print("AI chatbot loaded successfully!")

    def reset_chat(self):

        self.chat_history_ids = None

        return {
            "message": "Conversation history cleared."
        }

    def chat(self, message):

        if not isinstance(message, str) or not message.strip():
            raise ValueError(
                "Please enter a message."
            )

        message = message.strip()[:1000]

        # Convert user message into tokens
        new_input_ids = self.tokenizer.encode(
            message + self.tokenizer.eos_token,
            return_tensors="pt"
        ).to(self.device)

        # Add previous conversation
        if self.chat_history_ids is not None:

            bot_input_ids = torch.cat(
                [
                    self.chat_history_ids,
                    new_input_ids
                ],
                dim=-1
            )

        else:

            bot_input_ids = new_input_ids

        # Limit conversation history
        if bot_input_ids.shape[-1] > self.max_history_tokens:

            bot_input_ids = bot_input_ids[
                :, -self.max_history_tokens:
            ]

        # Generate response
        with torch.no_grad():

            output_ids = self.model.generate(
                bot_input_ids,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.92,
                repetition_penalty=1.1
            )

        # Save conversation history
        self.chat_history_ids = output_ids

        # Extract generated response
        response_ids = output_ids[
            :, bot_input_ids.shape[-1]:
        ]

        response = self.tokenizer.decode(
            response_ids[0],
            skip_special_tokens=True
        ).strip()

        if not response:
            response = "I could not generate a response. Please try again."

        return {
            "user_message": message,
            "bot_response": response
        }
