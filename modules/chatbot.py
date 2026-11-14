
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import CHATBOT_MODEL_NAME


class Chatbot:

    def __init__(self):

        self.model_name = CHATBOT_MODEL_NAME

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        # Load tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        # Load conversational language model.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

        # DialoGPT uses the EOS token as its padding token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_history_tokens = 700
        self.max_new_tokens = 100

    # ======================================
    # VALIDATE MESSAGE
    # ======================================

    @staticmethod
    def validate_message(message):

        if not isinstance(message, str):
            raise ValueError(
                "Message must be text."
            )

        message = message.strip()

        if not message:
            raise ValueError(
                "Message cannot be empty."
            )

        if len(message) > 1000:
            raise ValueError(
                "Message must be 1000 characters or fewer."
            )

        return message

    # ======================================
    # BUILD CONVERSATION HISTORY
    # ======================================

    def build_history(self, history):

        if not isinstance(history, list):
            history = []

        conversation_ids = []

        for item in history[-12:]:

            if not isinstance(item, dict):
                continue

            message = item.get("message", "")
            speaker = item.get("speaker", "")

            if speaker not in ("user", "assistant"):
                continue

            if not isinstance(message, str) or not message.strip():
                continue

            encoded = self.tokenizer.encode(
                message.strip(),
                add_special_tokens=False
            )

            if conversation_ids:
                conversation_ids.extend(encoded)
            else:
                conversation_ids = encoded

            conversation_ids.append(
                self.tokenizer.eos_token_id
            )

        # Keep recent conversation tokens only.
        conversation_ids = conversation_ids[
            -self.max_history_tokens:
        ]

        return conversation_ids

    # ======================================
    # GENERATE CHAT RESPONSE
    # ======================================

    def chat(self, message, history=None):

        message = self.validate_message(message)

        if history is None:
            history = []

        history_ids = self.build_history(history)

        user_ids = self.tokenizer.encode(
            message + self.tokenizer.eos_token,
            return_tensors="pt"
        ).to(self.device)

        if history_ids:
            history_tensor = torch.tensor(
                [history_ids],
                dtype=torch.long,
                device=self.device
            )

            input_ids = torch.cat(
                [history_tensor, user_ids],
                dim=-1
            )
        else:
            input_ids = user_ids

        input_ids = input_ids[
            :, -self.max_history_tokens:
        ]

        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():

            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )

        generated_ids = output[
            :, input_ids.shape[-1]:
        ]

        response = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        ).strip()

        if not response:
            response = "I'm sorry, I couldn't generate a response."

        updated_history = history[-10:] + [
            {
                "speaker": "user",
                "message": message
            },
            {
                "speaker": "assistant",
                "message": response
            }
        ]

        return {
            "response": response,
            "history": updated_history,
            "model": self.model_name
        }

    # ======================================
    # RESET CONVERSATION
    # ======================================

    @staticmethod
    def reset_chat():

        return []
