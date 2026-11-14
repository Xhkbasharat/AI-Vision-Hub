
import logging

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from config import CHATBOT_MODEL_NAME


logger = logging.getLogger("AI-Vision-Hub")


class Chatbot:
    """
    AI Vision Hub conversational chatbot.

    Uses Microsoft's DialoGPT model to generate
    conversational responses.
    """

    def __init__(self):

        logger.info(
            "Initializing chatbot: %s",
            CHATBOT_MODEL_NAME
        )

        # Select CPU or GPU.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            CHATBOT_MODEL_NAME
        )

        # Load language model.
        self.model = AutoModelForCausalLM.from_pretrained(
            CHATBOT_MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

        # Configure padding.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Generation settings.
        self.max_history_turns = 4
        self.max_new_tokens = 100
        self.max_input_tokens = 768

        logger.info(
            "Chatbot initialized successfully."
        )

    # ======================================
    # VALIDATE MESSAGE
    # ======================================

    @staticmethod
    def validate_message(message):

        if not isinstance(message, str):
            raise ValueError("Message must be text.")

        message = message.strip()

        if not message:
            raise ValueError("Message cannot be empty.")

        if len(message) > 1000:
            raise ValueError(
                "Message cannot exceed 1000 characters."
            )

        return message

    # ======================================
    # BUILD CONVERSATION HISTORY
    # ======================================

    def build_history(self, history):

        if not isinstance(history, list):
            return []

        valid_history = []

        for item in history:

            if not isinstance(item, dict):
                continue

            speaker = item.get("speaker")
            message = item.get("message")

            if speaker not in ["user", "assistant"]:
                continue

            if not isinstance(message, str):
                continue

            message = message.strip()

            if not message:
                continue

            valid_history.append({
                "speaker": speaker,
                "message": message[:1000]
            })

        return valid_history[
            -self.max_history_turns * 2:
        ]

    # ======================================
    # ENCODE CONVERSATION
    # ======================================

    def encode_conversation(self, history, message):

        conversation_text = ""

        for item in history:

            conversation_text += (
                item["message"] +
                self.tokenizer.eos_token
            )

        conversation_text += (
            message +
            self.tokenizer.eos_token
        )

        encoded = self.tokenizer.encode(
            conversation_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens
        )

        return encoded.to(self.device)

    # ======================================
    # GENERATE CHATBOT RESPONSE
    # ======================================

    def generate_response(self, input_ids):

        attention_mask = torch.ones_like(
            input_ids
        )

        with torch.inference_mode():

            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )

        generated_ids = output_ids[
            :,
            input_ids.shape[-1]:
        ]

        response = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        ).strip()

        if not response:

            response = (
                "I'm sorry, I couldn't generate a response. "
                "Please try asking another question."
            )

        return response

    # ======================================
    # MAIN CHAT FUNCTION
    # ======================================

    def chat(self, message, history=None):

        message = self.validate_message(message)

        conversation = self.build_history(
            history
        )

        try:

            input_ids = self.encode_conversation(
                conversation,
                message
            )

            response = self.generate_response(
                input_ids
            )

            updated_history = conversation + [
                {
                    "speaker": "user",
                    "message": message
                },
                {
                    "speaker": "assistant",
                    "message": response
                }
            ]

            updated_history = updated_history[
                -self.max_history_turns * 2:
            ]

            logger.info(
                "Chatbot response generated successfully."
            )

            return {
                "response": response,
                "history": updated_history,
                "model": CHATBOT_MODEL_NAME
            }

        except Exception:

            logger.exception(
                "Chatbot response generation failed."
            )

            raise RuntimeError(
                "The chatbot could not generate a response."
            )

    # ======================================
    # RESET CONVERSATION
    # ======================================

    @staticmethod
    def reset_chat():

        return {
            "response": "",
            "history": [],
            "message": "Conversation reset successfully."
        }


# ==========================================
# MODULE EXPORTS
# ==========================================

__all__ = [
    "Chatbot"
]
