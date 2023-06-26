import unittest
from unittest.mock import patch, MagicMock
from gpt_text_gym.gpt import GPTChatCompleter, openai_chat_completion_create, Message


class TestMessage(unittest.TestCase):
    def setUp(self):
        self.raw_message = {"role": "system", "content": "You are a helpful assistant."}
        self.message = Message("system", "You are a helpful assistant.")

    def test_message_from_dict(self):
        message = Message.from_dict(self.raw_message)
        self.assertEqual(message.role, self.message.role)
        self.assertEqual(message.content, self.message.content)

    def test_message_to_dict(self):
        raw_message = self.message.to_dict()
        self.assertEqual(raw_message, self.raw_message)

    def test_message_str(self):
        str_message = str(self.message)
        self.assertEqual(str_message, f"{self.message.role}: {self.message.content}")


class TestGPTChatCompleter(unittest.TestCase):
    def setUp(self):
        self.messages = [Message("system", "You are a helpful assistant.")]
        self.model = "gpt-3.5-turbo"
        self.n = 1
        self.temperature = 1.0
        self.max_tokens = None
        self.kwargs = {}

    @patch("dotenv.get_key", return_value="dummy_api_key")
    def test_gpt_chat_completer_init(self, mock_get_key):
        chat_completer = GPTChatCompleter()
        self.assertEqual(chat_completer.model, self.model)
        self.assertEqual(chat_completer.temperature, self.temperature)
        self.assertIsNone(chat_completer.max_tokens)
        self.assertEqual(chat_completer.n, self.n)
        self.assertEqual(chat_completer.chat_history, [])
        mock_get_key.assert_called_once()

    @patch("dotenv.get_key", return_value="dummy_api_key")
    def test_gpt_chat_completer_clear(self, mock_get_key):
        chat_completer = GPTChatCompleter()
        chat_completer.add_message(self.messages[0])
        chat_completer.clear()
        self.assertEqual(chat_completer.chat_history, [])

    @patch("dotenv.get_key", return_value="dummy_api_key")
    def test_gpt_chat_completer_add_message(self, mock_get_key):
        chat_completer = GPTChatCompleter()
        chat_completer.add_message(self.messages[0])
        self.assertEqual(chat_completer.chat_history, self.messages)

    @patch("openai.ChatCompletion.create")
    @patch("dotenv.get_key", return_value="dummy_api_key")
    def test_gpt_chat_completer_generate_chat_completion(
        self, mock_get_key, mock_create
    ):
        chat_completer = GPTChatCompleter()
        mock_create.return_value = {
            "choices": [{"message": self.messages[0].to_dict()}]
        }
        chat_completer.add_message(self.messages[0])
        res = chat_completer.generate_chat_completion()
        self.assertEqual(res, self.messages[0])
        mock_create.assert_called_once_with(
            model=self.model,
            messages=[msg.to_dict() for msg in self.messages],
            n=self.n,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.kwargs,
        )

    @patch("openai.ChatCompletion.create")
    def test_openai_chat_completion_create(self, mock_create):
        mock_create.return_value = {
            "choices": [{"message": self.messages[0].to_dict()}]
        }
        res = openai_chat_completion_create(
            model=self.model,
            messages=[msg.to_dict() for msg in self.messages],
            n=self.n,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.kwargs,
        )
        self.assertEqual(res, mock_create.return_value)
        mock_create.assert_called_once_with(
            model=self.model,
            messages=[msg.to_dict() for msg in self.messages],
            n=self.n,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.kwargs,
        )


if __name__ == "__main__":
    unittest.main()
