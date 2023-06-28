from gpt_text_gym.gpt import GPTChatCompleter, Message, default_system_message

prompts = [
    "Problem: Jane has 12 flowers. She gives 2 flowers to her mom and 3 flowers to her dad. How many flowers does she have left?",
    "Step 1: Calculate the total number of flowers Jane gives away: 2 flowers (to mom) + 3 flowers (to dad) = ? flowers given away",
    "Step 2: Subtract the total number of flowers given away from the initial number of flowers Jane had: 12 flowers (initial) - ? flowers given away = ? flowers left",
]


def main(argv):
    chatbot = GPTChatCompleter(
        model="gpt-4",
    )
    chatbot.add_message(default_system_message())
    for prompt in prompts:
        chatbot.add_message(Message(role="user", content=prompt))
    cot_reply = chatbot.generate_chat_completion()

    chatbot = GPTChatCompleter(
        model="gpt-4",
    )
    chatbot.add_message(default_system_message())
    chatbot.add_message(Message(role="user", content=prompts[0]))
    zs_reply = chatbot.generate_chat_completion()

    print("Chain of Thought reply: ")
    print(cot_reply)
    print("Zero-shot reply: ")
    print(zs_reply)


if __name__ == "__main__":
    main(0)
