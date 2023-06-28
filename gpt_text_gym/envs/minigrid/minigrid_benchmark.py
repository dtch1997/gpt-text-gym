class MinigridUnderstandingBenchmark:
    """Question answering benchmark for MiniGrid environments"""

    def __init__(self, env):
        self.env = env
        self.env.reset()

    def get_questions(self):
        """Returns a list of questions"""
        return [
            "There is a key in the same room as the agent. True or False?",
            "There is a door adjoining the room the agent is in. True or False?",
        ]

    def get_true_answer(self, question):
        """Returns the answer to a question"""
