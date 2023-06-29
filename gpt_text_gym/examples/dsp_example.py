import dsp
import openai
import dotenv

from gpt_text_gym import ROOT_DIR

LLM_MODEL = "text-davinci-002"
OPENAI_TEMPERATURE = 0.0
OPENAI_API_KEY = dotenv.get_key(ROOT_DIR / ".env", "API_KEY")


def vanilla_QA_LM(question: str) -> str:

    Question = dsp.Type(prefix="Question:", desc="${the question to be answered}")
    Answer = dsp.Type(
        prefix="Answer:",
        desc="${a short factoid answer, often between 1 and 5 words}",
        format=dsp.format_answers,
    )

    qa_template = dsp.Template(
        instructions="Answer questions with short factoid answers.",
        question=Question(),
        answer=Answer(),
    )

    # demos = dsp.sample()
    example = dsp.Example(question=question, demos=[])
    example, completions = dsp.generate(qa_template)(example, stage="qa")
    return completions[0].answer


if __name__ == "__main__":
    # Set up dsp
    language_model = dsp.GPT3(LLM_MODEL, OPENAI_API_KEY)
    dsp.settings.configure(lm=language_model)

    question = "What is the capital of the United States?"
    answer = vanilla_QA_LM(question)
    print(question)
    print(answer)
    print(language_model.inspect_history(n=1))
