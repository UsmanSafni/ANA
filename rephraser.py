from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

class QuestionRephraser:
    """
    A class to handle question rewriting using an LLM.
    """

    def __init__(self, model_name="gpt-4o", temperature=0):
        """
        Initializes the QuestionRephraser with the specified model and temperature.

        Args:
            model_name (str): The name of the LLM model to use.
            temperature (float): Sampling temperature for the LLM.
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.output_parser = StrOutputParser()

        # system prompt for rephrasing
        self.SYS_PROMPT = """Act as a question re-writer and perform the following task:
                             - Convert the following input question to a better version that is optimized for web search.
                             - When re-writing, look at the input question and try to reason about the underlying semantic intent/meaning.
                          """
        
        re_write_prompt = self.build_prompt()
        self.rephraser_chain = re_write_prompt | self.llm | self.output_parser


    def build_prompt(self):
        """
        Builds a ChatPromptTemplate for question rephrasing.
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.SYS_PROMPT),
                ("human", """Here is the initial question:
                             {question}

                             Formulate an improved question.
                          """),
            ]
        )

    

if __name__ == "__main__":
    # Initialize the rephraser
    rephraser = QuestionRephraser(model_name="gpt-4o", temperature=0)

    # Original question
    original_question = "What are the best practices for improving website SEO?"

    # Get the rephrased question
    rephrased_question = rephraser.rephrase_question(original_question)

    print(f"Original Question: {original_question}")
    print(f"Rephrased Question: {rephrased_question}")
