from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class DocGrader:
    """
    A class to handle document grading using an LLM.
    """

    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""
        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    def __init__(self, model_name="gpt-4o", temperature=0):
        """
        Initializes the DocumentGrader with the specified model and temperature.

        Args:
            model_name (str): The name of the LLM model to use.
            temperature (float): Sampling temperature for the LLM.
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_llm_grader = self.llm.with_structured_output(self.GradeDocuments)

        # Default system prompt for the grader
        self.SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.
        Follow these instructions for grading:
          - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
          - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not.
        """

        self.prompt = self.build_prompt()

        # Generate the grade
        self.grader_chain = self.prompt | self.structured_llm_grader

    def build_prompt(self):
        """
        Builds a ChatPromptTemplate for grading documents.
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.SYS_PROMPT),
                ("human", """Retrieved document:
                             {document}

                             User question:
                             {question}
                          """),
            ]
        )

    

if __name__ == "__main__":
# Initialize the grader
    doc_grader = DocGrader()

# Example document and question
    retrieved_document = """
This document discusses the effects of climate change on global agriculture.
It highlights the need for sustainable practices to adapt to shifting weather patterns.
"""
    user_question = "How does climate change affect agriculture?"

# Grade the document
    grade = doc_grader.grade_document(document=retrieved_document, question=user_question)

    print(f"Grade: {grade}")  # Output: 'yes' or 'no'
