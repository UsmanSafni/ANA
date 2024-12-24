from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

class QARAGChain:
    def __init__(self, model_name='gpt-3.5-turbo', temperature=0):
        """
        Initializes the QA RAG Chain with the model and prompt template.
        
        Args:
            model_name (str): The name of the language model to use.
            temperature (float): The sampling temperature for response generation.
        """
        # Create a prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a highly knowledgeable healthcare assistant specializing in Exercise, Diet, General Health, Sleep, Mental Health, Nutrition, and Drugs.
       Your sole purpose is to answer questions strictly related to these topics. 

       You are provided with the following retrieved context to answer the question.

       Guidelines:
       - If the question is unrelated to Exercise, Diet, General Health, Sleep, Mental Health, Nutrition, or Drugs, respond with: "I am only able to assist with healthcare-related topics and cannot answer this question."
       - If no context is present or if you don't know the answer, respond with: "I don't know the answer."
       - Do not generate an answer beyond the given context.
       - Provide concise, accurate, and context-based answers.

       Question:
       {question}

       Context:
       {context}

       Answer:
    """
        )
        
        # Initialize the language model
        self.chatgpt = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Define the RAG chain
        self.rag_chain = (
            {
                "context": (itemgetter('context') | RunnableLambda(self.format_docs)),
                "question": itemgetter('question'),
            }
            | self.prompt_template
            | self.chatgpt
            | StrOutputParser()
        )
    
    @staticmethod
    def format_docs(docs):
        """
        Formats the context documents by concatenating their content with new lines.
        
        Args:
            docs (list): List of document objects with 'page_content' attributes.
        
        Returns:
            str: Concatenated string of document contents.
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def run(self, question, context):
        """
        Executes the QA RAG chain with the provided question and context.
        
        Args:
            question (str): The input question.
            context (list): A list of context documents (each with a 'page_content' attribute).
        
        Returns:
            str: The generated answer from the RAG chain.
        """
        input_data = {"question": question, "context": context}
        return self.qa_rag_chain.invoke(input_data)

# Example usage
if __name__ == "__main__":
    # Initialize the QA RAG Chain
    qa_chain = QARAGChain(model_name="gpt-4o", temperature=0)
    
    # Example input
    question = "What is LangChain?"
    context = [{"page_content": "LangChain is a framework for building applications with LLMs."}]
    
    # Execute and print the output
    answer = qa_chain.run(question, context)
    print(answer)
