from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from graderr import doc_grader
from rephraserr import question_rewriter
from dta_loder import similarity_threshold_retriever
from utils import tv_search
from qa_rag_chainn import qa_rag_chain
from data_loader import DataLoader
from grader import DocGrader
from rephraser import QuestionRephraser
from qa_rag_chain import QARAGChain
from langgraph.graph import END, StateGraph


class GraphState(TypedDict):
    """Represents the state of our graph."""
    question: str
    generation: str
    web_search_needed: str
    documents: List[str]


class Agent:
    """Encapsulates all steps of the agent's workflow."""

    def __init__(self):
        self.data_loader = DataLoader()
        self.grader = DocGrader()
        self.rephraser = QuestionRephraser()
        self.ans_generator = QARAGChain()

    def retrieve(self, state: GraphState) -> GraphState:
        """Retrieve documents."""
        print("---RETRIEVAL FROM VECTOR DB---")
        question = state["question"]
        documents = self.data_loader.get_retriever().invoke(question)
        return {"documents": documents, "question": question}

    def grade_documents(self, state: GraphState) -> GraphState:
        """Grade documents for relevance."""
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        web_search_needed = "No"

        if documents:
            for doc in documents:
                score = self.grader.grader_chain.invoke(
                    {"question": question, "document": doc.page_content}
                )
                if score.binary_score == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(doc)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    web_search_needed = "Yes"
        else:
            print("---NO DOCUMENTS RETRIEVED---")
            web_search_needed = "Yes"

        return {"documents": filtered_docs, "question": question, "web_search_needed": web_search_needed}

    def rewrite_query(self, state: GraphState) -> GraphState:
        """Rewrite the query."""
        print("---REWRITE QUERY---")
        question = state["question"]
        better_question = self.rephraser.rephraser_chain.invoke({"question": question})
        return {"documents": state["documents"], "question": better_question}

    def web_search(self, state: GraphState) -> GraphState:
        """Perform a web search."""
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]
        web_results = tv_search.invoke(question)
        web_content = "\n\n".join([doc["content"] for doc in web_results])
        documents.append(Document(page_content=web_content))
        return {"documents": documents, "question": question}

    def generate_answer(self, state: GraphState) -> GraphState:
        """Generate an answer."""
        print("---GENERATE ANSWER---")
        question = state["question"]
        documents = state["documents"]
        generation = self.ans_generator.rag_chain.invoke(
            {"context": documents, "question": question}
        )
        return {"documents": documents, "question": question, "generation": generation}

    def decide_to_generate(self, state: GraphState) -> str:
        """Decide the next step."""
        print("---ASSESS GRADED DOCUMENTS---")
        if state["web_search_needed"] == "Yes":
            print("---DECISION: REWRITE QUERY---")
            return "rewrite_query"
        else:
            print("---DECISION: GENERATE RESPONSE---")
            return "generate_answer"


class AgenticRAG:
    """Sets up and executes the agent's workflow using a state graph."""

    def __init__(self):
        self.agent = Agent()
        self.graph = StateGraph(GraphState)

        # Define the nodes
        self.graph.add_node("retrieve", self.agent.retrieve)
        self.graph.add_node("grade_documents", self.agent.grade_documents)
        self.graph.add_node("rewrite_query", self.agent.rewrite_query)
        self.graph.add_node("web_search", self.agent.web_search)
        self.graph.add_node("generate_answer", self.agent.generate_answer)

        # Build graph
        self.graph.set_entry_point("retrieve")
        self.graph.add_edge("retrieve", "grade_documents")
        self.graph.add_conditional_edges(
            "grade_documents",
            self.agent.decide_to_generate,
            {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"},
        )
        self.graph.add_edge("rewrite_query", "web_search")
        self.graph.add_edge("web_search", "generate_answer")
        self.graph.add_edge("generate_answer", END)

        # Compile graph
        self.graph = self.graph.compile()

    def invoke(self, query: str):
        """Invoke the graph with the initial query."""
        response = self.graph.invoke({"question": query})
        return response


if __name__ == "__main__":
    query = "What is the capital of India?"
    agentic_rag = AgenticRAG()
    response = agentic_rag.invoke(query)
    print(response["generation"])
