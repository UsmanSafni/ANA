import os
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class DataLoader:
    def __init__(self, pdf_directory="docs", persist_directory="./wikipedia_db", chunk_size=2000, chunk_overlap=300):
        """
        Initializes the DataLoader class with required parameters.
        
        Args:
            pdf_directory (str): Directory containing PDF files.
            embedding_model: Embedding model for Chroma.
            persist_directory (str): Directory for storing the vector DB.
            chunk_size (int): Maximum size of text chunks.
            chunk_overlap (int): Overlap size between chunks.
        """
        self.pdf_directory = pdf_directory
        self.embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs = []
        self.chunked_docs = []
        self.chroma_db = None
        self.retriever = None

        self.load_pdfs()
        self.chunk_documents()
        self.setup_vector_db()
        self.setup_retriever()


    def load_pdfs(self):
        """
        Loads PDFs from the specified directory and converts them into Document objects.
        """
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith(".pdf"):  # Process only PDF files
                pdf_path = os.path.join(self.pdf_directory, filename)
                pdf_reader = PdfReader(pdf_path)

                # Combine text from all pages
                full_text = "".join([page.extract_text() for page in pdf_reader.pages])

                # Create a Document object for the PDF
                self.docs.append(
                    Document(
                        page_content=full_text,
                        metadata={"file_name": filename}  # Include file name as metadata
                    )
                )
        print(f"[INFO] Loaded {len(self.docs)} PDF(s) from {self.pdf_directory}.")

    def chunk_documents(self):
        """
        Splits the loaded documents into smaller chunks using RecursiveCharacterTextSplitter.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        self.chunked_docs = splitter.split_documents(self.docs)
        print(f"[INFO] Chunked documents into {len(self.chunked_docs)} smaller pieces.")

    def setup_vector_db(self):
        """
        Sets up the Chroma vector database using the chunked documents.
        """
        if not self.chunked_docs:
            raise ValueError("No chunked documents found. Please run `chunk_documents()` first.")
        
        self.chroma_db = Chroma.from_documents(
            documents=self.chunked_docs,
            collection_name='rag_wikipedia_db',
            embedding=self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=self.persist_directory
        )
        print("[INFO] Vector database setup completed and persisted.")

    def setup_retriever(self, k=3, score_threshold=0.3):
        """
        Sets up the similarity-based retriever.
        
        Args:
            k (int): Number of top results to return.
            score_threshold (float): Threshold for similarity score.
        """
        if not self.chroma_db:
            raise ValueError("Chroma DB not initialized. Please run `setup_vector_db()` first.")
        
        self.retriever = self.chroma_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold}
        )
        print("[INFO] Retriever setup completed.")
        

    def get_retriever(self):
        """
        Returns the configured retriever.
        """
        
        if not self.retriever:
            raise ValueError("Retriever not initialized. Please run `setup_retriever()` first.")
        print("[INFO] Retriever present.")
        return self.retriever

if __name__ == "__main__":
# Initialize DataLoader
    data_loader = DataLoader()


# Access the retriever
    retriever = data_loader.get_retriever()