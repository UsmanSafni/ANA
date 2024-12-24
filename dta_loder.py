
import os
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#openai embeddings
from langchain_openai import OpenAIEmbeddings

# details here: https://openai.com/blog/new-embedding-models-and-api-updates
openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')

# Directory containing PDF files
pdf_directory = "docs"

# Initialize a list to store documents
docs = []

print('Extract text from each PDF file')
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):  # Process only PDF files
        pdf_path = os.path.join(pdf_directory, filename)
        pdf_reader = PdfReader(pdf_path)

        # Combine text from all pages
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text()

        # Create a Document object for the PDF
        docs.append(
            Document(
                page_content=full_text,
                metadata={"file_name": filename}  # Include file name as metadata
            )
        )

print('Chunk the documents')
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
chunked_docs = splitter.split_documents(docs)

#set up vector db


# create vector DB of docs and embeddings - takes < 30s on Colab

chroma_db = Chroma.from_documents(documents=chunked_docs,
                                  collection_name='rag_wikipedia_db',
                                  embedding=openai_embed_model,
                                  # need to set the distance function to cosine else it uses euclidean by default
                                  # check https://docs.trychroma.com/guides#changing-the-distance-function
                                  collection_metadata={"hnsw:space": "cosine"},
                                  persist_directory="./wikipedia_db")

#set up db retriever


similarity_threshold_retriever = chroma_db.as_retriever(search_type="similarity_score_threshold",
                                                        search_kwargs={"k": 3,
                                                                       "score_threshold": 0.3})

