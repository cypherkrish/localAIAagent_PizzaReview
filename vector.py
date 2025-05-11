from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd


df = pd.read_csv('./src/docs/restaurant_reviews.csv')
embeddings = OllamaEmbeddings(model = "mxbai-embed-large")

chroma_db_path = "./chroma_langchain_db"
add_documents = not os.path.exists(chroma_db_path)

# Creating the data

if add_documents:
    documents = []
    ids = []
    
    # Column names: Title, Date, Rating, Review
    for i, row in df.iterrows():
        document = Document(
            page_content=row['Title'] + " " + row['Review'],
            metadata = {"rating" : row["Rating"], "date": row["Date"]},
            id = str(i)
        )
        documents.append(document)
        ids.append(str(i))
        
# Add the data to vector store

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=chroma_db_path,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids= ids)
    
retriever = vector_store.as_retriever(
    search_kwards = {"k": 5}
)