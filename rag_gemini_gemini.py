import os
import time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from qdrant_client import QdrantClient
import pdfplumber

# --- 1. Load environment variables ---
load_dotenv()

# --- 2. API keys and configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")

if not all([GOOGLE_API_KEY, QDRANT_API_KEY, QDRANT_URL]):
    raise ValueError("Please ensure GOOGLE_API_KEY, QDRANT_API_KEY, and QDRANT_URL are set in your .env file")

# --- 3. Initialize LLM (using Google Gemini API) ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=1.0,
)

# --- 4. Initialize Gemini Embedding ---
# Create a wrapper class to handle different embedding types for documents and queries
# I found that models/embedding-001 can only handle English, but gemini-embedding-001 can for multi-language
# But idk gemini-embedding-001 can't be set to 768 dimensions via output_dimensionality, default is 3072 dimensions.
class GeminiEmbeddingsWrapper(Embeddings):
    def __init__(self, google_api_key: str, model_name: str = "gemini-embedding-001", output_dimensionality: int = 768):
        self.doc_embedding = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=google_api_key,
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=output_dimensionality
        )
        self.query_embedding = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=google_api_key,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=output_dimensionality
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.doc_embedding.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.query_embedding.embed_query(text)

# Initialize the embedding function with the wrapper
embedding_function = GeminiEmbeddingsWrapper(google_api_key=GOOGLE_API_KEY)

# --- 5. PDF Document Loader ---
def load_pdf_document(pdf_path: str) -> list[Document]:
    """Load PDF document using pdfplumber, simplified version"""
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file '{pdf_path}' does not exist")
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File '{pdf_path}' is not a PDF")
        documents = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
 
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "document_type": "pdf"
                    }
                ))

        print(f"Successfully loaded PDF document: {pdf_path}, total {len(documents)} pages")
        return documents
    except Exception as e:
        raise ValueError(f"Error reading PDF file {pdf_path}: {str(e)}")

def get_or_create_vectorstore(doc_path: str = "chinesecookbook.pdf"):
    """Create or get vectorstore using PDF document (batch writing to avoid timeout)"""
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"PDF document '{doc_path}' does not exist. Please make sure the document file is in the current directory.")
    if not doc_path.lower().endswith('.pdf'):
        raise ValueError(f"File '{doc_path}' is not a PDF.")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        try:
            collection_info = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists. Connecting to existing collection.")
            if collection_info.points_count == 0:
                print(f"Collection '{QDRANT_COLLECTION_NAME}' is empty. Loading and vectorizing PDF document...")
                documents = load_pdf_document(doc_path)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=700,
                    chunk_overlap=200
                )
                docs = text_splitter.split_documents(documents)

                print(f"PDF document split into {len(docs)} text chunks.")
                vectorstore_instance = QdrantVectorStore.from_existing_collection(
                    embedding=embedding_function,
                    collection_name=QDRANT_COLLECTION_NAME,
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                )
                batch_size = 10
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i + batch_size]
                    print(f"Adding batch {i//batch_size + 1}, total {len(batch)} documents...")
                    try:
                        vectorstore_instance.add_documents(batch)
                        print(f"Batch {i//batch_size + 1} added successfully")
                    except Exception as e:
                        print(f"Batch {i//batch_size + 1} failed: {e}")
                        for doc in batch:
                            try:
                                vectorstore_instance.add_documents([doc])
                                print(f"Single document added successfully")
                            except Exception as single_e:
                                print(f"Single document failed: {single_e}")
                    time.sleep(5)
                print("All documents have been added to the collection.")
                return vectorstore_instance
            else:
                vectorstore_instance = QdrantVectorStore.from_existing_collection(
                    embedding=embedding_function,
                    collection_name=QDRANT_COLLECTION_NAME,
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                )
                return vectorstore_instance
        except Exception as e:
            print(f"Collection '{QDRANT_COLLECTION_NAME}' does not exist or cannot be accessed ({e}). Attempting to create a new collection.")
            documents = load_pdf_document(doc_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(documents)

            print(f"Vectorizing {len(docs)} document chunks and creating new collection '{QDRANT_COLLECTION_NAME}'...")
            vectorstore = QdrantVectorStore.from_documents(
                [],
                embedding=embedding_function,
                collection_name=QDRANT_COLLECTION_NAME,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            batch_size = 10
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                print(f"Adding batch {i//batch_size + 1}, total {len(batch)} documents...")
                try:
                    vectorstore.add_documents(batch)
                    print(f"Batch {i//batch_size + 1} added successfully")
                except Exception as e:
                    print(f"Batch {i//batch_size + 1} failed: {e}")
                    for doc in batch:
                        try:
                            vectorstore.add_documents([doc])
                            print(f"Single document added successfully")
                        except Exception as single_e:
                            print(f"Single document failed: {single_e}")
                time.sleep(5)
            print("New vector database collection created and populated!")
            return vectorstore
    except Exception as e:
        raise RuntimeError(f"Error creating or getting vectorstore: {str(e)}")

vectorstore = get_or_create_vectorstore()

# --- 6. Initialize Memory ---
memory = ConversationBufferWindowMemory(k=10, return_messages=True)

# --- 7. Build RAG Chain ---
def retrieve_chunks(query: str):
    print(f"\nRetrieving documents related to '{query}'...")
    try:
        results = vectorstore.similarity_search(query, k=3)
        if not results:
            print("No relevant documents found.")
            return ""
        formatted_results = []
        for i, doc in enumerate(results, 1):
            formatted_results.append(f"Document chunk {i}:")
            formatted_results.append(f"Content: {doc.page_content}")
            if doc.metadata:
                metadata_info = []
                if metadata_info:
                    formatted_results.append("Info: " + " | ".join(metadata_info))
            formatted_results.append("-" * 50)
        result_str = "\n".join(formatted_results)
        print(f"Found {len(results)} relevant document chunks.")
        print("Retrieval result:\n" + result_str)
        return result_str
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
        return ""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional {role}. But can be lively and interesting.
        If user query is in Chinese, answer with Chinese, vice versa for English too.
        Please answer the user's question naturally and helpfully based on the background information below. 
        If the information is not available in the background, 
        reply: 'Sry, I can't answer your question because it has reached my knowledge limit' 
        or translate it into Chinese if query is in Chinese.
        Do not explicitly mention 'background information'; instead, speak as if it comes from your own knowledge. 
        If no relevant info, you can answer based on your own, but must related to your role.
        Background Information:\n\n{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}")
])

rag_chain = (
    {
        "question": RunnablePassthrough(),
        "role": lambda x: "Chinese Food & Culture Expert",
        "context": lambda x: retrieve_chunks(x["question"]),
        "chat_history": lambda x: memory.load_memory_variables({})["history"]
    }
    | chat_prompt
    | llm
    | StrOutputParser()
)

# --- 8. CLI Interaction Loop ---
if __name__ == "__main__":
    print("AI Chat Assistant has initialized! (with Memory) - Gemini Version with Gemini Embedding")
    print("Type 'exit' or 'quit' to quit.")
    while True:
        user_input = input("\nYour Question: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Thank you! Goodbye.")
            break

        try:
            print("\nSearching for an answer...")
            # Call the chain, it will automatically load history from memory
            answer = rag_chain.invoke({"question": user_input})
            print("\nAnswer:")
            print(answer)

            # Save the new Q&A pair to memory
            memory.save_context({"input": user_input}, {"output": answer})

            # Print memory content for debugging and observation
            print("\n--- Current Memory Content (recent messages) ---")
            history_messages = memory.load_memory_variables({})["history"]
            for msg in history_messages:
                print(f"{type(msg).__name__}: {msg.content}")
            print("------------------------------------")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
