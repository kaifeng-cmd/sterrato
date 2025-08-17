import os
import time
import re
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from qdrant_client import QdrantClient
import pdfplumber

# --- 1. Load environment variables ---
load_dotenv()

# --- 2. API keys and configuration ---

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")

if not all([OPENROUTER_API_KEY, QDRANT_API_KEY, QDRANT_URL]):
    raise ValueError("Please ensure OPENROUTER_API_KEY, QDRANT_API_KEY, and QDRANT_URL are set in your .env file")

# --- 3. Custom Embedding class (using Hugging Face Inference Client) ---
class CustomIBMEmbeddings(Embeddings):
    def __init__(self, model_name="ibm-granite/granite-embedding-278m-multilingual"):
        self.client = InferenceClient(
            provider="hf-inference",
            model=model_name,
            api_key=os.environ.get("HF_TOKEN") # using Hugging Face Hub's inference API
        )        
        self.prefix = "passage: "

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Add prefix to documents
        prefixed_texts = [self.prefix + text for text in texts]
        embeddings = self.client.feature_extraction(prefixed_texts)
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        # Add prefix to query
        prefixed_text = "query: " + text
        embedding = self.client.feature_extraction(prefixed_text)
        return embedding.tolist()

# --- 4. Initialize LLM (using LangChain's ChatOpenAI, configured for OpenRouter) ---
llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=1.0
)

# --- 5. Initialize Embedding ---
embedding_function = CustomIBMEmbeddings()

# --- 6. Initialize Qdrant Vector Store ---
def load_pdf_document(pdf_path: str) -> list[Document]:
    """Load PDF document using pdfplumber, simplified version"""
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file '{pdf_path}' does not exist")
        
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File '{pdf_path}' is not a PDF")
        
        documents = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():  # Only process pages with content
                    # Extract page number
                    """ Holding off on page number extraction for now, as it may not be needed
                    or not suitable for this scenario."""
                    #page_number = extract_page_number(text)
                    
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

def extract_page_number(text: str) -> int:
    """Extract page number from text content"""
    patterns = [
        r'第(\d+)页',  # Chinese page number
        r'Page\s+(\d+)',  # English page number
        r'^\s*(\d+)\s+THE\s+CHINESE\s+COOK\s+BOOK',  # Specific format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    
    return None

def get_or_create_vectorstore(doc_path: str = "chinesecookbook.pdf"):
    """Create or get vectorstore using PDF document (batch writing to avoid timeout)"""
    
    # Check if PDF file exists
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"PDF document '{doc_path}' does not exist. Please make sure the document file is in the current directory.")
    
    if not doc_path.lower().endswith('.pdf'):
        raise ValueError(f"File '{doc_path}' is not a PDF.")
    
    try:
        # Try to connect to Qdrant client
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Check if collection exists
        try:
            collection_info = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists. Connecting to existing collection.")
            # Check if the collection is empty, if so, add documents
            if collection_info.points_count == 0:
                print(f"Collection '{QDRANT_COLLECTION_NAME}' is empty. Loading and vectorizing PDF document...")
                
                # Load PDF document
                documents = load_pdf_document(doc_path)
                
                # Optimize text splitting
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
                
                # Add documents in batches
                batch_size = 10  # Further reduce batch size
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i + batch_size]
                    print(f"Adding batch {i//batch_size + 1}, total {len(batch)} documents...")
                    try:
                        vectorstore_instance.add_documents(batch)
                        print(f"Batch {i//batch_size + 1} added successfully")
                    except Exception as e:
                        print(f"Batch {i//batch_size + 1} failed: {e}")
                        # Try adding individually
                        for doc in batch:
                            try:
                                vectorstore_instance.add_documents([doc])
                                print(f"Single document added successfully")
                            except Exception as single_e:
                                print(f"Single document failed: {single_e}")
                    time.sleep(5)  # Increase delay
                    
                print("All documents have been added to the collection.")
                return vectorstore_instance
            else:
                return QdrantVectorStore.from_existing_collection(
                    embedding=embedding_function,
                    collection_name=QDRANT_COLLECTION_NAME,
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                )
                
        except Exception as e:
            # If collection does not exist
            print(f"Collection '{QDRANT_COLLECTION_NAME}' does not exist or cannot be accessed ({e}). Attempting to create a new collection.")
            
            # Load PDF document
            documents = load_pdf_document(doc_path)
            
            # Optimize text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(documents)
            
            print(f"Vectorizing {len(docs)} document chunks and creating new collection '{QDRANT_COLLECTION_NAME}'...")
            
            # Create empty collection first
            vectorstore = QdrantVectorStore.from_documents(
                [],
                embedding=embedding_function,
                collection_name=QDRANT_COLLECTION_NAME,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            
            # Add documents in batches
            batch_size = 10  # Further reduce batch size
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                print(f"Adding batch {i//batch_size + 1}, total {len(batch)} documents...")
                try:
                    vectorstore.add_documents(batch)
                    print(f"Batch {i//batch_size + 1} added successfully")
                except Exception as e:
                    print(f"Batch {i//batch_size + 1} failed: {e}")
                    # Try adding individually
                    for doc in batch:
                        try:
                            vectorstore.add_documents([doc])
                            print(f"Single document added successfully")
                        except Exception as single_e:
                            print(f"Single document failed: {single_e}")
                time.sleep(5)  # Increase delay
            
            print("New vector database collection created and populated!")
            return vectorstore
            
    except Exception as e:
        raise RuntimeError(f"Error creating or getting vectorstore: {str(e)}")

# To initialize vectorstore
vectorstore = get_or_create_vectorstore()

# --- 8. Initialize Memory ---
# keep only the last  message pairs (5 rounds of conversation)
memory = ConversationBufferWindowMemory(k=10, return_messages=True)

# --- 9. Build RAG Chain ---
# Retrieval function
def retrieve_chunks(query: str):
    """Retrieve relevant document chunks"""
    print(f"\nRetrieving documents related to '{query}'...")
    
    try:
        results = vectorstore.similarity_search(query, k=3)
        
        if not results:
            print("No relevant documents found.")
            return ""
        
        # Format results
        formatted_results = []
        for i, doc in enumerate(results, 1):
            formatted_results.append(f"Document chunk {i}:")
            formatted_results.append(f"Content: {doc.page_content}")
            
            # Add basic metadata (page number display removed)
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

# New ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional {role}. 
                    Please answer the user's question based on the background information below. 
                    If there is no relevant information in the background, please answer 'Sry, I don't know'. 
                    Background Information:\n\n{context}"""),
    MessagesPlaceholder(variable_name="chat_history"), # Conversation history placeholder
    ("user", "{question}") # Current question
])

# Build LCEL chain
rag_chain = (
    {
        "question": RunnablePassthrough(), # Pass through the user question
        "role": lambda x: "AI Assistant", # Fixed role
        "context": lambda x: retrieve_chunks(x["question"]), # Retrieve context based on the question
        "chat_history": lambda x: memory.load_memory_variables({})["history"] # Load history from memory
    }
    | chat_prompt
    | llm
    | StrOutputParser()
)

# --- 10. CLI Interaction Loop ---
if __name__ == "__main__":
    print("AI Chat Assistant has initialized! (with Memory) - PDF Version")
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
