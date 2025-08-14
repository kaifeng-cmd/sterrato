import os
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
def get_or_create_vectorstore(doc_path: str = "sample.txt"):
    # Try to connect to Qdrant client
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Create a sample document if it doesn't exist
    if not os.path.exists(doc_path):
        print(f"Sample file {doc_path} not found. Creating a sample document...")
        sample_content = """
        人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        人工智能的研究历史有着一条从以“推理”为重点到以“知识”为重点，再到以“学习”为重点的自然、清晰的脉络。
        人工智能领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
        机器学习是人工智能的核心，是使计算机具有智能的根本途径，其应用遍及人工智能的各个领域。
        深度学习是机器学习中一种基于对数据进行表征学习的方法。观测值（例如图像、声音、文本）等原始数据可以使用多种方式来表示，如每个像素强度值，或者更抽象的表示如图像边缘、纹理和形状等。
        """
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(sample_content)

    # Check if collection exists
    try:
        collection_info = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists. Connecting to existing collection.")
        # Check if the collection is empty, if so, add documents
        if collection_info.points_count == 0:
            print(f"Collection '{QDRANT_COLLECTION_NAME}' is empty. Loading and vectorizing documents...")
            with open(doc_path, "r", encoding="utf-8") as f:
                text = f.read()
            documents = [Document(page_content=text)]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            
            vectorstore_instance = QdrantVectorStore.from_existing_collection(
                embedding=embedding_function,
                collection_name=QDRANT_COLLECTION_NAME,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            vectorstore_instance.add_documents(docs)
            print("Documents have been added to the existing collection.")
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
        
        print(f"Loading documents from {doc_path} to create a new collection...")
        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read()
        documents = [Document(page_content=text)]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        print(f"Vectorizing {len(docs)} document chunks and creating a new collection '{QDRANT_COLLECTION_NAME}'...")
        vectorstore = QdrantVectorStore.from_documents(
            docs,
            embedding=embedding_function,
            collection_name=QDRANT_COLLECTION_NAME,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        print("New vector database collection created and populated!")
        return vectorstore

# To initialize vectorstore
vectorstore = get_or_create_vectorstore()

# --- 8. Initialize Memory ---
# keep only the last 10 message pairs (5 rounds of conversation)
memory = ConversationBufferWindowMemory(k=10, return_messages=True)

# --- 9. Build RAG Chain ---
# Retrieval function
def retrieve_chunks(query: str):
    print(f"\nRetrieving documents related to '{query}'...")
    results = vectorstore.similarity_search(query, k=2)
    if not results:
        print("No relevant documents found.")
        return ""
    context = "\n\n".join([doc.page_content for doc in results])
    print(f"Found {len(results)} relevant document snippets.")
    return context

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
    print("AI Chat Assistant has initialized! (with Memory)")
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
