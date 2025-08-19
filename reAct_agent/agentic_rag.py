import os
import time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from datetime import datetime
import pdfplumber

# --- 1. Load environment variables ---
load_dotenv()

# --- 2. API keys and configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")

if not all([GOOGLE_API_KEY, OPENROUTER_API_KEY, QDRANT_API_KEY, QDRANT_URL]):
    raise ValueError("Please ensure all API keys are set in your .env file")

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

# --- 4. Initialize LLM (using Google Gemini API and OpenRouter API) ---
# Google Gemini LLM
llm_gemini = ChatGoogleGenerativeAI(
    # You can pick gemini-2.5-flash-lite, gemini-2.0-flash-001
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=1.0,
)

# OpenRouter LLM
llm_openrouter = ChatOpenAI(
    # You can pick others like moonshotai/kimi-k2:free, z-ai/glm-4.5-air:free
    model="deepseek/deepseek-chat-v3-0324:free",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=1.0
)

# Default LLM (can be changed by user)
llm = llm_openrouter

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

# --- 7. Initialize Tools ---
def create_tools():
    """Create tools for the agent."""
    tools = []
    
    # 1. Web search tool using ddgs
    def web_search(query: str) -> str:
        """Search the web using ddgs for latest information and real-time content."""
        try:
            from ddgs import DDGS
            
            print(f"\nSearching web for '{query}' using ddgs...")
            with DDGS() as ddgs:
                results = list(ddgs.text(query, region='wt-wt', safesearch='off', timelimit='y'))
                
                if not results:
                    return "No search results found."
                
                formatted_results = []
                for i, result in enumerate(results[:5], 1):  # Take for 5 results
                    formatted_results.append(f"Result {i}:")
                    formatted_results.append(f"Title: {result.get('title', 'N/A')}")
                    formatted_results.append(f"URL: {result.get('href', 'N/A')}")
                    formatted_results.append(f"Body: {result.get('body', 'N/A')[:200]}...")
                    formatted_results.append("-" * 50)
                
                print(f"Found {len(results)} search results.")
                return "\n".join(formatted_results)
                
        except ImportError:
            return "ddgs package not available. Please install with: pip install ddgs"
        except Exception as e:
            return f"Error during web search: {str(e)}"
    
    web_search_tool = Tool(
        name="web_search",
        func=web_search,
        description="Search the web for latest information and real-time content using ddgs"
    )
    tools.append(web_search_tool)
    
    # 2. Wikipedia search tool
    try:
        wikipedia = WikipediaQueryRun(
            name="wikipedia",
            api_wrapper=WikipediaAPIWrapper(top_k_results=3),
            description="Search Wikipedia for general knowledge"
        )
        tools.append(wikipedia)
    except Exception as e:
        print(f"⚠️ Wikipedia search tool failed: {e}")
    
    # 3. Document search tool
    def doc_search(query: str) -> str:
        """Search documents in the knowledge base about Chinese food and culture."""
        try:
            print(f"\nSearching documents related to '{query}'...")
            results = vectorstore.similarity_search(query, k=3)
            if not results:
                return "No relevant documents found in the knowledge base."
            
            formatted_results = []
            for i, doc in enumerate(results, 1):
                formatted_results.append(f"Document {i}:")
                formatted_results.append(doc.page_content)
                if doc.metadata:
                    formatted_results.append(f"Source: {doc.metadata}")
                formatted_results.append("-" * 50)
            
            result_str = "\n".join(formatted_results)
            print(f"Found {len(results)} relevant documents.")
            return result_str
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    doc_search_tool = Tool(
        name="doc_search",
        func=doc_search,
        description="Search documents in the knowledge base about Chinese food and culture"
    )
    tools.append(doc_search_tool)
    
    return tools

# --- 8. Create ReAct Agent with Prompt Template ---
def create_react_agent_with_prompt():
    """Create ReAct agent with custom prompt template."""
    
    # Custom ReAct prompt template
    react_prompt_template = """Answer the following questions as best you can. You are a Chinese Food & Culture Expert with access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Special Instructions:
    1. For questions about Chinese cuisine, cooking techniques, ingredients, or cultural aspects, use doc_search first
    2. For latest information, news, or real-time data, or the doc_search don't have the enough information to answer the user question, use web_search
    3. You can use multiple tools in sequence if needed
    4. Always provide comprehensive and accurate answers about Chinese food and culture
    5. Cite sources when using information from tools
    6. Current date: {current_date}

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}"""
    
    # Print the prompt template
    print("=" * 60)
    print("ReAct Prompt Template:")
    print("=" * 60)
    print(react_prompt_template)
    print("=" * 60)
    
    # Create prompt template
    react_prompt = PromptTemplate.from_template(react_prompt_template)
    
    # Create tools
    tools = create_tools()
    
    # Create ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return agent_executor

# --- 9. CLI Interaction Loop ---
if __name__ == "__main__":
    print("Agentic RAG System with ReAct Agent has initialized!")
    print("Domain: Chinese Food & Culture Expert")
    print("Document Source: chinesecookbook.pdf")
    print("Default mode is Openrouter LLM")
    print("\nType 'exit' or 'quit' to quit.")
    print("Type 'switch_gemini' to use Google Gemini LLM")
    print("Type 'switch_openrouter' to use OpenRouter LLM")
    print("\nBelow is the reAct prompt template for you to see & understand how agent work:\n")
    
    # Create agent executor
    agent_executor = create_react_agent_with_prompt()
    
    while True:
        user_input = input("\nYour Question: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Thank you! Goodbye.")
            break
        elif user_input.lower() == "switch_gemini":
            llm = llm_gemini
            print("Switched to Google Gemini LLM")
            # Recreate agent with new LLM
            agent_executor = create_react_agent_with_prompt()
            continue
        elif user_input.lower() == "switch_openrouter":
            llm = llm_openrouter
            print("Switched to OpenRouter LLM (z-ai/glm-4.5-air:free)")
            # Recreate agent with new LLM
            agent_executor = create_react_agent_with_prompt()
            continue
        
        try:
            print("\nProcessing your question with ReAct Agent...")
            # Call the agent with current date
            result = agent_executor.invoke({
                "input": user_input,
                "current_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            print("\nAnswer:")
            print(result["output"])

        except Exception as e:
            print(f"\nAn error occurred: {e}")
