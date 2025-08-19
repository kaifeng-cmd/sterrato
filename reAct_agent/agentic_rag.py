import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from qdrant_client import QdrantClient
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from datetime import datetime

# --- 1. Load environment variables ---
load_dotenv()

# --- 2. API keys and configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")

if not all([GOOGLE_API_KEY, QDRANT_API_KEY, QDRANT_URL]):
    raise ValueError("Please ensure GOOGLE_API_KEY, QDRANT_API_KEY, and QDRANT_URL are set in your .env file")

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

# --- 4. Initialize LLM (using Google Gemini API) ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=2.0,
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
        人工智能的研究历史有着一条从以"推理"为重点到以"知识"为重点，再到以"学习"为重点的自然、清晰的脉络。
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
    
    # 2. Wikipedia search tool (backup)
    try:
        wikipedia = WikipediaQueryRun(
            name="wikipedia",
            api_wrapper=WikipediaAPIWrapper(top_k_results=3),
            description="Search Wikipedia for general knowledge"
        )
        tools.append(wikipedia)
        print("✅ Wikipedia search tool initialized successfully")
    except Exception as e:
        print(f"⚠️ Wikipedia search tool failed: {e}")
    
    # 3. Document search tool
    def doc_search(query: str) -> str:
        """Search documents in the knowledge base."""
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
        description="Search documents in the knowledge base about AI, technology, and computer science"
    )
    tools.append(doc_search_tool)
    
    return tools

# --- 8. Create ReAct Agent with Prompt Template ---
def create_react_agent_with_prompt():
    """Create ReAct agent with custom prompt template."""
    
    # Custom ReAct prompt template
    react_prompt_template = """Answer the following questions as best you can. You have access to the following tools:

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
    1. For technical questions about AI, machine learning, or computer science, use doc_search first
    2. For latest information, news, or real-time data, use web_search
    3. You can use multiple tools in sequence if needed
    4. Always provide comprehensive and accurate answers
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
    print("Type 'exit' or 'quit' to quit.")
    
    # Create agent executor
    agent_executor = create_react_agent_with_prompt()
    
    while True:
        user_input = input("\nYour Question: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Thank you! Goodbye.")
            break
        
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
