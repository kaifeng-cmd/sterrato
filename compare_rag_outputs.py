import os
import time
import re
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from qdrant_client import QdrantClient
import pdfplumber
import json # Import json for parsing LLM response
import pandas as pd

# --- Helper Classes ---

# Gemini Embeddings Wrapper
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

# Custom IBM Embeddings
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

# PDF Document Loader
def load_pdf_document(pdf_path: str) -> list[Document]:
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file '{pdf_path}' does not exist")
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File '{pdf_path}' is not a PDF")
        documents = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
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

# Adapted Vector Store Initialization Function
def get_or_create_vectorstore_for_config(embedding_function, collection_name, doc_path="chinesecookbook.pdf"):
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    try:
        # Check if collection exists
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists. Connecting to existing collection.")
        
        # Check if collection is empty, if so, load and vectorize
        if collection_info.points_count == 0:
            print(f"Collection '{collection_name}' is empty. Loading and vectorizing PDF document...")
            documents = load_pdf_document(doc_path)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            print(f"PDF document split into {len(docs)} text chunks.")
            
            vectorstore_instance = QdrantVectorStore.from_existing_collection(
                embedding=embedding_function,
                collection_name=collection_name,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            
            batch_size = 10
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                print(f"Adding batch {i//batch_size + 1} to '{collection_name}', total {len(batch)} documents...")
                try:
                    vectorstore_instance.add_documents(batch)
                    print(f"Batch {i//batch_size + 1} added successfully to '{collection_name}'")
                except Exception as e:
                    print(f"Batch {i//batch_size + 1} failed for '{collection_name}': {e}")
                    # Try adding individually
                    for doc in batch:
                        try:
                            vectorstore_instance.add_documents([doc])
                            print(f"Single document added successfully to '{collection_name}'")
                        except Exception as single_e:
                            print(f"Single document failed for '{collection_name}': {single_e}")
                time.sleep(5)
            print(f"All documents added to collection '{collection_name}'.")
            return vectorstore_instance
        else:
            # Collection exists and is not empty, just connect
            vectorstore_instance = QdrantVectorStore.from_existing_collection(
                embedding=embedding_function,
                collection_name=collection_name,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            return vectorstore_instance
            
    except Exception as e: # Collection does not exist
        print(f"Collection '{collection_name}' does not exist or cannot be accessed ({e}). Attempting to create a new collection.")
        
        documents = load_pdf_document(doc_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        print(f"Vectorizing {len(docs)} document chunks and creating new collection '{collection_name}'...")
        
        # Create empty collection first
        vectorstore = QdrantVectorStore.from_documents(
            [],
            embedding=embedding_function,
            collection_name=collection_name,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        
        batch_size = 10
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            print(f"Adding batch {i//batch_size + 1} to '{collection_name}', total {len(batch)} documents...")
            try:
                vectorstore.add_documents(batch)
                print(f"Batch {i//batch_size + 1} added successfully to '{collection_name}'")
            except Exception as e:
                print(f"Batch {i//batch_size + 1} failed for '{collection_name}': {e}")
                for doc in batch:
                    try:
                        vectorstore.add_documents([doc])
                        print(f"Single document added successfully to '{collection_name}'")
                    except Exception as single_e:
                        print(f"Single document failed for '{collection_name}': {single_e}")
            time.sleep(5)
        
        print(f"New vector database collection '{collection_name}' created and populated!")
        return vectorstore

# --- RAG Execution Function ---
def run_rag_config(config, query):
    config_name = config["name"]
    print(f"\n--- Running configuration: {config_name} ---")
    
    try:
        # Initialize Vector Store for this configuration
        vectorstore = get_or_create_vectorstore_for_config(
            config["embedding_function"],
            config["qdrant_collection_name"]
        )
    except Exception as e:
        print(f"Error initializing vector store for {config_name}: {e}")
        return f"Error: Could not initialize vector store for {config_name}"

    # Initialize Memory for each run
    memory = ConversationBufferWindowMemory(k=30, return_messages=True)

    # Retrieval function (can be a shared helper function)
    def retrieve_chunks(query_text: str, vs):
        print(f"Retrieving documents related to '{query_text}'...")
        try:
            results = vs.similarity_search(query_text, k=3)
            if not results:
                print("No relevant documents found.")
                return ""
            # Format the retrieved chunks
            formatted_results = [
                f"Document chunk {i}:\nContent: {doc.page_content}"
                for i, doc in enumerate(results, 1)
            ]
            result_str = "\n" + "\n---\n".join(formatted_results)
            print(f"Found {len(results)} relevant document chunks.")
            return result_str
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return ""

    # Prompt Template (can be a shared template)
    """
    For comparison between LLM & Embedding Model combinations on RAG task, I used this prompt 
    which is different from others becuz I wanna test more on the embedding retrieval capability
    and whether LLM able to catch the context to deliver suitable answer.
    (the prompt only special at here.)
    """
    # For others like rag_gemini_ibm.py, I designed for chatbot use, approach towards real-world scenarios.
    # So, if retrieval chunks are not relevant, LLM will still can answer user query based on his role,
    # which is related to cooking and cuisine.
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional {role}. 
                        Please answer the user's question based on the background information below. 
                        If there is no relevant information in the background, please answer 'Sry, I don't know'. 
                        Background Information:\n\n{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}")
    ])

    # Build RAG Chain
    rag_chain = (
        {
            "question": RunnablePassthrough(),
            "role": lambda x: "Chinese Food & Culture Expert",
            "context": lambda x: retrieve_chunks(x["question"], vectorstore),
            "chat_history": lambda x: memory.load_memory_variables({})["history"]
        }
        | chat_prompt
        | config["llm"]
        | StrOutputParser()
    )

    # Execute the chain for the given query with error handling
    try:
        print(f"Querying: '{query}'")
        answer = rag_chain.invoke({"question": query})
        print(f"Answer: {answer}")
        return answer
    except Exception as e:
        print(f"An error occurred during RAG execution for {config_name}: {e}")
        return f"Error during RAG execution for {config_name}: {e}"

# --- Evaluation Function ---
def evaluate_answers(query: str, answers: dict, judge_llm):
    """
    Evaluates the answers from different RAG configurations using a judge LLM.
    Returns a list of dictionaries, each containing config_name, score, and explanation.
    """
    print(f"\n--- Evaluating answers for query: '{query}' ---")
    
    # Dynamically get the config order from the answers dictionary
    config_order = list(answers.keys())
    if len(config_order) < 3:
        print("Warning: Less than 3 answers provided for evaluation.")
        # Pad with dummy configs if needed, though the logic should handle this
        while len(config_order) < 3:
            config_order.append(f"dummy_config_{len(config_order)}")

    # Prepare the answers in a format suitable for the prompt, ensuring order
    # Pad with empty strings if any config failed to produce an answer
    padded_answers = {name: answers.get(name, "No answer produced.") for name in config_order}
    
    config_name_1, answer_1 = config_order[0], padded_answers.get(config_order[0], "N/A")
    config_name_2, answer_2 = config_order[1], padded_answers.get(config_order[1], "N/A")
    config_name_3, answer_3 = config_order[2], padded_answers.get(config_order[2], "N/A")

    # Prompt for the judge LLM
    judge_prompt_template = """
        You are an expert evaluator tasked with scoring the quality of answers 
        provided by different RAG systems to a given question.
        Your goal is to assess the accuracy, relevance, context suitability between question and answer, 
        consistency, and completeness of each answer based on the provided question.
        However, you need to know that not always lengthly answer is better, it must fit to the question context.

        Here are the scoring rules:
        1.  **Clear Ranking:** If there is a clear best answer, a clear second best, and a clear third answer:
            *   Best Answer: 1 point
            *   Second Best Answer: 0.5 points
            *   Third Answer: 0 points
        2.  **All Equally Good:** If all three answers are of very similar high quality and are indistinguishable in terms of relevance and accuracy:
            *   Each Answer: 0.33 points
        3.  **Two Tied for Best:** If two answers are of similar high quality and are clearly better than the third answer:
            *   The two best answers: Each gets 0.5 points
            *   The third answer: 0 points

        For each answer, please provide a brief explanation justifying your score.

        Present your evaluation in JSON format, with each entry containing the configuration name, the score, and the explanation.

        Example JSON output format:
        [
        {{"config_name": "Config A", "score": 1.0, "explanation": "Answer A is the most accurate, context fit, and comprehensive."}},
        {{"config_name": "Config B", "score": 0.5, "explanation": "Answer B is good but slightly less detailed and context fit than A."}},
        {{"config_name": "Config C", "score": 0.0, "explanation": "Answer C is irrelevant or inaccurate."}}
        ]

        ---
        Question: {question}

        RAG Configuration Answers:
        ---
        Config Name: {config_name_1}
        Answer: {answer_1}
        ---
        Config Name: {config_name_2}
        Answer: {answer_2}
        ---
        Config Name: {config_name_3}
        Answer: {answer_3}
        ---

        Please provide your evaluation in the specified JSON format:
        """
        
    prompt = judge_prompt_template.format(
        question=query,
        config_name_1=config_name_1,
        answer_1=answer_1,
        config_name_2=config_name_2,
        answer_2=answer_2,
        config_name_3=config_name_3,
        answer_3=answer_3
    )

    try:
        print("Calling judge LLM...")
        evaluation_response = judge_llm.invoke(prompt)
        evaluation_response_content = evaluation_response.content
        print(f"Judge LLM raw response content: {evaluation_response_content}")

        # Use regex to find the JSON block, making it robust against extra text
        json_match = re.search(r'\[.*\]', evaluation_response_content, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON array found in the judge LLM response.")
        
        json_str = json_match.group(0)
        print(f"Extracted JSON: {json_str}")
        
        # Attempt to parse the extracted JSON string
        evaluation_results = json.loads(json_str)
        
        # Validate the structure of the parsed results
        if not isinstance(evaluation_results, list):
            raise ValueError("Judge LLM response is not a list.")
        
        validated_results = []
        for item in evaluation_results:
            if not isinstance(item, dict) or "config_name" not in item or "score" not in item or "explanation" not in item:
                raise ValueError("Invalid item structure in judge LLM response.")
            validated_results.append(item)
            
        print("Evaluation successful.")
        return validated_results
        
    except Exception as e:
        print(f"Error during evaluation or parsing judge LLM response: {e}")
        # Return a placeholder indicating failure
        return [{"config_name": name, "score": "Error", "explanation": f"Evaluation failed: {e}"} for name in config_order]

def save_results_to_excel(results: dict, evaluations: dict, config_order: list, filename="rag_evaluation_results.xlsx"):
    """Formats and saves the final results and evaluations to an Excel file."""
    print(f"\n--- Saving results to {filename} ---")
    
    # Prepare data for DataFrame
    data_for_df = []
    
    # Populate data
    for query in evaluations.keys():
        evals_for_query = evaluations.get(query, [])
        eval_map = {item["config_name"]: (item["score"], item["explanation"]) for item in evals_for_query}
        
        for config_name in config_order:
            rag_answer = results.get(config_name, {}).get(query, "No answer produced.")
            score, explanation = eval_map.get(config_name, ("N/A", "N/A"))
            
            data_for_df.append({
                "Query": query,
                "Config": config_name,
                "Answer": rag_answer,
                "Score": score,
                "Explanation": explanation
            })

    if not data_for_df:
        print("No data to save.")
        return

    # Create DataFrame and save to Excel
    try:
        df = pd.DataFrame(data_for_df)
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"Successfully saved results to {filename}")
    except Exception as e:
        print(f"Error saving to Excel file: {e}")

def print_summary_table(evaluations: dict, config_order: list):
    """Calculates and prints a summary table of total scores for each configuration."""
    print("\n--- Final Score Summary ---")
    
    scores = {name: 0 for name in config_order}
    
    # Calculate total scores
    for query_evals in evaluations.values():
        for item in query_evals:
            config_name = item.get("config_name")
            score = item.get("score")
            if config_name in scores and isinstance(score, (int, float)):
                scores[config_name] += score

    # Prepare for printing
    print(f"{'Configuration':<35} | {'Total Score':<15}")
    print("-" * 55)
    for name, total_score in scores.items():
        print(f"{name:<35} | {total_score:<15.2f}")
    print("-" * 55)

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # API Keys (ensure these are set in .env)
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
    QDRANT_URL = os.environ.get("QDRANT_URL")
    HF_TOKEN = os.environ.get("HF_TOKEN") 

    # Validate essential API keys
    if not all([GOOGLE_API_KEY, OPENROUTER_API_KEY, QDRANT_API_KEY, QDRANT_URL, HF_TOKEN]):
        print("Error: Please ensure all API keys are set in your .env file")
        exit(1)

    # --- Centralized Model and Embedding Initialization ---
    llms = {
        "gemini": ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=1.0),
        "openrouter": ChatOpenAI(model="moonshotai/kimi-k2:free", openai_api_key=OPENROUTER_API_KEY, openai_api_base="https://openrouter.ai/api/v1", temperature=1.0),
        "judge_llm": ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", google_api_key=GOOGLE_API_KEY, temperature=0.2)
    }
    
    embeddings = {
        "gemini": GeminiEmbeddingsWrapper(google_api_key=GOOGLE_API_KEY),
        "huggingface": CustomIBMEmbeddings(model_name="ibm-granite/granite-embedding-278m-multilingual")
    }

    JUDGE_LLM = llms["judge_llm"]

    # --- Define RAG Configurations ---
    configurations = [
        {
            "name": "Moonshot/Kimi_HuggingFaceEmbeddings",
            "llm": llms["openrouter"],
            "embedding_function": embeddings["huggingface"],
            "qdrant_collection_name": "kai_Y"
        },
        {
            "name": "Gemini_HuggingFaceEmbeddings",
            "llm": llms["gemini"],
            "embedding_function": embeddings["huggingface"],
            "qdrant_collection_name": "kai_Y"
        },
        {
            "name": "Gemini_GeminiEmbeddings",
            "llm": llms["gemini"],
            "embedding_function": embeddings["gemini"],
            "qdrant_collection_name": "kai_U"
        }
    ]
    CONFIG_ORDER = [config["name"] for config in configurations]

    # --- Define Test Queries ---
    # Using user-provided queries
    test_queries = [
        "Can you explain what is yung chow min? the noodle..and the process of making it? from which page?",
        "处理食材通常有什么规矩还有禁忌？",
        "中式烹饪是怎么处理鱼类的？有什么讲究。",
        "What are the main ingredients for cooking chinese-style Chicken?",
        "牛肉适合拿来做什么料理/菜谱？通常都搭配什么食材？",
        "How to make steamed buns?",
        "What is the cooking time for roasted duck? what ingredients are needed?"
    ]

    results = {} # Stores RAG answers for each query and config

    # --- Orchestrate RAG Execution ---
    print("Starting RAG configuration comparison...")
    for config in configurations:
        config_name = config["name"]
        results[config_name] = {}
        print(f"\n===== Testing Configuration: {config_name} =====")
        
        for query in test_queries: # Use user-provided queries
            answer = run_rag_config(config, query)
            results[config_name][query] = answer
            # Print truncated answer for brevity during execution
            print(f"Query: '{query}' -> Answer: '{answer[:100]}...'" if len(answer) > 100 else f"Query: '{query}' -> Answer: '{answer}'")
        print(f"===== Finished testing {config_name} =====")

    # --- Evaluate Answers and Populate Table Data ---
    print("\n--- Starting LLM-based Evaluation ---")
    evaluation_results_all_queries = {}

    for query in test_queries:
        # Gather answers for the current query, maintaining the defined order
        query_answers = {
            config_name: results.get(config_name, {}).get(query, "No answer produced.")
            for config_name in CONFIG_ORDER
        }
        
        evaluation = evaluate_answers(query, query_answers, JUDGE_LLM)
        evaluation_results_all_queries[query] = evaluation
        
    # --- Save Detailed Results to Excel ---
    save_results_to_excel(results, evaluation_results_all_queries, CONFIG_ORDER)

    # --- Print Summary Table to Console ---
    print_summary_table(evaluation_results_all_queries, CONFIG_ORDER)

    print("\n--- Evaluation Complete ---")
