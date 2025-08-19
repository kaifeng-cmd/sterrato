# LLM-as-a-Judge on RAG evaluation & Agentic RAG (ReAct)
In this project, I first built different Retrieval-Augmented Generation (RAG) pipelines. For the language models, I used Gemini from Google AI Studio, Kimi K2, and DeepSeek V3 through the OpenRouter platform. For vector storage, I adopted Qdrant Cloud, and the embedding models included Gemini Embedding and IBM Granite Embedding. This setup allowed me to construct multiple RAG systems with varying combinations of LLMs and embeddings as the basis for evaluation.

To evaluate these pipelines, I designed a LLM-as-a-Judge framework. My approach is similar to ChatArena-style pairwise comparison, where the judge model compares multiple outputs side by side. On top of this, I introduced a fixed scoring rubric covering relevance, consistency, context fit, and suitability. This allows the judge to simulate human-like preference judgments while producing quantitative scores, making it easier to compare multiple systems under consistent evaluation criteria.

In addition to evaluation, I implemented an Agentic RAG system. Here, the RAG pipeline is embedded into a ReAct agent (Reasoning + Acting). In this paradigm, the agent alternates between three stages:

1. Reasoning – the agent explains its intermediate thought process in natural language.

2. Action – the agent decides which tool to call, such as retrieving from the vector database or querying external APIs.

3. Observation – the agent incorporates the tool’s response into its context before continuing the reasoning process.

This reasoning–action–observation loop enables the system to iteratively refine its answers. By combining retrieval with external tool usage, the agent acts as a “Chinese Food & Culture Expert” that can provide not only contextually relevant but also reasoning-driven and tool-augmented responses.

## LLM-as-a-Judge Method
| Dimension                | My method (Pairwise + Rubric Scoring)                                                                                                                                                                                     | Ground Truth Comparison                                                                                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core Idea**            | Given a fixed set of test queries, different RAG pipelines generate answers. The Judge LLM performs pairwise comparisons using a fixed rubric (relevance, consistency, context fit, suitability) to score and rank outputs. | The Judge LLM/system first generates ground-truth QA pairs, then evaluates each pipeline’s answers by comparing them against the ground-truth for factual correctness. |
| **Evaluation Approach**  | Multi-dimensional rubric scoring + pairwise comparison. Balances subjective judgment (like human judges) with quantitative scoring.                                                                                         | Objective evaluation based on similarity/correctness against ground-truth answers, focusing mainly on factual accuracy.                                                |
| **Resource Consumption** | Medium: does not require the LLM to read the full knowledge base or generate ground-truth; main cost lies in the scoring process.                                                                                           | High: requires generating a large ground-truth QA set and then comparing each output against it; token and compute costs are significant.                              |
| **Main Limitation**      | Lacks absolute guarantee of factual correctness and still involves subjectivity.                                                                                                                                            | Dependent on the quality and coverage of the ground-truth set; generating QA pairs is costly and constrained by token limits.                                          |

## Features
- **RAG Pipeline Comparison:** Build, evaluate, and compare the performance of different LLM and embedding model pairs using LangChain framework, OpenRouter API, Google AI Studio API, Qdrant cloud, and a judge LLM for scoring.

- **Evaluation Framework:** Includes an LLM-based scoring system to judge and rank RAG answer quality and saves detailed results to `rag_evaluation_results.xlsx`.

- **Agentic RAG System:** A LLM agent (ReAct agent) that acts as a knowledgeable Chinese Food & Culture Expert.

- **Multiple LLM Support:** Seamlessly switch between Google Gemini and various OpenRouter models (e.g., Kimi, Deepseek) for the agent.

- **Diverse Tool Integration:** The agent can utilize the available tools like web search (via `ddgs`), Wikipedia (via `WikipediaQueryRun`), and RAG document search over the `chinesecookbook.pdf`.

## Demo
Here are some examples to showcase the agent's capabilities:

### Document Search Example
![](reAct_agent\demo_screenshot\use_doc_search.png)
*Explanation: The agent using the `doc_search` tool to find top 3 relevant chunks within the knowledge base.*

### Web Search Example
![](reAct_agent\demo_screenshot\use_web_search.png)
*Explanation: The agent utilizing the `web_search` tool to retrieve latest, real-time information or details not found in the knowledge base.*

### Wikipedia Search Example
![](reAct_agent\demo_screenshot\use_wikipedia_tool.png)
*Explanation: Illustrates the agent querying Wikipedia based on the user's request.*

### Evaluation Scores Example
![](reAct_agent\demo_screenshot\eval_Score1.png)

![](reAct_agent\demo_screenshot\eval_Score2.png)
*Explanation: These tables display the result of the LLM-as-a-judge evaluation, showcasing total scores for different LLM_Embedding pairs. The judging explanations are available in `rag_evaluation_results.xlsx`. These two tables have different scores due to different set of user queries have been asked.*

## Project Structure

```
sterrato/
│
├── reAct_agent/                  (Agentic RAG)
│   ├── agentic_rag.py
│   └── demo_screenshot/
│       (Screenshot materials)
├── .gitignore
├── chinesecookbook.pdf
├── compare_rag_outputs.py        (LLM-as-a-Judge)
├── rag_evaluation_results.xlsx
├── rag_gemini_gemini.py          (RAG for Gemini's LLM & Embedding)
├── rag_gemini_ibm.py             (RAG for Gemini LLM & IBM Embedding)
├── rag_openrouter_ibm.py         (RAG for Kimi LLM & IBM Embedding)
├── requirements.txt
└── test_db.py                    (Test for Qdrant connection)
```

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/kaifeng-cmd/sterrato.git
cd sterrato
```

### 2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
Install all required Python packages using the provided `requirements.txt`.
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory of the project and add your API keys and configurations.

#### `.env` File Configuration
```dotenv
# API Keys
GOOGLE_API_KEY= your_google_api_key
OPENROUTER_API_KEY= your_openrouter_api_key
QDRANT_API_KEY= your_qdrant_api_key # I used Qdrant Cloud
HF_TOKEN= your_huggingface_token

# Qdrant Configuration
QDRANT_URL= your_qdrant_url
QDRANT_COLLECTION_NAME= kai_Y # example for mine
```
*Note: The `collection name`, you can just change at `.env` to switch between different collections (because we've Gemini & IBM embeddings).*

## How to Run

### Running individual RAG Pipeline
You can run `rag_openrouter_ibm.py`, `rag_gemini_ibm.py`, `rag_gemini_gemini.py` individually as they're standalone as playground, the AI's role is Chinese Food & Culture Expert.

### Running the RAG Comparison Script
This script orchestrates the pairwise comparison of different RAG pairs and saves the evaluation results.
```bash
python compare_rag_outputs.py
```
This command will execute predefined queries, evaluate the generated answers using a judge LLM, and output the results to `rag_evaluation_results.xlsx`. You can change the test queries. It supports English or Chinese language.

### Running the Agentic RAG System
Interact directly with the Chinese Food & Culture Expert agent.
```bash
python reAct_agent/agentic_rag.py
```
Once the agent is running, you can:
- Ask questions related to Chinese recipe, cuisine, ingredients, and cooking culture.
- Type `switch_gemini` to use the Google Gemini LLM for the agent.
- Type `switch_openrouter` to use an OpenRouter LLM (e.g., Kimi, Deepseek).
- Type `exit` or `quit` to terminate the agent.
