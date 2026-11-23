# NewsLens AI – Fake News Detection & Investigation

NewsLens AI is an AI-powered fake news detection and investigation system that combines traditional Machine Learning, Natural Language Processing (NLP), and locally running Large Language Models (LLMs).

The project analyzes news headlines using machine learning classifiers and provides an AI-powered investigation workflow that searches the web, retrieves accessible articles, and examines supporting evidence to generate an investigation report.

Rather than relying entirely on a machine learning prediction, NewsLens AI combines statistical classification with evidence-based investigation.

## Key Features

### 1. Machine Learning Classification
- Fake and real news headline classification.
- TF-IDF vectorization for text feature extraction.
- Two classification algorithms:
  - Logistic Regression
  - Multinomial Naive Bayes
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Feature influence explanations showing which words contributed towards a prediction.

### 2. Exploratory Data Analysis (EDA)
- Dataset overview and class distribution.
- Fake vs. real news distribution.
- Headline length analysis.
- Most frequently occurring words.
- Analysis of news sources and categories (when available).

### 3. News Topic Clustering
- K-Means clustering to group news headlines into similar topics.
- TF-IDF features for clustering.
- Identification of representative keywords for each cluster.

### 4. AI-Powered News Investigation
- Local LLM integration using Ollama.
- Web search using DuckDuckGo.
- Article retrieval and text extraction using Requests and BeautifulSoup.
- Semantic relevance ranking using local text embeddings.
- Evidence-based investigation reports.

The investigation agent evaluates claims using four possible verdicts:

| Verdict | Description |
|---|---|
| SUPPORTED | Evidence substantially supports the claim. |
| CONTRADICTED | Credible evidence directly contradicts the claim. |
| PARTIALLY SUPPORTED | Some elements are accurate, but important details are incorrect or missing. |
| INCONCLUSIVE | Available evidence is insufficient to reach a conclusion. |

### 5. Interactive AI Chat
- Ask follow-up questions about an investigation.
- Explore additional evidence through web search.
- Retrieve accessible articles for further examination.
- Maintain conversation context.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| NLP | TF-IDF |
| Clustering | K-Means |
| LLM Integration | LangChain |
| Local LLM Runtime | Ollama |
| LLM Model | Qwen 3.5 2B |
| Embeddings | Nomic Embed Text |
| Web Search | DuckDuckGo Search |
| Article Extraction | Requests, BeautifulSoup |
| Frontend | Streamlit |

## Project Structure

```text
NewsLens-AI/
│
├── app.py
├── backend.py
├── requirements.txt
│
└── dataset/
    └── fake_news_dataset.csv
```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/NewsLens-AI.git
cd NewsLens-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Download and install Ollama:

https://ollama.com/

Pull the required models:

```bash
ollama pull qwen3.5:2b
ollama pull nomic-embed-text
```

Make sure the Ollama service is running before starting the application.

### 5. Prepare the Dataset

Create a `dataset` folder in the project directory and place the dataset inside it:

```text
dataset/fake_news_dataset.csv
```

The dataset must contain the following columns:

- `title`: News headline.
- `label`: News category (`fake` or `real`).

Optional columns:
- `source`
- `category`

Dataset source: [Fake News Detection Dataset – Kaggle](https://www.kaggle.com/datasets/mahdimashayekhi/fake-news-detection-dataset)

### 6. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access NewsLens AI.

## How It Works

```text
             News Headline
                   |
                   v
          Text Preprocessing
                   |
                   v
             TF-IDF
                   |
                   v
          Machine Learning
          /             \
         v               v
   Naive Bayes     Logistic Regression
          \             /
           v           v
          Prediction & Explanation
                   |
                   v
          AI News Investigation
                   |
          +--------+---------+
          |                  |
          v                  v
      Web Search       Article Retrieval
          |                  |
          +--------+---------+
                   |
                   v
          Semantic Relevance Ranking
                   |
                   v
            Local LLM (Qwen)
                   |
                   v
          Evidence-Based Report
                   |
                   v
          Interactive AI Chat
```

## Important Limitations

- Machine learning predictions are based on patterns learned from the training dataset. They do not establish whether a news claim is factually true.
- The quality of classification depends on the dataset and its representativeness.
- Web search results are not automatically reliable evidence.
- Article extraction depends on website accessibility and may fail for certain websites.
- Semantic similarity measures relevance, not factual accuracy.
- Investigation results should be independently verified, especially for important or sensitive claims.

## Future Improvements

- Support full news article classification instead of headline-only classification.
- Improve source credibility assessment.
- Add more robust evidence verification and source comparison.
- Improve the AI agent's investigation workflow.
- Add model explainability and performance comparisons.
- Explore additional NLP and transformer-based classification models.

## Project Objective

The goal of NewsLens AI is to explore how traditional machine learning and modern AI engineering techniques can work together to build a more informative news analysis system.

**Note:** NewsLens AI is an educational project and should not be treated as a definitive fact-checking authority.
