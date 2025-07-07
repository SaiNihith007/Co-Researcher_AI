# 🧠 Co-Researcher.AI

**Your intelligent research assistant powered by multi-agent AI**

*Project by Sai Nihith Immaneni*

## Overview

Co-Researcher.AI is a sophisticated multi-agent research system that automatically finds, analyzes, and summarizes academic papers from arXiv and PubMed. It uses AI agents to deliver comprehensive research insights through an intuitive web interface.

## Features

### 🔍 **Intelligent Research Pipeline**
- **Smart Query Analysis**: Automatically extracts academic keywords and determines optimal data sources
- **Multi-Source Collection**: Searches arXiv and PubMed based on research domain
- **Advanced Filtering**: Uses embedding-based similarity and LLM reranking for quality papers
- **Full Paper Analysis**: Downloads and processes complete PDFs, not just abstracts

### 🤖 **Four Specialized AI Agents**
1. **Data Collection Agent**: Searches academic databases and processes papers
2. **Summarization Agent**: Creates detailed analysis of research papers
3. **Report Generator**: Produces comprehensive HTML and JSON reports
4. **RAG Chat Agent**: Answers follow-up questions about collected research

### 📊 **Comprehensive Outputs**
- **Interactive Reports**: Beautiful HTML reports with visualizations
- **Structured Data**: JSON exports for further analysis
- **RAG-Powered Chat**: Ask questions about your research results
- **Session Management**: Isolated research sessions with persistent storage

## Quick Start

### 1. Get Your OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key (starts with `sk-`)

### 2. Run the Application
- **Option A: Streamlit Cloud (Recommended)**
  - Visit our [live deployment](https://your-streamlit-app-url.streamlit.app)
  - Enter your API key in the sidebar
  - Start researching!

- **Option B: Local Development**
  ```bash
  git clone https://github.com/yourusername/co-researcher-ai.git
  cd co-researcher-ai
  pip install -r requirements.txt
  streamlit run app.py
  ```

### 3. Start Researching
1. Enter your research question
2. Configure parameters (max papers, years back)
3. Let the AI agents work their magic!
4. Review results and chat with your findings

## How It Works

### Research Pipeline
```
Query → Keyword Extraction → Multi-Source Search → Paper Collection → 
Embedding Filtering → LLM Reranking → Full Paper Analysis → 
Report Generation → RAG Chat Interface
```

### Paper Collection Strategy
- **arXiv**: For computer science, physics, mathematics papers
- **PubMed**: For biomedical and life science papers
- **Intelligent Funnel**: Collects 3-4x target papers, filters using embeddings, final LLM selection

### Quality Assurance
- **Embedding-based filtering**: Semantic similarity matching
- **LLM reranking**: Quality assessment and relevance scoring
- **Full paper processing**: Complete content analysis, not just abstracts
- **Validation agents**: Ensures research quality and accuracy

## Technical Architecture

### Core Components
- **Frontend**: Streamlit web interface
- **LLM Integration**: OpenAI GPT models
- **Vector Store**: FAISS for similarity search
- **PDF Processing**: Multi-method text extraction
- **Session Management**: Isolated research contexts

### Data Sources
- **arXiv**: Open access preprints
- **PubMed**: Biomedical literature database
- **PDF Downloads**: Direct paper access for full content

### AI Models
- **GPT-3.5/4**: Query analysis, summarization, reranking
- **Sentence Transformers**: Embedding generation for similarity
- **FAISS**: Vector similarity search

## Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_api_key_here  # Optional - can be set in UI
```

### Model Selection
- **GPT-3.5-turbo**: Fast, cost-effective (default)
- **GPT-4**: Higher quality, more expensive
- **GPT-4-turbo**: Latest model with enhanced capabilities

### Research Parameters
- **Max Papers**: 3-20 papers (default: 5)
- **Years Back**: 1-10 years (default: 3)
- **Sources**: Auto-selected based on query

## Deployment

### Streamlit Cloud Deployment
1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from your GitHub repository
4. Set secrets in Streamlit Cloud dashboard (optional)

### Local Development
```bash
# Clone repository
git clone https://github.com/yourusername/co-researcher-ai.git
cd co-researcher-ai

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py --server.port 8501
```

## File Structure

```
co-researcher-ai/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── router.py             # Query routing and analysis
├── requirements.txt      # Python dependencies
├── agents/              # AI agent implementations
│   ├── data_collection_agent.py
│   ├── summarization_agent.py
│   ├── report_generator.py
│   └── rag_chat_agent.py
├── modules/             # Core functionality modules
│   ├── llm_client.py
│   ├── vector_store.py
│   ├── pdf_processor.py
│   └── paper_reranker.py
└── outputs/             # Generated reports and data
    ├── reports/
    ├── papers/
    ├── cache/
    └── vector_db/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:
- Open a GitHub issue
- Email: your-email@example.com
- Documentation: See project wiki

## Acknowledgments

- OpenAI for GPT models
- Streamlit for the web framework
- arXiv and PubMed for research data access
- FAISS for vector similarity search

---

**Built with ❤️ by Sai Nihith Immaneni** 