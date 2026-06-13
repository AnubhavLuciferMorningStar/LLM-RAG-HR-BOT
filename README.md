# 🤖 HR Q&A Assistant — RAG-powered HR Policy Chatbot

An AI assistant that answers employee HR questions by searching and reasoning
over company policy documents. Built with Python, Streamlit (or Flask),
LangChain, Ollama, and ChromaDB.

---

## Architecture

```
Employee Question
       │
       ▼
┌─────────────────────┐
│   Streamlit / Flask │  ← Web UI (chat interface)
│      Frontend       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    RAG Pipeline     │
│  1. Query → Embed   │
│  2. Vector Search   │  ← ChromaDB / TF-IDF
│  3. Top-K Chunks    │
│  4. Build Prompt    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Ollama (Local)    │  ← llama3.2 / mistral / phi3
│      LLM Call       │
└────────┬────────────┘
         │
         ▼
  Grounded Answer + Sources
```

---

## Quick Start

### 1. Install Ollama and pull a model
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull a model (in a new terminal)
ollama pull llama3.2
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3A. Run with Streamlit (primary)
```bash
streamlit run streamlit_app.py
# Open: http://localhost:8501
```

### 3B. Run with Flask (fallback)
```bash
python app.py
# Open: http://localhost:5000
```

---

## File Structure

```
HR_BOT/
├── streamlit_app.py      # Main Streamlit app (primary per spec)
├── rag_engine.py         # Core RAG logic (TF-IDF + Ollama REST)
├── requirements.txt      # Python dependencies
├── data/
│   └── hr_policies.json  # Mock HR policy knowledge base (18 policies)
```

---

## How It Works

1. **Knowledge Base**: 18 HR policies across 7 categories (Leave, Compensation,
   Onboarding, WFH, Separation, Benefits, Compliance) stored in JSON.

2. **Embedding & Indexing**: Policies are converted to vectors using
   SentenceTransformer (`all-MiniLM-L6-v2`) and stored in ChromaDB.
   The fallback uses TF-IDF + cosine similarity (no external deps).

3. **Retrieval**: When an employee asks a question, the top-3 most relevant
   policy chunks are retrieved by semantic similarity.

4. **Generation**: The question + retrieved context are sent to Ollama (local
   LLM). The model is instructed to answer only from the provided context,
   avoiding hallucination.

5. **Multi-turn**: Chat history (last 3 turns) is included for follow-up questions.

---

## Assumptions

- Ollama is running locally on port 11434
- `llama3.2` is the default model (changeable in the UI sidebar)
- All policy data is mock/sample data — representative of a mid-size Indian IT company
- No authentication layer (prototype scope)
- Policies are pre-loaded at startup (no real-time document ingestion)

---

## Extending to Production

| Feature | Approach |
|---------|----------|
| Real policy PDFs | LangChain `PyPDFLoader` + chunking |
| Authentication | Azure AD / OAuth via Streamlit-Authenticator |
| Persistent vector store | ChromaDB persistent client or Pinecone |
| Conversation history | Redis or PostgreSQL |
| Feedback loop | Thumbs up/down → fine-tune retrieval |
| Deployment | Docker + cloud VM or Azure App Service |
