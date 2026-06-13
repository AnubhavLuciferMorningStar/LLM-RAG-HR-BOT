"""
HR Q&A Assistant — Streamlit Version
======================================
This is the canonical Streamlit prototype as described in the solution design.

Run:
    pip install streamlit langchain langchain-ollama langchain-community chromadb sentence-transformers
    streamlit run streamlit_app.py

NOTE: If Streamlit is unavailable in the environment, use app.py (Flask) instead —
identical RAG logic, different UI framework.
"""

# ── Standard imports (work without Streamlit for syntax checking) ────────────
import json, os, sys

# ── Try Streamlit ─────────────────────────────────────────────────────────────
try:
    import streamlit as st
    HAS_ST = True
except ImportError:
    HAS_ST = False

# ── Try LangChain stack ────────────────────────────────────────────────────────
try:
    from langchain_ollama import OllamaLLM
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
    HAS_LC = True
except ImportError:
    HAS_LC = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A: LangChain-based RAG (uses full stack when available)
# ═══════════════════════════════════════════════════════════════════════════════

def build_langchain_rag(policies_path: str, model_name: str = "llama3.2"):
    """
    Build a LangChain RetrievalQA chain backed by:
      - SentenceTransformer embeddings  →  ChromaDB vector store
      - OllamaLLM                       →  local inference
    """
    if not HAS_LC:
        raise ImportError("LangChain stack not installed.")

    with open(policies_path) as f:
        data = json.load(f)

    documents = []
    for pol in data["policies"]:
        text = f"{pol['title']}. Category: {pol['category']}. {pol['content']}"
        documents.append(Document(
            page_content=text,
            metadata={"id": pol["id"], "title": pol["title"], "category": pol["category"]}
        ))

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb   = Chroma.from_documents(documents, embedding=embeddings)
    retriever  = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = OllamaLLM(model=model_name, temperature=0.2)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )
    return chain


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B: Fallback RAG (pure Python — used when LangChain unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

# Import our custom engine from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from rag_engine import load_policies, answer_question, OllamaError, list_ollama_models, DEFAULT_MODEL


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C: Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_ST:

    # ── Page config ─────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="HR Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ───────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .main { background: #0f1117; }
    .stChatMessage { background: #1a1d27; border-radius: 12px; }
    .source-pill {
        display: inline-block; margin: 3px 4px;
        padding: 3px 10px; border-radius: 12px;
        background: rgba(91,127,255,.15); color: #7c9fff;
        font-size: 12px; border: 1px solid rgba(91,127,255,.3);
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Session state ────────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "store" not in st.session_state:
        st.session_state.store = None
    if "lc_chain" not in st.session_state:
        st.session_state.lc_chain = None

    # ── Sidebar ──────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🤖 HR Assistant")
        st.caption("Powered by Ollama + RAG")
        st.divider()

        # Model picker
        available_models = list_ollama_models()
        model = st.selectbox(
            "LLM Model",
            options=available_models if available_models else [DEFAULT_MODEL],
            index=0,
        )

        # Ollama status
        if available_models:
            st.success(f"✅ Ollama ready · {len(available_models)} model(s)")
        else:
            st.error("❌ Ollama not running\n```\nollama serve\nollama pull llama3.2\n```")

        st.divider()

        # Load knowledge base
        DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "hr_policies.json")

        if st.session_state.store is None:
            with st.spinner("Loading HR policies…"):
                # Prefer LangChain stack, fall back to custom engine
                if HAS_LC:
                    try:
                        st.session_state.lc_chain = build_langchain_rag(DATA_PATH, model)
                        st.caption("Using: LangChain + ChromaDB")
                    except Exception:
                        st.session_state.store = load_policies(DATA_PATH)
                        st.caption("Using: Custom TF-IDF engine")
                else:
                    st.session_state.store = load_policies(DATA_PATH)
                    st.caption("Using: Custom TF-IDF engine")

        # Policy index
        st.markdown("### 📚 Policy Index")
        try:
            with open(DATA_PATH) as f:
                pdata = json.load(f)
            cats: dict[str, list] = {}
            for p in pdata["policies"]:
                cats.setdefault(p["category"], []).append(p["title"])
            for cat, titles in cats.items():
                with st.expander(cat, expanded=False):
                    for t in titles:
                        if st.button(t, key=f"pol_{t}", use_container_width=True):
                            st.session_state["prefill"] = f"Tell me about the {t} policy"
        except Exception:
            st.warning("Could not load policy index.")

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Main UI ──────────────────────────────────────────────────────────────────
    st.title("HR Policy Q&A Assistant")
    st.caption("Ask me anything about company policies — leave, compensation, onboarding, benefits, and more.")

    # Suggestion chips on first load
    if not st.session_state.messages:
        st.markdown("**Quick questions:**")
        cols = st.columns(3)
        suggestions = [
            "How many sick days do I get?",
            "What is the WFH policy?",
            "When is salary credited?",
            "What documents are needed for onboarding?",
            "What is the notice period?",
            "How does health insurance work?",
        ]
        for i, s in enumerate(suggestions):
            with cols[i % 3]:
                if st.button(s, key=f"sug_{i}", use_container_width=True):
                    st.session_state["prefill"] = s

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                src_html = " ".join(f'<span class="source-pill">📄 {s["title"]}</span>'
                                    for s in msg["sources"])
                st.markdown(src_html, unsafe_allow_html=True)

    # Prefill handling
    prefill = st.session_state.pop("prefill", "")

    # Chat input
    prompt = st.chat_input("Ask an HR question…") or prefill

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching policies and generating answer…"):
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                try:
                    # Choose engine
                    if st.session_state.lc_chain:
                        result = st.session_state.lc_chain({"query": prompt})
                        answer = result["result"]
                        sources = [
                            {"title": d.metadata.get("title", ""), "category": d.metadata.get("category", "")}
                            for d in result.get("source_documents", [])
                        ]
                    else:
                        result = answer_question(
                            question=prompt,
                            store=st.session_state.store,
                            chat_history=history,
                            model=model,
                        )
                        answer  = result["answer"]
                        sources = result["sources"]

                    st.markdown(answer)
                    if sources:
                        src_html = " ".join(f'<span class="source-pill">📄 {s["title"]}</span>'
                                            for s in sources)
                        st.markdown(src_html, unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except OllamaError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error: {e}")

else:
    # ── No Streamlit: helpful message ────────────────────────────────────────────
    print("=" * 60)
    print("Streamlit is not installed in this environment.")
    print("Use the Flask app instead:")
    print("  cd hr_assistant && python app.py")
    print("Then open: http://localhost:5000")
    print("=" * 60)
    print()
    print("To install the full stack:")
    print("  pip install streamlit langchain langchain-ollama")
    print("  pip install langchain-community chromadb sentence-transformers")
    print()
    print("Self-test: verifying RAG engine with sample question…")
    try:
        store = load_policies(os.path.join(os.path.dirname(__file__), "data", "hr_policies.json"))
        hits = store.query("how many days of sick leave", top_k=2)
        print(f"  ✅ Retrieved {len(hits)} relevant policy chunks")
        for h in hits:
            print(f"     → [{h['score']:.3f}] {h['title']}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
