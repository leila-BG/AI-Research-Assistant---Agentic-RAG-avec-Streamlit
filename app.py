"""
app.py — Client graphique Streamlit de l'AI Research Assistant (TP 4.a)
Étape de SERVICE : relit la base Chroma créée par indexer.py
(outils, system prompt, agent ReAct + interface).

1) uv run python indexer.py            (une fois, ou après modification
                                         du corpus)
2) uv run streamlit run app.py
"""
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# set_page_config doit être le PREMIER appel Streamlit
st.set_page_config(page_title="AI Research Assistant", page_icon="🔎",
                   layout="wide")

# ═══════════════════════════════════════════════════════════════════
# PARTIE 1 — L'AGENT (aucun appel Streamlit ici)
# ═══════════════════════════════════════════════════════════════════

# ─── Clés API ─────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_OK = GROQ_API_KEY.startswith("gsk_") and len(GROQ_API_KEY) > 30
TAVILY_OK = TAVILY_API_KEY.startswith("tvly-") and len(TAVILY_API_KEY) > 20

# ─── Base vectorielle produite par indexer.py (paramètres identiques) ─
BASE_DIR = Path(__file__).parent
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION = "tp4_corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """Tu es un AI Research Assistant spécialisé en intelligence artificielle.
Tu disposes de 3 outils :
- `rag_search` : corpus IA local (Transformer, RAG, HNSW, LoRA, vector DBs).
- `tavily_web_search` : recherche web généraliste (infos récentes 2024-2026).
- `arxiv_search` : papiers scientifiques arXiv (publications académiques précises).

RÈGLES :
- Utilise `rag_search` en priorité pour les concepts fondamentaux IA déjà bien établis
  (Transformer, RAG, HNSW, LoRA, embeddings...).
- Utilise `tavily_web_search` pour les informations récentes (2024-2026) ou hors corpus local.
- Utilise `arxiv_search` quand la question concerne un papier précis ou des publications
  académiques.
- Refuse poliment les questions hors-domaine (cuisine, météo, etc.) en français,
  SANS appeler aucun outil.
- Tu as accès à l'historique de la conversation (messages précédents de
  l'utilisateur et tes réponses). Pour toute question sur la conversation
  elle-même (question précédente, prénom donné, ce qui a été dit...),
  réponds directement à partir de cet historique, SANS appeler d'outil
  et sans prétendre ne pas y avoir accès.
- Cite tes sources dans ta réponse finale.
- Réponds en français.
"""

# Questions du golden set, proposées comme exemples dans l'interface
EXEMPLES = [
    "Comment fonctionne l'algorithme HNSW pour la recherche approximative de plus proches voisins ?",
    "Quels nouveaux modèles d'embedding open-source sont sortis en 2025 ?",
    "Que dit le papier original LoRA (2021) sur l'efficacité paramétrique du fine-tuning ?",
    "Compare HNSW à des alternatives utilisées en production en 2024-2025.",
    "Quelle est la recette préférée de l'auteur du papier 'Attention is all you need' ?",
]


# ─── 1. Retriever : on RELIT la base persistée (aucune indexation) ─
def load_retriever(k: int = 4):
    # Contournement Windows : si une stratégie de contrôle d'application
    # bloque la DLL de grpc, on remplace l'exporteur de télémétrie de
    # chromadb (inutile en local).
    try:
        import grpc  # noqa: F401
    except ImportError:
        import sys
        import types
        faux = types.ModuleType(
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
        faux.OTLPSpanExporter = None
        sys.modules[faux.__name__] = faux

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    if not PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"Base introuvable ({PERSIST_DIR}). "
            "Lancez d'abord : uv run python indexer.py")

    # Le modèle d'embedding doit être le même qu'à l'indexation
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma(collection_name=COLLECTION,
                   embedding_function=embeddings,
                   persist_directory=str(PERSIST_DIR))

    n = store._collection.count()
    if n == 0:
        raise ValueError(
            f"La collection '{COLLECTION}' est vide. Relancez indexer.py")
    return store.as_retriever(search_kwargs={"k": k}), n


# ─── 2. Les trois outils ──────────────────────────────────────────
def build_tools(retriever):
    from langchain_core.tools import Tool
    from langchain_community.tools.arxiv.tool import ArxivQueryRun

    # --- Outil n°1 : rag_search (corpus local) ---
    def _rag_search(query: str) -> str:
        docs = retriever.invoke(query)
        # Sécurité : on élimine les chunks identiques éventuels
        vus, uniques = set(), []
        for d in docs:
            cle = (d.metadata.get("source"), d.page_content)
            if cle not in vus:
                vus.add(cle)
                uniques.append(d)
        docs = uniques
        if not docs:
            return "[Aucun document pertinent trouvé dans le corpus local]"
        return "\n\n---\n\n".join(
            f"[source: {d.metadata.get('source', '?')}]\n{d.page_content}"
            for d in docs
        )

    rag_tool = Tool(
        name="rag_search",
        description=(
            "Cherche des informations dans le corpus IA local (Transformer, RAG, "
            "fine-tuning, vector databases). À utiliser pour les questions techniques "
            "sur ces sujets quand l'information est susceptible d'être dans la base."
        ),
        func=_rag_search,
    )

    # --- Outil n°2 : tavily_web_search (web) ---
    if TAVILY_OK:
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
        desc = ("Recherche sur le web public. À utiliser pour les questions récentes, "
                "hors du corpus local, ou nécessitant des informations à jour.")
        try:  # API moderne (paquet langchain-tavily)
            from langchain_tavily import TavilySearch
            tavily_tool = TavilySearch(max_results=3,
                                       name="tavily_web_search",
                                       description=desc)
        except ImportError:  # API du TP (dépréciée mais fonctionnelle)
            from langchain_community.tools.tavily_search import TavilySearchResults
            tavily_tool = TavilySearchResults(max_results=3,
                                              name="tavily_web_search",
                                              description=desc)
    else:
        tavily_tool = Tool.from_function(
            func=lambda q: "[Tavily désactivé — pas de clé API]",
            name="tavily_web_search",
            description="Recherche web (désactivée).",
        )

    # --- Outil n°3 : arxiv_search (publications) ---
    arxiv_tool = ArxivQueryRun()
    arxiv_tool.name = "arxiv_search"
    arxiv_tool.description = (
        "Cherche des papiers scientifiques sur arXiv. À utiliser pour les questions "
        "concernant des publications académiques précises (auteur, titre, papier précis)."
    )

    return [rag_tool, tavily_tool, arxiv_tool]


# ─── 3. Agent ReAct ───────────────────────────────────────────────
def build_agent(tools, model: str = GROQ_MODEL, temperature: float = 0.0,
                system_prompt: str = SYSTEM_PROMPT):
    from langchain_groq import ChatGroq
    from langgraph.prebuilt import create_react_agent

    llm = ChatGroq(model=model, api_key=GROQ_API_KEY, temperature=temperature)
    return create_react_agent(llm, tools, prompt=system_prompt)


# ═══════════════════════════════════════════════════════════════════
# PARTIE 2 — L'INTERFACE STREAMLIT
# ═══════════════════════════════════════════════════════════════════

# ─── Ressources lourdes : construites UNE seule fois ──────────────
@st.cache_resource(show_spinner="Chargement de la base Chroma et du modèle d'embedding…")
def get_tools():
    retriever, n_chunks = load_retriever()
    return build_tools(retriever), n_chunks


@st.cache_resource(show_spinner=False)
def get_agent(model: str, temperature: float, prompt: str):
    return build_agent(get_tools()[0], model=model,
                       temperature=temperature, system_prompt=prompt)


# ─── Barre latérale : état, réglages, exemples ────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"Groq : {'✅ ok' if GROQ_OK else '❌ clé manquante'}")
    st.write(f"Tavily : {'✅ ok' if TAVILY_OK else '❌ désactivé'}")
    st.write(f"Base Chroma : {'✅ ' + str(PERSIST_DIR.name) if PERSIST_DIR.exists() else '❌ à créer (indexer.py)'}")

    model = st.selectbox("Modèle Groq",
                         [GROQ_MODEL, "llama-3.3-70b-versatile",
                          "llama-3.1-8b-instant"])
    temperature = st.slider("Température", 0.0, 1.0, 0.0, 0.1)
    # Valeur par défaut 10 (le PDF utilise 5 ; Q4 exige au moins 6)
    recursion_limit = st.slider("recursion_limit (garde-fou)", 2, 30, 10)
    memoire = st.toggle("Mémoire de conversation", value=False,
                        help="Si activé, l'agent reçoit aussi les échanges précédents.")

    with st.expander("📝 System prompt (modifiable)"):
        prompt = st.text_area("Prompt", SYSTEM_PROMPT, height=320,
                              label_visibility="collapsed")

    st.divider()
    st.subheader("💡 Questions du golden set")
    for i, q in enumerate(EXEMPLES, 1):
        if st.button(f"Q{i} — {q[:45]}…", key=f"ex{i}",
                     use_container_width=True):
            st.session_state.pending = q

    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ─── Démarrage protégé de la zone principale ──────────────────────
if not GROQ_OK:
    st.error("Clé GROQ_API_KEY absente ou invalide dans le fichier .env.")
    st.stop()

st.title("🔎 AI Research Assistant")
st.caption("Agent ReAct · rag_search (Chroma) · tavily_web_search · arxiv_search")

try:
    _, n_chunks = get_tools()
    agent = get_agent(model, temperature, prompt)
    st.caption(f"📚 Corpus : {n_chunks} chunks chargés depuis {PERSIST_DIR.name}/")
except (FileNotFoundError, ValueError) as e:
    st.error(f"❌ {e}")
    st.code("uv run python indexer.py", language="bash")
    st.stop()
except Exception as e:
    st.error(f"Impossible de construire l'agent : {e}")
    st.exception(e)
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []


# ─── Affichage d'une trace ReAct ──────────────────────────────────
def afficher_etape(etape):
    if etape["type"] == "action":
        st.markdown(f"🤖 **Action** : `{etape['tool']}({etape['args']})`")
    else:
        st.markdown(f"📥 **Observation** [`{etape['tool']}`]")
        st.code(etape["content"][:1500], language=None)


def afficher_metriques(m):
    c1, c2, c3 = st.columns(3)
    c1.metric("Outils", ", ".join(m["tools"]) or "aucun")
    c2.metric("Étapes", len(m["tools"]))
    c3.metric("Durée", f"{m['duration']} s")


# ─── Rejouer l'historique ─────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander(f"🔍 Trace ReAct ({len(msg['trace'])} événements)"):
                for e in msg["trace"]:
                    afficher_etape(e)
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            afficher_metriques(msg["metrics"])


# ─── Exécution de l'agent en streaming ────────────────────────────
def executer(question: str):
    messages = []
    if memoire:
        # On ne renvoie pas les messages d'erreur (⚠) : ils pollueraient le contexte
        messages = [(m["role"], m["content"]) for m in st.session_state.history
                    if not m["content"].startswith("⚠")]
    messages.append(("user", question))
    st.caption(f"📨 {len(messages)} message(s) envoyé(s) à l'agent"
               + (" (mémoire activée)" if memoire else " (sans mémoire)"))

    from langgraph.errors import GraphRecursionError

    trace, tools_used, final = [], [], ""
    t0 = time.time()

    with st.status("L'agent réfléchit…", expanded=True) as status:
        try:
            # stream_mode="updates" : un événement par passage dans un nœud (agent / tools)
            for chunk in agent.stream({"messages": messages},
                                      {"recursion_limit": recursion_limit},
                                      stream_mode="updates"):
                for _node, update in chunk.items():
                    for m in update.get("messages", []):
                        kind = m.__class__.__name__
                        if kind == "AIMessage" and m.tool_calls:
                            for tc in m.tool_calls:
                                e = {"type": "action", "tool": tc["name"],
                                     "args": tc["args"]}
                                trace.append(e)
                                tools_used.append(tc["name"])
                                afficher_etape(e)
                        elif kind == "AIMessage" and m.content:
                            final = m.content
                        elif kind == "ToolMessage":
                            e = {"type": "observation", "tool": m.name,
                                 "content": str(m.content)}
                            trace.append(e)
                            afficher_etape(e)
            status.update(label="Terminé", state="complete", expanded=False)
        except GraphRecursionError:
            final = (f"⚠ Garde-fou atteint : plus de {recursion_limit} "
                     "super-étapes (boucle probable).")
            status.update(label="Interrompu (recursion_limit)", state="error")
        except Exception as e:  # quota 429, réseau…
            final = f"⚠ Erreur : {e}"
            status.update(label="Erreur", state="error")

    # Détection de l'échec silencieux
    if not final:
        final = ("⚠ **Échec silencieux** : l'agent n'a produit ni réponse ni appel d'outil. "
                 "Reformulez la question ou relancez.")

    metrics = {"tools": tools_used, "duration": round(time.time() - t0, 2)}
    st.markdown(final)
    afficher_metriques(metrics)
    return {"role": "assistant", "content": final, "trace": trace,
            "metrics": metrics}


# ─── Saisie : zone de chat ou bouton d'exemple ────────────────────
question = st.chat_input("Posez une question de recherche en IA…")
if not question and st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        reponse = executer(question)
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append(reponse)