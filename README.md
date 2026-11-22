# 🤖 AI Research Assistant — Agentic RAG avec Streamlit

## 1. Présentation du projet

Ce projet consiste à développer un assistant intelligent de recherche spécialisé dans le domaine de l'Intelligence Artificielle (IA), basé sur une architecture **Agentic RAG (Retrieval-Augmented Generation)**.

L'application utilise un agent intelligent de type **ReAct (Reasoning + Acting)** capable d'analyser les questions de l'utilisateur, de choisir les outils appropriés et de générer des réponses à partir des informations récupérées.

L'application propose trois outils de recherche :

- **RAG Search (ChromaDB)** : recherche dans un corpus documentaire local.
- **Tavily Web Search** : recherche d'informations récentes sur Internet.
- **arXiv Search** : recherche de publications et d'articles scientifiques.

L'interface utilisateur est développée avec Streamlit et permet de communiquer avec l'agent, de consulter les réponses et de visualiser les traces de raisonnement et d'utilisation des outils.

---

## 2. Objectifs du projet

Les principaux objectifs sont :

- Comprendre le fonctionnement du RAG et de l'Agentic RAG.
- Charger et préparer des documents textuels.
- Découper les documents en petits morceaux (Chunks).
- Transformer les textes en vecteurs numériques (Embeddings).
- Stocker les embeddings dans une base vectorielle ChromaDB.
- Construire un agent intelligent avec LangChain et LangGraph.
- Intégrer plusieurs outils de recherche.
- Utiliser le modèle LLM via Groq.
- Développer une interface conversationnelle avec Streamlit.
- Afficher les actions, les observations et les métriques de l'agent.
- Gérer l'historique des conversations.

---

## 3. Architecture du projet

Le projet est organisé autour de deux programmes principaux :

- `indexer.py` : prépare les documents et construit la base vectorielle.
- `app.py` : construit l'agent intelligent et l'interface utilisateur.

### Arborescence

```text
agentic-rag/
│
├── corpus/
│   ├── transformers.md
│   ├── rag_systems.md
│   ├── finetuning.md
│   └── vector_databases.md
│
├── chroma_db/
│   └── Base vectorielle générée automatiquement
│
├── indexer.py
├── app.py
│
├── .env
├── .gitignore
│
├── pyproject.toml
├── uv.lock
│
└── README.md
```

### Description des fichiers

| Fichier / Dossier | Description |
|---|---|
| `corpus/` | Contient les documents Markdown utilisés comme sources de connaissances. |
| `transformers.md` | Informations sur l'architecture Transformer. |
| `rag_systems.md` | Informations sur les systèmes RAG. |
| `finetuning.md` | Informations sur le Fine-tuning et LoRA. |
| `vector_databases.md` | Informations sur les bases de données vectorielles et HNSW. |
| `chroma_db/` | Stocke la base vectorielle ChromaDB. |
| `indexer.py` | Charge, découpe, vectorise et indexe les documents. |
| `app.py` | Contient les outils, l'agent ReAct et l'interface Streamlit. |
| `.env` | Contient les clés API. |
| `.gitignore` | Définit les fichiers à exclure de Git. |
| `pyproject.toml` | Déclare les dépendances et la configuration Python. |
| `uv.lock` | Verrouille les versions des dépendances. |

---

## 4. Technologies utilisées

| Technologie | Rôle |
|---|---|
| Python ≥ 3.12 | Langage de programmation. |
| uv | Gestion de l'environnement virtuel et des dépendances. |
| LangChain | Construction des composants LLM et des outils. |
| LangGraph | Gestion du cycle d'exécution de l'agent ReAct. |
| Groq | Fournisseur du modèle de langage LLM. |
| ChromaDB | Stockage et recherche vectorielle. |
| HuggingFace | Génération des embeddings. |
| all-MiniLM-L6-v2 | Modèle d'embedding utilisé. |
| Tavily | Recherche sur Internet. |
| arXiv | Recherche d'articles scientifiques. |
| Streamlit | Création de l'interface Web conversationnelle. |

---

## 5. Fonctionnement du système

Le fonctionnement du projet se déroule en deux grandes phases.

### 5.1 Phase 1 : Ingestion et indexation (indexer.py)

Cette phase permet de préparer les documents pour la recherche sémantique.

Les étapes sont :

1. Chargement des fichiers Markdown depuis le dossier `corpus/`.
2. Découpage des documents en chunks.
3. Calcul des embeddings avec le modèle `all-MiniLM-L6-v2`.
4. Enregistrement des vecteurs dans ChromaDB.
5. Persistance de la base vectorielle dans le dossier `chroma_db/`.

**Paramètres d'indexation :**

```python
COLLECTION = "tp4_corpus"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 300

CHUNK_OVERLAP = 30
```

La collection créée est `tp4_corpus`.

### 5.2 Phase 2 : Construction de l'agent et interface (app.py)

Cette phase utilise la base ChromaDB créée précédemment.

Elle comprend :

- Le chargement du Retriever.
- La création des trois outils de recherche.
- La configuration du modèle LLM avec Groq.
- La création de l'agent ReAct.
- La gestion de la mémoire conversationnelle.
- La création de l'interface Streamlit.
- L'affichage des traces ReAct et des métriques.

### 5.3 Architecture de l'Agentic RAG

```text
                  UTILISATEUR
                       |
                       v
                INTERFACE STREAMLIT
                       |
                       v
                 QUESTION
                       |
                       v
                  AGENT REACT
                       |
                       v
                    LLM GROQ
                       |
                Analyse de la question
                       |
                  Choix d'un outil
                       |
           +-----------+-----------+
           |           |           |
           v           v           v
        ChromaDB     Tavily      arXiv
           |           |           |
           v           v           v
        Corpus      Recherche   Publications
         local        Web       scientifiques
           |           |           |
           +-----------+-----------+
                       |
                       v
                  OBSERVATION
                       |
                       v
                RAISONNEMENT
                       |
                       v
                 RÉPONSE FINALE
                       |
                       v
                STREAMLIT
```

---

## 6. Installation et configuration

### 6.1 Prérequis

Avant de commencer, il faut installer :

- Python 3.12 ou supérieur.
- uv.
- Git (facultatif).
- Un compte Groq pour obtenir une clé API.
- Un compte Tavily pour obtenir une clé API.

### 6.2 Créer le projet

```bash
mkdir agentic-rag

cd agentic-rag

uv init

uv venv
```

Activer l'environnement virtuel sous Ubuntu/Linux :

```bash
source .venv/bin/activate
```

### 6.3 Installer les dépendances

Exécuter la commande suivante :

```bash
uv add langchain langchain-community langchain-groq langgraph tavily-python langchain-tavily arxiv chromadb sentence-transformers tiktoken langchain-huggingface python-dotenv streamlit
```

Cette commande installe les bibliothèques nécessaires au fonctionnement du projet.

### 6.4 Configurer les clés API

Créer un fichier `.env` à la racine du projet.

Ajouter les variables suivantes :

```env
GROQ_API_KEY=your_groq_api_key

TAVILY_API_KEY=your_tavily_api_key
```

**Important :**

- `GROQ_API_KEY` : permet d'utiliser le modèle LLM hébergé chez Groq.
- `TAVILY_API_KEY` : permet d'utiliser le moteur de recherche Web Tavily.
- Ne jamais publier le fichier `.env` sur GitHub.
- Ne jamais partager ses clés API.

---

## 7. Préparer les documents

Placer les quatre fichiers Markdown dans le dossier `corpus/` :

```text
corpus/
│
├── transformers.md
├── rag_systems.md
├── finetuning.md
└── vector_databases.md
```

Ces fichiers constituent la base documentaire locale utilisée par l'agent.

---

## 8. Exécution du projet

L'exécution doit respecter un ordre précis.

### Étape 1 : construire la base vectorielle

Exécuter :

```bash
uv run python indexer.py
```

Ce script :

- Lit les documents du dossier `corpus/`.
- Découpe les documents en chunks.
- Calcule les embeddings.
- Enregistre les données dans ChromaDB.
- Effectue une recherche de vérification.

Après cette opération, le dossier `chroma_db/` est créé automatiquement.

**Remarque :** il faut réexécuter `indexer.py` après chaque modification du corpus documentaire.

### Étape 2 : lancer l'application Streamlit

Exécuter :

```bash
uv run streamlit run app.py
```

L'application sera accessible à l'adresse :

http://localhost:8501

Si le port 8501 est déjà utilisé, Streamlit peut sélectionner un autre port.

---

## 9. Fonctionnalités de l'application

### 9.1 Recherche documentaire locale

L'outil `rag_search` permet de rechercher des informations dans les documents stockés dans ChromaDB.

### 9.2 Recherche Web

L'outil `tavily_web_search` permet de rechercher des informations récentes sur Internet.

### 9.3 Recherche scientifique

L'outil `arxiv_search` permet de retrouver des publications scientifiques à partir d'une question.

### 9.4 Agent ReAct

L'agent analyse les questions, choisit les outils appropriés et observe leurs résultats avant de générer une réponse.

### 9.5 Mémoire conversationnelle

L'application propose une option permettant d'activer ou de désactiver la mémoire de conversation.

Lorsque la mémoire est activée, les derniers messages sont transmis à l'agent pour conserver le contexte des échanges.

### 9.6 Trace ReAct

L'application affiche les différentes étapes de l'exécution :

- Action : outil choisi par l'agent.
- Observation : résultat retourné par l'outil.
- Réponse finale : réponse générée par le LLM.

### 9.7 Métriques

L'interface affiche :

- Les outils utilisés.
- Le nombre d'appels d'outils.
- La durée d'exécution.

---

## 10. Tests et validation (Golden Set)

L'application propose cinq questions de test pour vérifier le comportement de l'agent.

| Test | Question | Comportement attendu |
|---|---|---|
| Q1 | Comment fonctionne l'algorithme HNSW ? | `rag_search` |
| Q2 | Quels nouveaux modèles d'embedding open-source sont sortis en 2025 ? | `tavily_web_search` |
| Q3 | Que dit le papier original LoRA (2021) ? | `arxiv_search` |
| Q4 | Compare HNSW aux alternatives utilisées en production en 2024-2025. | `rag_search` puis `tavily_web_search` |
| Q5 | Quelle est la recette préférée de l'auteur de Attention Is All You Need ? | Refus poli sans utiliser d'outil |

### Expériences supplémentaires

Plusieurs expériences permettent d'évaluer le comportement de l'agent :

1. **Recursion Limit :** comparer les résultats de Q4 avec les valeurs 5 et 6.
2. **Mémoire :** comparer les réponses avec la mémoire activée et désactivée.
3. **System Prompt :** modifier les instructions et observer le comportement sur Q5.
4. **Modèle LLM :** comparer différents modèles Groq.
5. **Température :** observer les variations des réponses avec une température de 0.8.

---

## 11. Sécurité et bonnes pratiques

- Ne jamais enregistrer les clés API directement dans les fichiers Python.
- Utiliser un fichier `.env` pour les variables sensibles.
- Ajouter `.env` au fichier `.gitignore`.
- Ne pas publier le dossier `.venv/`.
- Conserver les mêmes paramètres `COLLECTION` et `EMBEDDING_MODEL` dans `indexer.py` et `app.py`.
- Réexécuter l'indexation après une modification des documents.

Exemple de fichier `.gitignore` :

```gitignore
.env

.venv/

__pycache__/

*.pyc

chroma_db/

.streamlit/secrets.toml
```

---

## 12. Conclusion

Ce projet permet de mettre en pratique les concepts fondamentaux du RAG et de l'Agentic RAG.

Il montre comment combiner une base de données vectorielle, plusieurs outils de recherche et un modèle de langage afin de construire un assistant intelligent capable de rechercher, analyser et générer des réponses contextualisées.

L'utilisation de LangChain# AI-Research-Assistant---Agentic-RAG-avec-Streamlit
