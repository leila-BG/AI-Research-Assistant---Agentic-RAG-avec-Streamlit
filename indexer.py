"""
indexer.py — Étape d'INGESTION de l'AI Research Assistant (TP 4)

Son rôle est de :
1. Lire les documents Markdown (.md) présents dans le dossier corpus/.
2. Découper les documents en petits morceaux appelés CHUNKS.
3. Transformer chaque chunk en vecteur numérique appelé EMBEDDING.
4. Enregistrer ces vecteurs dans une base de données vectorielle ChromaDB.

Lancer :  uv run python indexer.py
(une fois, puis à chaque modification du corpus)
"""
import shutil
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import TokenTextSplitter

BASE_DIR = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "corpus"
PERSIST_DIR = BASE_DIR / "chroma_db"

# ⚠ Ces deux valeurs doivent être STRICTEMENT IDENTIQUES dans app.py
COLLECTION = "tp4_corpus" #Nom de la collection ChromaDB
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # model d'embedding qui transforme chaque texte en un vecteur de 384 dimensions.


def main():

     # ==========================================================
    # ÉTAPE 1 : LECTURE DES DOCUMENTS DU CORPUS
    # ==========================================================
   
    fichiers = sorted(CORPUS_DIR.glob("*.md")) 
    #cherche tous les fichiers qui possèdent l'extension .md dans le dossier corpus.
    
    if not fichiers: # si fichiers vide
        raise FileNotFoundError(f"Aucun fichier .md trouvé dans {CORPUS_DIR}")

    documents = [] # Créer une liste vide qui contiendra tous les documents chargés.
    for chemin in fichiers:
        docs = TextLoader(str(chemin), encoding="utf-8").load()  
        for d in docs:
            d.metadata["source"] = chemin.name # Ajouter le nom du fichier dans les métadonnées du document.
        documents.extend(docs)  # Ajouter les documents chargés à la liste générale documents.
    print(f"{len(documents)} fichier(s) lu(s) : {[f.name for f in fichiers]}") # Afficher le nombre de documents chargés et leurs noms.

     # ==========================================================
    # ÉTAPE 2 : DÉCOUPAGE DES DOCUMENTS EN CHUNKS
    # ==========================================================

    splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=30)

     #Découper en des chuncks Chaque chunk conserve également les métadonnées de son document d'origine.
    chunks = splitter.split_documents(documents)

     # Afficher le nombre total de chunks créés.
    print(f"{len(chunks)} chunks créés")

      # ==========================================================
    # ÉTAPE 3 : NETTOYAGE DE L'ANCIENNE BASE VECTORIELLE
    # ==========================================================
    if PERSIST_DIR.exists():

        # Supprimer complètement l'ancien dossier chroma_db.
        shutil.rmtree(PERSIST_DIR)

     # ==========================================================
    # ÉTAPE 4 : CALCUL DES EMBEDDINGS ET INDEXATION DANS CHROMA
    # ==========================================================


    #Ce modèle transforme chaque chunk textuel en vecteur numérique.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
     # Créer la base vectorielle Chroma à partir des chunks.
    store = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=COLLECTION,

        # Le dossier où les données seront enregistrées durablement.
        persist_directory=str(PERSIST_DIR),
    )
    print(f"✅ Collection '{COLLECTION}' enregistrée dans {PERSIST_DIR.name}/")


     # ==========================================================
    # ÉTAPE 5 : TEST DE CONTRÔLE DE LA RECHERCHE VECTORIELLE
    # ==========================================================

     # k=1 signifie que l'on souhaite récupérer le chunk le plus pertinent.
    res = store.similarity_search("How does HNSW work?", k=1)
    print("Test « How does HNSW work? » →", res[0].metadata.get("source"))


# ==========================================================
# POINT D'ENTRÉE DU PROGRAMME
# ==========================================================

if __name__ == "__main__":
    main()