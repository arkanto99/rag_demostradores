"""
Crea un índice de Elasticsearch con las noticias del corpus NOS.

Pasos:
  1. Leer las noticias del corpus.
  2. Dividir cada noticia en fragmentos (chunks).
  3. Calcular el embedding de cada fragmento.
  4. Guardar en Elasticsearch cada fragmento con su texto, su embedding y sus metadatos.

El mismo índice sirve para la búsqueda léxica (BM25, sobre el campo "text") y
para la búsqueda semántica (kNN, sobre el campo "text_embedding").

Ejemplo, desde la raíz del repositorio (prueba rápida con 1000 noticias por fuente):
    python -m indexer.build_index --corpus-dir data/NOS_Corpus_Periodistico --index noticias_nos --limit 1000
"""

import argparse
import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from indexer.chunking import chunk_by_paragraphs, chunk_by_window
from indexer.nos_corpus import read_articles


def parse_arguments():
    parser = argparse.ArgumentParser(description="Indexa el corpus NOS en Elasticsearch.")
    parser.add_argument("--corpus-dir", required=True,
                        help="Carpeta NOS_Corpus_Periodistico descomprimida")
    parser.add_argument("--index", required=True,
                        help="Nombre del índice de Elasticsearch")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3",
                        help="Modelo de embeddings de Hugging Face (por defecto: BAAI/bge-m3)")
    parser.add_argument("--chunking", choices=["window", "paragraph"], default="window",
                        help="Troceo: ventanas de palabras con solapamiento, o un fragmento por párrafo")
    parser.add_argument("--window-size", type=int, default=200,
                        help="Palabras por ventana (solo con --chunking window)")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Palabras compartidas entre ventanas seguidas (solo con --chunking window)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Máximo de noticias por fuente, empezando por las más recientes")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Fragmentos que se procesan y guardan de cada vez")
    parser.add_argument("--overwrite", action="store_true",
                        help="Borra el índice si ya existe y lo crea de nuevo")
    return parser.parse_args()


def connect_to_elasticsearch():
    """Conecta con Elasticsearch usando los datos del fichero .env."""
    url = os.getenv("ES_URL", "http://localhost:9200")
    user = os.getenv("ES_USER", "elastic")
    password = os.getenv("ES_PASSWORD")
    if not password:
        raise SystemExit("Falta ES_PASSWORD en el fichero .env (ver .env.example).")

    es = Elasticsearch(url, basic_auth=(user, password))
    es.info()  # si no hay conexión, el error salta aquí y no a mitad del proceso
    return es


def create_index(es, index_name, embedding_dim, overwrite):
    """Crea el índice con los campos que usa la demo."""
    if es.indices.exists(index=index_name):
        if not overwrite:
            print(f"El índice '{index_name}' ya existe: se añadirán o actualizarán fragmentos.")
            return
        print(f"Borrando el índice '{index_name}'...")
        es.indices.delete(index=index_name)

    mappings = {
        "properties": {
            # Texto del fragmento: lo usa la búsqueda léxica (BM25)
            "text": {"type": "text"},
            # Embedding del fragmento: lo usa la búsqueda semántica (kNN)
            "text_embedding": {
                "type": "dense_vector",
                "dims": embedding_dim,
                "index": True,
                "similarity": "cosine",
            },
            # Metadatos para citar la noticia de la que sale cada fragmento
            "doc_id": {"type": "keyword"},
            "title": {"type": "text"},
            "date": {"type": "date"},
            "url": {"type": "keyword"},
            "source": {"type": "keyword"},
            "chunk_id": {"type": "integer"},    # posición del fragmento en la noticia
            "num_chunks": {"type": "integer"},  # fragmentos totales de la noticia
        }
    }
    es.indices.create(index=index_name, mappings=mappings)
    print(f"Índice '{index_name}' creado.")


def split_article(article, args):
    """Divide una noticia en fragmentos según el troceo elegido."""
    if args.chunking == "paragraph":
        return chunk_by_paragraphs(article["paragraphs"])

    text = "\n".join(article["paragraphs"])
    return chunk_by_window(text, args.window_size, args.overlap)


def save_batch(es, model, index_name, chunks):
    """Calcula los embeddings de un lote de fragmentos y los guarda en Elasticsearch."""
    texts = []
    for chunk in chunks:
        texts.append(chunk["text"])
    embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True)

    actions = []
    for chunk, embedding in zip(chunks, embeddings):
        chunk["text_embedding"] = embedding.tolist()
        actions.append({
            "_index": index_name,
            # Identificador fijo por fragmento: si se vuelve a ejecutar el script,
            # los fragmentos se sobrescriben en lugar de duplicarse.
            "_id": f"{chunk['doc_id']}-{chunk['chunk_id']}",
            "_source": chunk,
        })
    helpers.bulk(es, actions)


def index_corpus(es, model, args):
    """Lee el corpus, lo trocea y lo guarda en Elasticsearch por lotes."""
    batch = []  # fragmentos pendientes de guardar
    num_articles = 0
    num_chunks = 0

    articles = read_articles(args.corpus_dir, args.limit)
    for article in tqdm(articles, desc="Noticias", unit=" noticias"):
        chunks = split_article(article, args)
        for position, chunk_text in enumerate(chunks):
            batch.append({
                "doc_id": article["id"],
                "title": article["title"],
                "date": article["date"],
                "url": article["url"],
                "source": article["source"],
                "text": chunk_text,
                "chunk_id": position,
                "num_chunks": len(chunks),
            })
        num_articles = num_articles + 1

        if len(batch) >= args.batch_size:
            save_batch(es, model, args.index, batch)
            num_chunks = num_chunks + len(batch)
            batch = []

    # Guardar los fragmentos que hayan quedado en el último lote
    if batch:
        save_batch(es, model, args.index, batch)
        num_chunks = num_chunks + len(batch)

    # Hace visibles para las búsquedas los fragmentos recién guardados
    es.indices.refresh(index=args.index)
    print(f"Listo: {num_articles} noticias y {num_chunks} fragmentos en '{args.index}'.")


def main():
    load_dotenv()
    args = parse_arguments()

    es = connect_to_elasticsearch()

    print(f"Cargando el modelo de embeddings {args.embedding_model}...")
    cache_dir = os.getenv("HF_CACHE_DIR") or None
    model = SentenceTransformer(args.embedding_model, cache_folder=cache_dir)
    embedding_dim = model.get_sentence_embedding_dimension()

    create_index(es, args.index, embedding_dim, args.overwrite)
    index_corpus(es, model, args)


if __name__ == "__main__":
    main()
