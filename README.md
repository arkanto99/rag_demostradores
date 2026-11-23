# Demostrador RAG

Sistema de pregunta-respuesta sobre noticias basado en RAG (*Retrieval-Augmented Generation*):

1. **Recuperación**: busca en Elasticsearch los fragmentos de noticias más parecidos a la pregunta, por palabras (BM25) o por significado (embeddings con BGE-M3).
2. **Reordenación** (opcional): un *reranker* puntúa de nuevo esos fragmentos y se queda con los mejores.
3. **Generación**: un modelo de lenguaje (Salamandra-7B) redacta la respuesta usando solo esos fragmentos.

Incluye una interfaz web (Gradio) para hacer preguntas, cambiar el recuperador y el reranker en directo y ver las fuentes de cada respuesta.

## Requisitos

- Linux con GPU NVIDIA (el LLM se carga cuantizado a 4 bits; el reranker y los embeddings también usan la GPU).
- Python 3.11 o superior.
- Docker, para Elasticsearch.
- Espacio en disco: unos 50 GB para los modelos de Hugging Face, más el corpus y los índices.

## Instalación

### 1. Entorno de Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuración

```bash
cp .env.example .env
```

Edita `.env` y pon una contraseña en `ES_PASSWORD`. Si quieres guardar los modelos en otra carpeta, indícala en `HF_CACHE_DIR`.

### 3. Elasticsearch

```bash
docker compose up -d
```

Comprueba que responde (te pedirá la contraseña de `ES_PASSWORD`):

```bash
curl -u elastic http://localhost:9200
```

### 4. Corpus

La demo usa el corpus periodístico de Proxecto Nós ([Zenodo](https://zenodo.org/records/17943274), licencia CC BY 4.0), con noticias de *Nós Diario* y *Praza Pública* (2012-2025):

```bash
mkdir -p data
wget -O data/NOS_Corpus_Periodistico.zip "https://zenodo.org/records/17943274/files/NOS_Corpus_Periodistico.zip?download=1"
unzip data/NOS_Corpus_Periodistico.zip -d data/
```

### 5. Índice

Para una prueba rápida, indexa solo las noticias más recientes (1000 por fuente, unos minutos con GPU):

```bash
python -m indexer.build_index \
    --corpus-dir data/NOS_Corpus_Periodistico \
    --index noticias_nos \
    --limit 1000
```

Para el corpus completo, quita `--limit` (puede tardar varias horas).

Opciones principales del indexador (`python -m indexer.build_index --help` para verlas todas):

| Opción | Qué hace | Por defecto |
|---|---|---|
| `--chunking` | `window`: ventanas de palabras con solapamiento; `paragraph`: un fragmento por párrafo | `window` |
| `--window-size`, `--overlap` | Tamaño de la ventana y palabras compartidas entre ventanas seguidas | 200 y 50 |
| `--embedding-model` | Modelo de embeddings | `BAAI/bge-m3` |
| `--limit` | Máximo de noticias por fuente, empezando por las más recientes | todas |
| `--overwrite` | Borra el índice y lo crea de nuevo | no |

Volver a ejecutar el indexador sobre el mismo índice no duplica fragmentos: los actualiza. Si cambias el troceo, usa `--overwrite` o un índice con otro nombre.

El nombre del índice tiene que coincidir con `elastic_index` en [`general_config.json`](demostrador/backend/configs/general_config.json).

## Uso

### Interfaz web

```bash
bash launch_rag.sh
```

Gradio muestra en la terminal la dirección local y un enlace público temporal.

### API

```bash
uvicorn api.main:app --port 8000
```

Documentación interactiva en http://127.0.0.1:8000/docs. Más detalles en [`api/readme.md`](api/readme.md).

## Configuración de modelos y prompts

Todo está en [`demostrador/backend/configs/general_config.json`](demostrador/backend/configs/general_config.json):

- `RETRIEVAL_MODELS`: recuperadores disponibles (BM25 y BGE-M3 usan el mismo índice).
- `RERANKER_MODELS`: rerankers disponibles (`None` para no reordenar).
- `GENERATION_MODELS`: modelo de lenguaje.
- `NUM_DOCS_RETRIEVAL` y `NUM_DOCS_RERANKER`: fragmentos que se recuperan y fragmentos que pasan al LLM.
- `PROMPTS`: instrucciones del sistema y respuesta fija cuando no hay información (`no_answer_text`).

## Estructura

```
├── indexer/              # Creación del índice de Elasticsearch
│   ├── build_index.py    #   script principal
│   ├── nos_corpus.py     #   lectura del corpus NOS
│   └── chunking.py       #   troceo de las noticias
├── demostrador/
│   ├── backend/          # Recuperador, reranker, LLM y configuración
│   └── interfaz/         # Interfaz web (Gradio)
├── api/                  # API REST (FastAPI)
├── evaluator/, ragas/    # Evaluación del sistema
├── docker-compose.yml    # Elasticsearch
└── .env.example          # Plantilla de configuración local
```

## Licencia

Código bajo licencia Apache 2.0 (ver [LICENSE](LICENSE)). El corpus NOS se distribuye bajo CC BY 4.0.
