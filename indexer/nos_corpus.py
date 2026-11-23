"""
Lectura del corpus periodístico de Proxecto Nós.

Corpus: https://zenodo.org/records/17943274 (licencia CC BY 4.0)
Al descomprimirlo, la carpeta tiene esta estructura:

    NOS_Corpus_Periodistico/
    ├── nosdiario/corpus/2012/ ... 2025/
    ├── prazapublica/2012/ ... 2025/
    └── gl_*.jsonl        (otras cabeceras, sin metadatos)

Solo se leen Nós Diario y Praza Pública, porque son las únicas fuentes con
título, fecha y URL, que la demo usa para citar las noticias.
"""

import html
import json
import re
from datetime import datetime
from pathlib import Path

# Carpeta del corpus -> nombre de la fuente que se muestra en la demo
SOURCES = {
    "nosdiario": "Nós Diario",
    "prazapublica": "Praza Pública",
}


def read_articles(corpus_dir, limit_per_source=None):
    """
    Recorre el corpus y devuelve las noticias de una en una (es un generador).

    Cada noticia es un diccionario con: id, title, date, url, source y paragraphs.
    Con `limit_per_source` se lee como máximo ese número de noticias de cada
    fuente, empezando por las más recientes (útil para hacer pruebas rápidas).
    """
    for folder, source_name in SOURCES.items():
        source_dir = Path(corpus_dir) / folder
        if not source_dir.exists():
            print(f"Aviso: no se encuentra la carpeta {source_dir}")
            continue

        count = 0
        for record in read_records(source_dir):
            article = parse_record(record, folder, source_name)
            if article is None:
                continue

            yield article
            count = count + 1
            if limit_per_source is not None and count >= limit_per_source:
                break


def read_records(source_dir):
    """
    Devuelve los registros JSON de una carpeta, empezando por los más recientes.

    Acepta ficheros .json (una noticia por fichero) y .jsonl (una noticia por línea).
    """
    files = []
    for path in source_dir.rglob("*"):
        if path.suffix in (".json", ".jsonl"):
            files.append(path)

    # Las carpetas van por año (2012 ... 2025): en orden inverso, primero lo más reciente
    files.sort(reverse=True)

    for path in files:
        with open(path, encoding="utf-8") as f:
            if path.suffix == ".jsonl":
                for line in f:
                    if line.strip():
                        yield json.loads(line)
            else:
                yield json.load(f)


def parse_record(record, folder, source_name):
    """
    Extrae de un registro del corpus los campos que usa la demo.

    Devuelve None si a la noticia le falta el identificador o el texto.
    """
    metadata = record.get("metadata", {})
    news = record.get("news", {})

    news_id = metadata.get("news_item_id") or metadata.get("url")
    paragraphs = extract_paragraphs(news)
    if not news_id or not paragraphs:
        return None

    return {
        # Se antepone la carpeta para que no coincidan ids de fuentes distintas
        "id": f"{folder}-{news_id}",
        "title": news.get("headline", ""),
        "date": parse_date(metadata.get("first_published")),
        "url": metadata.get("url", ""),
        "source": source_name,
        "paragraphs": paragraphs,
    }


def extract_paragraphs(news):
    """
    Devuelve la lista de párrafos de una noticia.

    El texto plano ("body") no siempre conserva los saltos de párrafo, así que
    se usa primero la versión HTML ("body_html"), donde cada párrafo va entre
    <p> y </p>. Si no hay HTML, se parte el texto plano por saltos de línea.
    """
    paragraphs = []

    body_html = news.get("body_html") or ""
    for fragment in re.findall(r"<p[^>]*>(.*?)</p>", body_html, flags=re.DOTALL):
        text = clean_html(fragment)
        if text:
            paragraphs.append(text)

    if not paragraphs:
        body = news.get("body") or ""
        for line in body.split("\n"):
            if line.strip():
                paragraphs.append(line.strip())

    return paragraphs


def clean_html(fragment):
    """Quita las etiquetas HTML y convierte entidades como &amp; en texto normal."""
    text = re.sub(r"<br\s*/?>", " ", fragment)  # un salto de línea separa palabras
    text = re.sub(r"<[^>]+>", "", text)         # el resto de etiquetas se eliminan
    text = html.unescape(text)
    return " ".join(text.split())               # deja un solo espacio entre palabras


def parse_date(value):
    """
    Convierte una fecha del corpus (p. ej. "20240201T072219+01:00") en "2024-02-01".

    Devuelve None si no hay fecha o si no tiene un formato reconocible.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return None
