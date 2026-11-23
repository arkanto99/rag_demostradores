"""
Funciones para dividir el texto de una noticia en fragmentos (chunks).

Un sistema RAG no recupera documentos completos, sino fragmentos: así el
contexto que recibe el LLM es más corto y se ajusta mejor a la pregunta.
"""


def chunk_by_paragraphs(paragraphs, min_words=5):
    """
    Cada párrafo es un fragmento.

    Se descartan los párrafos de menos de `min_words` palabras (pies de foto,
    firmas, etc.), que apenas aportan información.
    """
    chunks = []
    for paragraph in paragraphs:
        if len(paragraph.split()) >= min_words:
            chunks.append(paragraph)
    return chunks


def chunk_by_window(text, window_size=200, overlap=50):
    """
    Divide el texto en ventanas de `window_size` palabras.

    Dos ventanas seguidas comparten `overlap` palabras. Ese solapamiento evita
    que una idea quede partida justo en la frontera entre dos fragmentos.
    """
    if overlap >= window_size:
        raise ValueError("El solapamiento debe ser menor que el tamaño de la ventana.")

    words = text.split()
    step = window_size - overlap  # cuántas palabras avanza cada ventana

    chunks = []
    start = 0
    while start < len(words):
        end = start + window_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break  # esta ventana ya llega al final del texto
        start = start + step
    return chunks
