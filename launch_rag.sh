#!/bin/bash
# Lanza la interfaz web de la demo. Ejecutar desde la raíz del repositorio,
# con el entorno virtual activado y Elasticsearch en marcha.

CONFIG=demostrador/backend/configs/general_config.json

python3 -m demostrador.interfaz.rag_interface $CONFIG
