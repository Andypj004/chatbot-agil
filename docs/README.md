# Documentación — Agile Chatbot

## Documentos disponibles

| Documento | Contenido |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Diseño completo: capas, módulos, clases del dominio, flujo RAG, flujo socrático, árbol de decisión, esquema SQLite, construcción del prototipo, estándares y principios |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Despliegue con uvicorn y Docker Compose, variables de entorno (tabla completa), permisos, troubleshooting |
| [API.md](API.md) | Referencia completa de endpoints con ejemplos curl, modelos de request/response, códigos de error |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Instalación paso a paso, primer chat, uso de RAG, sesiones, streaming |
| [RUNTIME_BEHAVIOR.md](RUNTIME_BEHAVIOR.md) | Comportamiento en ejecución: caché, fallbacks, reindexado, warmup |
| [ISO25010_Quality_Matrix.md](ISO25010_Quality_Matrix.md) | Matriz de calidad ISO/IEC 25010 aplicada al prototipo |

## Lectura recomendada

1. [README.md](../README.md) del repositorio — visión general y quickstart.
2. [GETTING_STARTED.md](GETTING_STARTED.md) — instalación y primer uso.
3. [ARCHITECTURE.md](ARCHITECTURE.md) — diseño completo del sistema.
4. [API.md](API.md) — referencia de endpoints durante el desarrollo.
5. [DEPLOYMENT.md](DEPLOYMENT.md) — Docker, variables de entorno, producción.

## Entrypoint del sistema

```
src/main.py → FastAPI app
           → /api/v1/* (routers)
           → /app (frontend SPA)
           → /docs (Swagger UI)
           → /redoc (ReDoc)
```
