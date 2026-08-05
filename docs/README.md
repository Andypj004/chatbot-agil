# Documentación — Agile Chatbot

## Documentos disponibles

| Documento | Contenido |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Diseño completo: capas, módulos, clases del dominio, flujo RAG, flujo socrático, árbol de decisión, esquema SQLite, construcción del prototipo, estándares y principios |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Despliegue con uvicorn y Docker Compose, variables de entorno (tabla completa), permisos, troubleshooting |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Instalación paso a paso, primer chat, uso de RAG, sesiones, streaming |
| [RAG_EVALUATION.md](RAG_EVALUATION.md) | Harness de evaluación de calidad RAG: métricas de recuperación y generación, dataset de verdad fundamental |

## Lectura recomendada

1. [README.md](../README.md) del repositorio — visión general y quickstart.
2. [GETTING_STARTED.md](GETTING_STARTED.md) — instalación y primer uso.
3. [ARCHITECTURE.md](ARCHITECTURE.md) — diseño completo del sistema.
4. [RAG_EVALUATION.md](RAG_EVALUATION.md) — cómo se mide la calidad del RAG.
5. [DEPLOYMENT.md](DEPLOYMENT.md) — Docker, variables de entorno, producción.

## Entrypoint del sistema

```
src/main.py → FastAPI app
           → /api/v1/* (routers)
           → /app (frontend SPA)
           → /docs (Swagger UI)
           → /redoc (ReDoc)
```
