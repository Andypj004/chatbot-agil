# Agile Chatbot

Prototipo académico de chatbot tutor en español para metodologías ágiles (Scrum, Kanban, Lean, XP, SAFe, ABP). Construido con FastAPI, abstracción multi-LLM, RAG sobre ChromaDB y persistencia en SQLite.

---

## Características principales

- **Tutoría pedagógica** — responde siempre en español con tono educativo, adapta el nivel al perfil del estudiante.
- **Enrutamiento inteligente** — clasifica automáticamente si la pregunta pide conocimiento directo o guía socrática.
- **RAG** — recuperación desde documentos PDF/DOCX/TXT/MD indexados en ChromaDB.
- **Multi-LLM** — soporte nativo para OpenAI, Anthropic (Claude), Google (Gemini), DeepSeek y Ollama (local).
- **Sesiones persistentes** — historial de conversación, conceptos repetidos y citas de fuentes en SQLite.
- **Autenticación** — registro, login y perfil de usuario con cuestionario de adopción ágil.
- **Streaming** — respuestas token a token por Server-Sent Events.
- **Formularios conversacionales** — flujos de captura de datos multi-paso por chat.

---

## Arranque rápido

### Opción A — uvicorn local

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate        # Linux / WSL
# .venv\Scripts\activate         # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Crear configuración
cp .env.example .env
# Editar .env con tu clave de API (ver sección Variables de entorno)

# 4. Levantar
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Opción B — Docker Compose (incluye Ollama)

```bash
# 1. Configurar .env con las claves necesarias
cp .env.example .env

# 2. Levantar todos los servicios
docker compose up --build -d

# 3. Ver logs
docker compose logs -f chatbot-agil
```

### Verificación

```
http://localhost:8000/app        ← Interfaz web
http://localhost:8000/docs       ← Swagger UI interactivo
http://localhost:8000/api/v1/health
```

---

## Variables de entorno mínimas

```env
# Al menos una clave de proveedor LLM
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...

# Proveedor y modelo por defecto
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-4-turbo-preview
```

Ver la tabla completa en [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## Primer chat

```bash
# Sin autenticación, sin RAG
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"¿Qué es un Sprint Goal?","use_rag":false}'
```

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| API | FastAPI 0.136 + Uvicorn |
| LLM | LangChain 1.x + OpenAI / Anthropic / Google / DeepSeek / Ollama |
| Vectores | ChromaDB 1.5 + sentence-transformers |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` |
| Persistencia | SQLite (conversaciones) |
| Validación | Pydantic v2 |
| Tests | pytest + pytest-cov |

---

## Documentación

| Documento | Contenido |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Diseño completo: capas, clases, flujo RAG, flujo socrático, estándares |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Despliegue uvicorn y Docker, variables de entorno, troubleshooting |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Instalación paso a paso y primer uso |

---

## Comandos de desarrollo

```bash
# Tests
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html

# Calidad de código
black src/ tests/
flake8 src/
mypy src/
pylint src/
```

---

## Estructura del proyecto (resumen)

```
src/
├── main.py                  # Entrypoint FastAPI + lifespan
├── agents/chatbot_agent.py  # Orquestador principal
├── api/
│   ├── models.py            # Esquemas Pydantic
│   ├── dependencies.py      # Inyección de dependencias
│   └── routes/              # auth, chat, sessions, documents, config, forms, debug, admin
├── core/
│   ├── config.py            # Settings (pydantic-settings)
│   ├── prompt_manager.py    # Constructores de prompts
│   ├── question_classifier.py
│   ├── security.py
│   └── forms/               # FormManager + agile_adoption_assessment
├── llm/
│   ├── base.py              # BaseLLMProvider (abstracta)
│   ├── factory.py           # LLMFactory
│   ├── models.json          # Catálogo de modelos (editable)
│   └── providers/           # openai, anthropic, google, deepseek, ollama
├── rag/
│   ├── vector_store.py      # VectorStore (ChromaDB)
│   ├── document_processor.py
│   └── retriever.py         # RAGRetriever
└── memory/
    ├── session_manager.py   # SQLite: sesiones, mensajes, usuarios
    └── concept_tracker.py   # Detección de conceptos repetidos
```
