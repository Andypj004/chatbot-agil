# Arquitectura del Sistema — Agile Chatbot

## Índice

1. [Visión general](#1-visión-general)
2. [Diseño del sistema](#2-diseño-del-sistema)
3. [Capas y módulos](#3-capas-y-módulos)
4. [Clases del dominio](#4-clases-del-dominio)
5. [Esquema de base de datos SQLite](#5-esquema-de-base-de-datos-sqlite)
6. [Flujo de chat (ruta principal)](#6-flujo-de-chat-ruta-principal)
7. [Flujo RAG](#7-flujo-rag)
8. [Flujo socrático](#8-flujo-socrático)
9. [Árbol de decisión del agente](#9-árbol-de-decisión-del-agente)
10. [Construcción del prototipo](#10-construcción-del-prototipo)
11. [Estándares y principios aplicados](#11-estándares-y-principios-aplicados)
12. [Estructura de archivos completa](#12-estructura-de-archivos-completa)

---

## 1. Visión general

El sistema es un chatbot tutor de metodologías ágiles que responde siempre en español. Está diseñado como prototipo académico con arquitectura de producción: API REST, abstracción de proveedor LLM, recuperación aumentada por documentos (RAG) y sesiones persistentes.

**Frameworks soportados:** Scrum, Kanban.

**Capacidades clave:**

| Capacidad | Mecanismo |
|---|---|
| Respuesta directa | Prompt directo → LLM |
| Guía socrática | Clasificador de pregunta → prompt socrático → LLM |
| Conocimiento de base | RAG: ChromaDB → contexto → LLM |
| Análisis de imágenes | Base64 multimodal → Ollama (requiere descomentar el servicio en `docker-compose.yml`) / cloud LLM |
| Historial | SQLite `messages` con ventana deslizante de 12 mensajes |
| Cuestionario ágil | Cuestionario de evaluación de adopción ágil (5 preguntas a\|b\|c\|d) |
| Usuarios | Registro + login + cuestionario de nivel ágil |

---

## 2. Diseño del sistema

### 2.1 Diagrama de capas

```
┌──────────────────────────────────────────────────────────────┐
│                        CLIENTE                               │
│           Browser (/app)  ·  curl  ·  Swagger (/docs)        │
└────────────────────────────┬─────────────────────────────────┘
                             │ HTTP / SSE
┌────────────────────────────▼─────────────────────────────────┐
│                      CAPA API (FastAPI)                       │
│  auth · chat · sessions · documents · config · health        │
│  forms · debug · admin                                       │
│  Modelos Pydantic v2 · CORS · Dependency Injection           │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│              CAPA DE ORQUESTACIÓN (ChatbotAgent)              │
│  PromptManager · QuestionClassifier · ConceptTracker         │
│  CitationFilter · StreamingSupport                           │
└──────┬─────────────────┬─────────────────┬───────────────────┘
       │                 │                 │
┌──────▼──────┐  ┌───────▼───────┐  ┌─────▼──────────────────┐
│  CAPA LLM   │  │  CAPA RAG     │  │  CAPA DE PERSISTENCIA   │
│ LLMFactory  │  │ VectorStore   │  │  SessionManager         │
│ BaseLLM-    │  │ RAGRetriever  │  │  SQLite                 │
│ Provider    │  │ DocProcessor  │  │                         │
│ providers/  │  │ ChromaDB      │  │                         │
└─────────────┘  └───────────────┘  └─────────────────────────┘
       │                 │
┌──────▼──────┐  ┌───────▼───────┐
│  CAPA CORE  │  │  INFRA        │
│  config.py  │  │ chroma_db/    │
│  security   │  │ data/         │
│  logger     │  │ uploads/      │
└─────────────┘  └───────────────┘
```

### 2.2 Ciclo de vida de la aplicación

Al arrancar, `lifespan()` en `src/main.py` ejecuta en orden:

1. Inicializa el `VectorStore` (carga ChromaDB, descarga el modelo de embedding).
2. Si la colección está vacía, recorre `data/uploads/global/` y reindexea cada archivo soportado (PDF, DOCX, TXT, MD).
3. Opcionalmente instancia el proveedor LLM por defecto para calentarlo.
4. Registra todos los routers en FastAPI bajo el prefijo `/api/v1`.
5. Monta los archivos estáticos del frontend en `/static` y `/uploads`.

### 2.3 Patrón de inyección de dependencias

`src/api/dependencies.py` implementa singletons lazy y caché por parámetros:

```
get_vector_store()     → singleton VectorStore
get_document_processor() → singleton DocumentProcessor
get_session_manager()  → singleton SessionManager

_create_cached_llm_provider(provider, model, temp, max_tokens)
  └── @lru_cache(maxsize=32) — clave = (provider_normalizado, model, temp, tokens)

_create_cached_rag_retriever(provider, model, temp, max_tokens)
  └── @lru_cache(maxsize=32)

reset_runtime_caches() → limpia ambos caches (llama al actualizar config)
```

---

## 3. Capas y módulos

### 3.1 Capa API — `src/api/`

**`src/api/models.py`** — todos los esquemas Pydantic v2 del sistema.

**`src/api/dependencies.py`** — funciones de inyección de dependencias y singletons.

**`src/api/routes/`:**

| Archivo | Prefijo | Responsabilidad |
|---|---|---|
| `auth.py` | `/auth` | Registro, login, perfil del usuario autenticado, eliminación de cuenta |
| `chat.py` | `/chat` | Chat síncrono y streaming SSE |
| `sessions.py` | `/sessions` | CRUD de sesiones y historial de mensajes |
| `documents.py` | `/documents` | Subida y gestión de documentos globales y de sesión |
| `config.py` | `/config` | Lectura y actualización de la configuración runtime |
| `health.py` | `/health` | Estado del servicio y conteo de documentos |
| `forms.py` | `/forms` | Cuestionario de evaluación ágil (`agile_adoption_assessment`) |
| `debug.py` | `/debug` | Inspección de resultados RAG (solo desarrollo) |
| `admin.py` | `/admin` | Gestión de usuarios (solo administradores): listado y eliminación |

### 3.2 Capa de orquestación — `src/agents/`

**`src/agents/chatbot_agent.py`** — `ChatbotAgent` es el punto central de toda la lógica conversacional.

Responsabilidades:
- Invocar el `QuestionClassifier` para decidir modo de respuesta.
- Construir el contexto conversacional (ventana de historial).
- Obtener el `rag_hint` del `RAGRetriever` cuando aplica.
- Seleccionar y llamar al método de generación correcto (directo, socrático, multimodal).
- Puntuar y filtrar fuentes RAG contra el texto de respuesta generado.
- Soportar streaming token a token vía `chat_stream()`.

### 3.3 Capa LLM — `src/llm/`

**`src/llm/base.py` — `BaseLLMProvider` (abstracta)**

Define la interfaz común:
- `get_llm() → BaseLanguageModel` — retorna el objeto LangChain (chat o completion).
- `get_provider_name() → str`
- `validate_generation_params(temperature, max_tokens)` — valida rangos antes de construir.

**`src/llm/factory.py` — `LLMFactory`**

- Registra proveedores con `register_provider(name, class)`.
- Lee `src/llm/models.json` al importar y sobreescribe el catálogo interno.
- `create_provider(provider, model, temperature, max_tokens)` — crea instancias validadas.
- Normaliza alias: `claude → anthropic`, `gemini → google`.
- **Pilot lock**: `create_provider()` ignora cualquier `provider`/`model` solicitado que no coincida con `settings.default_llm_provider` y usa el default en su lugar (no lanza error). `get_available_providers()` y `get_available_models()` exponen únicamente el proveedor/modelo bloqueado. `get_registered_providers()` (nuevo) sí devuelve el catálogo completo de proveedores implementados, sin el lock, para introspección.

**`src/llm/models.json`** — catálogo de modelos por proveedor (editable sin tocar código). Este catálogo completo sigue existiendo y se usa para validar nombres de modelo y para `get_registered_providers()`, pero en runtime el pilot lock limita lo que `get_available_providers()`/`get_available_models()` exponen al proveedor/modelo default:

```json
{
  "openai":    ["gpt-4o-mini", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
  "anthropic": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
  "google":    ["gemini-3.1-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-pro"],
  "deepseek":  ["deepseek-chat", "deepseek-reasoner"],
  "ollama":    ["llama3.2:3b", "qwen2.5:3b", "phi3:mini", "llava:7b", "llava:13b", "gemma3:4b"]
}
```

**`src/llm/providers/`** — un archivo por proveedor:

| Archivo | Clase | Backend LangChain |
|---|---|---|
| `openai_provider.py` | `OpenAIProvider` | `ChatOpenAI` |
| `anthropic_provider.py` | `AnthropicProvider` | `ChatAnthropic` |
| `google_provider.py` | `GoogleProvider` | `ChatGoogleGenerativeAI` |
| `deepseek_provider.py` | `DeepseekProvider` | `ChatOpenAI` (endpoint DeepSeek) |
| `ollama_provider.py` | `OllamaProvider` | `OllamaLLM` (langchain_ollama) |

Los proveedores cloud envuelven `BaseChatModel` de LangChain; Ollama envuelve `BaseLLM` vía `langchain-ollama`. Ambos exponen `invoke()` con la misma firma.

### 3.4 Capa RAG — `src/rag/`

**`src/rag/vector_store.py` — `VectorStore`**

- Usa `chromadb.PersistentClient` + `langchain_chroma.Chroma`.
- El nombre de colección incluye un slug del modelo de embedding para evitar colisiones:
  `chatbot_documents_paraphrase_multilingual_minilm_l12_v2`.
- Patrón writable-fallback: si `chroma_db/` no tiene permisos de escritura (Docker → host), copia a `chroma_db.writable/`.
- Métodos clave: `add_documents`, `similarity_search`, `similarity_search_with_score`, `delete_by_metadata`, `list_indexed_documents`.

**`src/rag/document_processor.py` — `DocumentProcessor`**

- Carga documentos con loaders de LangChain (`PyPDFLoader`, `TextLoader`, `Docx2txtLoader`, `UnstructuredMarkdownLoader`).
- Trocea con `RecursiveCharacterTextSplitter` (chunk_size=1500, overlap=300).
- Añade metadatos por chunk: `source`, `filename`, `file_type`, `file_hash`, `scope`, `session_id`, `chunk_id`.
- Calcula hash MD5 del archivo para identificación consistente.

**`src/rag/retriever.py` — `RAGRetriever`**

- Recuperación combinada: globales (60 %) + sesión (40 %) dentro de cuota `top_k`.
- Algoritmo de ranking por alineación: score vectorial + score de alineación de tokens entre query y contenido/filename.
- Límite de chunks por archivo: `RAG_MAX_CHUNKS_PER_FILE` (default 5) para diversidad temática.
- Contexto acotado: `RAG_CONTEXT_MAX_CHARS` (default 10 000 chars).

### 3.5 Capa de persistencia — `src/memory/`

**`src/memory/session_manager.py` — `SessionManager`**

- SQLite con threading lock para acceso concurrente seguro.
- Patrón writable-fallback igual que ChromaDB.
- Gestiona: usuarios, sesiones, mensajes, documentos de sesión, conceptos, citas de fuentes y estados del cuestionario ágil.
- Métodos de autenticación: `create_user`, `authenticate_user`, `issue_user_token`, `get_user_by_token`.

**`src/memory/concept_tracker.py`**

- Lista de conceptos ágiles conocidos (scrum, kanban, sprint, backlog, etc.).
- `extract_concepts(text)` → lista de conceptos encontrados.
- `build_history_note(concepts, repeated)` → instrucción para el prompt que indica si repetir o profundizar.

**`src/memory/form_state.py` / `form_state_session.py`**

- Almacenan el estado en progreso del cuestionario de evaluación ágil.
- La versión `_session` usa `SessionManager.upsert_form_state()` para persistir en SQLite.

### 3.6 Capa core — `src/core/`

**`src/core/config.py` — `Settings`**

Carga variables de `.env` con `pydantic-settings`. Instancia global `settings` accesible en todo el proyecto.

**`src/core/prompt_manager.py` — `PromptManager`**

Centraliza la construcción de prompts. Contiene el `BASE_SYSTEM_PROMPT` que impone:
- Respuesta siempre en español.
- Tono educativo, nivel adaptado al perfil del estudiante (1-Ninguno → 4-Avanzado).
- No mencionar "contexto" ni "documentos" en la respuesta.
- Etiquetar preguntas fuera de alcance con "fuera de alcance".
- No mezclar frameworks entre sí sin evidencia explícita.

Expone dos métodos: `build_direct_prompt()` y `build_socratic_prompt()`.

**`src/core/question_classifier.py` — `QuestionClassifier`**

Módulo de clasificación basado en reglas léxicas. Determina el modo de respuesta sin llamar al LLM.

**`src/core/security.py`**

- `hash_password` / `verify_password`: PBKDF2-HMAC-SHA256, 390 000 iteraciones, salt de 16 bytes.
- `generate_token` / `hash_token`: token de 32 bytes urlsafe, almacenado como hash SHA256.
- `assess_agile_level(answers)`: calcula nivel de adopción ágil (1–4) desde respuestas del cuestionario.
- `build_user_profile_context(user)`: construye el bloque de perfil para el prompt.

**`src/core/forms/`**

- `form_manager.py` — `FormManager` + `FormSpec`: registro y ejecución de formularios con lógica de dependencias entre campos.
- `default_forms.py` — formulario precargado: `agile_adoption_assessment` (Cuestionario de Evaluación Ágil Sustentado).
- `validators.py` — validadores de campo y validadores cruzados.

---

## 4. Clases del dominio

```
                    ┌─────────────────────────────┐
                    │        Settings             │
                    │  (pydantic-settings)        │
                    │  + get_api_key(provider)    │
                    └─────────────────────────────┘
                                  ▲
                                  │ usa
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
┌────────▼────────┐    ┌──────────▼──────┐    ┌───────────▼────────┐
│  BaseLLMProvider│    │   VectorStore   │    │  SessionManager    │
│  (abstracta)    │    │                 │    │                    │
│ +get_llm()      │    │+add_documents() │    │+create_session()   │
│ +get_provider() │    │+similarity_     │    │+append_message()   │
│ +validate_      │    │  search()       │    │+get_messages()     │
│   params()      │    │+delete_by_      │    │+create_user()      │
└────────┬────────┘    │  metadata()     │    │+authenticate_user()│
         │             │+list_indexed_   │    │+record_concepts()  │
  ┌──────┴──────┐      │  documents()    │    │+record_citations() │
  │  Providers  │      └────────┬────────┘    └───────────────────┘
  │ OpenAI      │               │ usa
  │ Anthropic   │    ┌──────────▼──────────┐
  │ Google      │    │  DocumentProcessor  │
  │ DeepSeek    │    │                     │
  │ Ollama      │    │+load_document()     │
  └──────┬──────┘    │+chunk_documents()   │
         │            │+process_file()      │
         │            └──────────┬──────────┘
         │                       │
┌────────▼───────────────────────▼────────────────────────────┐
│                       RAGRetriever                          │
│                                                              │
│ +has_documents()                                             │
│ +retrieve_documents(query, k, filter, session_id)           │
│ +_retrieve_combined_documents(query, k, session_id)         │
│ +_build_context(documents) → str                            │
│ +_query_alignment_score(query, doc) → float                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             │ usa
┌────────────────────────────▼────────────────────────────────┐
│                       ChatbotAgent                          │
│                                                              │
│ +chat(message, use_rag, ...) → Dict                         │
│ +chat_stream(message, use_rag, ...) → Iterator[Dict]        │
│ -_build_direct_prompt(...) → str                            │
│ -_build_socratic_prompt(...) → str                          │
│ -_generate_direct_response(...) → str                       │
│ -_generate_socratic_response(...) → str                     │
│ -_generate_multimodal_response(prompt, paths) → str        │
│ -_filter_relevant_sources(response, sources) → List         │
│ -_extract_sources(raw) → List                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐    ┌─────────────────────────────┐
│      PromptManager          │    │   QuestionClassification    │
│                             │    │   (dataclass frozen)        │
│ +build_direct_prompt(...)   │    │                             │
│ +build_socratic_prompt(...) │    │ is_project_context: bool    │
└─────────────────────────────┘    │ confidence: float           │
                                   │ matched_signals: tuple      │
┌─────────────────────────────┐    └─────────────────────────────┘
│       FormManager           │
│                             │    ┌─────────────────────────────┐
│ _registry: Dict[str,FormSpec│    │         FormSpec            │
│ +register_form(spec)        │    │   (dataclass)               │
│ +start_form(id, session...) │    │                             │
│ +answer(id, session, ...)   │    │ form_id: str                │
│ -_current_question(...)     │    │ title: str                  │
└─────────────────────────────┘    │ fields: List[Dict]          │
                                   │ cross_validators: List      │
                                   └─────────────────────────────┘
```

### 4.1 Modelos de datos API (Pydantic v2)

| Clase | Dirección | Descripción |
|---|---|---|
| `ChatRequest` | → entrada | Mensaje, session_id, use_rag, stream, provider, model, temp, adjuntos |
| `ChatResponse` | ← salida | session_id, response, provider, model, used_rag, sources, error |
| `SourceCitation` | ← salida | document_id, filename, page, section, scope, relevance, excerpt |
| `UserRegistrationRequest` | → entrada | email, password, full_name, account_type, knowledge_level, questionnaire_answers |
| `UserLoginRequest` | → entrada | email, password |
| `AuthResponse` | ← salida | access_token, token_type, user (UserProfileResponse) |
| `UserProfileResponse` | ← salida | Perfil completo del usuario con nivel ágil e indicador `is_admin` |
| `UserListResponse` | ← salida | total + lista de `UserProfileResponse` (solo admin) |
| `QuestionnaireAnswer` | → entrada | question_number (1–5), answer (a\|b\|c\|d) |
| `DocumentUploadResponse` | ← salida | filename, document_id, chunks_created, scope, session_id |
| `SessionSummary` | ← salida | session_id, title, message_count, last_message, timestamps |
| `SessionHistoryResponse` | ← salida | messages paginados con metadata |
| `ConfigUpdateRequest` | → entrada | llm_provider, model_name, temperature, max_tokens |
| `ConfigResponse` | ← salida | Config activa + catálogos de proveedores y modelos |
| `HealthResponse` | ← salida | status, version, proveedores, rag_status, vector_store_documents |

---

## 5. Esquema de base de datos SQLite

Ruta: `CONVERSATION_DB_PATH` (default `./data/conversations.db`).  
La migración es in-place: `PRAGMA table_info` detecta columnas faltantes y las añade.

```sql
-- Usuarios del sistema
CREATE TABLE users (
    user_id                   TEXT PRIMARY KEY,
    email                     TEXT NOT NULL UNIQUE,
    full_name                 TEXT NOT NULL,
    account_type              TEXT NOT NULL,          -- "Estudiante" | "Profesor"
    knowledge_level           INTEGER NOT NULL,       -- 1-4 (declarado)
    agile_adoption_level      INTEGER NOT NULL,       -- 1-4 (estimado por cuestionario)
    agile_adoption_label      TEXT NOT NULL,          -- "Ninguno" | "Inicial" | ... | "Avanzado"
    questionnaire_answers_json TEXT NOT NULL,
    password_hash             TEXT NOT NULL,          -- PBKDF2-HMAC-SHA256
    password_salt             TEXT NOT NULL,
    auth_token_hash           TEXT,                   -- SHA256 del bearer token
    created_at                TEXT NOT NULL,
    updated_at                TEXT NOT NULL,
    last_login_at             TEXT
);

-- Sesiones de conversación
CREATE TABLE sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT,                                 -- NULL = sesión anónima
    title       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Mensajes de una sesión
CREATE TABLE messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions,
    role            TEXT NOT NULL,                    -- "user" | "assistant" | "system"
    text            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    provider        TEXT,
    model           TEXT,
    used_rag        INTEGER,                          -- 0 | 1
    sources_json    TEXT,                             -- JSON array de SourceCitation
    attachments_json TEXT
);

-- Documentos asociados a una sesión
CREATE TABLE session_documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions,
    document_id TEXT NOT NULL,
    filename    TEXT,
    source      TEXT,
    file_type   TEXT,
    file_hash   TEXT,
    uploaded_at TEXT NOT NULL,
    UNIQUE(session_id, document_id)
);

-- Conceptos ágiles mencionados en una sesión
CREATE TABLE session_concepts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions,
    concept         TEXT NOT NULL,
    mention_count   INTEGER NOT NULL DEFAULT 1,
    first_mentioned TEXT NOT NULL,
    last_mentioned  TEXT NOT NULL,
    UNIQUE(session_id, concept)
);

-- Citas de fuentes RAG para deduplicación
CREATE TABLE session_citations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions,
    citation_key  TEXT NOT NULL,                      -- hash normalizado de la cita
    document_id   TEXT,
    filename      TEXT,
    source        TEXT,
    page          INTEGER,
    section       TEXT,
    scope         TEXT,
    excerpt       TEXT,
    mention_count INTEGER NOT NULL DEFAULT 1,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    UNIQUE(session_id, citation_key)
);

-- Estado en progreso del cuestionario de evaluación ágil
CREATE TABLE form_states (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions,
    form_id     TEXT NOT NULL,
    state_json  TEXT NOT NULL,                        -- {current_index, answers:{}}
    updated_at  TEXT NOT NULL,
    UNIQUE(session_id, form_id)
);
```

---

## 6. Flujo de chat (ruta principal)

```
POST /api/v1/chat
        │
        ▼
[1] Resolver sesión
    ├── session_id = request.session_id ?? uuid4()
    ├── create_session(session_id, user_id)
    └── verificar ownership (403 si sesión de otro usuario)

        │
        ▼
[2] Cargar contexto
    ├── session_documents = get_session_documents(session_id, doc_ids)
    ├── history = get_messages(session_id, limit=12, offset=ultimo-12)
    └── user_profile_note = build_user_profile_context(current_user)

        │
        ▼
[3] Persistir mensaje del usuario
    └── append_message(session_id, role="user", text=message)
        (Si es el primer turno, generar título legible)

        │
        ▼
[4] Obtener proveedor LLM (desde caché lru_cache)
    └── get_llm_provider(provider, model, temperature)

        │
        ▼
[5] Obtener agente
    └── get_chatbot_agent(llm_provider, use_rag=True)
        ├── crea RAGRetriever si use_rag=True
        └── crea ChatbotAgent(llm_provider, rag_retriever)

        │
        ▼
[6] ChatbotAgent.chat() / .chat_stream()
    (ver sección 9 — árbol de decisión)

        │
        ▼
[7] Persistir respuesta del asistente
    ├── append_message(role="assistant", text, provider, model, used_rag, sources)
    ├── record_source_citations(session_id, sources)
    └── record_concepts(session_id, message + response)

        │
        ▼
[8] Retornar ChatResponse
    └── {session_id, response, provider, model, used_rag, sources}
```

**Modo streaming (stream=True):** La ruta devuelve `StreamingResponse(event_generator(), media_type="text/event-stream")`. El generador emite eventos `data: {"type":"delta","content":"..."}` y al final un evento `data: {"type":"final",...}` con metadatos completos.

---

## 7. Flujo RAG

```
ChatbotAgent.chat()
        │
        ├── [pre-condición] use_rag=True AND rag_retriever.has_documents()
        │
        ▼
[1] Construir query contextual
    ├── Si hay historial previo:
    │     contextual_message = "Contexto conversacional reciente:\n{historial}\n\nPregunta actual: {message}"
    └── Si no hay historial:
          contextual_message = message

        │
        ▼
[2] RAGRetriever.retrieve_documents(contextual_message, session_id)
        │
        ▼
[3] _retrieve_combined_documents(query, k=8, session_id)
    │
    ├── [A] Búsqueda global (filter: scope=global_rag)
    │     └── similarity_search_with_score(query, k=max(k*4,12))
    │
    ├── [B] Búsqueda de sesión (filter: session_id=X, scope=session_chat)
    │     └── similarity_search_with_score(query, k=max(k*2,8))
    │
    ├── [C] Scoring compuesto por chunk
    │     score = (1/(rank+1)) + _query_alignment_score(query, doc)
    │     alignment = token_overlap + phrase_bonus + filename_bonus + title_bonus
    │
    ├── [D] Agrupación por archivo
    │     └── max chunks_por_archivo = RAG_MAX_CHUNKS_PER_FILE (5)
    │
    └── [E] Selección final interleaved: un chunk por archivo por ronda
          hasta completar k=8 documentos

        │
        ▼
[4] _build_context(documents)
    └── concatena page_content hasta RAG_CONTEXT_MAX_CHARS (10 000)
        → rag_hint: str

        │
        ▼
[5] Inyectar rag_hint en el prompt
    └── PromptManager.build_direct_prompt(
            message, conversation_block, rag_hint, history_note, user_profile_note
        )
        Sección en el prompt: "Contexto verificado:\n{rag_hint}"

        │
        ▼
[6] LLM genera respuesta

        │
        ▼
[7] Puntuar fuentes contra respuesta generada
    ChatbotAgent._filter_relevant_sources(response_text, raw_sources)
    │
    ├── Para cada chunk fuente:
    │     score = _source_relevance_score(response_text, chunk_content)
    │     ├── Exact match → 1.0
    │     ├── Token overlap × 0.85
    │     ├── Phrase bonus (ventana de 4–12 tokens) + 0.2 a 0.95
    │     └── Section bonus si el título aparece en respuesta + 0.08
    │
    ├── Filtro mínimo: score ≥ 0.25
    │
    └── Ordenamiento: prioriza fuentes nuevas (no vistas en session_citations)
          → devuelve los top TOP_K_RESULTS (8)

        │
        ▼
[8] Retornar sources junto con la respuesta
```

**Cuando RAG no aplica:**
- `use_rag=False` en la request → se salta todo el flujo RAG.
- `rag_retriever.has_documents()` es False → colección vacía, respuesta directa.
- La pregunta es de tipo socrático → RAG se ejecuta pero el prompt cambia.

---

## 8. Flujo socrático

El modo socrático se activa cuando el `QuestionClassifier` detecta que el estudiante habla de su propio contexto, no que pide conocimiento general.

### Clasificador de preguntas

```
classify_question(message)
        │
        ▼
[1] Normalizar texto (lower, strip, collapse spaces)

        │
        ▼
[2] Buscar patrones de intención de conocimiento (_KNOWLEDGE_INTENT_PATTERNS)
    Ejemplos: "¿qué es?", "¿cómo funciona?", "¿cuál es la diferencia?",
              "explícame", "define", "¿para qué sirve?"
    Si hay match → QuestionClassification(is_project_context=False, confidence=0.85)
    (PRIORIDAD ABSOLUTA: una pregunta de definición siempre da respuesta directa)

        │
        ▼
[3] Si no hay intención de conocimiento: buscar señales de contexto de proyecto
    │
    ├── Keywords (_PROJECT_KEYWORDS):
    │     "mi proyecto", "nuestro equipo", "mi sprint", "para la entrega", ...
    │
    └── Patrones regex (_PROJECT_PATTERNS):
          "como podemos", "como deberiamos", "que podemos hacer", "como organizamos", ...

        │
        ▼
[4] Calcular confianza
    signals = keywords_matches + pattern_matches
    confidence = min(0.95, 0.55 + 0.1 × len(signals))
    is_project_context = len(signals) > 0
```

### Generación socrática

```
PromptManager.build_socratic_prompt(
    message, conversation_block, rag_hint, history_note, user_profile_note
)
        │
        ▼
El prompt incluye instrucciones adicionales:
  "No des la respuesta directa. Ayuda al estudiante con preguntas orientadoras
   y breves pistas para analizar su propio proyecto. Concéntrate en preguntas
   que revelen supuestos, prioridades, riesgos y próximos pasos accionables."

        │
        ▼
Instrucción de cierre:
  "Genera de 3 a 5 preguntas socráticas concretas y orientadas al proyecto,
   evitando repeticiones y manteniendo un tono de apoyo.
   Tras las preguntas, sugiere 1 o 2 pasos concretos (máx. 2 frases)."
```

**Diferencia clave entre modos:**

| Aspecto | Respuesta directa | Respuesta socrática |
|---|---|---|
| Prompt base | `build_direct_prompt()` | `build_socratic_prompt()` |
| RAG | Sí (si aplica) | Sí, como "Conocimiento de referencia" |
| Salida esperada | Explicación educativa | 3-5 preguntas + 1-2 pasos concretos |
| Tipo en `final` SSE | `"direct"` | `"socratic"` |

---

## 9. Árbol de decisión del agente

```
ChatbotAgent.chat(message, use_rag, ...)
        │
        ▼
        classify_question(message)
            │
            ├── is_project_context = True?
            │       │
            │       ├── Obtener rag_hint (si use_rag y hay docs)
            │       └── _generate_socratic_response(message, ..., rag_hint)
            │
            └── is_project_context = False
                    │
                    ├── ¿Hay documentos de imagen en session_documents?
                    │       │
                    │       └── SÍ → _build_multimodal_prompt()
                    │               └── _generate_multimodal_response(prompt, image_paths)
                    │                   ├── Ollama: POST /api/generate con images[]
                    │                   └── Cloud: HumanMessage con image_url base64
                    │
                    └── NO → Obtener rag_hint (si use_rag y hay docs)
                            └── _generate_direct_response(message, ..., rag_hint)
                                    └── _invoke_llm(prompt) → str
```

**ConceptTracker** (se ejecuta antes del switch):
- Extrae conceptos ágiles del mensaje.
- Consulta `session_manager.has_seen_concept()` para cada concepto.
- Si algún concepto es repetido → `build_history_note(repeated=True)` → instrucción al LLM para profundizar, no repetir la definición base.

**CitationDeduplication** (se ejecuta después de generar):
- Obtiene `recent_citation_keys` de `session_citations` en SQLite.
- `_filter_relevant_sources()` baja el ranking de fuentes ya vistas recientemente.

---

## 10. Construcción del prototipo

### Decisiones de diseño

**Por qué FastAPI**
FastAPI ofrece validación automática de esquemas con Pydantic v2, documentación OpenAPI generada, soporte nativo de async/await y StreamingResponse para SSE. Es la elección natural para APIs Python de producción.

**Por qué ChromaDB**
Permite persistencia local sin necesidad de infraestructura externa, embeddings con sentence-transformers multilingüe, y filtrado por metadatos clave como `scope` y `session_id` para separar documentos globales de sesión.

**Por qué la abstracción multi-LLM**
El patrón Factory + Strategy permite añadir o intercambiar proveedores sin tocar el código del agente — registrar uno nuevo solo requiere implementar `BaseLLMProvider` y llamar `register_provider()`. Esto es crítico en un contexto académico donde la disponibilidad de API keys varía. Ollama permite ejecución completamente local (servicio deshabilitado por defecto en `docker-compose.yml`, ver sección 3.3). En esta configuración piloto, el cambio de proveedor en runtime vía API está bloqueado (`LLMFactory` pilot lock): solo el proveedor en `DEFAULT_LLM_PROVIDER` puede usarse en cada momento; cambiar de proveedor activo requiere editar `.env` y reiniciar.

**Por qué SQLite y no una base de datos externa**
El prototipo prioriza la portabilidad y la facilidad de despliegue. SQLite no requiere un servidor separado y es suficiente para la carga esperada. El patrón writable-fallback resuelve el problema de permisos en entornos Docker sin infraestructura adicional.

**Por qué clasificación léxica y no LLM para clasificar preguntas**
La clasificación basada en reglas es determinista, instantánea (sin latencia adicional), y suficientemente precisa para el dominio acotado. Un clasificador LLM añadiría coste y latencia a cada request sin mejora significativa para preguntas en español sobre metodologías ágiles.

**Flujo de contexto RAG (rag_hint) vs respuesta directa RAG**
El sistema no usa un chain RetrievalQA clásico. En cambio, inyecta el contexto recuperado como sección `Contexto verificado` dentro del prompt directo. Esto da al `PromptManager` control total sobre el sistema prompt y evita que el LLM revele que está leyendo documentos.

**Deduplicación de fuentes**
El sistema puntúa cada fuente contra el texto generado (overlap de tokens + frase exacta + sección). Fuentes que aparecen en la respuesta con score ≥ 0.25 se muestran; fuentes que ya fueron citadas en turnos anteriores de la sesión se muestran al final para promover variedad.

### Tecnologías clave y versiones

| Tecnología | Versión | Rol |
|---|---|---|
| Python | 3.14 | Runtime |
| FastAPI | 0.136 | Framework HTTP |
| Uvicorn | 0.39 | Servidor ASGI |
| Pydantic | 2.13 | Validación y modelos |
| pydantic-settings | 2.14 | Carga de configuración |
| LangChain | 1.3 | Abstracción LLM y splitters |
| langchain-openai | 1.3.0 | Integración OpenAI |
| langchain-anthropic | 1.4.4 | Integración Anthropic |
| langchain-google-genai | 4.2.5 | Integración Google |
| langchain-ollama | 1.1.0 | Integración Ollama |
| ChromaDB | 1.5.9 | Vector store |
| sentence-transformers | 5.5.1 | Modelo de embedding |
| torch | 2.12.0 | Backend de embedding |
| SQLite | builtin | Persistencia |
| httpx | 0.28.1 | Cliente HTTP (Ollama multimodal) |
| loguru | 0.7.3 | Logging estructurado |

> **Nota:** los loaders de documentos (`PyPDFLoader`, `TextLoader`, `Docx2txtLoader`,
> `UnstructuredMarkdownLoader`) siguen viniendo de `langchain-community`, paquete que
> upstream está despriorizando en favor de integraciones dedicadas. No es bloqueante
> hoy, pero futuras migraciones deberían vigilar reemplazos dedicados para estos
> loaders.

---

## 11. Estándares y principios aplicados

### Principios SOLID

| Principio | Aplicación |
|---|---|
| **S** — Responsabilidad única | Cada clase tiene un rol claro: `VectorStore` solo maneja ChromaDB, `DocumentProcessor` solo procesa archivos, `SessionManager` solo gestiona SQLite. |
| **O** — Abierto/Cerrado | Se añaden proveedores LLM implementando `BaseLLMProvider` sin modificar `LLMFactory`. Se añaden modelos editando `models.json`. |
| **L** — Sustitución de Liskov | `OpenAIProvider`, `AnthropicProvider`, etc. son intercambiables: todos exponen `get_llm()` con la misma contrato. |
| **I** — Segregación de interfaces | `BaseLLMProvider` solo define `get_llm()` y `get_provider_name()`. Los proveedores no implementan métodos que no usan. |
| **D** — Inversión de dependencias | `ChatbotAgent` depende de `BaseLLMProvider` (abstracta) y `RAGRetriever`, no de implementaciones concretas. La inyección se hace desde `dependencies.py`. |

### Patrones de diseño

| Patrón | Dónde |
|---|---|
| Factory | `LLMFactory.create_provider()` — crea proveedores sin exponer clases concretas. |
| Strategy | Cada `Provider` concreto es una estrategia de invocación LLM. |
| Singleton / Pool | `get_vector_store()`, `get_session_manager()` con lazy init; `_create_cached_llm_provider` con `lru_cache`. |
| Template Method | `BaseLLMProvider.__init__()` valida params antes de que la subclase construya el cliente. |
| Writable Fallback | `VectorStore._resolve_persist_directory()` y `SessionManager._resolve_db_path()` — fallback a copia escribible. |

### Principios de calidad (ISO/IEC 25010)

| Característica | Mecanismo |
|---|---|
| **Funcionalidad** | Cobertura de frameworks ágiles (Scrum, Kanban). |
| **Usabilidad** | Prompt en español, nivel adaptativo (1–4), sin mencionar artefactos internos. |
| **Confiabilidad** | Threading lock en SQLite, writable-fallback para permisos, manejo de errores en todos los endpoints. |
| **Mantenibilidad** | `models.json` editable sin código, separación de capas, tests unitarios e integración. |
| **Portabilidad** | Docker Compose; soporte de Ollama local en el código para ejecución sin cloud (servicio comentado por defecto, descomentar para activarlo). |
| **Seguridad** | PBKDF2-HMAC-SHA256 para contraseñas, tokens almacenados como hash SHA256, ownership de sesiones. |
| **Eficiencia** | `lru_cache` para proveedores y retrievers, count cache en VectorStore, context window limitado a 12 mensajes. |

### Restricciones del prompt (disciplina pedagógica)

El `BASE_SYSTEM_PROMPT` impone tres reglas no negociables que varios tests verifican:

1. **Español siempre** — ninguna respuesta debe estar en otro idioma.
2. **Sin revelar fuentes** — nunca mencionar "contexto", "documentos", ni frases como "según el contexto".
3. **Sin contaminación entre frameworks** — no usar conocimiento de Kanban para responder preguntas específicas de Scrum (roles, eventos, artefactos) y viceversa, salvo que la evidencia lo respalde explícitamente.

---

## 12. Estructura de archivos completa

```
chatbot-agil/
├── src/
│   ├── __init__.py                    # __version__ = "1.0.0"
│   ├── main.py                        # FastAPI app + lifespan + routers
│   │
│   ├── agents/
│   │   └── chatbot_agent.py           # ChatbotAgent (orquestador)
│   │
│   ├── api/
│   │   ├── models.py                  # Todos los esquemas Pydantic v2
│   │   ├── dependencies.py            # DI, singletons, caché lru_cache
│   │   └── routes/
│   │       ├── auth.py                # /auth/register, /auth/login, /auth/me
│   │       ├── chat.py                # /chat (sync + SSE)
│   │       ├── sessions.py            # /sessions CRUD + history
│   │       ├── documents.py           # /documents upload/list/delete
│   │       ├── config.py              # /config GET + POST
│   │       ├── health.py              # /health
│   │       ├── forms.py               # /forms/start + /forms/answer (agile_adoption_assessment)
│   │       ├── debug.py               # /debug/rag (solo dev)
│   │       └── admin.py               # /admin/users (GET, DELETE — solo admins)
│   │
│   ├── core/
│   │   ├── config.py                  # Settings (pydantic-settings)
│   │   ├── logger.py                  # get_logger() con loguru
│   │   ├── prompt_manager.py          # PromptManager + BASE_SYSTEM_PROMPT
│   │   ├── question_classifier.py     # classify_question()
│   │   ├── security.py                # hash_password, tokens, assess_agile_level
│   │   └── forms/
│   │       ├── form_manager.py        # FormManager + FormSpec
│   │       ├── default_forms.py       # agile_adoption_assessment
│   │       └── validators.py          # Validadores de campo y cruzados
│   │
│   ├── llm/
│   │   ├── base.py                    # BaseLLMProvider (ABC)
│   │   ├── factory.py                 # LLMFactory + auto-registro
│   │   ├── models.json                # Catálogo de modelos (editable)
│   │   └── providers/
│   │       ├── openai_provider.py     # OpenAIProvider → ChatOpenAI
│   │       ├── anthropic_provider.py  # AnthropicProvider → ChatAnthropic
│   │       ├── google_provider.py     # GoogleProvider → ChatGoogleGenerativeAI
│   │       ├── deepseek_provider.py   # DeepseekProvider → ChatOpenAI (DeepSeek)
│   │       └── ollama_provider.py     # OllamaProvider → OllamaLLM (langchain_ollama)
│   │
│   ├── rag/
│   │   ├── vector_store.py            # VectorStore (ChromaDB + HuggingFace)
│   │   ├── document_processor.py      # DocumentProcessor (loaders + splitter)
│   │   └── retriever.py               # RAGRetriever (combined search + scoring)
│   │
│   ├── evaluation/
│   │   ├── dataset.py                 # EvalQuestion/RelevantChunkDescriptor + load_eval_dataset()
│   │   ├── retrieval_metrics.py       # Context Precision/Recall, MRR (deterministas)
│   │   └── judge.py                   # LLM-as-judge: Faithfulness, Answer Relevancy, Hallucination Rate
│   │
│   ├── memory/
│   │   ├── session_manager.py         # SessionManager (SQLite, all tables)
│   │   ├── concept_tracker.py         # extract_concepts, build_history_note
│   │   ├── form_state.py              # Estado del cuestionario ágil (en memoria)
│   │   └── form_state_session.py      # Estado del cuestionario ágil (SQLite)
│   │
│   ├── utils/
│   │   └── file_utils.py              # Utilidades de archivos
│   │
│   └── frontend/
│       ├── templates/index.html       # SPA React (servida por FastAPI en /app)
│       └── static/                    # CSS, JS compilados
│
├── data/
│   ├── uploads/
│   │   └── global/                    # Documentos globales (reindexados en arranque)
│   └── conversations.db               # SQLite (creado en runtime)
│
├── chroma_db/                         # Índice vectorial persistente
│
├── tests/                             # Suite de pruebas (pytest)
│   ├── conftest.py                    # Fixtures compartidos
│   ├── data/
│   │   └── agile_rag_eval_dataset.json # Dataset de verdad fundamental (16 preguntas, RAG_EVALUATION.md)
│   ├── test_api.py                    # Tests de integración de endpoints
│   ├── test_admin.py                  # Gestión de usuarios admin
│   ├── test_auth.py                   # Seguridad, CRUD de usuarios, login, personalización
│   ├── test_audit_smoke.py            # Smoke tests de la superficie crítica de la API
│   ├── test_document_processor.py     # DocumentProcessor (loaders + splitter)
│   ├── test_evaluation_metrics.py     # Métricas de retrieval y LLM judge (src/evaluation/)
│   ├── test_iso25010_quality.py       # Suite de evaluación de calidad ISO/IEC 25010
│   ├── test_llm_factory.py            # LLMFactory, incluyendo el pilot lock
│   ├── test_performance_pipeline.py   # Tests de rendimiento del pipeline real
│   ├── test_question_classifier.py    # Clasificación de preguntas y enrutamiento de prompts
│   ├── test_rag_retriever.py          # RAGRetriever (comportamiento dual-scope)
│   ├── test_session_concepts.py       # Seguimiento de conceptos repetidos en sesión
│   └── test_source_citations.py       # Normalización de citas de fuentes
│
├── scripts/                            # Utilidades de línea de comandos
│   ├── evaluate_rag_quality.py        # CLI de evaluación offline de calidad RAG (RAG_EVALUATION.md)
│   ├── dump_sqlite.py                 # Exporta una base SQLite (esquema + datos) a .sql
│   └── dump_sqlite_schema.py          # Exporta solo el esquema de una base SQLite a .sql
│
├── docs/                              # Documentación técnica
├── main.py                            # Entrypoint legacy: re-exporta src.main:app
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example

```
