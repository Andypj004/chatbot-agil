# Guía de Despliegue

## Opciones soportadas

| Opción | Cuándo usarla |
|---|---|
| uvicorn local | Desarrollo, debugging, iteración rápida |
| Docker Compose | Entorno reproducible (Ollama deshabilitado por defecto, reactivable comentando/descomentando el servicio) |
| Docker (solo API) | Despliegue sin Ollama |

---

## Prerequisitos

- **Python 3.14** o superior (para uvicorn local).
- **Docker y Docker Compose** (para la opción Docker).
- La clave de API del proveedor configurado en `DEFAULT_LLM_PROVIDER` (`.env`), o Ollama corriendo localmente si ese es el proveedor default. El pilot lock de `LLMFactory` restringe el sistema a un solo proveedor activo a la vez (ver [docs/ARCHITECTURE.md](ARCHITECTURE.md)).

---

## Opción A — uvicorn local

### 1. Crear entorno virtual

```bash
# Linux / WSL
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

> **WSL:** Siempre crea y activa el entorno desde dentro de WSL. Nunca uses un entorno creado en Windows desde WSL ni viceversa.

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

La primera ejecución descarga el modelo de embedding (`paraphrase-multilingual-MiniLM-L12-v2`, ~500 MB).

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tu editor favorito
```

Mínimo necesario para funcionar:

```env
OPENAI_API_KEY=sk-...          # Si usas OpenAI
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-4-turbo-preview
```

Ver la tabla completa en la sección [Variables de entorno](#variables-de-entorno).

### 4. Levantar la API

```bash
# Modo desarrollo con recarga automática
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Modo producción (sin recarga)
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### 5. Verificar

```bash
curl http://localhost:8000/api/v1/health
# → {"status":"healthy","version":"1.0.0",...}

# Interfaz web
open http://localhost:8000/app

# Swagger UI
open http://localhost:8000/docs
```

---

## Opción B — Docker Compose

El `docker-compose.yml` levanta el servicio de la API. Los servicios de Ollama están **comentados por defecto**:

| Servicio | Imagen | Función | Estado por defecto |
|---|---|---|---|
| `chatbot-agil` | Dockerfile local | API FastAPI | Activo |
| `ollama` | `ollama/ollama:latest` | LLM local | Comentado (descomentar para activar) |
| `ollama-pull` | `ollama/ollama:latest` | Descarga `llama3.2:3b` al inicio | Comentado (descomentar para activar) |

### 1. Configurar `.env`

```bash
cp .env.example .env
# Editar con claves API y configuración deseada
```

Si reactivas el servicio `ollama` descomentándolo en `docker-compose.yml` (junto con la línea `OLLAMA_BASE_URL` y el bloque `depends_on` del servicio `chatbot-agil`), `OLLAMA_BASE_URL` se sobreescribe automáticamente a `http://ollama:11434` (red interna de Docker).

### 2. Construir y levantar

```bash
# Primera vez (descarga imágenes y construye)
docker compose up --build

# Modo background
docker compose up --build -d

# Solo reconstruir la API (sin bajar Ollama)
docker compose build chatbot-agil && docker compose up -d
```

### 3. Comandos útiles

```bash
# Ver logs en tiempo real
docker compose logs -f chatbot-agil

# Ver estado de servicios
docker compose ps

# Detener (mantiene volúmenes)
docker compose down

# Detener y eliminar volúmenes (¡borra chroma_db y datos!)
docker compose down -v

# Reiniciar solo la API
docker compose restart chatbot-agil
```

### 4. Volúmenes persistentes

El `docker-compose.yml` monta estos volúmenes del host:

| Volumen host | Contenedor | Contenido |
|---|---|---|
| `./data` | `/app/data` | SQLite (conversaciones, usuarios) |
| `./logs` | `/app/logs` | Logs de aplicación |
| `./chroma_db` | `/app/chroma_db` | Índice vectorial ChromaDB |

Los directorios se crean automáticamente al arrancar Docker. El volumen `ollama-data` (gestionado por Docker, persiste los modelos descargados) solo aplica si reactivaste el servicio `ollama`, comentado por defecto junto con su declaración de volumen.

---

## Variables de entorno

Todas las variables se leen desde `.env` (o del entorno del sistema). Las variables con valor por defecto no son obligatorias.

### Claves de API (al menos una requerida)

| Variable | Proveedor |
|---|---|
| `OPENAI_API_KEY` | OpenAI (GPT-4, GPT-3.5) |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |
| `GOOGLE_API_KEY` | Google (Gemini) |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `OLLAMA_API_KEY` | Ollama (opcional, local generalmente no lo requiere) |

### Configuración del LLM

| Variable | Default | Descripción |
|---|---|---|
| `DEFAULT_LLM_PROVIDER` | `openai` | Proveedor usado cuando la request no especifica uno |
| `DEFAULT_MODEL` | `gpt-4-turbo-preview` | Modelo usado cuando la request no especifica uno |
| `TEMPERATURE` | `0.5` | Temperatura de muestreo (0.0–1.0) |
| `MAX_TOKENS` | `2000` | Tokens máximos en la respuesta |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL del servidor Ollama |

### Configuración RAG

| Variable | Default | Descripción |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modelo de embedding; cambiarlo crea una nueva colección |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Directorio de persistencia de ChromaDB |
| `CHUNK_SIZE` | `1500` | Tamaño de cada chunk de texto en caracteres |
| `CHUNK_OVERLAP` | `300` | Solapamiento entre chunks |
| `TOP_K_RESULTS` | `8` | Chunks recuperados por búsqueda |
| `RAG_CONTEXT_MAX_CHARS` | `10000` | Límite de caracteres del contexto RAG inyectado al prompt |
| `RAG_MAX_CHUNKS_PER_FILE` | `5` | Máximo de chunks del mismo archivo por búsqueda |

### Persistencia de sesiones

| Variable | Default | Descripción |
|---|---|---|
| `CONVERSATION_DB_PATH` | `./data/conversations.db` | Ruta del archivo SQLite |
| `CONVERSATION_CONTEXT_MESSAGES` | `12` | Mensajes anteriores inyectados en el prompt |
| `CONVERSATION_LIST_LIMIT` | `50` | Máximo de sesiones en el listado |

### API y servidor

| Variable | Default | Descripción |
|---|---|---|
| `API_HOST` | `0.0.0.0` | Host de escucha de uvicorn |
| `API_PORT` | `8000` | Puerto de escucha |
| `API_RELOAD` | `true` | Recarga automática en cambios (solo desarrollo) |
| `CORS_ORIGINS` | `["http://localhost:3000","http://localhost:8000"]` | Orígenes permitidos para CORS |
| `LOG_LEVEL` | `INFO` | Nivel de log (DEBUG, INFO, WARNING, ERROR) |

### Startup / warmup

| Variable | Default | Descripción |
|---|---|---|
| `WARMUP_VECTOR_STORE_ON_STARTUP` | `true` | Inicializa VectorStore al arrancar y reindexea si está vacío |
| `WARMUP_DEFAULT_PROVIDER_ON_STARTUP` | `false` | Pre-instancia el proveedor LLM por defecto al arrancar |

---

## Permisos y patrón writable-fallback

Cuando Docker crea los directorios de datos como `root`, el proceso del host puede perder permisos de escritura. El sistema resuelve esto automáticamente:

| Recurso | Path original | Fallback |
|---|---|---|
| SQLite | `data/conversations.db` | `data/conversations.writable.db` |
| ChromaDB | `chroma_db/` | `chroma_db.writable/` |

Si alternas entre `docker compose up` y `uvicorn` en el host:

```bash
# Opción 1: corregir propietario (requiere sudo)
sudo chown -R $(whoami):$(whoami) ./data ./chroma_db

# Opción 2: dejar que el sistema use el fallback .writable (automático)
# Opción 3: arrancar siempre desde Docker o siempre desde el host
```

---

## Agregar documentos al RAG

### Documentos globales (disponibles en todas las sesiones)

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@mi_guia_scrum.pdf"

# Via filesystem (se reindexan al arrancar si la colección está vacía)
cp mi_guia_scrum.pdf data/uploads/global/
```

### Documentos de sesión

```bash
curl -X POST http://localhost:8000/api/v1/documents/sessions/{session_id}/upload \
  -F "file=@plan_proyecto.pdf"
```

---

## Scripts de utilidad

`scripts/dump_sqlite.py` y `scripts/dump_sqlite_schema.py` exportan una base SQLite a `.sql` para backup o inspección — el primero incluye esquema y datos, el segundo solo el esquema (tablas, índices, triggers):

```bash
python scripts/dump_sqlite.py data/conversations.db
python scripts/dump_sqlite_schema.py data/conversations.db
```

---

## Producción

Para un despliegue de producción:

1. **HTTPS obligatorio** — coloca la app detrás de un reverse proxy (nginx, Caddy) con certificado TLS.
2. **CORS restringido** — limita `CORS_ORIGINS` a los dominios necesarios.
3. **Secretos** — guarda las claves API en un gestor de secretos (AWS Secrets Manager, HashiCorp Vault, etc.), no en `.env` en el sistema.
4. **Workers** — usa `--workers N` con N = número de CPUs para carga concurrente.
5. **Monitoring** — vigila `/api/v1/health`, latencia de respuestas y tamaño del índice vectorial.
6. **Backups** — programa backups periódicos de `data/` y `chroma_db/`.

---

## Troubleshooting

### La API no arranca

```bash
# Verificar que el entorno está activo
which python  # debe apuntar a .venv/bin/python

# Verificar dependencias
pip install -r requirements.txt

# Ver errores de importación
python -c "from src.main import app"
```

### ChromaDB o SQLite con error de solo lectura

```bash
# Ver si existen archivos .writable
ls data/
ls -la chroma_db* 2>/dev/null || echo "no existe"

# Corregir propietario
sudo chown -R $(whoami):$(whoami) data/ chroma_db/

# O limpiar y recrear (pierde los documentos indexados)
rm -rf chroma_db/ data/conversations.db
uvicorn src.main:app --reload
```

### Ollama no responde

Aplica solo si reactivaste los servicios `ollama`/`ollama-pull` (comentados por defecto en `docker-compose.yml`) y configuraste `DEFAULT_LLM_PROVIDER=ollama`:

```bash
# Verificar que Ollama está corriendo
curl http://localhost:11434/api/tags

# Descargar el modelo si falta
ollama pull llama3.2:3b

# Si usas Docker, verificar que ollama-pull completó
docker compose logs ollama-pull
```

### El modelo de embedding falla al cargar

```bash
# Verificar versiones de torch y transformers
pip show torch transformers sentence-transformers

# Reinstalar forzando versiones del requirements.txt
pip install --force-reinstall -r requirements.txt
```

### Documentos no aparecen en RAG

```bash
# Ver cuántos documentos tiene el índice
curl http://localhost:8000/api/v1/health
# Busca "vector_store_documents"

# Listar documentos indexados
curl http://localhost:8000/api/v1/documents

# Forzar reindexado: borra la colección y reinicia
rm -rf chroma_db/
uvicorn src.main:app --reload
```
