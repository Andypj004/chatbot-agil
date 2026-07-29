# Guía de inicio rápido

## Prerequisitos

- Python 3.14 o superior.
- Al menos una clave de API de proveedor LLM (OpenAI, Anthropic, Google, DeepSeek) **o** Ollama corriendo localmente.
- Si trabajas en WSL: crea y activa el entorno virtual desde dentro de WSL.

> **Python 3.14 sin `pip`:** algunas distribuciones (p. ej. Debian/WSL) instalan
> `python3.14` sin `pip` ni `ensurepip`. Si `python -m venv .venv` crea un entorno sin
> `pip`, créalo con `python3.14 -m venv --without-pip .venv` y luego instala `pip` con
> [`get-pip.py`](https://bootstrap.pypa.io/get-pip.py):
>
> ```bash
> curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
> .venv/bin/python /tmp/get-pip.py
> .venv/bin/python -m pip install --upgrade pip setuptools wheel
> ```

---

## Instalación local paso a paso

### Paso 1 — Clonar y entrar al proyecto

```bash
git clone https://github.com/Andypj004/chatbot-agil.git
cd chatbot-agil
```

### Paso 2 — Crear entorno virtual

```bash
# Linux / WSL
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Paso 3 — Instalar dependencias

```bash
pip install -r requirements.txt
```

> La primera instalación descarga el modelo de embedding (`paraphrase-multilingual-MiniLM-L12-v2`, ~500 MB). Es una operación única.

### Paso 4 — Crear el archivo `.env`

```bash
cp .env.example .env
```

Editar `.env` con al menos una clave de API:

```env
# OpenAI (recomendado para empezar)
OPENAI_API_KEY=sk-...

# Proveedor por defecto
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-4-turbo-preview
```

Para una lista completa de variables, ver [DEPLOYMENT.md](DEPLOYMENT.md#variables-de-entorno).

### Paso 5 — Levantar la API

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Esperar hasta ver en la consola:
```
INFO: Application startup complete.
```

### Paso 6 — Verificar

| URL | Descripción |
|---|---|
| `http://localhost:8000/app` | Interfaz web |
| `http://localhost:8000/docs` | Swagger UI interactivo |
| `http://localhost:8000/api/v1/health` | Estado del sistema |

---

## Primer chat (sin autenticación)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"¿Qué es un Sprint Goal?","use_rag":false}'
```

Respuesta esperada:

```json
{
  "session_id": "...",
  "response": "El Sprint Goal es el objetivo acordado...",
  "provider": "openai",
  "model": "gpt-4-turbo-preview",
  "used_rag": false,
  "sources": []
}
```

---

## Registro y login

### Registrar usuario

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alumno@correo.com",
    "password": "mipassword123",
    "full_name": "Ana García",
    "account_type": "Estudiante",
    "knowledge_level": 2,
    "questionnaire_answers": [
      {"question_number": 1, "answer": "b"},
      {"question_number": 2, "answer": "c"},
      {"question_number": 3, "answer": "a"},
      {"question_number": 4, "answer": "b"},
      {"question_number": 5, "answer": "c"}
    ]
  }'
```

El campo `questionnaire_answers` es opcional y evalúa el nivel de adopción ágil.

Guardar el `access_token` de la respuesta para usarlo en las siguientes requests.

### Login

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"alumno@correo.com","password":"mipassword123"}'
```

---

## Agregar documentos al RAG

### Documentos globales

```bash
# Subir un PDF al RAG global
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@guia_scrum.pdf"
```

### Consultar con RAG activo

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"¿Cuáles son los eventos de Scrum?","use_rag":true}'
```

La respuesta incluirá `"used_rag": true` y el campo `sources` con las fuentes utilizadas.

---

## Sesiones con historial

```bash
SESSION_ID="mi-sesion-1"

# Primera pregunta
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"Explícame Kanban\",\"session_id\":\"$SESSION_ID\"}"

# Segunda pregunta (continúa la sesión)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"¿Cómo difiere del sistema Pull en Scrum?\",\"session_id\":\"$SESSION_ID\"}"

# Ver historial
curl http://localhost:8000/api/v1/sessions/$SESSION_ID/history
```

---

## Streaming de respuestas

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Explica el manifiesto ágil en detalle","stream":true}' \
  --no-buffer
```

Los eventos llegan como `data: {...}` de tipo `delta` (fragmentos) y `final` (metadatos).

---

## Cuestionario de Evaluación Ágil

```bash
SESSION_ID="sesion-evaluacion-1"

# Iniciar cuestionario
curl -X POST http://localhost:8000/api/v1/forms/start \
  -H "Content-Type: application/json" \
  -d "{\"form_id\":\"agile_adoption_assessment\",\"session_id\":\"$SESSION_ID\"}"

# Responder pregunta
curl -X POST http://localhost:8000/api/v1/forms/answer \
  -H "Content-Type: application/json" \
  -d "{\"form_id\":\"agile_adoption_assessment\",\"session_id\":\"$SESSION_ID\",\"field\":\"question_1\",\"value\":\"b\"}"
```

---

## Cambiar proveedor/modelo LLM

Los selects de proveedor y modelo del chat se pueblan con `GET /api/v1/config` → `available_providers` / `available_models`. Un proveedor aparece ahí si está registrado en `LLMFactory` **y** tiene una API key utilizable en `.env`; Ollama, que no necesita key, aparece sólo si su servidor responde. Cada mensaje puede usar el proveedor/modelo elegido en el toolbar.

```bash
# Qué se puede elegir ahora mismo
curl -s http://localhost:8000/api/v1/config | jq '{available_providers, available_models}'
```

`POST /api/v1/config` (solo admin) cambia el default global; acepta cualquier proveedor disponible:

```bash
curl -X POST http://localhost:8000/api/v1/config \
  -H "Content-Type: application/json" -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"llm_provider":"anthropic","model_name":"claude-3-5-haiku-latest"}'
```

Pedir un proveedor sin key disponible responde `400`; pedir un modelo fuera del catálogo del proveedor cae al modelo default sin error.

### Modo piloto (`PILOT_LOCK`)

Con `PILOT_LOCK=true` en `.env` (por defecto `false`), solo el proveedor y modelo de `DEFAULT_LLM_PROVIDER`/`DEFAULT_MODEL` quedan expuestos y seleccionables: los selects muestran una única opción, `POST /api/v1/config` con otro proveedor responde `400 "Invalid provider"`, y cualquier `llm_provider`/`model_name` distinto enviado en `/api/v1/chat` se ignora en favor del default. Cambiar el proveedor activo requiere entonces editar `.env` y reiniciar el servidor.

---

## Resolución de problemas frecuentes

### Error de permisos en ChromaDB o SQLite

```bash
# Corregir propietario (en Docker)
sudo chown -R $(whoami):$(whoami) data/ chroma_db/

# O limpiar y recrear
rm -rf chroma_db/ data/conversations.db
```

### Ollama no responde

Aplica solo si reactivaste el servicio `ollama` (comentado por defecto en `docker-compose.yml`) y lo configuraste como `DEFAULT_LLM_PROVIDER`:

```bash
# Verificar que Ollama está corriendo
curl http://localhost:11434/api/tags

# Descargar modelo
ollama pull llama3.2:3b
```

### Inspeccionar o respaldar la base SQLite

`scripts/dump_sqlite.py` exporta una base SQLite completa (esquema + datos) a un archivo `.sql`; `scripts/dump_sqlite_schema.py` exporta solo el esquema (tablas, índices, triggers), sin datos:

```bash
python scripts/dump_sqlite.py data/conversations.db
python scripts/dump_sqlite_schema.py data/conversations.db
```

### Modelo no encontrado en el catálogo

Editar `src/llm/models.json` y agregar el modelo al array del proveedor correspondiente. No requiere reiniciar si se hace antes del arranque; si el servidor está corriendo, reiniciarlo.

---

## Lectura siguiente

- [docs/ARCHITECTURE.md](ARCHITECTURE.md) — diseño completo del sistema
- [Swagger UI](http://localhost:8000/docs) — referencia interactiva de todos los endpoints (con el servidor corriendo)
- [docs/DEPLOYMENT.md](DEPLOYMENT.md) — despliegue con Docker y configuración completa
