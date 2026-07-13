# Evaluación de Calidad del Pipeline RAG

## Índice

1. [Objetivo](#1-objetivo)
2. [Por qué se descartó ragas](#2-por-qué-se-descartó-ragas)
3. [Metodología del conjunto de verdad fundamental (ground truth)](#3-metodología-del-conjunto-de-verdad-fundamental-ground-truth)
4. [Definiciones de métricas](#4-definiciones-de-métricas)
5. [El prompt del juez](#5-el-prompt-del-juez)
6. [Cómo ejecutar](#6-cómo-ejecutar)
7. [Cómo interpretar los resultados](#7-cómo-interpretar-los-resultados)
8. [Limitaciones](#8-limitaciones)
9. [Código](#9-código)

---

## 1. Objetivo

Este documento describe el arnés (harness) de evaluación de calidad del pipeline RAG (Retrieval-Augmented Generation) del chatbot de prácticas ágiles. El objetivo es medir, de forma reproducible y con un conjunto de verdad fundamental verificado manualmente, qué tan bien recupera el sistema los fragmentos de documentos relevantes para una pregunta (métricas de recuperación, deterministas) y qué tan bien genera respuestas fundamentadas y relevantes a partir de ese contexto (métricas de generación, evaluadas mediante un LLM como juez).

El arnés está compuesto por dos capas independientes:

- Una capa de pruebas rápidas y deterministas (`pytest`), sin llamadas a red, que valida la lógica de las métricas y del parseo de la respuesta del juez con datos sintéticos.
- Un script de evaluación manual (`scripts/evaluate_rag_quality.py`) que ejecuta el pipeline real contra ChromaDB y, opcionalmente, contra el LLM configurado en `.env`, generando reportes en JSON y Markdown.

## 2. Por qué se descartó ragas

Se evaluó usar la librería `ragas` para esta tarea, pero se descartó porque sus métricas de recuperación y generación dependen de llamadas a un LLM externo para casi todo (incluyendo la propia descomposición de la pregunta en "claims"), lo cual introduce no determinismo y costo en pruebas que deberían poder ejecutarse en CI sin red. Además, su acoplamiento a la versión de LangChain y a wrappers específicos de embeddings/LLM complicaba la integración con la abstracción de proveedores ya existente en este proyecto (`src/llm/factory.py`). Se optó por implementar métricas de recuperación propias, deterministas y basadas en texto, y reservar el LLM-como-juez únicamente para las métricas de generación que genuinamente lo requieren.

## 3. Metodología del conjunto de verdad fundamental (ground truth)

El conjunto de verdad fundamental vive en `tests/data/agile_rag_eval_dataset.json` y contiene actualmente **16 preguntas** repartidas en **8 subtemas** (2 preguntas por subtema): `scrum`, `kanban`, `historias_usuario`, `retrospectivas`, `metricas_agiles`, `roles`, `eventos` y `artefactos`.

### Esquema de cada pregunta

```json
{
  "id": "scrum-01",
  "subtopic": "scrum",
  "question": "¿Qué es un Sprint en Scrum?",
  "reference_answer": "Un Sprint es un evento de duración fija de un mes o menos...",
  "relevant_chunks": [
    {
      "source_contains": "The Scrum Guide",
      "anchor_contains": ["Sprints are the heartbeat of Scrum, where ideas are turned into value."]
    },
    {
      "source_contains": "Essential Scrum",
      "anchor_contains": ["Scrum organizes work in iterations or cycles of up to a calendar month called sprints"]
    }
  ]
}
```

Campos:

- `id`: identificador único de la pregunta (`<subtopic>-NN`).
- `subtopic`: subtema agile al que pertenece, usado para agregar métricas por categoría.
- `question`: la pregunta en español que se envía al pipeline RAG.
- `reference_answer`: una respuesta de referencia redactada por un humano, usada como insumo opcional para el juez (no se compara textualmente, solo se incluye en el prompt).
- `relevant_chunks`: lista de **descriptores** (no IDs de chunk) que identifican qué documento(s) recuperado(s) cuentan como relevantes para esta pregunta. Cada descriptor tiene:
  - `source_contains`: una subcadena que debe aparecer en el nombre de archivo (`filename`) o, si no existe, en el metadato `source` del documento recuperado.
  - `anchor_contains`: una lista de subcadenas que deben aparecer **todas** en el contenido del chunk recuperado.

Un documento recuperado se considera relevante para una pregunta si **algún** descriptor de su lista de `relevant_chunks` matchea completamente (lógica OR entre descriptores, AND entre `source_contains` y todos los `anchor_contains` dentro de un mismo descriptor). La comparación de texto se hace sobre versiones normalizadas (ver `normalize_text` en `src/evaluation/retrieval_metrics.py`): minúsculas, normalización Unicode NFKD con remoción de tildes/diacríticos, eliminación de cualquier carácter que no sea alfanumérico o espacio (puntuación, paréntesis, etc.), y colapso de espacios — por lo que el matching es insensible a mayúsculas, acentos y puntuación.

### Por qué no se usa `chunk_id`

El dataset deliberadamente no referencia los chunks por un identificador interno de ChromaDB (`chunk_id`) o por número de página/índice de chunk. La razón es que esos identificadores son inestables frente a cualquier cambio en el pipeline de ingestión: si se ajusta el tamaño de chunk, el `chunk_overlap`, el splitter, o simplemente se reingestan los documentos, los IDs y offsets cambian aunque el contenido relevante siga existiendo en el corpus. Usar descriptores de texto (`source_contains` + `anchor_contains`) hace que el ground truth sea estable mientras el contenido fuente no cambie, independientemente de cómo se haya chunkeado o de qué ID interno le haya asignado ChromaDB al fragmento.

### Cómo se eligieron y verificaron las 16 preguntas

Las preguntas se diseñaron para cubrir los subtemas centrales del corpus de 14 PDFs ya ingeridos (la Scrum Guide, Essential Scrum, el glosario asociado, la Kanban Guide, y la guía condensada de Kanban en español, entre otros). Cada pregunta tiene **dos descriptores** en `relevant_chunks`, normalmente provenientes de dos documentos distintos del corpus que abordan la misma pregunta desde ángulos o fuentes diferentes (por ejemplo, la definición canónica de "The Scrum Guide" junto con una explicación complementaria de "Essential Scrum"). Cada `anchor_contains` fue verificado manualmente contra el índice real de ChromaDB: se confirmó que la frase ancla aparece literalmente (tras normalización) en el contenido de un chunk real del corpus, y que `source_contains` matchea el nombre de archivo real del PDF ingerido. Esto significa que el ground truth no es hipotético — fue calibrado contra el corpus real, no contra una versión idealizada del contenido. No todos los descriptores rankean igual de alto en la búsqueda por similitud: algunos se verificaron como genuinamente relevantes y presentes en el corpus aun sabiendo que, en una corrida con `k` pequeño, podrían no llegar a aparecer entre los documentos recuperados (ver sección 8).

## 4. Definiciones de métricas

### Métricas de recuperación (deterministas)

Implementadas en `src/evaluation/retrieval_metrics.py`, no requieren ningún LLM — son funciones puras sobre los documentos recuperados y los descriptores de ground truth.

**Context Precision**

Fracción de los documentos recuperados que son relevantes (matchean al menos un descriptor):

```
context_precision = (# documentos recuperados relevantes) / (# documentos recuperados)
```

Caso borde: si no se recuperó ningún documento (`retrieved_docs` vacío), el valor es `0.0`.

**Techo estructural de Context Precision con el dataset actual**: las 16 preguntas del dataset tienen exactamente **dos descriptores** en `relevant_chunks` (dos chunks "dorados" marcados y verificados a mano por pregunta, cada uno en un documento distinto cuando es posible). Con `k` documentos recuperados y `m` chunks dorados distintos presentes en el top-k (`m` puede ser `0`, `1` o `2`), el máximo valor alcanzable de `context_precision` es `m/k` — los documentos recuperados que no matchean ninguno de los descriptores nunca contarán como relevantes según esta métrica, aun cuando en la práctica traten el mismo tema. Con `k=8` (`TOP_K_RESULTS=8` en `.env`), `context_precision` solo puede tomar los valores `0.0`, `0.125` (`1/8`, un solo chunk dorado encontrado) o `0.25` (`2/8`, ambos chunks dorados encontrados) en este dataset — nunca un valor intermedio ni mayor a `0.25`. Esto no es indicio de que el retriever traiga contenido irrelevante; es una consecuencia directa de tener un número fijo y pequeño de positivos verdaderos por pregunta. Ver la sección 8 para un ejemplo numérico real.

**Context Recall**

Fracción de los **descriptores** de ground truth que fueron cubiertos por al menos un documento recuperado:

```
context_recall = (# descriptores cubiertos por algún doc recuperado) / (# descriptores totales)
```

Aclaración de granularidad: el recall se calcula por **descriptor**, no por documento recuperado. Las 16 preguntas del dataset tienen dos descriptores cada una en `relevant_chunks`, así que el recall de cada pregunta solo puede ser `0.0` (ningún descriptor cubierto), `0.5` (uno de los dos cubierto) o `1.0` (ambos cubiertos) — no importa cuántos documentos irrelevantes se recuperaron también, ni importa si cada descriptor fue matcheado por el primer o el último documento de la lista. El recall decae específicamente cuando alguno de los dos descriptores no es cubierto por ningún documento recuperado.

Caso borde documentado explícitamente en el código: si la lista de descriptores está vacía (`descriptors == []`), `context_recall` devuelve `1.0` por **verdad vacua** (no hay nada que cubrir, por lo tanto está "todo" cubierto):

```python
def context_recall(retrieved_docs, descriptors) -> float:
    if not descriptors:
        return 1.0
    ...
```

**Mean Reciprocal Rank (MRR)**

Para una pregunta individual, es el inverso de la posición (1-indexada) del primer documento relevante en la lista de recuperados:

```
mrr(pregunta) = 1 / rank(primer_doc_relevante)
```

Si ningún documento recuperado es relevante, o la lista está vacía, el valor es `0.0`. La agregación entre preguntas (`aggregate_mrr`) es la media aritmética simple de los MRR por pregunta; con lista vacía de valores también devuelve `0.0`.

**Cómo leer un MRR global "bajo"**: conviene descomponerlo en dos factores antes de interpretarlo como un problema de ranking. (1) La fracción de preguntas para las que ninguno de los chunks dorados aparece en el top-k (esas preguntas aportan `0.0` al promedio y son indistinguibles, en el agregado, de un acierto en el último puesto). (2) El MRR promedio condicionado *solo* a las preguntas donde sí se encontró al menos un chunk dorado, que indica en qué posición típica aparece cuando la recuperación funciona. Un MRR global moderado puede ocultar una recuperación con buen ranking (MRR condicional alto) arrastrada hacia abajo simplemente por una tasa de aciertos incompleta — son dos causas distintas con remedios distintos (subir `k`/mejorar el embedding vs. ajustar el orden de ranking). Ver la sección 8 para un ejemplo numérico real con los resultados de la corrida completa.

### Métricas de generación (basadas en juez LLM)

Implementadas en `src/evaluation/judge.py`, requieren invocar un LLM (el configurado vía `LLMFactory`) que actúa como evaluador imparcial.

- **Faithfulness** (Fidelidad): ¿la respuesta generada está fundamentada y es consistente con el contexto recuperado?
- **Answer Relevancy** (Relevancia de la respuesta): ¿la respuesta responde directa y completamente a la pregunta?
- **Hallucination Rate** (Tasa de alucinación): ¿qué proporción de la respuesta contiene afirmaciones no verificables contra el contexto ni respaldadas por conocimiento general ampliamente aceptado sobre metodologías ágiles?

Cada una se puntúa en `[0.0, 1.0]` con decimales permitidos, junto con una justificación textual en español.

**Por qué Faithfulness/Answer Relevancy/Hallucination Rate son evaluadas por el juez, y Context Precision/Recall no lo son:** Context Precision y Context Recall son medibles de forma objetiva y determinista porque dependen únicamente de si ciertas subcadenas de texto (ya conocidas de antemano en el ground truth) aparecen en los documentos recuperados — no requieren juicio semántico. Faithfulness, Answer Relevancy y Hallucination Rate, en cambio, requieren evaluar si una respuesta en lenguaje natural generada por el LLM es consistente, completa o inventada respecto a un contexto también en lenguaje natural — esto es un juicio semántico que no puede reducirse a coincidencia de subcadenas, y por eso se delega a un LLM evaluador. El módulo del juez documenta explícitamente esta separación de responsabilidades para evitar que existan dos números distintos con el mismo nombre:

> "Context Precision/Recall are intentionally NOT requested from the judge — those remain deterministic metrics from `retrieval_metrics.py`, to avoid having two different numbers sharing the same name." (`src/evaluation/judge.py`)

## 5. El prompt del juez

El siguiente es el contenido **verbatim** de `JUDGE_PROMPT_TEMPLATE` en `src/evaluation/judge.py`, con los cuatro placeholders (`{question}`, `{context}`, `{generated_answer}`, `{reference_answer}`) sustituidos en tiempo de ejecución por `build_judge_prompt()`:

```
Eres un evaluador experto e imparcial de sistemas de respuesta a preguntas basados en recuperación de información (RAG) sobre metodologías ágiles.

Se te proporciona:
1. Una PREGUNTA realizada por un estudiante.
2. El CONTEXTO recuperado por el sistema (fragmentos de documentos) que se usó para generar la respuesta.
3. La RESPUESTA GENERADA por el sistema.
4. Opcionalmente, una RESPUESTA DE REFERENCIA (elaborada por un experto humano) para comparación.

Evalúa la RESPUESTA GENERADA según los siguientes tres criterios, cada uno con un puntaje entre 0.0 y 1.0 (puedes usar decimales), y una breve justificación en español para cada uno:

- "faithfulness" (Fidelidad): ¿La respuesta generada está fundamentada y es consistente con la información presente en el CONTEXTO? 1.0 = todas las afirmaciones están respaldadas por el contexto; 0.0 = la respuesta contradice o no tiene relación alguna con el contexto.
- "answer_relevancy" (Relevancia de la respuesta): ¿La respuesta generada responde directamente y de forma completa a la PREGUNTA realizada? 1.0 = responde completa y directamente; 0.0 = no responde a la pregunta o es irrelevante.
- "hallucination_rate" (Tasa de alucinación): ¿Qué proporción de la respuesta contiene afirmaciones que NO se pueden verificar con el CONTEXTO ni son hechos generales y ampliamente aceptados sobre metodologías ágiles? 1.0 = la respuesta está plagada de afirmaciones inventadas o no verificables; 0.0 = no hay ninguna afirmación inventada.

PREGUNTA:
{question}

CONTEXTO RECUPERADO:
{context}

RESPUESTA GENERADA:
{generated_answer}

RESPUESTA DE REFERENCIA (puede estar vacía):
{reference_answer}

Responde EXCLUSIVAMENTE con un objeto JSON válido, sin texto adicional antes ni después, sin bloques de código markdown, con exactamente esta estructura:

{
  "faithfulness": {"score": 0.0, "justification": "..."},
  "answer_relevancy": {"score": 0.0, "justification": "..."},
  "hallucination_rate": {"score": 0.0, "justification": "..."}
}
```

La respuesta cruda del LLM se parsea de forma defensiva en `parse_judge_response()`: se eliminan bloques de código Markdown (` ```json ... ``` `) si están presentes, se intenta `json.loads` directo, y si falla se extrae por regex el primer bloque `{...}` y se reintenta. Los puntajes fuera de `[0, 1]` se recortan (clamp) a ese rango, los campos faltantes reciben puntaje `0.0` y una justificación sintética, y si el parseo falla por completo se devuelve un `JudgeResult` con las tres puntuaciones en `0.0` y el campo `parse_error` poblado — la función nunca lanza una excepción.

## 6. Cómo ejecutar

### Capa 1: pruebas rápidas (sin costo, sin red)

```bash
pytest tests/test_evaluation_metrics.py -v
```

Esta capa valida la lógica de `context_precision`, `context_recall`, `mean_reciprocal_rank`, `aggregate_mrr`, `is_relevant_document`, `build_judge_prompt` y `parse_judge_response` con documentos y respuestas de juez sintéticos (LLMs simulados mediante dobles de prueba `_DummyLLM`/`_DummyProvider`/`_PlainStringLLM`). No realiza ninguna llamada a ChromaDB ni a ningún proveedor LLM real, por lo que es gratuita y determinista — apta para ejecutarse en CI.

### Capa 2: pipeline real (con costo de API)

```bash
python scripts/evaluate_rag_quality.py --mode both --k 8 \
    --dataset tests/data/agile_rag_eval_dataset.json \
    --output reports/rag_evaluation/
```

**Advertencia de costo y API key**: este script ejecuta consultas reales contra el ChromaDB ya poblado y, en los modos `generation` y `both`, realiza llamadas reales al proveedor LLM configurado (el mismo `LLMFactory.create_provider()` usado por el resto de la aplicación, que lee `default_llm_provider`/`default_model` y la API key correspondiente desde `.env`). Esto consume cuota/tokens de la API configurada — no es una prueba gratuita ni offline, y requiere que `.env` tenga una API key válida para el proveedor activo.

Argumentos del CLI (definidos en `parse_args()` de `scripts/evaluate_rag_quality.py`):

| Flag | Tipo / opciones | Default | Descripción |
|---|---|---|---|
| `--mode` | `retrieval` \| `generation` \| `both` | `both` | Qué fases ejecutar. |
| `--k` | `int` | `None` (cae a `settings.top_k_results`, que por defecto es `8`) | Número de documentos a recuperar por pregunta. |
| `--dataset` | ruta | `tests/data/agile_rag_eval_dataset.json` | Archivo JSON del dataset de evaluación. |
| `--output` | ruta de directorio | `reports/rag_evaluation/` | Directorio donde se escriben los reportes generados. |
| `--no-markdown` | flag (booleano) | `False` (es decir, por defecto SÍ se escribe el Markdown) | Si se pasa, omite la escritura del reporte Markdown legible. |
| `--question-id` | string | `None` | Filtra a una sola pregunta por su `id`, útil para iterar de forma económica sobre un caso puntual. |

El script siempre escribe un reporte JSON en `<output>/eval_<timestamp>.json` (timestamp con formato `%Y%m%d_%H%M%S`), y además un reporte Markdown en `<output>/eval_<timestamp>.md` salvo que se pase `--no-markdown`. El directorio por defecto `reports/` está excluido de git (`.gitignore`), por lo que estos reportes no se versionan.

Ejemplo para iterar rápido sobre una sola pregunta solo en la fase de recuperación (sin gastar tokens del LLM generador/juez):

```bash
python scripts/evaluate_rag_quality.py --mode retrieval --question-id scrum-01
```

## 7. Cómo interpretar los resultados

Tanto el reporte JSON como el Markdown comparten la misma estructura lógica:

- **Metadatos de la corrida**: `timestamp`, `mode`, `k`, `dataset_path`, `num_questions`, y (si se ejecutó la fase de generación) `judge_parse_error_count`.
- **Agregados de recuperación** (`retrieval.aggregate`), si `mode` incluye `retrieval`: media de `context_precision`, `context_recall` y `mrr` a nivel global (`overall`) y por subtema (`by_subtopic`).
- **Agregados de generación** (`generation.aggregate`), si `mode` incluye `generation`: media de `faithfulness`, `answer_relevancy` y `hallucination_rate`, también `overall` y `by_subtopic`.
- **Detalle por pregunta** (`questions`): para cada pregunta, su `id`, `subtopic`, el texto de la pregunta, y según el modo, el bloque `retrieval` (métricas + `num_retrieved` + `retrieved_sources`) y/o el bloque `generation` (la respuesta generada, las tres puntuaciones del juez con sus justificaciones, y `parse_error` si el parseo de esa pregunta falló).

El reporte Markdown presenta lo mismo en formato legible: tablas para los agregados globales y por subtema, y una sección "Per-Question Detail" con la pregunta, las métricas de recuperación/generación y la respuesta generada completa.

Interpretación de los valores (todos en `[0, 1]`, salvo `mrr` que también está en `[0,1]` por construcción ya que es `1/rank`):

| Métrica | Valor alto (≈1.0) significa | Valor bajo (≈0.0) significa |
|---|---|---|
| `context_precision` | La mayoría de los documentos recuperados son relevantes; poco "ruido" en el contexto. | Se está recuperando mucho contenido irrelevante junto con (o en lugar de) lo relevante. |
| `context_recall` | El/los descriptor(es) de ground truth de la pregunta fueron encontrados entre los documentos recuperados. | El sistema de recuperación no trajo el chunk relevante esperado para esta pregunta. |
| `mrr` | El primer documento relevante aparece muy arriba en el ranking (idealmente en la posición 1). | El documento relevante, si aparece, está enterrado lejos en el ranking, o no aparece en absoluto (`0.0`). |
| `faithfulness` | La respuesta generada está bien fundamentada en el contexto recuperado, sin contradicciones. | La respuesta contradice el contexto o no guarda relación con él. |
| `answer_relevancy` | La respuesta aborda directa y completamente la pregunta formulada. | La respuesta es tangencial, incompleta o no responde lo que se preguntó. |
| `hallucination_rate` | (Cuidado: aquí alto es MALO) Gran parte de la respuesta son afirmaciones inventadas o no verificables. | No hay afirmaciones inventadas; todo lo dicho es verificable contra el contexto o es conocimiento general aceptado. |

Una corrida saludable debería mostrar `context_precision`/`context_recall`/`mrr`/`faithfulness`/`answer_relevancy` altos y `hallucination_rate` bajo, de forma consistente entre subtemas. Un `judge_parse_error_count` mayor a cero indica que, para esa(s) pregunta(s), la respuesta cruda del LLM juez no pudo interpretarse como el JSON esperado (las tres puntuaciones de esas preguntas quedan en `0.0` con `parse_error` poblado) — esto es una falla del formato de salida del juez, no necesariamente una mala calidad real de la respuesta evaluada, y conviene revisarlo manualmente antes de sacar conclusiones agregadas.

## 8. Limitaciones

- **Subjetividad y no determinismo del juez**: las métricas de generación (`faithfulness`, `answer_relevancy`, `hallucination_rate`) dependen de un LLM evaluando texto en lenguaje natural; dos corridas con el mismo prompt pueden producir puntuaciones ligeramente distintas. Esto se ve acentuado por `settings.temperature` (definido en `src/core/config.py`, por defecto `0.7`), que también afecta la llamada al juez en `invoke_judge()` — no hay ningún override de temperatura específico para el juez que la fuerce a `0` o a un valor determinista, por lo que el juez hereda la misma temperatura configurada para el resto del chatbot.
- **Inestabilidad de `chunk_id` / necesidad de re-verificación manual**: como se explica en la sección 3, el ground truth usa descriptores de texto en vez de IDs de chunk precisamente para evitar acoplarse a una configuración de ingestión específica. Sin embargo, esto no elimina el riesgo por completo: si se reingestan los documentos con un splitter o `chunk_overlap` distinto, es responsabilidad de quien mantiene el dataset volver a verificar manualmente que cada `anchor_contains` siga apareciendo íntegro dentro de un único chunk recuperable (y no partido entre dos chunks), y que `source_contains` siga matcheando el nombre de archivo real.
- **Muestra pequeña**: el dataset cubre solo 16 preguntas distribuidas en 8 subtemas (2 preguntas por subtema). Esto es suficiente para detectar regresiones evidentes o problemas sistemáticos, pero no constituye una muestra estadísticamente robusta del desempeño del sistema sobre la totalidad de preguntas que un usuario real podría formular.
- **Sesgo de juez único**: todas las puntuaciones de generación provienen de un solo modelo evaluador (el configurado como proveedor/modelo por defecto en `.env` al momento de ejecutar el script). No hay validación cruzada con un segundo juez ni con evaluación humana, por lo que cualquier sesgo sistemático de ese modelo (por ejemplo, tendencia a puntuar generosamente, o preferencias estilísticas) se traslada directamente a los resultados.
- **Resultados no comparables entre configuraciones distintas de `.env`**: dado que tanto el LLM generador como el LLM juez se instancian vía `LLMFactory.create_provider()` leyendo la configuración activa en `.env` (proveedor, modelo, temperatura), los resultados de una corrida con un proveedor/modelo no son directamente comparables con los de una corrida con otro proveedor/modelo distinto. Para comparar configuraciones es necesario volver a ejecutar el script completo bajo cada configuración deseada.
- **Context Precision tiene un techo bajo por diseño, y un Context Recall/MRR moderados pueden venir de una sola causa concentrada**: con dos descriptores por pregunta en el dataset actual (sección 4), `context_precision` solo puede valer `0.0`, `1/k` o `2/k` — nunca más. Ejemplo real de una corrida completa con `k=8` (`TOP_K_RESULTS=8` en `.env`) sobre las 16 preguntas:
  - `context_precision` tomó únicamente los valores `0.0`, `0.125` (`1/8`) y `0.25` (`2/8`) en las 16 preguntas (nunca un valor intermedio ni mayor a `0.25`), confirmando el techo estructural. El promedio global de `context_precision` fue `0.1562` y el de `context_recall` fue `0.625`.
  - Esos promedios se explican por una distribución discreta en tres grupos: **7 de las 16 preguntas (43.75 %)** encontraron ambos chunks dorados en el top-8 (`context_precision=0.25`, `context_recall=1.0`); **6 de las 16 (37.5 %)** encontraron exactamente uno de los dos (`context_precision=0.125`, `context_recall=0.5`); y **3 de las 16 (18.75 %)** no encontraron ninguno (`context_precision = context_recall = mrr = 0.0`). Ninguna pregunta cae fuera de estos tres grupos porque, con solo dos positivos verdaderos por pregunta, no hay otra combinación posible.
  - El MRR global (`0.5719`) se descompone como: `13/16` preguntas con al menos un acierto (las 7 con ambos descriptores más las 6 con uno solo), con un MRR promedio condicional de `0.70` entre esas 13 (equivalente a encontrar un chunk dorado típicamente en el puesto 1–2). El resto de la caída hacia `0.5719` viene enteramente de las 3 preguntas sin ningún acierto, no de que los chunks dorados aparezcan consistentemente en posiciones bajas cuando sí son recuperados.
  - Inspeccionando las 3 preguntas sin ningún acierto (`scrum-01`, `historias_usuario-02`, `retrospectivas-02`), el retriever sí trajo chunks reales y temáticamente correctos del mismo libro o capítulo (p. ej. para `scrum-01` trajo otro pasaje de "Essential Scrum" sobre el cierre del capítulo de sprints, la entrada de glosario de "sprint review", y la definición general de Scrum de "The Scrum Guide") — simplemente ninguno de esos ocho documentos coincidía verbatim con los dos anclajes específicos marcados en el ground truth para esa pregunta. En el caso de `historias_usuario-02` (criterio INVEST), el desajuste es principalmente de vocabulario entre idiomas: la pregunta está en español y el acrónimo "INVEST" aparece explícitamente solo en un par de chunks en inglés que, pese a tratar el mismo tema, no rankearon dentro del top-8 frente a otros chunks en inglés sobre historias de usuario en general. Esto refleja un límite real de la combinación embedding multilingüe + corpus mayoritariamente en inglés + `k` pequeño, no evidencia de que el retriever devuelva contenido irrelevante.
  - Implicación práctica: antes de interpretar un `context_precision`/`mrr` "bajo" como un problema real de retrieval, conviene (a) verificar cuántos de los descriptores de la pregunta puntual fueron encontrados (un acierto parcial de `1/2` ya descarta un fallo total de recuperación), y (b) probar con un `--k` más alto en el script de evaluación para distinguir entre "ningún chunk dorado aparece nunca" y "aparece pero en una posición baja".

## 9. Código

Archivos que componen esta funcionalidad:

| Archivo | Rol |
|---|---|
| `src/evaluation/__init__.py` | Paquete Python vacío que marca `src/evaluation` como módulo importable. |
| `src/evaluation/dataset.py` | Dataclasses `RelevantChunkDescriptor` y `EvalQuestion`, y `load_eval_dataset()` para parsear el JSON del dataset. |
| `src/evaluation/retrieval_metrics.py` | `normalize_text`, `is_relevant_document`, `context_precision`, `context_recall`, `mean_reciprocal_rank`, `aggregate_mrr` — todas las métricas deterministas de recuperación. |
| `src/evaluation/judge.py` | `JUDGE_PROMPT_TEMPLATE`, `build_judge_prompt`, `invoke_judge`, `parse_judge_response`, y las dataclasses `JudgeScore`/`JudgeResult` — el juez LLM y su parseo defensivo. |
| `tests/test_evaluation_metrics.py` | Pruebas unitarias rápidas (sin red) de todo lo anterior, usando dobles de prueba para el LLM. |
| `scripts/evaluate_rag_quality.py` | CLI manual que ejecuta el pipeline real (ChromaDB + LLM real) y genera reportes JSON/Markdown. |
| `tests/data/agile_rag_eval_dataset.json` | El conjunto de verdad fundamental: 16 preguntas con sus descriptores de relevancia y respuestas de referencia. |
