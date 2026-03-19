const { useEffect, useMemo, useRef, useState } = React;

const DEFAULT_API_BASE = `${window.location.origin}/api/v1`;
const DEFAULT_SESSION = localStorage.getItem("sessionId") || null;
const MAX_SESSION_LABEL_LENGTH = 56;

function truncateText(text, max = MAX_SESSION_LABEL_LENGTH) {
  const compact = (text || "").trim();
  if (compact.length <= max) {
    return compact;
  }
  return `${compact.slice(0, max).trim()}...`;
}

function getSessionLabel(session) {
  if (!session) {
    return "Nueva conversacion";
  }

  const title = truncateText(session.title || "");
  if (title) {
    return title;
  }

  const fallback = truncateText(session.last_message || "", 42);
  if (fallback) {
    return fallback;
  }

  return `Chat ${String(session.session_id || "").slice(0, 8)}`;
}

function SourceReferences({ sources = [] }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!sources.length) {
    return null;
  }

  return (
    <div className="sources-block">
      <button className="sources-toggle" onClick={() => setIsOpen((value) => !value)}>
        <span>{isOpen ? "Ocultar" : "Ver"} fuentes</span>
        <span className="sources-count">{sources.length}</span>
      </button>

      {isOpen && (
        <div className="sources-panel">
          {sources.map((source, sourceIndex) => {
            const sourceTitle = source.filename || source.source || "Documento";
            const excerpt = source.excerpt || "Sin extracto disponible.";
            const sourceHint = source.source || source.document_id || "Referencia";

            return (
              <article key={`${sourceTitle}-${sourceIndex}`} className="source-card">
                <div className="source-card-head">
                  <strong>{sourceTitle}</strong>
                  <span className="source-pill">{sourceIndex + 1}</span>
                </div>
                <p className="source-meta">{sourceHint}</p>
                <p className="source-excerpt">{excerpt}</p>
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}

function App() {
  const [apiBase, setApiBase] = useState(localStorage.getItem("apiBase") || DEFAULT_API_BASE);
  const [theme, setTheme] = useState(localStorage.getItem("theme") || "dark");

  const [providers, setProviders] = useState([]);
  const [modelsByProvider, setModelsByProvider] = useState({});
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");

  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(2000);
  const [useRag, setUseRag] = useState(true);

  const [sessions, setSessions] = useState([]);
  const [sessionSearch, setSessionSearch] = useState("");
  const [sessionId, setSessionId] = useState(DEFAULT_SESSION);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [docsOpen, setDocsOpen] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [documentsSummary, setDocumentsSummary] = useState("No cargado");
  const [deleteDocId, setDeleteDocId] = useState("");
  const [health, setHealth] = useState({ status: "...", version: "-", rag_status: "-", vector_store_documents: 0 });

  const [alert, setAlert] = useState(null);

  const fileInputRef = useRef(null);
  const chatViewportRef = useRef(null);

  const modelsForCurrentProvider = useMemo(() => modelsByProvider[provider] || [], [modelsByProvider, provider]);
  const activeSession = useMemo(() => sessions.find((item) => item.session_id === sessionId) || null, [sessions, sessionId]);

  const request = async (path, options = {}) => {
    const response = await fetch(`${apiBase}${path}`, options);
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const data = await response.json();
        detail = data.detail || JSON.stringify(data);
      } catch (_) {
        // Preserve generic detail when body is not JSON.
      }
      throw new Error(detail);
    }
    if (response.status === 204) {
      return null;
    }
    return response.json();
  };

  const notify = (message, type = "ok") => {
    setAlert({ message, type });
    setTimeout(() => setAlert(null), 4000);
  };

  const loadConfig = async () => {
    const config = await request("/config");
    setProviders(config.available_providers || []);
    setModelsByProvider(config.available_models || {});
    setProvider(config.llm_provider || "");
    setModel(config.model_name || "");
    setTemperature(Number(config.temperature ?? 0.7));
    setMaxTokens(Number(config.max_tokens ?? 2000));
  };

  const saveAdvancedConfig = async () => {
    const payload = {
      llm_provider: provider,
      model_name: model,
      temperature: Number(temperature),
      max_tokens: Number(maxTokens)
    };
    const data = await request("/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    setProvider(data.llm_provider);
    setModel(data.model_name);
    setTemperature(data.temperature);
    setMaxTokens(data.max_tokens);
    notify("Configuracion actualizada", "ok");
  };

  const loadHealth = async () => {
    const data = await request("/health");
    setHealth(data);
  };

  const loadSessions = async (query = "") => {
    const qs = query ? `?q=${encodeURIComponent(query)}` : "";
    const data = await request(`/sessions${qs}`);
    setSessions(data.sessions || []);
  };

  const loadHistory = async (id) => {
    const data = await request(`/sessions/${encodeURIComponent(id)}/history?limit=300`);
    const mapped = (data.messages || []).map((item) => ({
      id: item.id,
      role: item.role,
      content: item.text,
      sources: item.sources || []
    }));
    setMessages(mapped);
    setSessionId(id);
    localStorage.setItem("sessionId", id);
  };

  const createNewConversation = () => {
    setSessionId(null);
    localStorage.removeItem("sessionId");
    setMessages([]);
  };

  const uploadDocument = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${apiBase}/documents/upload`, {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const data = await response.json();
        detail = data.detail || JSON.stringify(data);
      } catch (_) {
        // Keep generic detail.
      }
      throw new Error(detail);
    }
    return response.json();
  };

  const handleQuickUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleQuickUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    try {
      const result = await uploadDocument(file);
      notify(`Documento ${result.filename} cargado (${result.chunks_created} chunks)`, "ok");
      await Promise.all([loadHealth(), loadDocuments()]);
    } catch (error) {
      notify(`Error al subir documento: ${error.message}`, "error");
    } finally {
      event.target.value = "";
    }
  };

  const loadDocuments = async () => {
    const data = await request("/documents");
    setDocuments(data.documents || []);
    setDocumentsSummary(`Total en vector store: ${data.total_documents}`);
  };

  const clearDocuments = async () => {
    await request("/documents", { method: "DELETE" });
    notify("Documentos eliminados", "ok");
    await Promise.all([loadDocuments(), loadHealth()]);
  };

  const deleteDocumentById = async () => {
    if (!deleteDocId.trim()) {
      return;
    }
    await request(`/documents/${encodeURIComponent(deleteDocId.trim())}`, { method: "DELETE" });
    notify(`Documento ${deleteDocId} eliminado`, "ok");
    setDeleteDocId("");
    await Promise.all([loadDocuments(), loadHealth()]);
  };

  const deleteSession = async (id) => {
    await request(`/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (sessionId === id) {
      createNewConversation();
    }
    await loadSessions(sessionSearch);
  };

  const renameSession = async (session) => {
    const currentLabel = session.title || getSessionLabel(session);
    const nextTitle = window.prompt("Nuevo titulo para esta conversacion:", currentLabel);
    if (nextTitle === null) {
      return;
    }

    const cleanedTitle = nextTitle.trim();
    if (!cleanedTitle) {
      notify("El titulo no puede estar vacio", "error");
      return;
    }

    if (cleanedTitle === session.title) {
      return;
    }

    await request(`/sessions/${encodeURIComponent(session.session_id)}/title`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: cleanedTitle })
    });

    setSessions((current) => current.map((item) => (
      item.session_id === session.session_id
        ? { ...item, title: cleanedTitle }
        : item
    )));

    notify("Titulo actualizado", "ok");
  };

  const streamChat = async (payload, assistantIndex) => {
    const response = await fetch(`${apiBase}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload, stream: true })
    });

    if (!response.ok || !response.body) {
      let detail = `HTTP ${response.status}`;
      try {
        const data = await response.json();
        detail = data.detail || JSON.stringify(data);
      } catch (_) {
        // Keep generic detail.
      }
      throw new Error(detail);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalPayload = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });

      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() || "";

      chunks.forEach((rawEvent) => {
        const line = rawEvent.split("\n").find((item) => item.startsWith("data: "));
        if (!line) {
          return;
        }
        try {
          const event = JSON.parse(line.slice(6));
          if (event.type === "delta") {
            setMessages((current) => {
              const next = [...current];
              const previous = next[assistantIndex] || { role: "assistant", content: "", sources: [] };
              next[assistantIndex] = {
                ...previous,
                content: `${previous.content}${event.content || ""}`
              };
              return next;
            });
          }
          if (event.type === "final") {
            finalPayload = event;
            if (event.session_id) {
              setSessionId(event.session_id);
              localStorage.setItem("sessionId", event.session_id);
            }
          }
        } catch (_) {
          // Ignore malformed stream chunks.
        }
      });
    }

    if (finalPayload?.sources) {
      setMessages((current) => {
        const next = [...current];
        next[assistantIndex] = {
          ...(next[assistantIndex] || { role: "assistant", content: "" }),
          sources: finalPayload.sources
        };
        return next;
      });
    }
  };

  const sendMessage = async () => {
    const message = input.trim();
    if (!message || isSending) {
      return;
    }

    const payload = {
      message,
      session_id: sessionId,
      use_rag: useRag,
      llm_provider: provider || null,
      model_name: model || null,
      temperature: Number(temperature)
    };

    const userMessage = { role: "user", content: message, sources: [] };
    const assistantPlaceholder = { role: "assistant", content: "", sources: [] };
    const assistantIndex = messages.length + 1;

    setMessages((current) => [...current, userMessage, assistantPlaceholder]);
    setInput("");
    setIsSending(true);

    try {
      await streamChat(payload, assistantIndex);
      await loadSessions(sessionSearch);
    } catch (error) {
      notify(`Error de chat: ${error.message}`, "error");
      setMessages((current) => {
        const next = [...current];
        next[assistantIndex] = {
          role: "assistant",
          content: `No se pudo completar la respuesta en streaming: ${error.message}`,
          sources: []
        };
        return next;
      });
    } finally {
      setIsSending(false);
    }
  };

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem("apiBase", apiBase);
  }, [apiBase]);

  useEffect(() => {
    if (!provider || modelsForCurrentProvider.length === 0) {
      return;
    }
    if (!modelsForCurrentProvider.includes(model)) {
      setModel(modelsForCurrentProvider[0]);
    }
  }, [provider, modelsForCurrentProvider, model]);

  useEffect(() => {
    Promise.allSettled([loadConfig(), loadSessions(), loadHealth()]);
  }, []);

  useEffect(() => {
    if (sessionId) {
      loadHistory(sessionId).catch(() => {
        setMessages([]);
      });
    }
  }, []);

  useEffect(() => {
    if (chatViewportRef.current) {
      chatViewportRef.current.scrollTop = chatViewportRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="app-shell">
      <aside className="history-sidebar">
        <div className="sidebar-top">
          <h1>Agile Chat</h1>
          <button className="ghost-btn" onClick={createNewConversation}>Nueva charla</button>
        </div>

        <input
          className="search-input"
          placeholder="Buscar historial"
          value={sessionSearch}
          onChange={(event) => setSessionSearch(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              loadSessions(sessionSearch).catch((error) => notify(error.message, "error"));
            }
          }}
        />

        <div className="session-list">
          {sessions.map((item) => (
            <div key={item.session_id} className={`session-item ${item.session_id === sessionId ? "active" : ""}`}>
              <button className="session-select" onClick={() => loadHistory(item.session_id)}>
                <strong>{getSessionLabel(item)}</strong>
                <span>{item.last_message || "Sin mensajes"}</span>
              </button>
              <button className="session-rename" onClick={() => renameSession(item)} title="Renombrar charla">✎</button>
              <button className="session-delete" onClick={() => deleteSession(item.session_id)}>x</button>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <p>Estado: {health.status}</p>
          <p>RAG: {health.rag_status}</p>
        </div>
      </aside>

      <main className="chat-main">
        <header className="chat-main-header">
          <div>
            <h2>Agile Assistant</h2>
            <p>Sesion: {activeSession ? getSessionLabel(activeSession) : (sessionId ? `Chat ${sessionId.slice(0, 8)}` : "Nueva conversacion")}</p>
          </div>

          <div className="header-controls">
            <select value={provider} onChange={(event) => setProvider(event.target.value)}>
              {(providers || []).map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              {(modelsForCurrentProvider || []).map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
            <button className="ghost-btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? "Claro" : "Oscuro"}</button>
            <button className="ghost-btn" onClick={() => setSettingsOpen(true)}>Configuracion</button>
            <button className="ghost-btn" onClick={() => { setDocsOpen(true); loadDocuments().catch(() => {}); }}>Documentos</button>
          </div>
        </header>

        <section ref={chatViewportRef} className="chat-viewport">
          {messages.length === 0 && (
            <div className="empty-chat">
              <h3>Inicia una conversacion</h3>
              <p>Usa el selector de proveedor y modelo aqui. La configuracion avanzada esta en su propia seccion.</p>
            </div>
          )}

          {messages.map((item, index) => (
            <article key={`${item.role}-${index}`} className={`bubble ${item.role}`}>
              <p>{item.content}</p>
              <SourceReferences sources={item.sources} />
            </article>
          ))}
        </section>

        <footer className="chat-input-area">
          <textarea
            rows={2}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Escribe tu mensaje sobre metodologias agiles..."
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
              }
            }}
          />

          <div className="input-actions">
            <label className="checkbox-row">
              <input type="checkbox" checked={useRag} onChange={(event) => setUseRag(event.target.checked)} />
              <span>RAG</span>
            </label>

            <input ref={fileInputRef} type="file" onChange={handleQuickUpload} style={{ display: "none" }} />
            <button className="ghost-btn" onClick={handleQuickUploadClick}>Adjuntar</button>
            <button className="primary-btn" onClick={sendMessage} disabled={isSending}>{isSending ? "Pensando..." : "Enviar"}</button>
          </div>
        </footer>
      </main>

      {alert && <div className={`alert ${alert.type}`}>{alert.message}</div>}

      {settingsOpen && (
        <div className="overlay" onClick={() => setSettingsOpen(false)}>
          <section className="panel" onClick={(event) => event.stopPropagation()}>
            <h3>Configuracion avanzada</h3>
            <label>API Base URL</label>
            <input value={apiBase} onChange={(event) => setApiBase(event.target.value)} />

            <label>Temperature: {temperature}</label>
            <input type="range" min="0" max="1" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />

            <label>Max tokens</label>
            <input type="number" min="1" value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value))} />

            <div className="panel-actions">
              <button className="ghost-btn" onClick={() => setSettingsOpen(false)}>Cerrar</button>
              <button className="primary-btn" onClick={saveAdvancedConfig}>Guardar</button>
            </div>
          </section>
        </div>
      )}

      {docsOpen && (
        <div className="overlay" onClick={() => setDocsOpen(false)}>
          <section className="panel" onClick={(event) => event.stopPropagation()}>
            <h3>Gestion de documentos</h3>
            <p>{documentsSummary}</p>

            <div className="doc-list">
              {documents.length === 0 ? (
                <p>No hay metadatos detallados en el backend actual.</p>
              ) : (
                documents.map((item, index) => <div key={`${item.filename}-${index}`}>{item.filename}</div>)
              )}
            </div>

            <div className="doc-delete-row">
              <input
                value={deleteDocId}
                placeholder="Document ID"
                onChange={(event) => setDeleteDocId(event.target.value)}
              />
              <button className="ghost-btn" onClick={deleteDocumentById}>Eliminar por ID</button>
            </div>

            <div className="panel-actions">
              <button className="ghost-btn" onClick={() => setDocsOpen(false)}>Cerrar</button>
              <button className="danger-btn" onClick={clearDocuments}>Vaciar todo</button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
