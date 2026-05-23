const { useEffect, useMemo, useRef, useState } = React;

const DEFAULT_API_BASE = `${window.location.origin}/api/v1`;
const DEFAULT_SESSION = localStorage.getItem("sessionId") || null;
const MAX_SESSION_LABEL_LENGTH = 56;
const DEFAULT_MARKDOWN_RENDERING = true;
const hasMarked = typeof window.marked !== "undefined";
const hasDomPurify = typeof window.DOMPurify !== "undefined";

const escapeHtml = (value) => (
  String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;")
);

if (hasMarked) {
  const markdownRenderer = new window.marked.Renderer();

  markdownRenderer.link = (hrefOrToken, title, text) => {
    const href = typeof hrefOrToken === "object" && hrefOrToken !== null ? hrefOrToken.href : hrefOrToken;
    const linkTitle = typeof hrefOrToken === "object" && hrefOrToken !== null ? hrefOrToken.title : title;
    const linkText = typeof hrefOrToken === "object" && hrefOrToken !== null ? hrefOrToken.text : text;

    const safeHref = String(href || "").trim();
    const isSafeProtocol = /^(https?:|mailto:|#|\/)/i.test(safeHref);
    const finalHref = isSafeProtocol ? safeHref : "#";
    const safeTitle = linkTitle ? ` title=\"${escapeHtml(linkTitle)}\"` : "";
    return `<a href="${escapeHtml(finalHref)}" target="_blank" rel="noopener noreferrer"${safeTitle}>${linkText || ""}</a>`;
  };

  markdownRenderer.html = (html) => escapeHtml(html);

  window.marked.setOptions({
    gfm: true,
    breaks: true,
    renderer: markdownRenderer
  });
}

function sanitizeUrl(url) {
  const candidate = String(url || "").trim();
  if (!candidate) {
    return "#";
  }
  return /^(https?:|mailto:|#|\/)/i.test(candidate) ? candidate : "#";
}

function normalizeMarkdownSource(value) {
  return String(value || "")
    .replace(/\\n/g, "\n")
    .replace(/\\r/g, "\r")
    .replace(/\\([`*_#>\[\]\(\)\-])/g, "$1");
}

function normalizeOrderedListsHtml(html) {
  try {
    const container = document.createElement('div');
    container.innerHTML = html;
    const nodes = Array.from(container.childNodes);
    let i = 0;
    while (i < nodes.length) {
      const node = nodes[i];
      if (node.nodeType === Node.ELEMENT_NODE && node.tagName === 'P') {
        const text = node.textContent || '';
        if (/^\s*\d+\.\s+/.test(text)) {
          // Start collecting consecutive numbered paragraphs
          const ol = document.createElement('ol');
          while (i < nodes.length) {
            const current = nodes[i];
            if (!(current.nodeType === Node.ELEMENT_NODE && current.tagName === 'P')) break;
            const curText = current.textContent || '';
            const m = curText.match(/^\s*\d+\.\s+(.*)$/s);
            if (!m) break;
            const li = document.createElement('li');
            li.innerHTML = current.innerHTML.replace(/^\s*\d+\.\s+/, '');
            ol.appendChild(li);
            const next = current.nextSibling;
            container.removeChild(current);
            nodes.splice(i, 1);
            if (next == null) break;
            // refresh nodes reference
            // nodes array will be rebuilt in next loop iteration if needed
          }
          // insert ol at position i
          const refNode = container.childNodes[i] || null;
          container.insertBefore(ol, refNode);
          // rebuild nodes and continue after inserted ol
          const newNodes = Array.from(container.childNodes);
          nodes.length = 0;
          Array.prototype.push.apply(nodes, newNodes);
          i += 1;
          continue;
        }
      }
      i += 1;
    }
    return container.innerHTML;
  } catch (e) {
    return html;
  }
}

function renderInlineMarkdown(value) {
  let text = escapeHtml(value);

  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_m, label, href) => {
    const safeHref = escapeHtml(sanitizeUrl(href));
    return `<a href="${safeHref}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  });

  return text;
}

function renderBasicMarkdown(rawText) {
  const isTableSeparator = (value) => /^\s*\|?\s*[:\-\|\s]+\|?\s*$/.test(value || "");
  const splitTableRow = (value) => String(value || "").trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
  const lines = String(rawText || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    if (trimmed.startsWith("```")) {
      const codeLines = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i += 1;
      }
      html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      if (i < lines.length) {
        i += 1;
      }
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,4})\s+(.*)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      html.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
      i += 1;
      continue;
    }

    if (/^>\s+/.test(trimmed)) {
      html.push(`<blockquote>${renderInlineMarkdown(trimmed.replace(/^>\s+/, ""))}</blockquote>`);
      i += 1;
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ""));
        i += 1;
      }
      html.push(`<ol>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ol>`);
      continue;
    }

    if (/^[-*]\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*]\s+/, ""));
        i += 1;
      }
      html.push(`<ul>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
      continue;
    }

    if (trimmed.includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const headers = splitTableRow(lines[i]);
      i += 2;
      const bodyRows = [];

      while (i < lines.length && lines[i].trim().includes("|") && lines[i].trim()) {
        bodyRows.push(splitTableRow(lines[i]));
        i += 1;
      }

      const thead = `<thead><tr>${headers.map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join("")}</tr></thead>`;
      const tbody = bodyRows.length
        ? `<tbody>${bodyRows.map((row) => `<tr>${row.map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>`).join("")}</tbody>`
        : "";
      html.push(`<table>${thead}${tbody}</table>`);
      continue;
    }

    const paragraphLines = [trimmed];
    i += 1;
    while (i < lines.length && lines[i].trim() && !/^(#{1,4})\s+/.test(lines[i].trim()) && !/^\s*\d+\.\s+/.test(lines[i].trim()) && !/^\s*[-*]\s+/.test(lines[i].trim()) && !/^>\s+/.test(lines[i].trim()) && !lines[i].trim().startsWith("```")) {
      paragraphLines.push(lines[i].trim());
      i += 1;
    }
    html.push(`<p>${renderInlineMarkdown(paragraphLines.join(" "))}</p>`);
  }

  return html.join("\n");
}

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

function isImageAttachment(item) {
  return /^(png|jpg|jpeg|webp|gif)$/i.test(String(item?.file_type || ""));
}

function attachmentLabel(item) {
  return item?.filename || item?.document_id || "Documento";
}

function attachmentSourceUrl(item) {
  return String(item?.source_url || item?.previewUrl || item?.source || "").trim();
}

function attachmentKindLabel(item) {
  return /^(png|jpg|jpeg|webp|gif)$/i.test(String(item?.file_type || ""))
    ? "Imagen"
    : String(item?.file_type || "Documento").toUpperCase();
}

function SourceReferences({ sources = [] }) {
  const [isOpen, setIsOpen] = useState(false);
  const sortedSources = useMemo(() => {
    return [...sources].sort((left, right) => Number(right.relevance || 0) - Number(left.relevance || 0));
  }, [sources]);

  if (!sortedSources.length) {
    return null;
  }

  return (
    <div className="sources-block">
      <button className="sources-toggle" onClick={() => setIsOpen((value) => !value)}>
        <span>{isOpen ? "Ocultar" : "Ver"} fuentes</span>
        <span className="sources-count">{sortedSources.length}</span>
      </button>

      {isOpen && (
        <div className="sources-panel">
          {sortedSources.map((source, sourceIndex) => {
            const sourceTitle = source.filename || source.source || "Documento";
            const excerpt = source.excerpt || "Sin extracto disponible.";
            const sourceHint = source.source || source.document_id || "Referencia";
            const scopeLabel = source.scope === "session_chat" ? "Sesion" : "Global";
            const pageLabel = source.page ? `Pagina ${source.page}` : null;
            const sectionLabel = source.section ? `Seccion: ${source.section}` : null;

            return (
              <article key={`${sourceTitle}-${sourceIndex}`} className="source-card">
                <div className="source-card-head">
                  <strong>{sourceTitle}</strong>
                  <div className="source-card-pills">
                    <span className="source-pill">{scopeLabel}</span>
                    <span className="source-pill">{sourceIndex + 1}</span>
                  </div>
                </div>
                <p className="source-meta">{sourceHint}</p>
                {(pageLabel || sectionLabel) && (
                  <p className="source-location">
                    {[pageLabel, sectionLabel].filter(Boolean).join(" · ")}
                  </p>
                )}
                <p className="source-excerpt">{excerpt}</p>
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}

function MarkdownContent({ content = "", enabled = true }) {
  const safeHtml = useMemo(() => {
    const raw = normalizeMarkdownSource(content);
    const fallbackHtml = renderBasicMarkdown(raw);

    if (!enabled || !hasMarked || !hasDomPurify) {
      return fallbackHtml;
    }

    let parsed = window.marked.parse(raw);
    parsed = normalizeOrderedListsHtml(parsed);
    return window.DOMPurify.sanitize(parsed, {
      USE_PROFILES: { html: true },
      FORBID_TAGS: ["script", "style", "iframe", "object", "embed", "form", "input", "button", "textarea", "svg", "math"],
      FORBID_ATTR: ["onerror", "onload", "onclick", "onmouseover", "style"]
    });
  }, [content, enabled]);

  return <div className="markdown-content" dangerouslySetInnerHTML={{ __html: safeHtml }} />;
}

function AttachmentPreview({ item, onOpen }) {
  const isImage = String(item?.file_type || "").toLowerCase().startsWith("image/") || /^(png|jpg|jpeg|webp|gif)$/i.test(String(item?.file_type || ""));
  const previewUrl = attachmentSourceUrl(item);
  const label = attachmentLabel(item);
  const handleOpen = () => {
    if (typeof onOpen === "function") {
      onOpen(item);
    }
  };

  return (
    <figure
      className={`message-attachment ${isImage ? "image" : "document"}`}
      tabIndex={isImage ? 0 : -1}
      role={isImage ? "button" : undefined}
      onClick={isImage ? handleOpen : undefined}
      onKeyDown={isImage ? (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          handleOpen();
        }
      } : undefined}
    >
      {isImage && previewUrl ? (
        <img className="message-attachment-image" src={previewUrl} alt={label} />
      ) : (
        <div className="message-attachment-fallback" aria-hidden="true">📎</div>
      )}
      <figcaption className="message-attachment-caption">
        <strong>{label}</strong>
        <span>{attachmentKindLabel(item)}</span>
      </figcaption>
    </figure>
  );
}

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [helpMenuOpen, setHelpMenuOpen] = useState(false);

  const [apiBase, setApiBase] = useState(localStorage.getItem("apiBase") || DEFAULT_API_BASE);
  const [theme, setTheme] = useState(localStorage.getItem("theme") || "dark");
  const [markdownRendering, setMarkdownRendering] = useState(DEFAULT_MARKDOWN_RENDERING);

  const [providers, setProviders] = useState([]);
  const [modelsByProvider, setModelsByProvider] = useState({});
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");

  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(2000);
  const useRag = true;

  const [sessions, setSessions] = useState([]);
  const [sessionSearch, setSessionSearch] = useState("");
  const [sessionId, setSessionId] = useState(DEFAULT_SESSION);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [pendingAttachments, setPendingAttachments] = useState([]);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [docsOpen, setDocsOpen] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [sessionDocuments, setSessionDocuments] = useState([]);
  const [isUploadingGlobalDocs, setIsUploadingGlobalDocs] = useState(false);
  const [isUploadingSessionDocs, setIsUploadingSessionDocs] = useState(false);
  const [documentsSummary, setDocumentsSummary] = useState("No cargado");
  const [deleteDocId, setDeleteDocId] = useState("");
  const [health, setHealth] = useState({ status: "...", version: "-", rag_status: "-", vector_store_documents: 0 });
  const [attachmentViewer, setAttachmentViewer] = useState(null);

  const [alert, setAlert] = useState(null);

  const globalFileInputRef = useRef(null);
  const sessionFileInputRef = useRef(null);
  const chatViewportRef = useRef(null);
  const shouldStickToBottomRef = useRef(true);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);

  const modelsForCurrentProvider = useMemo(() => modelsByProvider[provider] || [], [modelsByProvider, provider]);
  const activeSession = useMemo(() => sessions.find((item) => item.session_id === sessionId) || null, [sessions, sessionId]);
  const hasMessages = messages.length > 0;

  const scrollToConversationBottom = () => {
    if (!chatViewportRef.current) {
      return;
    }
    chatViewportRef.current.scrollTop = chatViewportRef.current.scrollHeight;
  };

  const updateScrollState = () => {
    const viewport = chatViewportRef.current;
    if (!viewport) {
      return;
    }

    const distanceToBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;
    const isNearBottom = distanceToBottom < 44;
    shouldStickToBottomRef.current = isNearBottom;
    setShowScrollToBottom(!isNearBottom && hasMessages);
  };

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
      sources: item.sources || [],
      attachments: (item.attachments || []).map((attachment) => ({
        ...attachment,
        source_url: attachment.source_url || attachment.source || ""
      }))
    }));
    setMessages(mapped);
    setSessionId(id);
    localStorage.setItem("sessionId", id);
    setPendingAttachments([]);
    setAttachmentViewer(null);
    await loadSessionDocuments(id);
  };

  const createNewConversation = () => {
    setSessionId(null);
    localStorage.removeItem("sessionId");
    setMessages([]);
    setSessionDocuments([]);
    setPendingAttachments([]);
    setAttachmentViewer(null);
    setHelpMenuOpen(false);
  };

  const ensureSessionContextId = () => {
    if (sessionId) {
      return sessionId;
    }
    const generatedId = window.crypto?.randomUUID?.() || `session-${Date.now()}`;
    setSessionId(generatedId);
    localStorage.setItem("sessionId", generatedId);
    return generatedId;
  };

  const uploadGlobalDocument = async (file) => {
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

  const uploadSessionDocument = async (targetSessionId, file) => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${apiBase}/documents/sessions/${encodeURIComponent(targetSessionId)}/upload`, {
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

  const uploadSessionFiles = async (targetSessionId, files) => {
    const results = await Promise.all(files.map((file) => uploadSessionDocument(targetSessionId, file)));
    await Promise.all([loadSessionDocuments(targetSessionId), loadSessions(sessionSearch)]);
    return results;
  };

  const publicUploadUrl = (targetSessionId, filename) => (
    `/uploads/sessions/${encodeURIComponent(targetSessionId)}/${encodeURIComponent(filename)}`
  );

  const openAttachmentViewer = (item) => {
    const sourceUrl = attachmentSourceUrl(item);
    if (!sourceUrl) {
      return;
    }
    setAttachmentViewer({
      src: sourceUrl,
      label: attachmentLabel(item)
    });
  };

  const closeAttachmentViewer = () => {
    setAttachmentViewer(null);
  };

  const removePendingAttachment = (index) => {
    setPendingAttachments((current) => {
      const next = [...current];
      const [removed] = next.splice(index, 1);
      if (removed?.previewUrl?.startsWith("blob:")) {
        URL.revokeObjectURL(removed.previewUrl);
      }
      return next;
    });
  };

  const handleGlobalUploadClick = () => {
    if (isUploadingGlobalDocs) {
      return;
    }
    if (globalFileInputRef.current) {
      globalFileInputRef.current.click();
    }
  };

  const handleGlobalUpload = async (event) => {
    const files = Array.from(event.target.files || []);
    if (!files.length) {
      return;
    }
    setIsUploadingGlobalDocs(true);
    try {
      const results = await Promise.all(files.map((file) => uploadGlobalDocument(file)));
      notify(`Documentos globales cargados: ${results.length}`, "ok");
      await Promise.all([loadHealth(), loadDocuments()]);
    } catch (error) {
      notify(`Error al subir documento: ${error.message}`, "error");
    } finally {
      setIsUploadingGlobalDocs(false);
      event.target.value = "";
    }
  };

  const handleSessionUploadClick = () => {
    if (isUploadingSessionDocs) {
      return;
    }
    if (sessionFileInputRef.current) {
      sessionFileInputRef.current.click();
    }
  };

  const handleSessionUpload = async (event) => {
    const files = Array.from(event.target.files || []);
    if (!files.length) {
      return;
    }

    const targetSessionId = ensureSessionContextId();
    const attachments = files.map((file) => ({
      filename: file.name,
      file_type: file.type || file.name.split(".").pop() || "document",
      previewUrl: URL.createObjectURL(file),
      source_url: publicUploadUrl(targetSessionId, file.name),
    }));
    setPendingAttachments(attachments);
    setIsUploadingSessionDocs(true);
    try {
      const results = await uploadSessionFiles(targetSessionId, files);
      setPendingAttachments((current) => current.map((item, index) => ({
        ...item,
        document_id: results[index]?.document_id || item.document_id,
        source_url: publicUploadUrl(targetSessionId, item.filename),
      })));
      notify(`Adjuntos de sesion cargados: ${files.length}`, "ok");
    } catch (error) {
      notify(`Error al subir adjunto de sesion: ${error.message}`, "error");
    } finally {
      setIsUploadingSessionDocs(false);
      event.target.value = "";
    }
  };

  const handleSessionPaste = async (event) => {
    const items = Array.from(event.clipboardData?.items || []);
    const imageFiles = items
      .filter((item) => item.kind === "file" && String(item.type || "").startsWith("image/"))
      .map((item, index) => {
        const file = item.getAsFile();
        if (!file) {
          return null;
        }
        const extension = String(file.type || "image/png").split("/")[1] || "png";
        const filename = `clipboard-image-${Date.now()}-${index}.${extension}`;
        return new File([file], filename, { type: file.type || "image/png" });
      })
      .filter(Boolean);

    if (!imageFiles.length) {
      return;
    }

    event.preventDefault();
    const targetSessionId = ensureSessionContextId();
    const attachments = imageFiles.map((file) => ({
      filename: file.name,
      file_type: file.type || "image/png",
      previewUrl: URL.createObjectURL(file),
      source_url: publicUploadUrl(targetSessionId, file.name),
    }));
    setPendingAttachments(attachments);
    setIsUploadingSessionDocs(true);
    try {
      const results = await uploadSessionFiles(targetSessionId, imageFiles);
      setPendingAttachments((current) => current.map((item, index) => ({
        ...item,
        document_id: results[index]?.document_id || item.document_id,
        source_url: publicUploadUrl(targetSessionId, item.filename),
      })));
      notify(`Imagen pegada y adjuntada: ${imageFiles.length}`, "ok");
    } catch (error) {
      notify(`Error al pegar imagen: ${error.message}`, "error");
    } finally {
      setIsUploadingSessionDocs(false);
    }
  };

  const loadSessionDocuments = async (targetSessionId) => {
    if (!targetSessionId) {
      setSessionDocuments([]);
      return;
    }

    const data = await request(`/documents/sessions/${encodeURIComponent(targetSessionId)}`);
    setSessionDocuments(data.documents || []);
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

  const handleDeleteDocument = async (documentId) => {
    if (!documentId) return;
    const first = window.confirm("¿Eliminar este documento? Esta acción no se puede deshacer.");
    if (!first) return;
    const second = window.confirm("¿Estás seguro? Confirma nuevamente para eliminar.");
    if (!second) return;
    try {
      await request(`/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" });
      notify(`Documento ${documentId} eliminado`, "ok");
      await Promise.all([loadDocuments(), loadHealth()]);
    } catch (err) {
      notify(`Error al eliminar documento: ${err.message}`, "error");
    }
  };

  const deleteSession = async (id) => {
    await request(`/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (sessionId === id) {
      createNewConversation();
    }
    await loadSessions(sessionSearch);
  };

  const deleteSessionDocument = async (documentId) => {
    if (!sessionId || !documentId) {
      return;
    }
    await request(`/documents/sessions/${encodeURIComponent(sessionId)}/${encodeURIComponent(documentId)}`, {
      method: "DELETE"
    });
    notify("Adjunto eliminado de la sesion", "ok");
    await loadSessionDocuments(sessionId);
  };

  const clearSessionDocuments = async () => {
    if (!sessionId) {
      return;
    }
    await request(`/documents/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    notify("Adjuntos de sesion eliminados", "ok");
    setPendingAttachments([]);
    await loadSessionDocuments(sessionId);
  };

  const toggleHelpMenu = () => {
    if (!sidebarOpen) {
      setSidebarOpen(true);
      setHelpMenuOpen(true);
      return;
    }
    setHelpMenuOpen((prev) => !prev);
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

    const latestSessionId = finalPayload?.session_id || payload.session_id;
    if (latestSessionId) {
      await loadSessionDocuments(latestSessionId);
    }
  };

  const sendMessage = async () => {
    const message = input.trim();
    if (!message || isSending) {
      return;
    }

    const payload = {
      message,
      session_id: sessionId || ensureSessionContextId(),
      use_rag: useRag,
      llm_provider: provider || null,
      model_name: model || null,
      temperature: Number(temperature),
      session_document_ids: sessionDocuments.map((item) => item.document_id).filter(Boolean),
      session_attachments: pendingAttachments.map((item) => ({
        document_id: item.document_id || null,
        filename: item.filename || null,
        file_type: item.file_type || null,
        source_url: attachmentSourceUrl(item) || null,
      }))
    };

    const userMessage = { role: "user", content: message, sources: [] };
    const draftAttachments = pendingAttachments.map((item) => ({ ...item }));
    const attachmentMessage = draftAttachments.length > 0
      ? {
          role: "system",
          content: "",
          sources: [],
          attachments: draftAttachments
        }
      : null;
    const assistantPlaceholder = { role: "assistant", content: "", sources: [] };
    const assistantIndex = messages.length + 1 + (attachmentMessage ? 1 : 0);

    setMessages((current) => [...current, userMessage, ...(attachmentMessage ? [attachmentMessage] : []), assistantPlaceholder]);
    setInput("");
    setIsSending(true);
    shouldStickToBottomRef.current = true;

    try {
      await streamChat(payload, assistantIndex);
      await loadSessions(sessionSearch);
      setPendingAttachments([]);
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
    localStorage.setItem("enableMarkdownRendering", String(markdownRendering));
  }, [markdownRendering]);

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
    loadSessionDocuments(sessionId).catch(() => {
      setSessionDocuments([]);
    });
  }, [sessionId]);

  useEffect(() => {
    if (shouldStickToBottomRef.current) {
      scrollToConversationBottom();
    }
    updateScrollState();
  }, [messages]);


  useEffect(() => {
    updateScrollState();
  }, [hasMessages]);

  return (
    <div className="app-shell">
      <aside className={`history-sidebar ${sidebarOpen ? "" : "collapsed"}`}>
        <div className="sidebar-top">
          <div className="sidebar-header-row">
            <button className="sidebar-toggle" onClick={() => { setSidebarOpen((prev) => !prev); setHelpMenuOpen(false); }}>
              {sidebarOpen ? "☰" : "☷"}
            </button>
            {sidebarOpen && <h1>Menu</h1>}
          </div>
          {sidebarOpen && <button className="new-chat-btn" onClick={createNewConversation}>Nueva conversacion</button>}
        </div>

        { /* Search input removed — historial search disabled in sidebar */ }

        <div className="session-list">
          {sidebarOpen && <div className="section-title">Conversaciones</div>}
          {sessions.map((item) => (
            <div key={item.session_id} className={`session-item ${item.session_id === sessionId ? "active" : ""}`}>
              {sidebarOpen ? (
                <>
                  <button className="session-select" onClick={() => loadHistory(item.session_id)}>
                    <strong>{truncateText(getSessionLabel(item), 32)}</strong>
                  </button>
                  <button className="session-rename" onClick={() => renameSession(item)} title="Renombrar charla">✎</button>
                  <button className="session-delete" onClick={() => deleteSession(item.session_id)}>x</button>
                </>
              ) : (
                <button className="session-dot" onClick={() => loadHistory(item.session_id)} title={getSessionLabel(item)}>●</button>
              )}
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-tools">
            <button className="sidebar-menu-trigger" onClick={toggleHelpMenu} title="Configuracion y ayuda">
              ⚙
            </button>
            {sidebarOpen && <span>Configuracion</span>}
          </div>

          {helpMenuOpen && (
            <div className="sidebar-help-menu">
              <button onClick={() => { setSettingsOpen(true); setHelpMenuOpen(false); }}>Configuracion avanzada</button>
              <button onClick={() => { setDocsOpen(true); loadDocuments().catch(() => {}); setHelpMenuOpen(false); }}>Gestion de documentos</button>
              <button onClick={() => { setTheme(theme === "dark" ? "light" : "dark"); setHelpMenuOpen(false); }}>
                Tema: {theme === "dark" ? "Claro" : "Oscuro"}
              </button>
            </div>
          )}

          {sidebarOpen && (
            <>
              <p>Estado: {health.status}</p>
              <p>RAG: {health.rag_status}</p>
            </>
          )}
        </div>
      </aside>

      {!sidebarOpen && (
        <button className="floating-new-chat" onClick={createNewConversation} title="Nueva charla">＋</button>
      )}

      <main className="chat-main">
        <div className="app-brand">Chatbot Agil</div>

        <section ref={chatViewportRef} className="chat-viewport" onScroll={updateScrollState}>
          {!hasMessages && (
            <div className="welcome-panel">
              <div className="welcome-kicker">Hola</div>
              <h2>¿Cómo puedo ayudarte hoy?</h2>
            </div>
          )}

          {hasMessages && (
            <div className="conversation-headline">
              {activeSession ? getSessionLabel(activeSession) : "Nueva conversacion"}
            </div>
          )}

          <div className="messages-stack">
            {messages.map((item, index) => (
              <article key={`${item.role}-${index}`} className={`bubble ${item.role}`}>
                {String(item.role || "").toLowerCase() !== "user" && String(item.content || "").trim().length > 0
                  ? <MarkdownContent content={item.content} enabled={markdownRendering} />
                  : String(item.role || "").toLowerCase() === "user"
                    ? <p>{item.content}</p>
                    : null}
                {Array.isArray(item.attachments) && item.attachments.length > 0 && (
                  <div className="message-attachments">
                    {item.attachments.map((attachment, attachmentIndex) => (
                      <AttachmentPreview
                        key={`${attachment.filename || attachmentIndex}-${attachmentIndex}`}
                        item={attachment}
                        onOpen={openAttachmentViewer}
                      />
                    ))}
                  </div>
                )}
                <SourceReferences sources={item.sources} />
              </article>
            ))}
          </div>
        </section>

        <footer className="chat-input-area">
          <div className="input-shell">
            <input
              ref={sessionFileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt,.doc,.docx,.md,.markdown,.png,.jpg,.jpeg,.webp"
              onChange={handleSessionUpload}
              style={{ display: "none" }}
            />
            <button
              className="input-icon"
              onClick={handleSessionUploadClick}
              title="Adjuntar a esta sesion"
              disabled={isUploadingSessionDocs}
            >
              {isUploadingSessionDocs ? "…" : "＋"}
            </button>

            <textarea
              rows={1}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Pregunta a Agile Assistant"
              onPaste={handleSessionPaste}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  sendMessage();
                }
              }}
            />

            <div className="session-attachments">
              {isUploadingSessionDocs && (
                <span className="uploading-inline">
                  <span className="spinner" aria-hidden="true" />
                  Subiendo adjuntos...
                </span>
              )}
              {pendingAttachments.map((item, index) => {
                const image = String(item.file_type || "").toLowerCase().startsWith("image/") || /^(png|jpg|jpeg|webp|gif)$/i.test(String(item.file_type || ""));
                return (
                  <div key={`${item.filename || index}-${index}`} className={`attachment-card preview ${image ? "image" : "document"}`}>
                    <span className="attachment-card-body">
                      <span className="attachment-name">{attachmentLabel(item)}</span>
                      <span className="attachment-kind">{attachmentKindLabel(item)}</span>
                    </span>
                    {image && attachmentSourceUrl(item) && (
                      <button
                        type="button"
                        className="attachment-thumb-button"
                        onClick={() => openAttachmentViewer(item)}
                        title="Ver imagen"
                      >
                        <img className="attachment-thumb" src={attachmentSourceUrl(item)} alt={attachmentLabel(item)} />
                      </button>
                    )}
                    {!image && <span className="attachment-icon" aria-hidden="true">📎</span>}
                    <button
                      type="button"
                      className="attachment-remove-btn"
                      onClick={() => removePendingAttachment(index)}
                      title="Quitar adjunto"
                      aria-label={`Quitar ${attachmentLabel(item)}`}
                    >
                      ×
                    </button>
                  </div>
                );
              })}
            </div>

            <div className="input-toolbar">
              <select value={provider} onChange={(event) => setProvider(event.target.value)}>
                {(providers || []).map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
              <select value={model} onChange={(event) => setModel(event.target.value)}>
                {(modelsForCurrentProvider || []).map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
              <button className="primary-btn" onClick={sendMessage} disabled={isSending}>{isSending ? "Pensando..." : "Enviar"}</button>
            </div>
          </div>
        </footer>

        {showScrollToBottom && (
          <button className="scroll-bottom-btn" onClick={scrollToConversationBottom} title="Ir al final">
            Ir al final
          </button>
        )}
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

            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={markdownRendering}
                onChange={(event) => setMarkdownRendering(event.target.checked)}
              />
              Render Markdown seguro en respuestas
            </label>

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
            <input
              ref={globalFileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt,.doc,.docx,.md,.markdown"
              onChange={handleGlobalUpload}
              style={{ display: "none" }}
            />
            <button className="primary-btn" onClick={handleGlobalUploadClick} disabled={isUploadingGlobalDocs}>
              {isUploadingGlobalDocs ? "Subiendo..." : "Subir documentos globales RAG"}
            </button>
            {isUploadingGlobalDocs && (
              <p className="uploading-inline">
                <span className="spinner" aria-hidden="true" />
                Subiendo documentos al RAG...
              </p>
            )}

            <div className="doc-list">
              {documents.length === 0 ? (
                <p>No hay documentos indexados para mostrar.</p>
              ) : (
                documents.map((item, index) => {
                  const documentId = item.file_hash || item.document_id || item.id || "Sin ID";
                  return (
                    <article key={`${item.filename}-${index}`} className="doc-entry">
                      <div className="doc-entry-head">
                        <strong>{item.filename || "Documento sin nombre"}</strong>
                        <span className="doc-type-pill">{(item.file_type || "unknown").toUpperCase()}</span>
                      </div>
                      <p><span className="doc-label">Origen:</span> {item.source || "Sin origen"}</p>
                      <p><span className="doc-label">ID:</span> <code>{documentId}</code></p>
                      <div className="doc-actions">
                        <button className="danger-btn" onClick={() => handleDeleteDocument(documentId)}>Eliminar</button>
                      </div>
                    </article>
                  );
                })
              )}
            </div>

            {/* Eliminacion por ID eliminada; ahora cada documento tiene su boton de borrar con doble confirmacion */}

            <div className="panel-actions">
              <button className="ghost-btn" onClick={() => setDocsOpen(false)}>Cerrar</button>
              <button className="danger-btn" onClick={clearDocuments}>Vaciar todo</button>
            </div>
          </section>
        </div>
      )}

      {attachmentViewer && (
        <div className="overlay attachment-viewer-overlay" onClick={closeAttachmentViewer}>
          <section className="panel attachment-viewer-panel" onClick={(event) => event.stopPropagation()}>
            <div className="attachment-viewer-header">
              <strong>{attachmentViewer.label}</strong>
              <button type="button" className="ghost-btn" onClick={closeAttachmentViewer}>Cerrar</button>
            </div>
            <img className="attachment-viewer-image" src={attachmentViewer.src} alt={attachmentViewer.label} />
          </section>
        </div>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
