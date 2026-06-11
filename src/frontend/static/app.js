const { useEffect, useMemo, useRef, useState } = React;

const DEFAULT_API_BASE = `${window.location.origin}/api/v1`;
const MAX_SESSION_LABEL_LENGTH = 56;
const DEFAULT_MARKDOWN_RENDERING = true;
const AUTH_TOKEN_KEY = "authToken";
const AUTH_USER_KEY = "authUser";
const AGILE_QUESTIONNAIRE = [
  {
    id: 1,
    statement: "Ante una modificación imprevista en los requisitos del software a mitad del ciclo de desarrollo, ¿cuál considera que es la postura metodológica correcta?",
    options: [
      { value: "a", label: "a) Evitar o penalizar el cambio porque rompe la planificación inicial y pone en riesgo el cronograma acordado." },
      { value: "b", label: "b) Aceptar el cambio por exigencia, aunque genere frustración y desorganización interna al alterar el alcance ya pactado." },
      { value: "c", label: "c) Mantener una actitud de bienvenida hacia el cambio, entendiéndolo como parte de un proceso de aprendizaje continuo para maximizar el valor real entregado al cliente." },
      { value: "d", label: "d) No conozco" },
    ],
  },
  {
    id: 2,
    statement: "Con respecto a la frecuencia de las entregas y la planificación del producto:",
    options: [
      { value: "a", label: "a) Se planifica todo el proyecto al inicio y se realiza una única entrega formal y completa al finalizar el proceso." },
      { value: "b", label: "b) Se entrega software en periodos fijos, pero el feedback del cliente se procesa tarde, afectando poco la planificación de los siguientes ciclos." },
      { value: "c", label: "c) Se entrega software funcional de manera temprana y frecuente para obtener retroalimentación crucial que moldee el alcance y la dirección de la siguiente planificación." },
      { value: "d", label: "d) No conozco" },
    ],
  },
  {
    id: 3,
    statement: "¿Cómo se concibe la dinámica de trabajo, la asignación de tareas y las interacciones dentro del equipo?",
    options: [
      { value: "a", label: "a) Las tareas son asignadas y supervisadas de forma individual y centralizada por un líder o gerente de proyecto." },
      { value: "b", label: "b) El equipo se reúne para revisar tareas, pero la toma de decisiones y la responsabilidad siguen dependiendo de un control externo." },
      { value: "c", label: "c) El éxito se basa en las personas y sus interacciones; el equipo es multifuncional, se autogestiona y colabora diariamente de forma transparente." },
      { value: "d", label: "d) No conozco" },
    ],
  },
  {
    id: 4,
    statement: "Para asegurar la sostenibilidad del software en entornos de ritmo rápido, ¿cuándo se define que una funcionalidad está realmente concluida?",
    options: [
      { value: "a", label: "a) Cuando el desarrollador termina de escribir el código en su máquina local, delegando las pruebas a terceros." },
      { value: "b", label: "b) Cuando la funcionalidad pasa filtros básicos de pruebas individuales, aunque queden pendientes integraciones o revisiones de calidad global." },
      { value: "c", label: "c) Cuando cumple estrictamente con un compromiso de calidad y código limpio (Definition of Done), estando totalmente integrado, probado y listo para producción." },
      { value: "d", label: "d) No conozco" },
    ],
  },
  {
    id: 5,
    statement: "¿Cómo debe ser la relación e interacción con el cliente y los stakeholders durante el desarrollo?",
    options: [
      { value: "a", label: "a) Contractual y limitada a puntos específicos del proyecto (inicio y entrega final) para evitar corrupciones en el alcance." },
      { value: "b", label: "b) Intermitente; se le consulta al cliente únicamente cuando surgen dudas puntuales o en demostraciones programadas al final de hitos largos." },
      { value: "c", label: "c) Significativa, frecuente y colaborativa a lo largo de todo el esfuerzo de desarrollo para asegurar que el producto satisfaga las necesidades reales del negocio." },
      { value: "d", label: "d) No conozco" },
    ],
  },
];
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
    return "Nueva conversación";
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

function getSessionStorageKey(userId) {
  return userId ? `sessionId:${userId}` : "sessionId:anonymous";
}

function formatApiDetail(payload, fallbackStatus) {
  if (!payload) {
    return `HTTP ${fallbackStatus}`;
  }

  const detail = payload.detail ?? payload;

  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    const lines = detail.map((item) => {
      const location = Array.isArray(item?.loc) ? item.loc.join(".") : "campo";
      const message = item?.msg || JSON.stringify(item);
      return `${location}: ${message}`;
    });
    return lines.join("\n");
  }

  if (typeof detail === "object") {
    return JSON.stringify(detail);
  }

  return `HTTP ${fallbackStatus}`;
}

function validateRegistrationForm(authForm) {
  if (!String(authForm.email || "").trim()) {
    return "El email es obligatorio.";
  }

  if (!String(authForm.password || "").trim()) {
    return "La contraseña es obligatoria.";
  }

  if (String(authForm.password || "").trim().length < 8) {
    return "La contraseña debe tener al menos 8 caracteres.";
  }

  const firstName = String(authForm.first_name || "").trim();
  const lastName = String(authForm.last_name || "").trim();
  if (!firstName || !lastName) {
    return "El nombre y el apellido son obligatorios.";
  }
  if (firstName.length < 2 || lastName.length < 2) {
    return "Nombre y apellido deben tener al menos 2 caracteres cada uno.";
  }

  return null;
}

function SourceReferences({ sources = [] }) {
  const sortedSources = useMemo(() => {
    return [...sources]
      .sort((left, right) => Number(right.relevance || 0) - Number(left.relevance || 0));
  }, [sources]);

  if (!sortedSources.length) {
    return null;
  }

  return (
    <div className="source-chips">
      {sortedSources.map((source, index) => {
        const title = source.filename || source.source || "Documento";
        const excerpt = source.excerpt
          ? source.excerpt.slice(0, 120) + (source.excerpt.length > 120 ? "…" : "")
          : "Sin extracto disponible.";
        return (
          <span key={`${title}-${index}`} className="source-chip">
            <span className="source-chip-num">{index + 1}</span>
            {title}
            <span className="source-chip-tooltip">
              <strong>{title}</strong>
              {excerpt}
            </span>
          </span>
        );
      })}
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
  const previewUrl = String(item?.previewUrl || "").trim() || attachmentSourceUrl(item);
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

function groupSessionsByDate(sessions) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const weekAgo = new Date(today);
  weekAgo.setDate(weekAgo.getDate() - 7);

  const groups = { "Hoy": [], "Ayer": [], "Esta semana": [], "Anteriores": [] };

  sessions.forEach((session) => {
    const date = new Date(session.created_at || session.updated_at || 0);
    date.setHours(0, 0, 0, 0);
    if (date >= today) {
      groups["Hoy"].push(session);
    } else if (date >= yesterday) {
      groups["Ayer"].push(session);
    } else if (date >= weekAgo) {
      groups["Esta semana"].push(session);
    } else {
      groups["Anteriores"].push(session);
    }
  });

  return Object.entries(groups).filter(([, items]) => items.length > 0);
}

function getInitials(name) {
  if (!name) return "?";
  const parts = String(name).trim().split(/\s+/);
  if (parts.length >= 2) return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  return parts[0].slice(0, 2).toUpperCase();
}


function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [helpMenuOpen, setHelpMenuOpen] = useState(false);

  const [apiBase, setApiBase] = useState(localStorage.getItem("apiBase") || DEFAULT_API_BASE);
  const [theme, setTheme] = useState(localStorage.getItem("theme") || "dark");
  const [markdownRendering, setMarkdownRendering] = useState(DEFAULT_MARKDOWN_RENDERING);
  const [authToken, setAuthToken] = useState(localStorage.getItem(AUTH_TOKEN_KEY) || "");
  const [currentUser, setCurrentUser] = useState(() => {
    try {
      const raw = localStorage.getItem(AUTH_USER_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (_) {
      return null;
    }
  });
  const [authMode, setAuthMode] = useState("login");
  const [authReady, setAuthReady] = useState(false);
  const [authSubmitting, setAuthSubmitting] = useState(false);
  const [authMessage, setAuthMessage] = useState("");
  const [authError, setAuthError] = useState("");
  const [registerStep, setRegisterStep] = useState(1);
  const [authForm, setAuthForm] = useState({
    email: "",
    password: "",
    first_name: "",
    last_name: "",
    full_name: "",
    account_type: "Estudiante",
    knowledge_level: 1,
    questionnaire_answers: ["", "", "", "", ""]
  });

  const [providers, setProviders] = useState([]);
  const [modelsByProvider, setModelsByProvider] = useState({});
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");

  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(2000);
  const useRag = true;

  const [sessions, setSessions] = useState([]);
  const [sessionSearch, setSessionSearch] = useState("");
  const [sessionId, setSessionId] = useState(null);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const abortControllerRef = React.useRef(null);
  const [pendingAttachments, setPendingAttachments] = useState([]);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [docsOpen, setDocsOpen] = useState(false);
  const [deleteAccountOpen, setDeleteAccountOpen] = useState(false);
  const [deleteAccountConfirmText, setDeleteAccountConfirmText] = useState("");
  const [isDeletingAccount, setIsDeletingAccount] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [sessionDocuments, setSessionDocuments] = useState([]);
  const [isUploadingGlobalDocs, setIsUploadingGlobalDocs] = useState(false);
  const [isUploadingSessionDocs, setIsUploadingSessionDocs] = useState(false);
  const [documentsSummary, setDocumentsSummary] = useState("No cargado");
  const [deleteDocId, setDeleteDocId] = useState("");
  const [health, setHealth] = useState({ status: "...", version: "-", rag_status: "-", vector_store_documents: 0 });
  const [attachmentViewer, setAttachmentViewer] = useState(null);

  const [alert, setAlert] = useState(null);

  const [renamingSessionId, setRenamingSessionId] = useState(null);
  const [renameValue, setRenameValue] = useState("");
  const [confirmDeleteDocId, setConfirmDeleteDocId] = useState(null);
  const [confirmClearDocs, setConfirmClearDocs] = useState(false);

  const globalFileInputRef = useRef(null);
  const sessionFileInputRef = useRef(null);
  const chatViewportRef = useRef(null);
  const shouldStickToBottomRef = useRef(true);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const settingsPanelRef = useRef(null);
  const docsPanelRef = useRef(null);
  const deleteAccountPanelRef = useRef(null);
  const scrollRafRef = useRef(null);

  const modelsForCurrentProvider = useMemo(() => modelsByProvider[provider] || [], [modelsByProvider, provider]);
  const activeSession = useMemo(() => sessions.find((item) => item.session_id === sessionId) || null, [sessions, sessionId]);
  const hasMessages = messages.length > 0;
  const sessionStorageKey = getSessionStorageKey(currentUser?.user_id);
  const isAdmin = Boolean(currentUser?.is_admin);

  const scrollToConversationBottom = () => {
    if (!chatViewportRef.current) {
      return;
    }
    chatViewportRef.current.scrollTop = chatViewportRef.current.scrollHeight;
  };

  const updateScrollState = () => {
    if (scrollRafRef.current) return;
    scrollRafRef.current = requestAnimationFrame(() => {
      scrollRafRef.current = null;
      const viewport = chatViewportRef.current;
      if (!viewport) return;
      const distanceToBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;
      const isNearBottom = distanceToBottom < 44;
      shouldStickToBottomRef.current = isNearBottom;
      setShowScrollToBottom(!isNearBottom && hasMessages);
    });
  };

  const persistAuthState = (token, user) => {
    setAuthToken(token || "");
    setCurrentUser(user || null);

    if (token) {
      localStorage.setItem(AUTH_TOKEN_KEY, token);
    } else {
      localStorage.removeItem(AUTH_TOKEN_KEY);
    }

    if (user) {
      localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
    } else {
      localStorage.removeItem(AUTH_USER_KEY);
    }
  };

  const authHeaders = (extraHeaders = {}) => {
    const headers = { ...extraHeaders };
    if (authToken) {
      headers.Authorization = `Bearer ${authToken}`;
    }
    return headers;
  };

  const request = async (path, options = {}) => {
    const response = await fetch(`${apiBase}${path}`, {
      ...options,
      headers: authHeaders(options.headers || {}),
    });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const data = await response.json();
        detail = formatApiDetail(data, response.status);
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

  const requestWithoutAuth = async (path, options = {}) => {
    const response = await fetch(`${apiBase}${path}`, options);
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const data = await response.json();
        detail = formatApiDetail(data, response.status);
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
    notify("Configuración actualizada", "ok");
  };

  const loadHealth = async () => {
    const data = await request("/health");
    setHealth(data);
  };

  const loadCurrentUser = async () => {
    if (!authToken) {
      setAuthReady(true);
      setCurrentUser(null);
      return null;
    }

    try {
      const profile = await requestWithoutAuth("/auth/me", {
        headers: authHeaders(),
      });
      persistAuthState(authToken, profile);
      setAuthError("");
      return profile;
    } catch (error) {
      persistAuthState("", null);
      setAuthError("Sesion expirada. Inicia sesion de nuevo.");
      return null;
    } finally {
      setAuthReady(true);
    }
  };

  const logout = async () => {
    persistAuthState("", null);
    setMessages([]);
    setSessions([]);
    setSessionDocuments([]);
    setPendingAttachments([]);
    setAttachmentViewer(null);
    setInput("");
    setSessionId(null);
    localStorage.removeItem(sessionStorageKey);
    setAuthForm({ email: "", password: "", first_name: "", last_name: "", full_name: "", account_type: "Estudiante", knowledge_level: 1, questionnaire_answers: ["","","","",""] });
    setAuthMode("login");
    setRegisterStep(1);
    setAuthMessage("Sesion cerrada");
    setAuthError("");
    setHelpMenuOpen(false);
  };

  const deleteMyAccount = async () => {
    setIsDeletingAccount(true);
    try {
      await request("/auth/me", { method: "DELETE" });
      setDeleteAccountOpen(false);
      setDeleteAccountConfirmText("");
      await logout();
      setAuthMessage("Tu cuenta fue eliminada exitosamente");
      notify("Cuenta eliminada exitosamente", "ok");
    } catch (error) {
      notify(`Error al eliminar la cuenta: ${error.message}`, "error");
    } finally {
      setIsDeletingAccount(false);
    }
  };

  const updateAuthField = (field, value) => {
    setAuthForm((current) => ({ ...current, [field]: value }));
  };

  const updateQuestionnaireAnswer = (index, value) => {
    setAuthForm((current) => {
      const next = [...current.questionnaire_answers];
      next[index] = value;
      return { ...current, questionnaire_answers: next };
    });
  };

  const validateRegisterStep = (step) => {
    if (step === 1) {
      if (!authForm.first_name || authForm.first_name.trim().length < 2) {
        setAuthError("Ingresa tu nombre (mínimo 2 caracteres).");
        return false;
      }
      if (!authForm.last_name || authForm.last_name.trim().length < 2) {
        setAuthError("Ingresa tu apellido (mínimo 2 caracteres).");
        return false;
      }
      if (!authForm.email || !authForm.email.includes("@")) {
        setAuthError("Ingresa un correo electrónico válido.");
        return false;
      }
      if (!authForm.password || authForm.password.length < 8) {
        setAuthError("La contraseña debe tener al menos 8 caracteres.");
        return false;
      }
    }
    setAuthError(null);
    return true;
  };

  const submitAuth = async () => {
    setAuthSubmitting(true);
    setAuthError("");
    setAuthMessage("");

    try {
      if (authMode === "login") {
        const data = await requestWithoutAuth("/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email: authForm.email,
            password: authForm.password,
          }),
        });

        persistAuthState(data.access_token, data.user);
        const nextKey = getSessionStorageKey(data.user.user_id);
        const storedSessionId = localStorage.getItem(nextKey);
        setSessionId(storedSessionId || null);
        setAuthMessage(`Bienvenido, ${data.user.full_name}`);
        await Promise.all([loadConfig(), loadHealth(), loadSessions()]);
        if (storedSessionId) {
          await loadHistory(storedSessionId);
        }
      } else {
        const validationError = validateRegistrationForm(authForm);
        if (validationError) {
          throw new Error(validationError);
        }

        const questionnaireAnswers = AGILE_QUESTIONNAIRE.map((question, index) => ({
          question_number: question.id,
          answer: authForm.questionnaire_answers[index] || "d",
        }));
        const data = await requestWithoutAuth("/auth/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email: authForm.email,
            password: authForm.password,
            full_name: `${authForm.first_name.trim()} ${authForm.last_name.trim()}`.trim() || authForm.full_name,
            account_type: authForm.account_type,
            knowledge_level: Number(authForm.knowledge_level),
            questionnaire_answers: questionnaireAnswers,
          }),
        });

        persistAuthState(data.access_token, data.user);
        const nextKey = getSessionStorageKey(data.user.user_id);
        localStorage.removeItem(nextKey);
        setSessionId(null);
        setAuthMessage(`Cuenta creada. Nivel estimado: ${data.user.agile_adoption_label}`);
        await Promise.all([loadConfig(), loadHealth(), loadSessions()]);
      }
    } catch (error) {
      setAuthError(error.message);
    } finally {
      setAuthSubmitting(false);
    }
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
    if (currentUser?.user_id) {
      localStorage.setItem(getSessionStorageKey(currentUser.user_id), id);
    }
    setPendingAttachments([]);
    setAttachmentViewer(null);
    await loadSessionDocuments(id);
  };

  const createNewConversation = () => {
    setSessionId(null);
    if (currentUser?.user_id) {
      localStorage.removeItem(getSessionStorageKey(currentUser.user_id));
    }
    setMessages([]);
    setSessionDocuments([]);
    setPendingAttachments([]);
    setAttachmentViewer(null);
    setHelpMenuOpen(false);
  };

  const ensureSessionContextId = () => {
    if (!currentUser?.user_id) {
      return null;
    }
    if (sessionId) {
      return sessionId;
    }
    const generatedId = window.crypto?.randomUUID?.() || `session-${Date.now()}`;
    setSessionId(generatedId);
    localStorage.setItem(getSessionStorageKey(currentUser.user_id), generatedId);
    return generatedId;
  };

  const uploadGlobalDocument = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${apiBase}/documents/upload`, {
      method: "POST",
      headers: authHeaders(),
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
      headers: authHeaders(),
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
    if (!currentUser || !authToken) {
      notify("Inicia sesion para adjuntar documentos de sesion", "error");
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

    if (!currentUser || !authToken) {
      notify("Inicia sesion para adjuntar documentos de sesion", "error");
      event.target.value = "";
      return;
    }

    const targetSessionId = ensureSessionContextId();
    if (!targetSessionId) {
      notify("No se pudo crear una sesion de usuario", "error");
      event.target.value = "";
      return;
    }
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

    if (!currentUser || !authToken) {
      notify("Inicia sesion para pegar imagenes en una sesion", "error");
      return;
    }

    event.preventDefault();
    const targetSessionId = ensureSessionContextId();
    if (!targetSessionId) {
      notify("No se pudo crear una sesion de usuario", "error");
      return;
    }
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

  const handleDeleteDocument = (documentId) => {
    if (!documentId) return;
    setConfirmDeleteDocId(documentId);
  };

  const confirmDeleteDocument = async () => {
    const id = confirmDeleteDocId;
    if (!id) return;
    setConfirmDeleteDocId(null);
    try {
      await request(`/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
      notify("Documento eliminado", "ok");
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

  const startRename = (session) => {
    setRenamingSessionId(session.session_id);
    setRenameValue(session.title || getSessionLabel(session));
  };

  const commitRename = async () => {
    const id = renamingSessionId;
    if (!id) return;
    setRenamingSessionId(null);
    const cleaned = renameValue.trim();
    if (!cleaned) return;
    const session = sessions.find((s) => s.session_id === id);
    if (!session || cleaned === session.title) return;
    try {
      await request(`/sessions/${encodeURIComponent(id)}/title`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: cleaned })
      });
      setSessions((current) => current.map((item) =>
        item.session_id === id ? { ...item, title: cleaned } : item
      ));
      notify("Título actualizado", "ok");
    } catch (err) {
      notify(`Error al renombrar: ${err.message}`, "error");
    }
  };

  const streamChat = async (payload, assistantIndex) => {
    abortControllerRef.current = new AbortController();
    const response = await fetch(`${apiBase}/chat`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ ...payload, stream: true }),
      signal: abortControllerRef.current.signal
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
              if (currentUser?.user_id) {
                localStorage.setItem(getSessionStorageKey(currentUser.user_id), event.session_id);
              }
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

  const cancelStream = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const sendMessage = async () => {
    const message = input.trim();
    if (!message || isSending) {
      return;
    }

    if (!currentUser || !authToken) {
      notify("Inicia sesion para usar el chat", "error");
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
      if (error.name !== "AbortError") {
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
      }
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
    Promise.allSettled([loadConfig(), loadHealth()]);
    if (authToken) {
      loadCurrentUser().catch(() => {});
    } else {
      setAuthReady(true);
    }
  }, []);

  useEffect(() => {
    if (!currentUser) {
      setSessions([]);
      setMessages([]);
      setSessionDocuments([]);
      return;
    }
    loadSessions().catch(() => {});
    const storedSessionId = localStorage.getItem(getSessionStorageKey(currentUser.user_id));
    if (storedSessionId) {
      loadHistory(storedSessionId).catch(() => {
        setMessages([]);
      });
    } else {
      setSessionId(null);
      setMessages([]);
      setSessionDocuments([]);
    }
  }, [currentUser]);

  useEffect(() => {
    if (!currentUser || !sessionId) {
      setSessionDocuments([]);
      return;
    }
    loadSessionDocuments(sessionId).catch(() => {
      setSessionDocuments([]);
    });
  }, [sessionId, currentUser]);

  useEffect(() => {
    if (shouldStickToBottomRef.current) {
      scrollToConversationBottom();
    }
    updateScrollState();
  }, [messages]);


  useEffect(() => {
    updateScrollState();
  }, [hasMessages]);

  useEffect(() => {
    if (!settingsOpen || !settingsPanelRef.current) return;
    const panel = settingsPanelRef.current;
    const getFocusable = () => Array.from(panel.querySelectorAll(
      'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    ));
    const focusable = getFocusable();
    if (focusable.length) focusable[0].focus();
    const onKey = (e) => {
      if (e.key === "Escape") { setSettingsOpen(false); return; }
      if (e.key !== "Tab") return;
      const items = getFocusable();
      if (!items.length) return;
      if (e.shiftKey) {
        if (document.activeElement === items[0]) { e.preventDefault(); items[items.length - 1].focus(); }
      } else {
        if (document.activeElement === items[items.length - 1]) { e.preventDefault(); items[0].focus(); }
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [settingsOpen]);

  useEffect(() => {
    if (!deleteAccountOpen || !deleteAccountPanelRef.current) return;
    const panel = deleteAccountPanelRef.current;
    const getFocusable = () => Array.from(panel.querySelectorAll(
      'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    ));
    const focusable = getFocusable();
    if (focusable.length) focusable[0].focus();
    const onKey = (e) => {
      if (e.key === "Escape") { if (!isDeletingAccount) setDeleteAccountOpen(false); return; }
      if (e.key !== "Tab") return;
      const items = getFocusable();
      if (!items.length) return;
      if (e.shiftKey) {
        if (document.activeElement === items[0]) { e.preventDefault(); items[items.length - 1].focus(); }
      } else {
        if (document.activeElement === items[items.length - 1]) { e.preventDefault(); items[0].focus(); }
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [deleteAccountOpen, isDeletingAccount]);

  useEffect(() => {
    if (!docsOpen || !docsPanelRef.current) return;
    const panel = docsPanelRef.current;
    const getFocusable = () => Array.from(panel.querySelectorAll(
      'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    ));
    const focusable = getFocusable();
    if (focusable.length) focusable[0].focus();
    const onKey = (e) => {
      if (e.key === "Escape") { setDocsOpen(false); return; }
      if (e.key !== "Tab") return;
      const items = getFocusable();
      if (!items.length) return;
      if (e.shiftKey) {
        if (document.activeElement === items[0]) { e.preventDefault(); items[items.length - 1].focus(); }
      } else {
        if (document.activeElement === items[items.length - 1]) { e.preventDefault(); items[0].focus(); }
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [docsOpen]);

  return (
    <div className="app-shell">
      <aside className={`history-sidebar ${sidebarOpen ? "" : "collapsed"}`}>
        <div className="sidebar-header">
          {sidebarOpen
            ? <span className="sidebar-logo">AGILE</span>
            : <button className="sidebar-toggle" onClick={() => setSidebarOpen(true)} title="Abrir menú" aria-label="Abrir menú">☰</button>
          }
          {sidebarOpen && (
            <>
              <button className="new-chat-btn" onClick={createNewConversation} title="Nueva conversación" aria-label="Nueva conversación">+</button>
            </>
          )}
        </div>

        {sidebarOpen && (
          <button className="sidebar-toggle sidebar-close" style={{ position: "absolute", top: "12px", right: "50px" }}
            onClick={() => { setSidebarOpen(false); setHelpMenuOpen(false); }} title="Cerrar menú" aria-label="Cerrar menú">✕</button>
        )}

        <div className="session-list">
          {sidebarOpen
            ? groupSessionsByDate(sessions).map(([label, items]) => (
                <div key={label} className="session-group">
                  <div className="session-group-label">{label}</div>
                  {items.map((item) => (
                    <div
                      key={item.session_id}
                      className={`session-item ${item.session_id === sessionId ? "active" : ""}`}
                    >
                      {renamingSessionId === item.session_id ? (
                        <input
                          className="session-rename-input"
                          value={renameValue}
                          autoFocus
                          aria-label="Nuevo nombre de conversación"
                          onChange={(e) => setRenameValue(e.target.value)}
                          onKeyDown={(e) => { if (e.key === "Enter") commitRename(); if (e.key === "Escape") setRenamingSessionId(null); }}
                          onBlur={commitRename}
                        />
                      ) : (
                        <button
                          className="session-select"
                          onClick={() => loadHistory(item.session_id)}
                          title={getSessionLabel(item)}
                          aria-current={item.session_id === sessionId ? "true" : undefined}
                        >
                          {truncateText(getSessionLabel(item), 30)}
                        </button>
                      )}
                      <span className="session-item-actions">
                        <button className="session-rename" onClick={() => startRename(item)} title="Renombrar" aria-label={`Renombrar: ${getSessionLabel(item)}`}>✎</button>
                        <button className="session-delete" onClick={() => deleteSession(item.session_id)} title="Eliminar" aria-label={`Eliminar: ${getSessionLabel(item)}`}>×</button>
                      </span>
                    </div>
                  ))}
                </div>
              ))
            : sessions.map((item) => (
                <div key={item.session_id}
                  className={`session-item ${item.session_id === sessionId ? "active" : ""}`}
                  style={{ padding: "8px", display: "flex", justifyContent: "center" }}
                >
                  <button className="session-dot" onClick={() => loadHistory(item.session_id)} title={getSessionLabel(item)} aria-label={getSessionLabel(item)}>●</button>
                </div>
              ))
          }
        </div>

        <div className="sidebar-footer">
          {sidebarOpen && (
            <div className="sidebar-tools">
              <button className="sidebar-menu-trigger" onClick={toggleHelpMenu} title="Configuración y ayuda" aria-label="Configuración y ayuda" aria-expanded={helpMenuOpen}>⚙</button>
              <span className="sidebar-tools-label">Configuración</span>
            </div>
          )}

          {helpMenuOpen && (
            <div className="sidebar-help-menu">
              <button onClick={() => { setSettingsOpen(true); setHelpMenuOpen(false); }}>{isAdmin ? "Configuración avanzada" : "Configuración"}</button>
              {isAdmin && (
                <button onClick={() => { setDocsOpen(true); loadDocuments().catch(() => {}); setHelpMenuOpen(false); }}>Gestión de documentos</button>
              )}
              <button onClick={() => { setTheme(theme === "dark" ? "light" : "dark"); setHelpMenuOpen(false); }}>
                Tema: {theme === "dark" ? "Claro" : "Oscuro"}
              </button>
              <div className="sidebar-help-menu-divider" role="separator" />
              <button
                className="sidebar-help-menu-danger"
                onClick={() => { setDeleteAccountOpen(true); setDeleteAccountConfirmText(""); setHelpMenuOpen(false); }}
              >
                Eliminar mi cuenta
              </button>
            </div>
          )}

          {sidebarOpen && currentUser && (
            <div className="user-card">
              <div className="avatar">{getInitials(currentUser.full_name)}</div>
              <div className="user-info">
                <div className="user-name">{currentUser.full_name}</div>
                <div className="user-role">{currentUser.agile_adoption_label || "Agile learner"}</div>
              </div>
              <button className="sidebar-logout-btn" onClick={logout} title="Cerrar sesión" aria-label="Cerrar sesión">↪</button>
            </div>
          )}

          {sidebarOpen && !currentUser && (
            <div className="user-card empty">
              <div className="avatar">?</div>
              <div className="user-info">
                <div className="user-name">Sin sesión</div>
              </div>
            </div>
          )}
        </div>
      </aside>

      {!sidebarOpen && (
        <button className="floating-new-chat" onClick={createNewConversation} title="Nueva conversación" aria-label="Nueva conversación">＋</button>
      )}

      <main className="chat-main">
        <div className="app-brand" style={{ display: "none" }}></div>

        {currentUser && (
          <div className="chat-header">
            <span className="chat-session-title">
              {activeSession ? getSessionLabel(activeSession) : "Agile Assistant"}
            </span>
          </div>
        )}

        {authReady && !currentUser && (
          <section className="auth-gate">
            <div className="auth-card">
              <div className="auth-brand">
                <div className="auth-logo-text">AGILE ASSISTANT</div>
                <div className="auth-tagline">Tu tutor de metodologías ágiles</div>
              </div>

              <div className="tab-switcher">
                <button
                  className={`tab-btn ${authMode === "login" ? "active" : ""}`}
                  onClick={() => { setAuthMode("login"); setRegisterStep(1); setAuthError(null); setAuthMessage(""); setAuthForm({ email: "", password: "", first_name: "", last_name: "", full_name: "", account_type: "Estudiante", knowledge_level: 1, questionnaire_answers: ["","","","",""] }); }}
                >Iniciar sesión</button>
                <button
                  className={`tab-btn ${authMode === "register" ? "active" : ""}`}
                  onClick={() => { setAuthMode("register"); setRegisterStep(1); setAuthError(null); setAuthMessage(""); setAuthForm({ email: "", password: "", first_name: "", last_name: "", full_name: "", account_type: "Estudiante", knowledge_level: 1, questionnaire_answers: ["","","","",""] }); }}
                >Registrarse</button>
              </div>

              {authError && <p className="auth-status error">{authError}</p>}
              {authMessage && <p className="auth-status ok">{authMessage}</p>}

              {authMode === "login" && (
                <div className="field-group">
                  <div className="field">
                    <label htmlFor="login-email">Correo electrónico</label>
                    <input id="login-email" type="email" value={authForm.email} onChange={(e) => updateAuthField("email", e.target.value)} placeholder="usuario@correo.com" />
                  </div>
                  <div className="field">
                    <label htmlFor="login-password">Contraseña</label>
                    <input id="login-password" type="password" value={authForm.password} onChange={(e) => updateAuthField("password", e.target.value)} placeholder="••••••••" />
                  </div>
                </div>
              )}

              {authMode === "register" && (
                <>
                  <div className="progress-block">
                    <div className="progress-header">
                      <span className="progress-label">
                        {registerStep === 1 && "Tu cuenta"}
                        {registerStep === 2 && "Preguntas 1 – 3"}
                        {registerStep === 3 && "Preguntas 4 – 5"}
                        {registerStep === 4 && "Nivel general"}
                      </span>
                      <span className="progress-count">{registerStep} / 4</span>
                    </div>
                    <div
                      className="progress-track"
                      role="progressbar"
                      aria-valuenow={registerStep}
                      aria-valuemin={1}
                      aria-valuemax={4}
                      aria-label={`Paso ${registerStep} de 4`}
                    >
                      <div className="progress-fill" style={{ width: `${(registerStep / 4) * 100}%` }}></div>
                    </div>
                  </div>

                  <div className="step-dots">
                    {[1, 2, 3, 4].map((n) => (
                      <div key={n} className={`step-dot ${n === registerStep ? "active" : n < registerStep ? "done" : ""}`} />
                    ))}
                  </div>

                  {registerStep === 1 && (
                    <div className="field-group">
                      <div className="field-row">
                        <div className="field">
                          <label htmlFor="reg-first-name">Nombre</label>
                          <input id="reg-first-name" type="text" value={authForm.first_name} onChange={(e) => updateAuthField("first_name", e.target.value)} placeholder="Ej. Andy" />
                        </div>
                        <div className="field">
                          <label htmlFor="reg-last-name">Apellido</label>
                          <input id="reg-last-name" type="text" value={authForm.last_name} onChange={(e) => updateAuthField("last_name", e.target.value)} placeholder="Ej. Macas" />
                        </div>
                      </div>
                      <div className="field">
                        <label htmlFor="reg-email">Correo electrónico</label>
                        <input id="reg-email" type="email" value={authForm.email} onChange={(e) => updateAuthField("email", e.target.value)} placeholder="usuario@correo.com" />
                      </div>
                      <div className="field">
                        <label htmlFor="reg-password">Contraseña</label>
                        <input id="reg-password" type="password" value={authForm.password} onChange={(e) => updateAuthField("password", e.target.value)} placeholder="Mínimo 8 caracteres" />
                      </div>
                    </div>
                  )}

                  {registerStep === 2 && (
                    <div className="questionnaire-block">
                      {AGILE_QUESTIONNAIRE.slice(0, 3).map((question, index) => (
                        <div key={question.id} className="question-item">
                          <div className="q-number">Pregunta {question.id}</div>
                          <p className="question-text" id={`q2-${question.id}-label`}>{question.statement}</p>
                          <div className="question-options" role="radiogroup" aria-labelledby={`q2-${question.id}-label`}>
                            {question.options.map((option) => (
                              <button
                                key={option.value}
                                type="button"
                                className={`q-option ${authForm.questionnaire_answers[index] === option.value ? "selected" : ""}`}
                                aria-pressed={authForm.questionnaire_answers[index] === option.value}
                                onClick={() => updateQuestionnaireAnswer(index, option.value)}
                              >
                                <div className="q-radio" aria-hidden="true" />
                                {option.label}
                              </button>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {registerStep === 3 && (
                    <div className="questionnaire-block">
                      {AGILE_QUESTIONNAIRE.slice(3, 5).map((question, index) => (
                        <div key={question.id} className="question-item">
                          <div className="q-number">Pregunta {question.id}</div>
                          <p className="question-text" id={`q3-${question.id}-label`}>{question.statement}</p>
                          <div className="question-options" role="radiogroup" aria-labelledby={`q3-${question.id}-label`}>
                            {question.options.map((option) => (
                              <button
                                key={option.value}
                                type="button"
                                className={`q-option ${authForm.questionnaire_answers[index + 3] === option.value ? "selected" : ""}`}
                                aria-pressed={authForm.questionnaire_answers[index + 3] === option.value}
                                onClick={() => updateQuestionnaireAnswer(index + 3, option.value)}
                              >
                                <div className="q-radio" aria-hidden="true" />
                                {option.label}
                              </button>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {registerStep === 4 && (
                    <div className="level-selector" role="radiogroup" aria-label="Nivel de conocimiento ágil">
                      {[
                        { level: 1, text: "Principiante — No conozco las metodologías ágiles" },
                        { level: 2, text: "Básico — Conozco los conceptos fundamentales" },
                        { level: 3, text: "Intermedio — Aplico frameworks en proyectos reales" },
                        { level: 4, text: "Avanzado — Lidero equipos ágiles o soy coach certificado" },
                      ].map(({ level, text }) => (
                        <button
                          key={level}
                          type="button"
                          className={`level-card ${authForm.knowledge_level === level ? "selected" : ""}`}
                          aria-pressed={authForm.knowledge_level === level}
                          onClick={() => updateAuthField("knowledge_level", level)}
                        >
                          <div className="level-num" aria-hidden="true">{level}</div>
                          <div className="level-text">{text}</div>
                        </button>
                      ))}
                    </div>
                  )}
                </>
              )}

              {authMode === "login" ? (
                <button className="cta-btn" onClick={submitAuth} disabled={authSubmitting}>
                  {authSubmitting ? "Procesando..." : "Entrar"}
                </button>
              ) : registerStep < 4 ? (
                <button className="cta-btn" onClick={() => { if (validateRegisterStep(registerStep)) setRegisterStep((s) => s + 1); }}>
                  Continuar →
                </button>
              ) : (
                <button className="cta-btn" onClick={submitAuth} disabled={authSubmitting}>
                  {authSubmitting ? "Procesando..." : "✓ Crear mi cuenta"}
                </button>
              )}

              {authMode === "register" && registerStep > 1 && (
                <button className="cta-secondary" onClick={() => { setRegisterStep((s) => s - 1); setAuthError(null); }}>
                  ← Atrás
                </button>
              )}
              {authMode === "register" && registerStep === 1 && (
                <button className="cta-secondary" onClick={() => setAuthMode("login")}>
                  Ya tengo cuenta
                </button>
              )}
            </div>
          </section>
        )}

        <section ref={chatViewportRef} className="chat-viewport" onScroll={updateScrollState}>
          {currentUser && !hasMessages && (
            <div className="welcome-panel">
              <div className="welcome-kicker">Hola, {currentUser.full_name}</div>
              <h2>¿Cómo puedo ayudarte hoy?</h2>
              <p>Tu nivel detectado en Agilidad es {currentUser.agile_adoption_label}.</p>
            </div>
          )}

          {currentUser && hasMessages && (
            <div className="conversation-headline">
              {activeSession ? getSessionLabel(activeSession) : "Nueva conversación"}
            </div>
          )}

          <div className="messages-stack">
            {messages.map((item, index) => (
              <article key={`${item.role}-${index}`} className={`bubble ${item.role}`}>
                {String(item.role || "").toLowerCase() === "user"
                  ? <p>{item.content}</p>
                  : (
                    <div className="assistant-row">
                      <div className="bot-avatar" aria-hidden="true">✦</div>
                      <div className="bubble-text">
                        {String(item.content || "").trim().length > 0
                          ? <MarkdownContent content={item.content} enabled={markdownRendering} />
                          : item.role === "assistant" && isSending && index === messages.length - 1
                            ? (
                                <div className="streaming-row">
                                  <div className="typing-indicator">
                                    <div className="typing-dot"></div>
                                    <div className="typing-dot"></div>
                                    <div className="typing-dot"></div>
                                  </div>
                                  <button className="cancel-stream-btn" onClick={cancelStream}>✕ Cancelar</button>
                                </div>
                              )
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
                      </div>
                    </div>
                  )
                }
                {String(item.role || "").toLowerCase() === "user" && Array.isArray(item.attachments) && item.attachments.length > 0 && (
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
                {String(item.role || "").toLowerCase() === "user" && <SourceReferences sources={item.sources} />}
              </article>
            ))}
          </div>
        </section>

        <footer className="chat-input-area">
          {!currentUser && (
            <div className="auth-inline-banner">
              Inicia sesión para usar el chat y conservar tu historial personal.
            </div>
          )}
          <div className="input-shell">
            <input
              ref={sessionFileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt,.doc,.docx,.md,.markdown,.png,.jpg,.jpeg,.webp"
              onChange={handleSessionUpload}
              style={{ display: "none" }}
            />

            <textarea
              rows={1}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder={currentUser ? "Pregunta sobre metodologías ágiles…" : "Inicia sesión para usar el chat"}
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
                    {image && (item.previewUrl || attachmentSourceUrl(item)) && (
                      <button type="button" className="attachment-thumb-button" onClick={() => openAttachmentViewer(item)} title="Ver imagen">
                        <img className="attachment-thumb" src={item.previewUrl || attachmentSourceUrl(item)} alt={attachmentLabel(item)} />
                      </button>
                    )}
                    {!image && <span className="attachment-icon" aria-hidden="true">📎</span>}
                    <button type="button" className="attachment-remove-btn" onClick={() => removePendingAttachment(index)} title="Quitar adjunto" aria-label={`Quitar ${attachmentLabel(item)}`}>×</button>
                  </div>
                );
              })}
            </div>

            <div className="input-toolbar">
              <div className="toolbar-left">
                <button className="toolbar-chip" onClick={handleSessionUploadClick} disabled={isUploadingSessionDocs} title="Adjuntar archivo">
                  📎 Adjuntar
                </button>
                <select className="toolbar-model-select" value={provider} onChange={(e) => setProvider(e.target.value)}>
                  {(providers || []).map((item) => <option key={item} value={item}>{item}</option>)}
                </select>
                <select className="toolbar-model-select" value={model} onChange={(e) => setModel(e.target.value)}>
                  {(modelsForCurrentProvider || []).map((item) => <option key={item} value={item}>{item}</option>)}
                </select>
              </div>
              <button className="send-btn" onClick={sendMessage} disabled={isSending} title={isSending ? "Generando..." : "Enviar"} aria-label={isSending ? "Generando respuesta" : "Enviar mensaje"}>
                {isSending ? "…" : "↑"}
              </button>
            </div>
          </div>
        </footer>

        {showScrollToBottom && (
          <button className="scroll-bottom-btn" onClick={scrollToConversationBottom} title="Ir al final">
            Ir al final
          </button>
        )}
      </main>

      {alert && <div className={`alert ${alert.type}`} role="alert" aria-live="assertive">{alert.message}</div>}

      {settingsOpen && (
        <div className="overlay" onClick={() => setSettingsOpen(false)}>
          <section
            ref={settingsPanelRef}
            className="panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-dialog-title"
            onClick={(event) => event.stopPropagation()}
          >
            <h3 id="settings-dialog-title">{isAdmin ? "Configuración avanzada" : "Configuración"}</h3>
            {isAdmin && (
              <>
                <label htmlFor="settings-api-base">API Base URL</label>
                <input id="settings-api-base" value={apiBase} onChange={(event) => setApiBase(event.target.value)} />

                <label htmlFor="settings-temperature">Temperature: {temperature}</label>
                <input id="settings-temperature" type="range" min="0" max="1" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />

                <label htmlFor="settings-max-tokens">Max tokens</label>
                <input id="settings-max-tokens" type="number" min="1" value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value))} />
              </>
            )}

            <label className="checkbox-row">
              <input
                id="settings-markdown"
                type="checkbox"
                checked={markdownRendering}
                onChange={(event) => setMarkdownRendering(event.target.checked)}
              />
              Render Markdown seguro en respuestas
            </label>

            <div className="panel-actions">
              <button className="ghost-btn" onClick={() => setSettingsOpen(false)}>Cerrar</button>
              {isAdmin && (
                <button className="primary-btn" onClick={saveAdvancedConfig}>Guardar</button>
              )}
            </div>
          </section>
        </div>
      )}

      {deleteAccountOpen && (
        <div className="overlay" onClick={() => !isDeletingAccount && setDeleteAccountOpen(false)}>
          <section
            ref={deleteAccountPanelRef}
            className="panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-account-dialog-title"
            onClick={(event) => event.stopPropagation()}
          >
            <h3 id="delete-account-dialog-title">Eliminar mi cuenta</h3>
            <p>
              Esta acción es permanente y no se puede deshacer. Se eliminarán tu cuenta,
              tus conversaciones, mensajes y documentos adjuntos.
            </p>

            <label htmlFor="delete-account-confirm">Escribe ELIMINAR para confirmar</label>
            <input
              id="delete-account-confirm"
              value={deleteAccountConfirmText}
              onChange={(event) => setDeleteAccountConfirmText(event.target.value)}
              disabled={isDeletingAccount}
              autoComplete="off"
            />

            <div className="panel-actions">
              <button className="ghost-btn" onClick={() => setDeleteAccountOpen(false)} disabled={isDeletingAccount}>Cancelar</button>
              <button
                className="danger-btn"
                onClick={deleteMyAccount}
                disabled={deleteAccountConfirmText.trim() !== "ELIMINAR" || isDeletingAccount}
              >
                {isDeletingAccount ? "Eliminando…" : "Eliminar definitivamente"}
              </button>
            </div>
          </section>
        </div>
      )}

      {isAdmin && docsOpen && (
        <div className="overlay" onClick={() => setDocsOpen(false)}>
          <section
            ref={docsPanelRef}
            className="panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="docs-dialog-title"
            onClick={(event) => event.stopPropagation()}
          >
            <h3 id="docs-dialog-title">Gestión de documentos</h3>
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
                  const isPendingDelete = confirmDeleteDocId === documentId;
                  return (
                    <article key={`${item.filename}-${index}`} className="doc-entry">
                      <div className="doc-entry-head">
                        <strong>{item.filename || "Documento sin nombre"}</strong>
                        <span className="doc-type-pill">{(item.file_type || "unknown").toUpperCase()}</span>
                      </div>
                      <p><span className="doc-label">Origen:</span> {item.source || "Sin origen"}</p>
                      <p><span className="doc-label">ID:</span> <code>{documentId}</code></p>
                      <div className="doc-actions">
                        {isPendingDelete ? (
                          <div className="confirm-delete-row" role="group" aria-label="Confirmar eliminación">
                            <span>¿Eliminar este documento?</span>
                            <button className="danger-btn" onClick={confirmDeleteDocument}>Confirmar</button>
                            <button className="ghost-btn" onClick={() => setConfirmDeleteDocId(null)}>Cancelar</button>
                          </div>
                        ) : (
                          <button className="danger-btn" onClick={() => handleDeleteDocument(documentId)} aria-label={`Eliminar ${item.filename || "documento"}`}>Eliminar</button>
                        )}
                      </div>
                    </article>
                  );
                })
              )}
            </div>

            <div className="panel-actions">
              <button className="ghost-btn" onClick={() => setDocsOpen(false)}>Cerrar</button>
              {confirmClearDocs ? (
                <div className="confirm-delete-row" role="group" aria-label="Confirmar vaciado">
                  <span>¿Eliminar todos?</span>
                  <button className="danger-btn" onClick={() => { setConfirmClearDocs(false); clearDocuments(); }}>Confirmar</button>
                  <button className="ghost-btn" onClick={() => setConfirmClearDocs(false)}>Cancelar</button>
                </div>
              ) : (
                <button className="danger-btn" onClick={() => setConfirmClearDocs(true)}>Vaciar todo</button>
              )}
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

      <div className="sr-only" role="status" aria-live="polite" aria-atomic="false">
        {messages.length > 0 && messages[messages.length - 1].role === "assistant"
          ? messages[messages.length - 1].content
          : ""}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
