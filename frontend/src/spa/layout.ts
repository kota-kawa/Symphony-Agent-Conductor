import { $, $$ } from "./dom-utils";

type ViewKey = "general" | "browser" | "iot" | "chat" | "schedule";

type ViewActivationPayload = {
  view: ViewKey;
  isBrowserView: boolean;
  isChatView: boolean;
  isIotView: boolean;
  isGeneralView: boolean;
  isSchedulerView: boolean;
};

type GeneralProxyRenderPayload = {
  agent: string | null;
  view: ViewKey | null;
  currentView: ViewKey;
};

type GeneralProxyAgentPayload = {
  previousAgent: string | null;
  agent: string | null;
  targetView: ViewKey | null;
};

let initialized = false;

let layoutEl: HTMLElement | null = null;
let sidebarEl: HTMLElement | null = null;
let sidebarToggle: HTMLButtonElement | null = null;

let views: Record<ViewKey, HTMLElement | null> = {
  general: null,
  browser: null,
  iot: null,
  chat: null,
  schedule: null,
};

let appTitle: HTMLElement | null = null;
let navButtons: HTMLButtonElement[] = [];
let sidebarChatTitle: HTMLElement | null = null;
let sidebarChatIcon: HTMLElement | null = null;
let sidebarChatTitleTextNode: ChildNode | null = null;

let generalDefaultContent: HTMLElement | null = null;
let generalProxyStatus: HTMLElement | null = null;
let generalProxyContainer: HTMLElement | null = null;
let generalViewPanel: HTMLElement | null = null;

let generalProxyFrame: HTMLDivElement | null = null;
let generalProxyIframe: HTMLIFrameElement | null = null;
let generalProxyIframeSrc = "";

let generalBrowserSurface: HTMLDivElement | null = null;
let generalBrowserStage: HTMLDivElement | null = null;
let generalBrowserFullscreenBtn: HTMLButtonElement | null = null;

let sidebarTogglePositionRaf: number | null = null;

const viewPlacements = new Map<HTMLElement, { parent: HTMLElement | null; placeholder: Comment }>();

const ICONS = {
  generalChat: `<svg viewBox="0 0 24 24" fill="currentColor" focusable="false"><path d="M4 4h16a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H8l-4 4V5a1 1 0 0 1 1-1z"/></svg>`,
  chat: `<svg viewBox="0 0 24 24" fill="currentColor" focusable="false"><path d="M3 10v11h6v-7h6v7h6v-11L12,3z"/></svg>`,
  browser: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" focusable="false"><circle cx="12" cy="12" r="9"/><path d="M12 3c-4 0-4 18 0 18 4 0 4-18 0-18"/><path d="M3 12c0-4 18-4 18 0 0 4-18 4-18 0"/></svg>`,
  iot: `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M9 2h6v2h2v2h2v6h-2v2h-2v2h-6v-2H7v-2H5V6h2V4h2V2zm0 4v2H7v6h2v2h6v-2h2V8h-2V6H9z"/></svg>`,
  scheduler: `<svg viewBox="0 0 24 24" fill="currentColor" focusable="false"><path d="M19 4h-1V2h-2v2H8V2H6v2H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2Zm0 14H5V10h14Zm0-12v2H5V6Z"/></svg>`,
};

const AGENT_TO_VIEW_MAP: Record<string, ViewKey> = {
  browser: "browser",
  browser_agent: "browser",
  web: "browser",
  web_agent: "browser",
  navigator: "browser",
  iot: "iot",
  iot_agent: "iot",
  lifestyle: "chat",
  life_style: "chat",
  "life-style": "chat",
  faq: "chat",
  qa: "chat",
  qa_agent: "chat",
  "qa-agent": "chat",
  knowledge: "chat",
  knowledge_base: "chat",
  docs: "chat",
  gemini: "chat",
  faq_gemini: "chat",
  chat: "chat",
  scheduler: "schedule",
  scheduler_agent: "schedule",
};

const AGENT_RESULT_TARGETS: Record<string, string> = {
  browser: "browser",
  browser_agent: "browser",
  web: "browser",
  web_agent: "browser",
  navigator: "browser",
  iot: "iot",
  iot_agent: "iot",
  lifestyle: "lifestyle",
  life_style: "lifestyle",
  "life-style": "lifestyle",
  faq: "lifestyle",
  qa: "lifestyle",
  qa_agent: "lifestyle",
  "qa-agent": "lifestyle",
  knowledge: "lifestyle",
  knowledge_base: "lifestyle",
  docs: "lifestyle",
  gemini: "lifestyle",
  faq_gemini: "lifestyle",
  chat: "lifestyle",
  scheduler: "scheduler",
  scheduler_agent: "scheduler",
};

const AGENT_RESULT_PATHS: Record<string, string> = {
  browser: "/agent-result",
  lifestyle: "/agent-result",
  iot: "/agent_result.html",
  scheduler: "/agent-result",
};

const GENERAL_PROXY_AGENT_LABELS: Record<string, string> = {
  lifestyle: "Life-Styleエージェント",
  "life-style": "Life-Styleエージェント",
  life_style: "Life-Styleエージェント",
  browser: "ブラウザエージェント",
  browser_agent: "ブラウザエージェント",
  iot: "IoT エージェント",
  iot_agent: "IoT エージェント",
  scheduler: "Scheduler エージェント",
  scheduler_agent: "Scheduler エージェント",
  chat: "要約チャット",
};

const BROWSER_AGENT_FINAL_MARKER = "[browser-agent-final]";

function sanitizeBase(value: unknown): string {
  return typeof value === "string" ? value.trim().replace(/\/+$/, "") : "";
}

function readQueryParam(name: string): string {
  try {
    return new URLSearchParams(window.location.search).get(name) || "";
  } catch {
    return "";
  }
}

function resolveBrowserAgentResultBase(): string {
  const sources = [
    sanitizeBase(readQueryParam("browser_agent_base")),
    sanitizeBase((window as any).BROWSER_AGENT_API_BASE),
    sanitizeBase(document.querySelector<HTMLMetaElement>("meta[name='browser-agent-api-base']")?.content),
  ];
  for (const base of sources) {
    if (base) return base;
  }
  return "http://localhost:5005";
}

function resolveLifestyleAgentResultBase(): string {
  const sources = [
    sanitizeBase(readQueryParam("lifestyle_agent_base")),
    sanitizeBase((window as any).LIFESTYLE_AGENT_BASE),
    sanitizeBase(document.querySelector<HTMLMetaElement>("meta[name='lifestyle-agent-api-base']")?.content),
  ];
  for (const base of sources) {
    if (base) return base;
  }
  if (window.location.origin && window.location.origin !== "null") {
    return `${window.location.origin.replace(/\/+$/, "")}/lifestyle_agent`;
  }
  return "http://localhost:5000";
}

function resolveIotAgentResultBase(): string {
  const sources = [
    sanitizeBase(readQueryParam("iot_agent_base")),
    sanitizeBase((window as any).IOT_AGENT_API_BASE),
    sanitizeBase(document.querySelector<HTMLMetaElement>("meta[name='iot-agent-api-base']")?.content),
  ];
  for (const base of sources) {
    if (base) return base;
  }
  if (window.location.origin && window.location.origin !== "null") {
    return `${window.location.origin.replace(/\/+$/, "")}/iot_agent`;
  }
  return "https://iot-agent.project-kk.com";
}

function resolveSchedulerAgentResultBase(): string {
  const sources = [
    sanitizeBase(readQueryParam("scheduler_agent_base")),
    sanitizeBase((window as any).SCHEDULER_AGENT_BASE),
    sanitizeBase(document.querySelector<HTMLMetaElement>("meta[name='scheduler-agent-api-base']")?.content),
  ];
  for (const base of sources) {
    if (base) return base;
  }
  if (window.location.origin && window.location.origin !== "null") {
    return `${window.location.origin.replace(/\/+$/, "")}/scheduler_agent`;
  }
  return "http://localhost:5010";
}

const AGENT_RESULT_BASES = {
  browser: resolveBrowserAgentResultBase(),
  lifestyle: resolveLifestyleAgentResultBase(),
  iot: resolveIotAgentResultBase(),
  scheduler: resolveSchedulerAgentResultBase(),
};

function resolveAgentResultBase(agentKey: string): string {
  if (typeof agentKey !== "string") return "";
  const normalized = agentKey.trim().toLowerCase();
  const target = AGENT_RESULT_TARGETS[normalized] || normalized;
  return (AGENT_RESULT_BASES as Record<string, string>)[target] || "";
}

function buildAgentResultUrl(agentKey: string): string {
  if (!agentKey) return "";
  const normalized = agentKey.trim().toLowerCase();
  const target = AGENT_RESULT_TARGETS[normalized] || normalized;
  const path = AGENT_RESULT_PATHS[target] || "/agent-result";
  const base = resolveAgentResultBase(normalized);
  if (!base) return path;
  if (/^https?:/i.test(path)) return path;
  const cleanedBase = base.replace(/\/+$/, "");
  const cleanedPath = path.startsWith("/") ? path : `/${path}`;
  return `${cleanedBase}${cleanedPath}`;
}

function resolveAgentLabel(agentKey: string): string {
  if (!agentKey) return "";
  const normalized = agentKey.trim().toLowerCase();
  const directLabel = GENERAL_PROXY_AGENT_LABELS[normalized];
  if (directLabel) return directLabel;
  const target = AGENT_RESULT_TARGETS[normalized] || normalized;
  return GENERAL_PROXY_AGENT_LABELS[target] || agentKey;
}

function initAgentResultHosts() {
  const hosts = $$(".agent-result-view");
  if (!hosts.length) return;
  hosts.forEach((host) => {
    if (!host) return;
    const agentKey = (host as HTMLElement).dataset.agent || "";
    if (!agentKey) return;
    const url = buildAgentResultUrl(agentKey);
    if (!url) return;
    let iframe = host.querySelector<HTMLIFrameElement>("iframe");
    if (!iframe) {
      iframe = document.createElement("iframe");
      iframe.setAttribute("loading", "lazy");
      iframe.setAttribute("allow", "fullscreen");
      host.appendChild(iframe);
    }
    iframe.setAttribute("title", `${resolveAgentLabel(agentKey) || agentKey} 結果`);
    if (iframe.src !== url) {
      iframe.src = url;
    }
  });
}

export function containsBrowserAgentFinalMarker(text: string): boolean {
  return typeof text === "string" && text.includes(BROWSER_AGENT_FINAL_MARKER);
}

let generalProxyTargetView: ViewKey | null = null;
let generalProxyAgentKey: string | null = null;
let generalProxyViewKey: ViewKey | null = null;
let currentViewKey: ViewKey = "general";
let viewActivationHook: ((payload: ViewActivationPayload) => void) | null = null;
let generalProxyRenderHook: ((payload: GeneralProxyRenderPayload) => void) | null = null;
let generalProxyAgentHook: ((payload: GeneralProxyAgentPayload) => void) | null = null;

export function getInitialActiveView(): ViewKey {
  const active = document.querySelector<HTMLButtonElement>(".nav-btn.active")?.dataset.view;
  const key = (active as ViewKey) || "general";
  return Object.prototype.hasOwnProperty.call(views, key) ? key : "general";
}

export function registerViewActivationHook(handler: ((payload: ViewActivationPayload) => void) | null) {
  viewActivationHook = typeof handler === "function" ? handler : null;
}

export function registerGeneralProxyRenderHook(handler: ((payload: GeneralProxyRenderPayload) => void) | null) {
  generalProxyRenderHook = typeof handler === "function" ? handler : null;
}

export function registerGeneralProxyAgentHook(handler: ((payload: GeneralProxyAgentPayload) => void) | null) {
  generalProxyAgentHook = typeof handler === "function" ? handler : null;
}

function resolveAgentToView(agentKey: string): ViewKey | null {
  if (typeof agentKey !== "string") return null;
  const normalized = agentKey.trim().toLowerCase();
  if (!normalized) return null;
  return AGENT_TO_VIEW_MAP[normalized] || null;
}

function ensureGeneralProxyFrame(): HTMLIFrameElement | null {
  if (!generalProxyContainer) return null;
  if (!generalProxyFrame) {
    generalProxyFrame = document.createElement("div");
    generalProxyFrame.className = "general-view__proxy-frame";
  }
  if (!generalProxyIframe) {
    generalProxyIframe = document.createElement("iframe");
    generalProxyIframe.className = "general-view__proxy-iframe";
    generalProxyIframe.setAttribute("title", "エージェント結果");
    generalProxyIframe.setAttribute("loading", "lazy");
    generalProxyIframe.setAttribute("allow", "fullscreen");
    generalProxyFrame.appendChild(generalProxyIframe);
  }
  if (generalProxyFrame.parentElement !== generalProxyContainer) {
    generalProxyContainer.innerHTML = "";
    generalProxyContainer.appendChild(generalProxyFrame);
  }
  return generalProxyIframe;
}

function updateGeneralProxyFrame(agentKey: string) {
  const iframe = ensureGeneralProxyFrame();
  if (!iframe) return;
  const nextSrc = buildAgentResultUrl(agentKey);
  if (!nextSrc) return;
  const label = resolveAgentLabel(agentKey) || "エージェント";
  iframe.setAttribute("title", `${label} 結果`);
  if (generalProxyIframeSrc !== nextSrc) {
    generalProxyIframeSrc = nextSrc;
    iframe.src = nextSrc;
  }
}

function clearGeneralProxyFrame() {
  generalProxyIframeSrc = "";
  if (generalProxyFrame && generalProxyFrame.parentElement) {
    generalProxyFrame.parentElement.removeChild(generalProxyFrame);
  }
}

function ensureViewPlacement(viewEl: HTMLElement): { parent: HTMLElement | null; placeholder: Comment } {
  let placement = viewPlacements.get(viewEl);
  if (!placement) {
    placement = {
      parent: viewEl.parentElement as HTMLElement | null,
      placeholder: document.createComment(`placeholder:${viewEl.id || ""}`),
    };
    viewPlacements.set(viewEl, placement);
  }
  return placement;
}

function restoreView(viewKey: ViewKey) {
  if (viewKey === "browser") {
    deactivateGeneralBrowserProxy();
    return;
  }

  const viewEl = views[viewKey];
  if (!viewEl) return;
  const placement = viewPlacements.get(viewEl);
  if (!placement) return;
  const { parent, placeholder } = placement;
  if (!parent) return;
  if (placeholder.parentNode) {
    placeholder.parentNode.replaceChild(viewEl, placeholder);
  } else {
    parent.appendChild(viewEl);
  }
  viewEl.classList.remove("general-proxy-active");
}

function moveViewToGeneral(viewKey: ViewKey) {
  if (viewKey === "browser") {
    if (generalProxyViewKey && generalProxyViewKey !== viewKey) {
      restoreView(generalProxyViewKey);
      generalProxyViewKey = null;
    }
    activateGeneralBrowserProxy();
    generalProxyViewKey = viewKey;
    return;
  }

  const viewEl = views[viewKey];
  if (!generalProxyContainer || !viewEl) return;
  if (generalProxyViewKey && generalProxyViewKey !== viewKey) {
    restoreView(generalProxyViewKey);
    generalProxyViewKey = null;
  }
  const placement = ensureViewPlacement(viewEl);
  if (!placement || !placement.parent) return;
  if (viewEl.parentElement !== generalProxyContainer) {
    placement.parent.replaceChild(placement.placeholder, viewEl);
    generalProxyContainer.appendChild(viewEl);
  }
  viewEl.classList.add("general-proxy-active");
  generalProxyViewKey = viewKey;
}

function clearGeneralProxy() {
  generalProxyViewKey = null;
  clearGeneralProxyFrame();
  if (generalProxyContainer) {
    generalProxyContainer.innerHTML = "";
  }
}

function updateGeneralViewProxy() {
  const hasProxyAgent = Boolean(generalProxyAgentKey);
  if (hasProxyAgent && generalProxyAgentKey) {
    updateGeneralProxyFrame(generalProxyAgentKey);
  } else {
    clearGeneralProxy();
  }

  const shouldShowProxy = currentViewKey === "general" && hasProxyAgent;

  if (generalViewPanel) {
    generalViewPanel.classList.toggle("general-view--has-proxy", shouldShowProxy);
  }
  if (generalDefaultContent) {
    generalDefaultContent.hidden = shouldShowProxy;
  }
  if (generalProxyContainer) {
    generalProxyContainer.hidden = !shouldShowProxy;
  }
  if (generalProxyStatus) {
    if (shouldShowProxy && generalProxyAgentKey) {
      const agentLabel = resolveAgentLabel(generalProxyAgentKey);
      const labelText = `オーケストレーターは現在「${agentLabel}」を使用しています。`;
      generalProxyStatus.textContent = `${labelText}下の結果画面で進行状況を確認できます。`;
      generalProxyStatus.hidden = false;
    } else {
      generalProxyStatus.hidden = true;
      generalProxyStatus.textContent = "";
    }
  }

  if (!shouldShowProxy) {
    if (typeof generalProxyRenderHook === "function") {
      generalProxyRenderHook({
        agent: generalProxyAgentKey,
        view: null,
        currentView: currentViewKey,
      });
    }
    return;
  }

  if (typeof generalProxyRenderHook === "function") {
    generalProxyRenderHook({
      agent: generalProxyAgentKey,
      view: generalProxyTargetView,
      currentView: currentViewKey,
    });
  }
}

export function isGeneralProxyAgentBrowser(): boolean {
  return generalProxyAgentKey === "browser";
}

export function setGeneralProxyAgent(agentKey: string | null) {
  const normalizedAgent = typeof agentKey === "string" ? agentKey.trim().toLowerCase() : "";
  const targetView = resolveAgentToView(normalizedAgent);
  const previousAgent = generalProxyAgentKey;
  generalProxyAgentKey = targetView ? normalizedAgent : null;
  generalProxyTargetView = targetView;
  if (typeof generalProxyAgentHook === "function") {
    generalProxyAgentHook({
      previousAgent,
      agent: generalProxyAgentKey,
      targetView: generalProxyTargetView,
    });
  }
  updateGeneralViewProxy();
}

export function determineGeneralProxyAgentFromResult(result: any): string | null {
  if (!result || typeof result !== "object") return null;
  const executions = Array.isArray(result.executions) ? result.executions : [];
  for (let index = executions.length - 1; index >= 0; index -= 1) {
    const agent = executions[index]?.agent;
    if (typeof agent === "string" && agent.trim()) {
      return agent.trim().toLowerCase();
    }
  }
  const tasks = Array.isArray(result.tasks) ? result.tasks : [];
  const nextTask = tasks.find((task: any) => typeof task?.agent === "string" && task.agent.trim());
  return nextTask ? nextTask.agent.trim().toLowerCase() : null;
}

export function activateView(viewKey: ViewKey) {
  const target = Object.prototype.hasOwnProperty.call(views, viewKey) ? viewKey : "browser";
  currentViewKey = target;
  navButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.view === target);
  });
  Object.entries(views).forEach(([key, el]) => {
    if (!el) return;
    el.classList.toggle("active", key === target);
  });
  const titles: Record<ViewKey, string> = {
    general: "一般ビュー",
    browser: "リモートブラウザ",
    iot: "IoT ダッシュボード",
    chat: "Life",
    schedule: "Schedule",
  };
  if (appTitle) {
    appTitle.textContent = titles[target] ?? "リモートブラウザ";
  }

  if (sidebarChatTitleTextNode && sidebarChatIcon) {
    const isGeneralViewActive = target === "general";
    let titleText = isGeneralViewActive ? " 共通チャット" : " Life-Style エージェント";
    let iconSvg = isGeneralViewActive ? ICONS.generalChat : ICONS.chat;

    if (target === "browser") {
      titleText = " ブラウザエージェント";
      iconSvg = ICONS.browser;
    } else if (target === "iot") {
      titleText = " IoTエージェント";
      iconSvg = ICONS.iot;
    } else if (target === "schedule") {
      titleText = " Scheduler-Agent";
      iconSvg = ICONS.scheduler;
    }

    if (sidebarChatTitleTextNode.nodeType === Node.TEXT_NODE) {
      sidebarChatTitleTextNode.textContent = titleText;
    }
    sidebarChatIcon.innerHTML = iconSvg;
  }

  const isBrowserView = target === "browser";
  const isChatView = target === "chat";
  const isIotView = target === "iot";
  const isGeneralView = target === "general";
  const isSchedulerView = target === "schedule";

  if (typeof viewActivationHook === "function") {
    viewActivationHook({
      view: target,
      isBrowserView,
      isChatView,
      isIotView,
      isGeneralView,
      isSchedulerView,
    });
  }
  updateGeneralViewProxy();
  scheduleSidebarTogglePosition();
}

function updateSidebarTogglePosition() {
  if (!layoutEl || !sidebarEl) return;

  const sidebarRect = sidebarEl.getBoundingClientRect();
  const layoutRect = layoutEl.getBoundingClientRect();
  if (sidebarRect.height <= 0) return;

  const offset = sidebarRect.top - layoutRect.top + sidebarRect.height / 2;
  layoutEl.style.setProperty("--sidebar-toggle-top", `${offset}px`);
}

const scheduleSidebarTogglePosition = () => {
  if (!layoutEl || !sidebarEl) return;
  if (sidebarTogglePositionRaf !== null) return;

  sidebarTogglePositionRaf = requestAnimationFrame(() => {
    sidebarTogglePositionRaf = null;
    updateSidebarTogglePosition();
  });
};

/* ---------- Browser stage (noVNC 風) ---------- */
const noVncControllers = new Set<any>();
let generalBrowserController: any = null;
let mainBrowserController: any = null;

const ALLOWED_RESIZE_VALUES = new Set(["scale", "remote", "off"]);
const DEFAULT_NOVNC_PARAMS = {
  autoconnect: "1",
  resize: "scale",
  scale: "auto",
  view_clip: "false",
};

function normalizeBrowserEmbedUrl(value: string): string {
  if (!value) return value;

  try {
    const url = new URL(value, window.location.origin);
    const params = url.searchParams;

    for (const [key, defaultValue] of Object.entries(DEFAULT_NOVNC_PARAMS)) {
      const currentValue = params.get(key);
      if (key === "resize") {
        if (!currentValue || !ALLOWED_RESIZE_VALUES.has(currentValue)) {
          params.set(key, defaultValue);
        }
        continue;
      }

      if (key === "view_clip") {
        if (currentValue?.toLowerCase() !== defaultValue) {
          params.set(key, defaultValue);
        }
        continue;
      }

      if (!currentValue) {
        params.set(key, defaultValue);
      }
    }

    return url.toString();
  } catch {
    return value;
  }
}

function resolveBrowserEmbedUrl(): string {
  const sanitize = (value: unknown) => (typeof value === "string" ? value.trim() : "");
  const hasProtocol = (value: string) => /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(value);
  const localHosts = new Set(["localhost", "127.0.0.1", "::1"]);
  const isLocalHost = (host: string) => typeof host === "string" && localHosts.has(host.toLowerCase());
  const preferredProtocol = () => {
    if (window.location.protocol === "http:" || window.location.protocol === "https:") {
      return window.location.protocol;
    }
    return "http:";
  };

  let queryValue = "";
  try {
    queryValue = new URLSearchParams(window.location.search).get("browser_embed_url") || "";
  } catch {
    queryValue = "";
  }

  const sources = [
    sanitize(queryValue),
    sanitize((window as any).BROWSER_EMBED_URL),
    sanitize(document.querySelector<HTMLMetaElement>("meta[name='browser-embed-url']")?.content),
  ];

  for (const candidate of sources) {
    if (!candidate) continue;
    if (hasProtocol(candidate)) {
      try {
        const parsed = new URL(candidate);
        if (!isLocalHost(window.location.hostname) && isLocalHost(parsed.hostname)) {
          parsed.hostname = window.location.hostname;
          if (!parsed.port) {
            parsed.port = "7900";
          }
          parsed.protocol = preferredProtocol();
        }
        return normalizeBrowserEmbedUrl(parsed.toString());
      } catch {
        return normalizeBrowserEmbedUrl(candidate);
      }
    }
    try {
      const absolute = new URL(candidate, window.location.origin);
      if (!isLocalHost(window.location.hostname) && isLocalHost(absolute.hostname)) {
        absolute.hostname = window.location.hostname;
        if (!absolute.port) {
          absolute.port = "7900";
        }
        absolute.protocol = preferredProtocol();
      }
      return normalizeBrowserEmbedUrl(absolute.toString());
    } catch {
      continue;
    }
  }

  const fallbackBase =
    !isLocalHost(window.location.hostname)
      ? `${preferredProtocol()}//${window.location.hostname}:7900/`
      : "http://127.0.0.1:7900/";
  return normalizeBrowserEmbedUrl(
    `${fallbackBase}vnc_lite.html?autoconnect=1&resize=scale&scale=auto&view_clip=false`,
  );
}

const BROWSER_EMBED_URL = resolveBrowserEmbedUrl();

function reloadBrowserIframeWithCacheBust(iframe: HTMLIFrameElement | null) {
  if (!iframe) return;
  const base = iframe.src || BROWSER_EMBED_URL;
  try {
    const url = new URL(base, window.location.origin);
    url.searchParams.set("_ts", Date.now().toString(36));
    iframe.src = url.toString();
  } catch {
    iframe.src = base;
  }
}

function createNoVncController({ stage, fullscreenButton, context = "default" }: { stage: HTMLElement | null; fullscreenButton?: HTMLButtonElement | null; context?: string; }) {
  if (!stage) return null;

  const state = {
    iframe: null as HTMLIFrameElement | null,
    origin: "*",
    deferredRaf: null as number | null,
    deferredReloadFallback: false,
    stageResizeObserver: null as ResizeObserver | null,
    stageResizeRaf: null as number | null,
    statusEl: null as HTMLElement | null,
    connectTimer: null as number | null,
    connectAttempts: 0,
    lastReadyAt: 0,
  };

  const controller = {
    context,
    ensureIframe,
    requestSync,
    sync,
    reload,
    matchesWindow,
    markReady,
    getStage: () => stage,
    getIframe: () => state.iframe,
  };

  const CONNECT_CHECK_MS = 8000;
  const CONNECT_RETRY_DELAYS = [2000, 4000, 7000];

  function ensureStatusEl(): HTMLElement | null {
    if (state.statusEl) return state.statusEl;
    const stageEl = stage as HTMLElement;
    let fallback = stageEl.querySelector<HTMLElement>(".stage-fallback");
    if (!fallback) {
      fallback = document.createElement("p");
      fallback.className = "stage-fallback";
      stageEl.appendChild(fallback);
    }
    state.statusEl = fallback;
    return fallback;
  }

  function setStatus({ message, kind = "loading", hidden = false }: { message?: string; kind?: "loading" | "error"; hidden?: boolean; } = {}) {
    const el = ensureStatusEl();
    if (!el) return;
    el.textContent = message || "";
    el.classList.toggle("stage-fallback--error", kind === "error");
    el.classList.toggle("stage-fallback--loading", kind === "loading");
    el.hidden = Boolean(hidden);
  }

  function clearConnectTimer() {
    if (state.connectTimer) {
      clearTimeout(state.connectTimer);
      state.connectTimer = null;
    }
  }

  function getEmbedHostLabel() {
    try {
      const parsed = new URL(BROWSER_EMBED_URL, window.location.origin);
      return parsed.host || parsed.hostname || "127.0.0.1:7900";
    } catch {
      return "127.0.0.1:7900";
    }
  }

  function buildFailureMessage({ retrying = false }: { retrying?: boolean } = {}) {
    const hostLabel = getEmbedHostLabel();
    const lines = [
      "リモートブラウザに接続できませんでした。",
      `noVNC(Websockify) が ${hostLabel} で起動しているか確認してください。`,
      "必要なら設定の BROWSER_EMBED_URL を正しいホストに変更してください。",
    ];
    if (window.location.protocol === "https:" && BROWSER_EMBED_URL.startsWith("http://")) {
      lines.push("https で開いている場合は https の埋め込みURLを指定してください。");
    }
    if (retrying) {
      lines.push("再接続を試しています…");
    }
    return lines.join(" ");
  }

  function scheduleConnectionCheck(delay = CONNECT_CHECK_MS) {
    clearConnectTimer();
    state.connectTimer = window.setTimeout(() => {
      if (state.lastReadyAt) {
        return;
      }
      const attempt = state.connectAttempts;
      if (attempt < CONNECT_RETRY_DELAYS.length) {
        state.connectAttempts += 1;
        setStatus({ kind: "error", message: buildFailureMessage({ retrying: true }) });
        const retryDelay = CONNECT_RETRY_DELAYS[attempt] || 0;
        setTimeout(() => {
          controller.reload();
          scheduleConnectionCheck(CONNECT_CHECK_MS);
        }, retryDelay);
      } else {
        setStatus({ kind: "error", message: buildFailureMessage() });
      }
    }, delay);
  }

  function markReady() {
    state.lastReadyAt = Date.now();
    state.connectAttempts = 0;
    clearConnectTimer();
    setStatus({ hidden: true });
  }

  function ensureIframe() {
    if (!stage || !BROWSER_EMBED_URL) {
      return null;
    }

    let iframe = stage.querySelector<HTMLIFrameElement>("iframe");
    const titleSuffix = context === "general-proxy" ? " (一般ビュー)" : "";
    if (!iframe) {
      stage.innerHTML = "";
      iframe = document.createElement("iframe");
      iframe.setAttribute("title", `埋め込みブラウザ${titleSuffix}`);
      iframe.setAttribute("allow", "fullscreen");
      iframe.addEventListener("load", () => {
        state.lastReadyAt = 0;
        state.connectAttempts = 0;
        setStatus({ kind: "loading", message: "リモートブラウザを読み込んでいます…" });
        scheduleConnectionCheck();
        controller.requestSync();
      });
      stage.appendChild(iframe);
    }

    if (iframe.src !== BROWSER_EMBED_URL) {
      iframe.src = BROWSER_EMBED_URL;
    }

    try {
      const parsed = new URL(iframe.src, window.location.origin);
      state.origin = parsed.origin || "*";
    } catch {
      state.origin = "*";
    }

    state.iframe = iframe;
    controller.requestSync({ reloadFallback: true });
    return iframe;
  }

  function sync({ reloadFallback = false }: { reloadFallback?: boolean } = {}) {
    if (!state.iframe) {
      ensureIframe();
      if (!state.iframe) {
        return;
      }
    }

    const rect = (stage as HTMLElement)?.getBoundingClientRect?.();
    const width = Math.round((rect && rect.width) || (stage as HTMLElement)?.clientWidth || 0);
    const height = Math.round((rect && rect.height) || (stage as HTMLElement)?.clientHeight || 0);
    if (width <= 0 || height <= 0) {
      if (reloadFallback) {
        controller.requestSync();
      }
      return;
    }

    const payload = {
      source: "multi-agent-platform",
      type: "novnc.viewport.sync",
      width,
      height,
      stageWidth: Math.round((stage as HTMLElement)?.clientWidth || width),
      stageHeight: Math.round((stage as HTMLElement)?.clientHeight || height),
      devicePixelRatio: Number(window.devicePixelRatio) || 1,
      innerWidth: typeof window.innerWidth === "number" ? window.innerWidth : width,
      innerHeight: typeof window.innerHeight === "number" ? window.innerHeight : height,
      timestamp: Date.now(),
      context,
    };

    let posted = false;
    try {
      state.iframe.contentWindow?.postMessage(payload, state.origin || "*");
      posted = true;
    } catch {
      posted = false;
    }

    if (reloadFallback && !posted) {
      controller.reload();
    }
  }

  function requestSync({ reloadFallback = false }: { reloadFallback?: boolean } = {}) {
    state.deferredReloadFallback = state.deferredReloadFallback || reloadFallback;
    if (state.deferredRaf !== null) {
      return;
    }

    state.deferredRaf = requestAnimationFrame(() => {
      state.deferredRaf = requestAnimationFrame(() => {
        const shouldReload = state.deferredReloadFallback;
        state.deferredReloadFallback = false;
        state.deferredRaf = null;
        controller.sync({ reloadFallback: shouldReload });
      });
    });
  }

  function reload() {
    const iframe = state.iframe;
    if (!iframe) {
      return;
    }

    const lastReload = Number((iframe as any).dataset?.novncReloadTs || "0");
    const now = Date.now();
    if (!lastReload || now - lastReload > 1500) {
      if ((iframe as any).dataset) {
        (iframe as any).dataset.novncReloadTs = String(now);
      }
      reloadBrowserIframeWithCacheBust(iframe);
    }
  }

  function matchesWindow(win: Window | null) {
    return Boolean(state.iframe && win === state.iframe.contentWindow);
  }

  if (typeof ResizeObserver === "function") {
    state.stageResizeObserver = new ResizeObserver((entries) => {
      if (!entries || entries.length === 0) return;
      const entry = entries[0];
      const { width, height } = entry.contentRect || {};
      const roundedWidth = Math.round(width || 0);
      const roundedHeight = Math.round(height || 0);
      if (roundedWidth <= 0 || roundedHeight <= 0) return;

      if (state.stageResizeRaf !== null) {
        cancelAnimationFrame(state.stageResizeRaf);
        state.stageResizeRaf = null;
      }

      state.stageResizeRaf = requestAnimationFrame(() => {
        state.stageResizeRaf = null;
        controller.requestSync();
      });
    });
    state.stageResizeObserver.observe(stage as Element);
  }

  if (fullscreenButton) {
    fullscreenButton.addEventListener("click", () => {
      const el = state.iframe ?? stage;
      if (!el) return;
      if (document.fullscreenElement) (document as any).exitFullscreen();
      else (el as any).requestFullscreen?.();
    });
  }

  noVncControllers.add(controller);
  return controller;
}

function ensureGeneralBrowserProxyElements() {
  if (generalBrowserSurface && generalBrowserStage && generalBrowserFullscreenBtn) {
    return {
      surface: generalBrowserSurface,
      stage: generalBrowserStage,
      fullscreenBtn: generalBrowserFullscreenBtn,
    };
  }

  const surface = document.createElement("div");
  surface.className = "no-vnc-surface general-browser-surface";
  surface.hidden = true;

  const stage = document.createElement("div");
  stage.className = "stage";
  stage.id = "generalBrowserStage";
  stage.setAttribute("role", "region");
  stage.setAttribute("aria-label", "リモートブラウザ (一般ビュー)");

  const fallback = document.createElement("p");
  fallback.className = "stage-fallback";
  fallback.textContent = "リモートブラウザを読み込んでいます…";
  stage.appendChild(fallback);

  const toolbar = document.createElement("div");
  toolbar.className = "browser-toolbar";

  const fullscreenBtn = document.createElement("button");
  fullscreenBtn.type = "button";
  fullscreenBtn.id = "generalFullscreenBtn";
  fullscreenBtn.className = "btn subtle";
  fullscreenBtn.title = "フルスクリーン";
  fullscreenBtn.setAttribute("aria-label", "フルスクリーン");
  fullscreenBtn.textContent = "⤢";

  toolbar.appendChild(fullscreenBtn);

  surface.appendChild(stage);
  surface.appendChild(toolbar);

  generalBrowserSurface = surface;
  generalBrowserStage = stage;
  generalBrowserFullscreenBtn = fullscreenBtn;

  return { surface, stage, fullscreenBtn };
}

function ensureGeneralNoVncController() {
  if (!generalBrowserStage || !generalBrowserFullscreenBtn) {
    const elements = ensureGeneralBrowserProxyElements();
    if (!elements.stage || !elements.fullscreenBtn) {
      return null;
    }
  }

  if (!generalBrowserController) {
    generalBrowserController = createNoVncController({
      stage: generalBrowserStage,
      fullscreenButton: generalBrowserFullscreenBtn,
      context: "general-proxy",
    });
  }

  return generalBrowserController;
}

function activateGeneralBrowserProxy() {
  if (!generalProxyContainer) {
    return;
  }

  const { surface } = ensureGeneralBrowserProxyElements();
  if (!surface) {
    return;
  }

  if (surface.parentElement !== generalProxyContainer) {
    generalProxyContainer.innerHTML = "";
    generalProxyContainer.appendChild(surface);
  }

  surface.hidden = false;

  const controller = ensureGeneralNoVncController();
  controller?.ensureIframe();
}

function deactivateGeneralBrowserProxy() {
  if (!generalProxyContainer || !generalBrowserSurface) {
    return;
  }

  if (generalBrowserSurface.parentElement === generalProxyContainer) {
    generalProxyContainer.removeChild(generalBrowserSurface);
  }

  generalBrowserSurface.hidden = true;
}

export function requestMainBrowserViewportSync({ reloadFallback = false }: { reloadFallback?: boolean } = {}) {
  if (!mainBrowserController) {
    return;
  }
  mainBrowserController.requestSync({ reloadFallback });
}

function requestGeneralBrowserViewportSync({ reloadFallback = false }: { reloadFallback?: boolean } = {}) {
  const controller = ensureGeneralNoVncController();
  if (!controller) {
    return;
  }
  controller.ensureIframe();
  if (generalBrowserSurface?.isConnected) {
    controller.requestSync({ reloadFallback });
  }
}

function bindNavButtons() {
  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      activateView((btn.dataset.view as ViewKey) || "general");
    });
  });
}

function bindSidebarToggle() {
  if (layoutEl && sidebarToggle && sidebarEl) {
    const setSidebarCollapsed = (collapsed: boolean) => {
      layoutEl?.classList.toggle("sidebar-collapsed", collapsed);
      const label = collapsed ? "サイドバーを表示する" : "サイドバーを折りたたむ";
      sidebarToggle?.setAttribute("aria-expanded", String(!collapsed));
      sidebarToggle?.setAttribute("aria-label", label);
      sidebarToggle?.setAttribute("title", label);
      scheduleSidebarTogglePosition();
    };

    setSidebarCollapsed(false);

    sidebarToggle.addEventListener("click", () => {
      const collapsed = !layoutEl?.classList.contains("sidebar-collapsed");
      setSidebarCollapsed(Boolean(collapsed));
    });

    const mq = window.matchMedia("(max-width: 960px)");
    const handleMq = (event: MediaQueryListEvent | MediaQueryList) => {
      if (("matches" in event) && event.matches) {
        setSidebarCollapsed(false);
      }
    };

    handleMq(mq);
    if (typeof mq.addEventListener === "function") mq.addEventListener("change", handleMq);
    else if (typeof (mq as any).addListener === "function") (mq as any).addListener(handleMq);

    window.addEventListener("resize", scheduleSidebarTogglePosition);
    window.addEventListener("scroll", scheduleSidebarTogglePosition, { passive: true });

    if (typeof ResizeObserver === "function" && sidebarEl) {
      const sidebarResizeObserver = new ResizeObserver(scheduleSidebarTogglePosition);
      sidebarResizeObserver.observe(sidebarEl);
    }
  }
}

function bindNoVncWindowMessages() {
  window.addEventListener("message", (event) => {
    const data = (event as MessageEvent).data as any;
    if (!data || typeof data !== "object") {
      return;
    }

    const { type } = data;
    if (typeof type !== "string") {
      return;
    }

    const normalizedType = type.toLowerCase();
    if (
      normalizedType === "novnc.viewport.request" ||
      normalizedType === "novnc.viewport.requestsync" ||
      normalizedType === "novnc.viewport.ready" ||
      normalizedType === "novnc.ready"
    ) {
      let handled = false;
      for (const controller of noVncControllers) {
        if (!controller) continue;
        if (!event.source || controller.matchesWindow(event.source as Window)) {
          controller.markReady?.();
          controller.requestSync();
          handled = true;
        }
      }
      if (!handled) {
        for (const controller of noVncControllers) {
          controller?.markReady?.();
          controller?.requestSync();
        }
      }
      return;
    }

    if (
      normalizedType === "novnc.viewport.reload" || normalizedType === "novnc.reload"
    ) {
      for (const controller of noVncControllers) {
        if (!controller) continue;
        if (!event.source || controller.matchesWindow(event.source as Window)) {
          controller.reload();
        }
      }
    }
  });
}

export function initLayout() {
  if (initialized) return;
  initialized = true;

  layoutEl = $(".layout");
  sidebarEl = $(".sidebar");
  sidebarToggle = $(".sidebar-toggle");

  views = {
    general: $("#view-general"),
    browser: $("#view-browser"),
    iot: $("#view-iot"),
    chat: $("#view-chat"),
    schedule: $("#view-schedule"),
  };

  appTitle = $("#appTitle");
  navButtons = $$(".nav-btn") as HTMLButtonElement[];
  sidebarChatTitle = $(".sidebar-chat-title");
  sidebarChatIcon = $(".sidebar-chat-icon");
  sidebarChatTitleTextNode = sidebarChatTitle
    ? Array.from(sidebarChatTitle.childNodes).find(
        (node) => node.nodeType === Node.TEXT_NODE && node.textContent?.trim(),
      ) || null
    : null;

  generalDefaultContent = $("#generalDefaultContent");
  generalProxyStatus = $("#generalProxyStatus");
  generalProxyContainer = $("#generalProxyContainer");
  generalViewPanel = views.general?.querySelector(".general-view") ?? null;

  initAgentResultHosts();
  bindNavButtons();
  bindSidebarToggle();

  const browserStage = $("#browserStage") as HTMLElement | null;
  const browserFullscreenBtn = $("#fullscreenBtn") as HTMLButtonElement | null;
  mainBrowserController = createNoVncController({
    stage: browserStage,
    fullscreenButton: browserFullscreenBtn,
    context: "browser-view",
  });

  if (mainBrowserController) {
    mainBrowserController.ensureIframe();
  }

  bindNoVncWindowMessages();
}

export function ensureGeneralProxyView(viewKey: ViewKey | null) {
  if (!viewKey) return;
  moveViewToGeneral(viewKey);
}

export function setCurrentViewKey(viewKey: ViewKey) {
  currentViewKey = viewKey;
}

export function requestGeneralBrowserSync() {
  requestGeneralBrowserViewportSync({ reloadFallback: false });
}

export function updateProxyViewForAgent(agentKey: string | null) {
  if (agentKey) {
    const view = resolveAgentToView(agentKey);
    if (view) {
      moveViewToGeneral(view);
    }
  }
}
