import { $, $$ } from "./dom-utils";
import { applyAgentStatusPayload } from "./agent-status";

let settingsBtn: HTMLButtonElement | null = null;
let dialog: HTMLDialogElement | null = null;
let form: HTMLFormElement | null = null;
let closeBtn: HTMLButtonElement | null = null;
let refreshBtn: HTMLButtonElement | null = null;
let saveBtn: HTMLButtonElement | null = null;
let memoryToggle: HTMLInputElement | null = null;
let chatCountValue: HTMLElement | null = null;
let chatCountNote: HTMLElement | null = null;
let statusMessage: HTMLElement | null = null;

let agentToggleBrowser: HTMLInputElement | null = null;
let agentToggleLifestyle: HTMLInputElement | null = null;
let agentToggleIot: HTMLInputElement | null = null;
let agentToggleScheduler: HTMLInputElement | null = null;

let modelSelectOrchestrator: HTMLSelectElement | null = null;
let modelSelectBrowser: HTMLSelectElement | null = null;
let modelSelectLifestyle: HTMLSelectElement | null = null;
let modelSelectIot: HTMLSelectElement | null = null;
let modelSelectScheduler: HTMLSelectElement | null = null;
let modelSelectMemory: HTMLSelectElement | null = null;

let shortTermTtlInput: HTMLInputElement | null = null;
let shortTermGraceInput: HTMLInputElement | null = null;
let shortTermActiveHoldInput: HTMLInputElement | null = null;
let shortTermPromoteScoreInput: HTMLInputElement | null = null;
let shortTermPromoteImportanceInput: HTMLInputElement | null = null;

let longTermGrid: HTMLElement | null = null;
let shortTermGrid: HTMLElement | null = null;

const agentToggleInputs: Record<string, HTMLInputElement | null> = {};
const modelSelectInputs: Record<string, HTMLSelectElement | null> = {};

const DEFAULT_AGENT_CONNECTIONS = {
  browser: true,
  lifestyle: true,
  iot: true,
  scheduler: true,
};

const DEFAULT_AGENT_STATUS: Record<string, { available: boolean | null; enabled: boolean }> = {
  browser: { available: null, enabled: true },
  lifestyle: { available: null, enabled: true },
  iot: { available: null, enabled: true },
  scheduler: { available: null, enabled: true },
};

const LONG_TERM_CATEGORIES = [
  "profile",
  "preference",
  "health",
  "work",
  "hobby",
  "relationship",
  "life",
  "travel",
  "food",
  "general",
];

const SHORT_TERM_CATEGORIES = [
  "active_task",
  "pending_questions",
  "recent_entities",
  "emotional_context",
  "general",
];

const CATEGORY_LABELS: Record<string, string> = {
  profile: "基本情報",
  preference: "好み・嗜好",
  health: "健康",
  work: "仕事・学業",
  hobby: "趣味",
  relationship: "人間関係",
  life: "生活",
  travel: "旅行",
  food: "食事",
  general: "その他・メモ",
  active_task: "現在進行中のタスク",
  pending_questions: "未解決の質問",
  recent_entities: "直近の話題・キーワード",
  emotional_context: "現在の感情・雰囲気",
};

const PLACEHOLDER: Record<string, string> = {
  profile: "例: 名前は山田太郎。東京在住。30代。エンジニアとして働いている。",
  preference: "例: 返答は簡潔が好き。敬体が好み。長文より箇条書きが助かる。",
  health: "例: 毎日朝にジョギング。カフェイン控えめを希望。",
  work: "例: プロジェクトXの締切は毎週金曜。リモート勤務中心。",
  hobby: "例: ロードバイクと写真が趣味。休日は多摩川沿いを走る。",
  relationship: "例: 佐藤さんとは同僚。田中さんはメンター。",
  life: "例: 早朝型。家事は週末にまとめて行う。",
  travel: "例: 夏に北海道旅行を計画中。温泉が好き。",
  food: "例: 和食とコーヒーが好き。辛すぎる料理は苦手。",
  general: "例: 雑多なメモや、まだ分類できていない情報。",
  active_task: "例: タスク: 旅行の計画を立てる (ステータス: 進行中)",
  pending_questions: "例: 質問: 次回の会議はいつ？\n質問: あのレストランの名前は？",
  recent_entities: "例: キーワード: React, Python, 温泉",
  emotional_context: "例: 気分: 落ち着いている。少し急ぎ。",
};

const state: {
  loading: boolean;
  saving: boolean;
  modelOptions: any[];
  agentStatus: Record<string, { available: boolean | null; enabled: boolean }>;
  memoryValues: { long: Record<string, string>; short: Record<string, string> };
  memoryFull: { long: Record<string, any>; short: Record<string, any> };
} = {
  loading: false,
  saving: false,
  modelOptions: [],
  agentStatus: { ...DEFAULT_AGENT_STATUS },
  memoryValues: {
    long: {},
    short: {},
  },
  memoryFull: {
    long: {},
    short: {},
  },
};

function formatShortTermValue(category: string, summaryText: string, fullMemory: any) {
  if (category === "active_task") {
    const task = fullMemory.active_task || {};
    if (task.goal) {
      return `タスク: ${task.goal}\nステータス: ${task.status || "active"}`;
    }
  }
  if (category === "pending_questions") {
    const questions = fullMemory.pending_questions || [];
    if (Array.isArray(questions) && questions.length > 0) {
      return questions.map((q: string) => `質問: ${q}`).join("\n");
    }
  }
  if (category === "recent_entities") {
    const entities = fullMemory.recent_entities || [];
    if (Array.isArray(entities) && entities.length > 0) {
      const names = entities.map((e: any) => e.name).filter((n: string) => n);
      if (names.length > 0) {
        return `キーワード: ${names.join(", ")}`;
      }
    }
  }
  if (category === "emotional_context") {
    if (fullMemory.emotional_context) {
      return `気分: ${fullMemory.emotional_context}`;
    }
  }

  return summaryText || "";
}

async function fetchMemory() {
  const response = await fetch("/api/memory", { method: "GET" });
  if (!response.ok) {
    throw new Error(`メモリの取得に失敗しました (${response.status})`);
  }
  const data = await response.json();
  return {
    longTermCategories: data?.long_term_categories ?? {},
    shortTermCategories: data?.short_term_categories ?? {},
    longTermFull: data?.long_term_full ?? {},
    shortTermFull: data?.short_term_full ?? {},
    enabled: data?.enabled ?? true,
    shortTermTtlMinutes: data?.short_term_ttl_minutes ?? 45,
    shortTermGraceMinutes: data?.short_term_grace_minutes ?? 0,
    shortTermActiveHoldMinutes: data?.short_term_active_task_hold_minutes ?? 0,
    shortTermPromoteScore: data?.short_term_promote_score ?? 2,
    shortTermPromoteImportance: data?.short_term_promote_importance ?? 0.65,
  };
}

function setMemoryData(longCategories: Record<string, string>, shortCategories: Record<string, string>, longFull: any, shortFull: any) {
  state.memoryValues.long = { ...longCategories };
  state.memoryValues.short = { ...shortCategories };
  state.memoryFull.long = longFull || {};
  state.memoryFull.short = shortFull || {};
}

function renderMemoryGrid(type: "long" | "short") {
  const grid = type === "long" ? longTermGrid : shortTermGrid;
  const categories = type === "long" ? LONG_TERM_CATEGORIES : SHORT_TERM_CATEGORIES;
  const summaries = state.memoryValues[type] || {};
  const fullMemory = state.memoryFull[type] || {};

  if (!grid) return;

  while (grid.firstChild) {
    grid.removeChild(grid.firstChild);
  }

  categories.forEach((cat) => {
    const wrapper = document.createElement("div");
    wrapper.className = "settings-memory-card";

    const label = document.createElement("label");
    label.className = "settings-memory-label";
    label.textContent = CATEGORY_LABELS[cat] || cat;

    const textarea = document.createElement("textarea");
    textarea.className = "form-control settings-memory-input";
    textarea.rows = 3;
    textarea.placeholder = PLACEHOLDER[cat] || "";
    (textarea as any).dataset.category = cat;
    (textarea as any).dataset.memoryType = type;

    let initialValue = "";
    if (type === "short" && cat !== "general") {
      initialValue = formatShortTermValue(cat, summaries[cat] || "", fullMemory);
    } else {
      initialValue = summaries[cat] || "";
    }
    textarea.value = initialValue;

    const updateValue = (e: Event) => {
      const target = e.target as HTMLTextAreaElement;
      state.memoryValues[type][cat] = target.value;
    };
    textarea.addEventListener("input", updateValue);
    textarea.addEventListener("change", updateValue);

    wrapper.appendChild(label);
    wrapper.appendChild(textarea);

    grid.appendChild(wrapper);
  });
}

function syncMemoryValuesFromInputs() {
  const inputs = document.querySelectorAll<HTMLTextAreaElement>(".settings-memory-input[data-category]");
  inputs.forEach((input) => {
    const category = input.dataset.category;
    if (!category) return;
    const type = input.dataset.memoryType === "short" ? "short" : "long";
    if (!state.memoryValues[type]) state.memoryValues[type] = {};
    state.memoryValues[type][category] = input.value;
  });
}

async function loadSettingsData() {
  if (state.loading) return;
  state.loading = true;
  setStatus("データを読み込み中…", "muted");
  refreshBtn?.setAttribute("aria-busy", "true");
  if (refreshBtn) refreshBtn.disabled = true;

  try {
    const memoryPromise = fetchMemory();
    const chatCountPromise = fetchChatCount();
    const agentPromise = fetchAgentConnections();
    const modelPromise = fetchModelSettings();
    const agentStatusPromise = fetchAgentStatus();

    const memoryResult = await memoryPromise
      .then((value) => ({ status: "fulfilled" as const, value }))
      .catch((reason) => ({ status: "rejected" as const, reason }));

    const errors: string[] = [];

    if (memoryResult.status === "fulfilled") {
      const m = memoryResult.value;
      setMemoryData(m.longTermCategories, m.shortTermCategories, m.longTermFull, m.shortTermFull);

      if (memoryToggle) {
        memoryToggle.checked = m.enabled;
        updateSwitchAria(memoryToggle);
      }
      if (shortTermTtlInput) shortTermTtlInput.value = String(m.shortTermTtlMinutes ?? "");
      if (shortTermGraceInput) shortTermGraceInput.value = String(m.shortTermGraceMinutes ?? "");
      if (shortTermActiveHoldInput) shortTermActiveHoldInput.value = String(m.shortTermActiveHoldMinutes ?? "");
      if (shortTermPromoteScoreInput) shortTermPromoteScoreInput.value = String(m.shortTermPromoteScore ?? "");
      if (shortTermPromoteImportanceInput) shortTermPromoteImportanceInput.value = String(m.shortTermPromoteImportance ?? "");

      renderMemoryGrid("long");
      renderMemoryGrid("short");
    } else {
      errors.push((memoryResult as any).reason?.message || "メモリ取得エラー");
    }

    const [chatCountResult, agentResult, modelResult, agentStatusResult] = await Promise.allSettled([
      chatCountPromise,
      agentPromise,
      modelPromise,
      agentStatusPromise,
    ]);

    if (chatCountResult.status === "fulfilled") {
      updateChatCount(chatCountResult.value as number);
    } else {
      updateChatCount(undefined as any);
    }

    if (agentResult.status === "fulfilled") {
      setAgentConnections(agentResult.value as any);
    } else {
      setAgentConnections(DEFAULT_AGENT_CONNECTIONS);
    }

    if (agentStatusResult.status === "fulfilled") {
      applyAgentStatus(agentStatusResult.value as any);
    } else {
      applyAgentStatus({ agents: DEFAULT_AGENT_STATUS } as any);
    }

    if (modelResult.status === "fulfilled") {
      renderModelOptions((modelResult.value as any).options);
      setModelSelection((modelResult.value as any).selection);
    } else {
      renderModelOptions({ providers: [] });
    }

    applyModelAvailability();

    if (errors.length) {
      setStatus(errors[0], "error");
    } else {
      setStatus("最新のデータを読み込みました。", "success");
    }
  } catch (error) {
    console.error("設定データの取得に失敗しました:", error);
    setStatus("設定データの取得に失敗しました。", "error");
  } finally {
    state.loading = false;
    refreshBtn?.removeAttribute("aria-busy");
    if (refreshBtn) refreshBtn.disabled = false;
  }
}

async function saveMemory() {
  syncMemoryValuesFromInputs();

  const payload: any = {
    enabled: memoryToggle?.checked ?? true,
    long_term_memory: state.memoryValues.long,
    short_term_memory: state.memoryValues.short,
  };

  const ttl = readIntInput(shortTermTtlInput, { min: 5, max: 720 });
  if (typeof ttl === "number") payload.short_term_ttl_minutes = ttl;
  const grace = readIntInput(shortTermGraceInput, { min: 0, max: 240 });
  if (typeof grace === "number") payload.short_term_grace_minutes = grace;
  const hold = readIntInput(shortTermActiveHoldInput, { min: 0, max: 240 });
  if (typeof hold === "number") payload.short_term_active_task_hold_minutes = hold;
  const promoteScore = readIntInput(shortTermPromoteScoreInput, { min: 0, max: 10 });
  if (typeof promoteScore === "number") payload.short_term_promote_score = promoteScore;
  const promoteImportance = readFloatInput(shortTermPromoteImportanceInput, { min: 0, max: 1, precision: 2 });
  if (typeof promoteImportance === "number") payload.short_term_promote_importance = promoteImportance;

  const response = await fetchWithTimeout("/api/memory", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`保存に失敗しました (${response.status})`);
  }
  return response.json();
}

async function fetchChatCount() {
  const response = await fetch("/chat_history", { method: "GET" });
  if (!response.ok) throw new Error("History fetch failed");
  const data = await response.json();
  if (Array.isArray(data)) return data.length;
  if (data && Array.isArray(data.history)) return data.history.length;
  return 0;
}

async function fetchAgentConnections() {
  const response = await fetch("/api/agent_connections", { method: "GET" });
  if (!response.ok) throw new Error("Agent fetch failed");
  const data = await response.json();
  return data?.agents && typeof data.agents === "object" ? data.agents : data;
}

async function fetchModelSettings() {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch("/api/model_settings", { method: "GET", signal: controller.signal });
    if (!response.ok) throw new Error("Model fetch failed");
    const data = await response.json();
    return { selection: data?.selection || {}, options: data?.options || {} };
  } finally {
    window.clearTimeout(timeoutId);
  }
}

async function fetchAgentStatus() {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch("/api/agent_status", { method: "GET", signal: controller.signal });
    if (!response.ok) throw new Error("Agent status fetch failed");
    return response.json();
  } finally {
    window.clearTimeout(timeoutId);
  }
}

async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = 15000) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function setStatus(message: string, kind: "muted" | "success" | "error" = "muted") {
  if (!statusMessage) return;
  statusMessage.textContent = message || "";
  (statusMessage as HTMLElement).dataset.kind = kind || "muted";
  statusMessage.hidden = !message;
}

function updateSwitchAria(input: HTMLInputElement) {
  if (!input) return;
  input.setAttribute("aria-checked", input.checked ? "true" : "false");
}

function updateChatCount(count: number | undefined) {
  if (!chatCountValue || !chatCountNote) return;
  if (Number.isFinite(count)) {
    const safeCount = Math.max(0, Math.trunc(count as number));
    chatCountValue.textContent = safeCount.toLocaleString("ja-JP");
    chatCountNote.textContent = safeCount === 0 ? "履歴はまだありません。" : "保存済みのメッセージ総数です。";
  } else {
    chatCountValue.textContent = "-";
    chatCountNote.textContent = "履歴の取得に失敗しました。";
  }
}

function setAgentConnections(connections: Record<string, boolean>) {
  const merged = { ...DEFAULT_AGENT_CONNECTIONS } as Record<string, boolean>;
  if (connections && typeof connections === "object") {
    Object.keys(merged).forEach((key) => {
      if (typeof connections[key] === "boolean") {
        merged[key] = connections[key];
      }
    });
  }
  Object.entries(agentToggleInputs).forEach(([agent, input]) => {
    if (!input) return;
    input.checked = merged[agent];
    updateSwitchAria(input);
  });
}

function readAgentConnections() {
  const connections: Record<string, boolean> = {};
  Object.entries(agentToggleInputs).forEach(([agent, input]) => {
    if (!input) return;
    connections[agent] = Boolean(input.checked);
  });
  return connections;
}

async function saveAgentConnections() {
  const connections = readAgentConnections();
  const response = await fetchWithTimeout("/api/agent_connections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ agents: connections }),
  });
  if (!response.ok) throw new Error(`接続設定の保存に失敗しました (${response.status})`);
  return response.json();
}

async function saveModelSettings() {
  const payload = readModelSelection();
  const response = await fetchWithTimeout("/api/model_settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(`モデル設定の保存に失敗しました (${response.status})`);
  return response.json();
}

async function saveSettings() {
  const results = await Promise.allSettled([saveMemory(), saveAgentConnections(), saveModelSettings()]);
  const rejected = results.filter((result) => result.status === "rejected");
  const aborts = rejected.filter((result) => (result as any).reason?.name === "AbortError");
  const nonAbortErrors = rejected.filter((result) => (result as any).reason?.name !== "AbortError");

  if (nonAbortErrors.length) {
    const errors = nonAbortErrors.map((result: any) => result.reason?.message || "保存に失敗しました。");
    const error: any = new Error(errors[0]);
    error.messages = errors;
    throw error;
  }

  return { results, timedOut: aborts.length > 0 };
}

function applyAgentStatus(payload: any) {
  applyAgentStatusPayload(payload);

  const agents = payload?.agents && typeof payload.agents === "object" ? payload.agents : payload;
  if (!agents || typeof agents !== "object") {
    state.agentStatus = { ...DEFAULT_AGENT_STATUS };
    return;
  }
  const nextStatus = { ...DEFAULT_AGENT_STATUS } as Record<string, { available: boolean | null; enabled: boolean }>;
  Object.keys(nextStatus).forEach((key) => {
    const entry = agents[key];
    if (!entry || typeof entry !== "object") return;
    nextStatus[key] = {
      available: entry.available ?? nextStatus[key].available,
      enabled: entry.enabled ?? nextStatus[key].enabled,
    };
  });
  state.agentStatus = nextStatus;
}

function ensureSelectStatusPlaceholder(select: HTMLSelectElement, message: string) {
  const existing = Array.from(select.options || []).find((option) => (option as any).dataset?.status === "unavailable");
  if (existing) {
    existing.textContent = message;
    select.value = existing.value;
    return;
  }
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = message;
  (placeholder as any).dataset.status = "unavailable";
  select.prepend(placeholder);
  select.value = "";
}

function clearSelectStatusPlaceholder(select: HTMLSelectElement) {
  Array.from(select.options || []).forEach((option) => {
    if ((option as any).dataset?.status === "unavailable") {
      option.remove();
    }
  });
}

function applyModelAvailability() {
  Object.entries(modelSelectInputs).forEach(([agent, select]) => {
    if (!select) return;
    if (agent === "orchestrator" || agent === "memory") {
      return;
    }
    const status = state.agentStatus?.[agent];
    if (status && status.available === false) {
      ensureSelectStatusPlaceholder(select, "起動していません");
      select.disabled = true;
      return;
    }
    clearSelectStatusPlaceholder(select);
    if (state.modelOptions.length) {
      select.disabled = false;
    }
  });
}

function renderModelOptions(options: any) {
  state.modelOptions = Array.isArray(options?.providers) ? options.providers : [];
  Object.values(modelSelectInputs).forEach((select) => {
    if (!select) return;
    select.innerHTML = "";
    if (!state.modelOptions.length) {
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "利用可能なモデルがありません";
      select.appendChild(placeholder);
      select.disabled = true;
      return;
    }
    select.disabled = false;
    state.modelOptions.forEach((provider: any) => {
      const group = document.createElement("optgroup");
      group.label = provider.label || provider.id;
      (provider.models || []).forEach((model: any) => {
        const option = document.createElement("option");
        option.value = `${provider.id}::${model.id}`;
        (option as any).dataset.provider = provider.id;
        (option as any).dataset.model = model.id;
        option.textContent = model.label || model.id;
        group.appendChild(option);
      });
      select.appendChild(group);
    });
  });
}

function setModelSelection(selection: Record<string, { provider: string; model: string }>) {
  const safeSelection = selection && typeof selection === "object" ? selection : {};
  Object.entries(modelSelectInputs).forEach(([agent, select]) => {
    if (!select) return;
    const value = (safeSelection as any)[agent] || {};
    const provider = value.provider || "";
    const model = value.model || "";
    const match = Array.from(select.options || []).find(
      (option) => (option as any).dataset?.provider === provider && (option as any).dataset?.model === model,
    );
    if (match) {
      select.value = match.value;
    } else if (select.options.length) {
      select.selectedIndex = 0;
    }
  });
}

function readModelSelection() {
  const selection: Record<string, { provider: string; model: string }> = {};
  Object.entries(modelSelectInputs).forEach(([agent, select]) => {
    if (!select) return;
    const option = select.selectedOptions && select.selectedOptions[0];
    if (!option) return;
    selection[agent] = {
      provider: (option as any).dataset?.provider || "",
      model: (option as any).dataset?.model || option.value,
    };
  });
  return { selection };
}

function readIntInput(input: HTMLInputElement | null, { min, max }: { min?: number; max?: number } = {}) {
  if (!input) return null;
  const val = parseInt(input.value, 10);
  if (Number.isNaN(val)) return null;
  if (min !== undefined && val < min) return min;
  if (max !== undefined && val > max) return max;
  return val;
}

function readFloatInput(input: HTMLInputElement | null, { min, max, precision }: { min?: number; max?: number; precision?: number } = {}) {
  if (!input) return null;
  let val = parseFloat(input.value);
  if (Number.isNaN(val)) return null;
  if (min !== undefined && val < min) val = min;
  if (max !== undefined && val > max) val = max;
  if (precision !== undefined) {
    const factor = Math.pow(10, precision);
    val = Math.round(val * factor) / factor;
  }
  return val;
}

function closeDialog() {
  if (!dialog) return;
  if (dialog.open) {
    dialog.close();
  }
  setStatus("", "muted");
}

function bindDomReferences() {
  settingsBtn = $("#settingsBtn") as HTMLButtonElement | null;
  dialog = $("#settingsDialog") as HTMLDialogElement | null;
  form = $("#settingsForm") as HTMLFormElement | null;
  closeBtn = $("#settingsCloseBtn") as HTMLButtonElement | null;
  refreshBtn = $("#settingsRefreshBtn") as HTMLButtonElement | null;
  saveBtn = $("#settingsSaveBtn") as HTMLButtonElement | null;
  memoryToggle = $("#settingsMemoryToggle") as HTMLInputElement | null;
  chatCountValue = $("#chatCountValue");
  chatCountNote = $("#chatCountNote");
  statusMessage = $("#settingsStatusMessage");

  agentToggleBrowser = $("#agentToggleBrowser") as HTMLInputElement | null;
  agentToggleLifestyle = $("#agentToggleLifestyle") as HTMLInputElement | null;
  agentToggleIot = $("#agentToggleIot") as HTMLInputElement | null;
  agentToggleScheduler = $("#agentToggleScheduler") as HTMLInputElement | null;

  modelSelectOrchestrator = $("#modelSelectOrchestrator") as HTMLSelectElement | null;
  modelSelectBrowser = $("#modelSelectBrowser") as HTMLSelectElement | null;
  modelSelectLifestyle = $("#modelSelectLifestyle") as HTMLSelectElement | null;
  modelSelectIot = $("#modelSelectIot") as HTMLSelectElement | null;
  modelSelectScheduler = $("#modelSelectScheduler") as HTMLSelectElement | null;
  modelSelectMemory = $("#modelSelectMemory") as HTMLSelectElement | null;

  shortTermTtlInput = $("#settingsShortTermTtl") as HTMLInputElement | null;
  shortTermGraceInput = $("#settingsShortTermGrace") as HTMLInputElement | null;
  shortTermActiveHoldInput = $("#settingsShortTermActiveHold") as HTMLInputElement | null;
  shortTermPromoteScoreInput = $("#settingsShortTermPromoteScore") as HTMLInputElement | null;
  shortTermPromoteImportanceInput = $("#settingsShortTermPromoteImportance") as HTMLInputElement | null;

  longTermGrid = $("#settingsLongTermGrid");
  shortTermGrid = $("#settingsShortTermGrid");

  agentToggleInputs.browser = agentToggleBrowser;
  agentToggleInputs.lifestyle = agentToggleLifestyle;
  agentToggleInputs.iot = agentToggleIot;
  agentToggleInputs.scheduler = agentToggleScheduler;

  modelSelectInputs.orchestrator = modelSelectOrchestrator;
  modelSelectInputs.browser = modelSelectBrowser;
  modelSelectInputs.lifestyle = modelSelectLifestyle;
  modelSelectInputs.iot = modelSelectIot;
  modelSelectInputs.scheduler = modelSelectScheduler;
  modelSelectInputs.memory = modelSelectMemory;
}

let initialized = false;

export function initSettingsModal() {
  if (initialized) return;
  bindDomReferences();
  if (!settingsBtn || !dialog || !form) return;

  initialized = true;

  if (memoryToggle) {
    updateSwitchAria(memoryToggle);
    memoryToggle.addEventListener("change", () => updateSwitchAria(memoryToggle!));
  }
  Object.values(agentToggleInputs).forEach((input) => {
    if (!input) return;
    updateSwitchAria(input);
    input.addEventListener("change", () => updateSwitchAria(input));
  });

  settingsBtn.addEventListener("click", () => {
    if (!dialog?.open) {
      dialog?.showModal();
    }
    loadSettingsData();
  });
  closeBtn?.addEventListener("click", () => closeDialog());
  dialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeDialog();
  });
  refreshBtn?.addEventListener("click", () => loadSettingsData());

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.saving) return;
    state.saving = true;
    let savedOk = false;
    if (saveBtn) {
      saveBtn.disabled = true;
      saveBtn.textContent = "保存中…";
    }
    setStatus("保存しています…", "muted");

    try {
      const saveResult = await saveSettings();
      savedOk = true;
      if ((saveResult as any)?.timedOut) {
        setStatus("保存しました。（応答が遅延しました）", "success");
      } else {
        setStatus("保存しました。", "success");
      }
      if (saveBtn) {
        saveBtn.textContent = "保存完了";
      }
    } catch (error: any) {
      console.error("設定の保存に失敗しました:", error);
      const message = error?.messages?.[0] || error?.message || "保存に失敗しました。";
      setStatus(message, "error");
    } finally {
      state.saving = false;
      if (saveBtn) {
        if (savedOk) {
          window.setTimeout(() => {
            if (state.saving) return;
            saveBtn!.disabled = false;
            saveBtn!.textContent = "保存";
          }, 1200);
        } else {
          saveBtn.disabled = false;
          saveBtn.textContent = "保存";
        }
      }
    }
  });

  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      closeDialog();
    }
  });
}
