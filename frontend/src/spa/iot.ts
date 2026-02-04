import { $ } from "./dom-utils";
import { markAgentAvailable, markAgentUnavailable } from "./agent-status";

/* ---------- IoT Dashboard ---------- */

let deviceGrid: HTMLElement | null = null;
let iotNotice: HTMLElement | null = null;
let registerDeviceBtn: HTMLButtonElement | null = null;
let refreshDevicesBtn: HTMLButtonElement | null = null;

let registerDialog: HTMLDialogElement | null = null;
let registerForm: HTMLFormElement | null = null;
let registerIdInput: HTMLInputElement | null = null;
let registerNameInput: HTMLInputElement | null = null;
let registerNoteInput: HTMLInputElement | null = null;
let registerMessageEl: HTMLElement | null = null;
let registerCancelBtn: HTMLButtonElement | null = null;
let registerSubmitBtn: HTMLButtonElement | null = null;

const IOT_DEVICE_ICON = `<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="3" stroke="currentColor" stroke-width="1.6" fill="none" /><path d="M7 9h10M7 13h6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" /></svg>`;

const IOT_FETCH_INTERVAL = 6000;

const PUBLIC_IOT_AGENT_BASE = "https://iot-agent.project-kk.com";

let REGISTER_MESSAGE_DEFAULT = "エッジデバイスで使用する識別子を入力し、必要に応じて表示名やメモを設定してください。";

const iotState: {
  devices: any[];
  fetching: boolean;
  initialized: boolean;
  pollTimer: number | null;
} = {
  devices: [],
  fetching: false,
  initialized: false,
  pollTimer: null,
};

let lastRegisteredDevice: { id: string; name: string } | null = null;

function resolveIotAgentBase() {
  const sanitize = (value: unknown) => (typeof value === "string" ? value.trim().replace(/\/+$/, "") : "");
  let queryBase = "";
  try {
    queryBase = new URLSearchParams(window.location.search).get("iot_agent_base") || "";
  } catch {
    queryBase = "";
  }
  const sources = [
    sanitize(queryBase),
    sanitize((window as any).IOT_AGENT_API_BASE),
    sanitize(document.querySelector<HTMLMetaElement>("meta[name='iot-agent-api-base']")?.content),
  ];
  for (const base of sources) {
    if (base) return base;
  }
  if (window.location.origin && window.location.origin !== "null") {
    return `${window.location.origin.replace(/\/+$/, "")}/iot_agent`;
  }
  if (PUBLIC_IOT_AGENT_BASE) {
    return PUBLIC_IOT_AGENT_BASE;
  }
  return "/iot_agent";
}

const IOT_AGENT_API_BASE = resolveIotAgentBase();

function buildIotAgentUrl(path: string) {
  if (!path) {
    return IOT_AGENT_API_BASE || "/iot_agent";
  }
  if (/^https?:/i.test(path)) {
    return path;
  }
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const base = IOT_AGENT_API_BASE || "";
  if (!base) {
    return normalizedPath;
  }
  if (/^https?:/i.test(base)) {
    return `${base.replace(/\/+$/, "")}${normalizedPath}`;
  }
  return `${base.replace(/\/+$/, "")}${normalizedPath}` || normalizedPath;
}

export async function iotAgentRequest(
  path: string,
  { method = "GET", headers = {}, body, signal }: { method?: string; headers?: Record<string, string>; body?: BodyInit | null; signal?: AbortSignal } = {},
) {
  const url = buildIotAgentUrl(path);
  const finalHeaders: Record<string, string> = { ...headers };
  const hasBody = body !== undefined && body !== null;
  const isFormData = typeof FormData !== "undefined" && body instanceof FormData;
  if (hasBody && !isFormData && !finalHeaders["Content-Type"]) {
    finalHeaders["Content-Type"] = "application/json";
  }

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      headers: finalHeaders,
      body,
      signal,
      mode: /^https?:/i.test(url) ? "cors" : "same-origin",
      credentials: /^https?:/i.test(url) ? "include" : "same-origin",
    });
  } catch (error: any) {
    markAgentUnavailable("iot", error?.message || "接続に失敗しました。");
    return { data: { status: "unavailable", message: "IoT エージェントに接続できません。", error: error?.message }, status: 0, unavailable: true };
  }

  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  let data: any;
  try {
    data = isJson ? await response.json() : await response.text();
  } catch {
    data = isJson ? {} : "";
  }

  if (!response.ok) {
    const message = typeof data === "string" && data
      ? data
      : (data && typeof data.error === "string")
        ? data.error
        : `${response.status} ${response.statusText}`;
    if (response.status >= 500) {
      markAgentUnavailable("iot", message);
      return { data: { status: "unavailable", message: "IoT エージェントに接続できません。", error: message }, status: response.status, unavailable: true };
    }
    const error = new Error(message) as any;
    error.status = response.status;
    error.data = data;
    throw error;
  }

  const payload = typeof data === "string" ? { message: data } : data;
  if (payload && payload.status === "unavailable") {
    markAgentUnavailable("iot", payload.error || payload.message);
    return { data: payload, status: response.status, unavailable: true };
  }
  markAgentAvailable("iot");
  return { data: payload, status: response.status };
}

function showIotNotice(message: string, kind = "info") {
  if (!iotNotice) return;
  iotNotice.hidden = false;
  iotNotice.textContent = message;
  (iotNotice as HTMLElement).dataset.kind = kind;
}

function hideIotNotice() {
  if (!iotNotice) return;
  iotNotice.hidden = true;
  iotNotice.textContent = "";
  delete (iotNotice as HTMLElement).dataset.kind;
}

function iotDisplayName(device: any) {
  if (!device) return "";
  const meta = device.meta || {};
  const candidates = [meta.display_name, meta.note, meta.label, meta.location];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) {
      return candidate.trim();
    }
  }
  return device.device_id;
}

function formatIotTimestamp(ts: number) {
  if (!ts && ts !== 0) return "-";
  const date = new Date(ts * 1000);
  if (Number.isNaN(date.getTime())) {
    return String(ts);
  }
  return date.toLocaleString("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function formatIotRelativeTime(ts: number) {
  if (!ts && ts !== 0) return "未記録";
  const date = new Date(ts * 1000);
  if (Number.isNaN(date.getTime())) {
    return String(ts);
  }
  const diff = Date.now() - date.getTime();
  if (diff < 0) return formatIotTimestamp(ts);
  const sec = Math.floor(diff / 1000);
  if (sec < 5) return "たった今";
  if (sec < 60) return `${sec}秒前`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}分前`;
  const hours = Math.floor(min / 60);
  if (hours < 24) return `${hours}時間前`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}日前`;
  return formatIotTimestamp(ts);
}

function formatIotMetaValue(value: any) {
  if (value === null) return "null";
  if (value === undefined) return "-";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function createIotStat(label: string, value: any) {
  const wrapper = document.createElement("div");
  wrapper.className = "device-stat";
  const labelEl = document.createElement("div");
  labelEl.className = "device-stat__label";
  labelEl.textContent = label;
  const valueEl = document.createElement("div");
  valueEl.className = "device-stat__value";
  const textValue = value == null ? "-" : String(value);
  valueEl.textContent = textValue;
  valueEl.title = textValue;
  wrapper.appendChild(labelEl);
  wrapper.appendChild(valueEl);
  return wrapper;
}

function createCollapsibleText(text: string, { maxLength = 180 }: { maxLength?: number } = {}) {
  const str = text == null ? "" : String(text);
  const wrapper = document.createElement("div");
  wrapper.className = "collapsible-text";
  const content = document.createElement("div");
  content.className = "collapsible-text__content";
  content.textContent = str;
  content.title = str;
  wrapper.appendChild(content);

  if (str.length <= maxLength) {
    (wrapper as any).dataset.state = "expanded";
    return wrapper;
  }

  const fullText = str;
  const truncated = fullText.slice(0, maxLength).trimEnd() + "…";
  let collapsed = true;

  const toggleBtn = document.createElement("button");
  toggleBtn.type = "button";
  toggleBtn.className = "collapsible-text__toggle";
  toggleBtn.textContent = "もっと見る";
  toggleBtn.setAttribute("aria-expanded", "false");

  const applyState = () => {
    if (collapsed) {
      content.textContent = truncated;
      (wrapper as any).dataset.state = "collapsed";
      toggleBtn.textContent = "もっと見る";
      toggleBtn.setAttribute("aria-expanded", "false");
      toggleBtn.setAttribute("aria-label", "全文を表示");
    } else {
      content.textContent = fullText;
      (wrapper as any).dataset.state = "expanded";
      toggleBtn.textContent = "閉じる";
      toggleBtn.setAttribute("aria-expanded", "true");
      toggleBtn.setAttribute("aria-label", "折りたたむ");
    }
  };

  toggleBtn.addEventListener("click", () => {
    collapsed = !collapsed;
    applyState();
  });

  wrapper.appendChild(toggleBtn);
  applyState();
  return wrapper;
}

function renderIotCapabilities(capabilities: any[]) {
  if (!Array.isArray(capabilities) || capabilities.length === 0) {
    return null;
  }
  const names: string[] = [];
  for (const cap of capabilities) {
    if (cap && typeof cap.name === "string" && cap.name.trim()) {
      names.push(cap.name.trim());
    }
  }
  if (!names.length) {
    return null;
  }
  const section = document.createElement("div");
  section.className = "device-section";
  const label = document.createElement("div");
  label.className = "device-section__label";
  label.textContent = "提供機能";
  section.appendChild(label);
  const list = document.createElement("div");
  list.className = "device-section__body";
  const maxChips = 8;
  names.slice(0, maxChips).forEach((name) => {
    const chip = document.createElement("span");
    chip.className = "capability-badge";
    chip.textContent = name;
    list.appendChild(chip);
  });
  if (names.length > maxChips) {
    const rest = document.createElement("span");
    rest.className = "capability-badge";
    rest.textContent = `+${names.length - maxChips}`;
    rest.title = names.slice(maxChips).join(", ");
    list.appendChild(rest);
  }
  section.appendChild(list);
  return section;
}

function renderIotLastResult(result: any) {
  if (!result || typeof result !== "object") return null;
  const section = document.createElement("div");
  section.className = "device-section";
  const label = document.createElement("div");
  label.className = "device-section__label";
  label.textContent = "最後のジョブ";
  section.appendChild(label);

  const box = document.createElement("div");
  box.className = "device-last-result";

  const statusLine = document.createElement("div");
  statusLine.className = "device-last-result__meta";
  const statusText = result.ok ? "成功" : "失敗";
  const statusParts = [`ステータス: ${statusText}`];
  if (result.job_id) {
    statusParts.push(`ジョブID: ${result.job_id}`);
  }
  if (result.completed_at) {
    statusParts.push(`完了: ${formatIotTimestamp(result.completed_at)}`);
  }
  statusLine.textContent = statusParts.join(" / ");
  box.appendChild(statusLine);

  if (Object.prototype.hasOwnProperty.call(result, "return_value")) {
    const returnLine = document.createElement("div");
    returnLine.appendChild(createCollapsibleText(formatIotMetaValue(result.return_value)));
    box.appendChild(returnLine);
  }
  if (result.error || result.message) {
    const errorLine = document.createElement("div");
    errorLine.appendChild(createCollapsibleText(result.error || result.message));
    box.appendChild(errorLine);
  }
  if (result.output) {
    const outputLine = document.createElement("div");
    outputLine.appendChild(createCollapsibleText(formatIotMetaValue(result.output)));
    box.appendChild(outputLine);
  }
  section.appendChild(box);
  return section;
}

function renderIotDevices() {
  const grid = deviceGrid;
  if (!grid) return;
  grid.innerHTML = "";

  if (!iotState.devices.length) {
    const empty = document.createElement("div");
    empty.className = "device-empty";
    empty.innerHTML = "<p>登録されたデバイスがありません。</p><p>右上の「デバイス登録」から登録してください。</p>";
    grid.appendChild(empty);
    return;
  }

  iotState.devices.forEach((device) => {
    const card = document.createElement("article");
    card.className = "device-card";
    (card as any).dataset.deviceId = device.device_id;

    const header = document.createElement("div");
    header.className = "device-card-header";

    const summary = document.createElement("div");
    summary.className = "device-summary";
    const icon = document.createElement("div");
    icon.className = "device-icon";
    icon.innerHTML = IOT_DEVICE_ICON;
    summary.appendChild(icon);

    const metaWrap = document.createElement("div");
    metaWrap.className = "device-meta";
    const nameEl = document.createElement("div");
    nameEl.className = "device-name";
    nameEl.textContent = iotDisplayName(device);
    const idEl = document.createElement("div");
    idEl.className = "device-id";
    idEl.textContent = device.device_id;
    metaWrap.appendChild(nameEl);
    metaWrap.appendChild(idEl);
    summary.appendChild(metaWrap);

    header.appendChild(summary);

    const actions = document.createElement("div");
    actions.className = "device-actions";

    const renameBtn = document.createElement("button");
    renameBtn.type = "button";
    renameBtn.className = "icon-btn";
    (renameBtn as any).dataset.action = "rename";
    (renameBtn as any).dataset.deviceId = device.device_id;
    renameBtn.title = "名称変更";
    renameBtn.setAttribute("aria-label", `${iotDisplayName(device)} の名前を変更`);
    renameBtn.textContent = "✎";
    actions.appendChild(renameBtn);

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "icon-btn";
    (deleteBtn as any).dataset.action = "delete";
    (deleteBtn as any).dataset.deviceId = device.device_id;
    deleteBtn.title = "デバイスを削除";
    deleteBtn.setAttribute("aria-label", `${iotDisplayName(device)} を削除`);
    deleteBtn.textContent = "🗑";
    actions.appendChild(deleteBtn);

    header.appendChild(actions);
    card.appendChild(header);

    const body = document.createElement("div");
    body.className = "device-body";

    const stats = document.createElement("div");
    stats.className = "device-stats";
    stats.appendChild(createIotStat("最終アクセス", formatIotRelativeTime(device.last_seen)));
    stats.appendChild(createIotStat("登録日時", formatIotTimestamp(device.registered_at)));
    const queueDepth = Number.isFinite(Number(device.queue_depth)) ? `${Number(device.queue_depth)}件` : "-";
    stats.appendChild(createIotStat("待機ジョブ", queueDepth));
    body.appendChild(stats);

    const capabilities = renderIotCapabilities(device.capabilities);
    if (capabilities) {
      body.appendChild(capabilities);
    }
    const lastResult = renderIotLastResult(device.last_result);
    if (lastResult) {
      body.appendChild(lastResult);
    }

    card.appendChild(body);
    grid.appendChild(card);
  });
}

async function fetchIotDevices({ silent = false }: { silent?: boolean } = {}) {
  if (iotState.fetching) return;
  iotState.fetching = true;
  try {
    const { data, unavailable } = await iotAgentRequest("/api/devices");
    if (unavailable || data?.status === "unavailable") {
      iotState.devices = [];
      renderIotDevices();
      if (!silent) {
        showIotNotice(data?.message || "IoT エージェントに接続できません。", "error");
      }
      return;
    }
    if (Array.isArray(data.devices)) {
      iotState.devices = data.devices;
    } else {
      iotState.devices = [];
    }
    renderIotDevices();
    if (iotNotice && (iotNotice as any).dataset.kind === "error") {
      hideIotNotice();
    }
  } catch (error: any) {
    console.error("Failed to fetch devices", error);
    if (!silent) {
      showIotNotice(`デバイス一覧の取得に失敗しました: ${error.message}`, "error");
    }
  } finally {
    iotState.fetching = false;
  }
}

async function updateIotDeviceDisplayName(deviceId: string, displayName: string) {
  const payload = { display_name: displayName || null };
  const { data, unavailable } = await iotAgentRequest(`/api/devices/${encodeURIComponent(deviceId)}/name`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
  if (unavailable || data?.status === "unavailable") {
    showIotNotice(data?.message || "IoT エージェントに接続できません。", "error");
    return null;
  }
  return data?.device || null;
}

async function deleteIotDevice(deviceId: string) {
  const { data, unavailable } = await iotAgentRequest(`/api/devices/${encodeURIComponent(deviceId)}`, {
    method: "DELETE",
  });
  if (unavailable || data?.status === "unavailable") {
    showIotNotice(data?.message || "IoT エージェントに接続できません。", "error");
    return;
  }
  return data;
}

function updateLocalDevice(updated: any) {
  if (!updated) return;
  const index = iotState.devices.findIndex((device) => device.device_id === updated.device_id);
  if (index !== -1) {
    iotState.devices[index] = updated;
  }
}

function setRegisterMessage(message: string, kind: "info" | "error" | "success" = "info") {
  if (!registerMessageEl) return;
  registerMessageEl.textContent = message;
  registerMessageEl.className = "dialog-message";
  if (kind === "error") {
    registerMessageEl.classList.add("error");
  } else if (kind === "success") {
    registerMessageEl.classList.add("success");
  }
}

function resetRegisterDialog() {
  registerForm?.reset();
  if (registerSubmitBtn) {
    registerSubmitBtn.disabled = false;
    registerSubmitBtn.textContent = "登録";
  }
  setRegisterMessage(REGISTER_MESSAGE_DEFAULT);
}

async function handleRegisterSubmit(event: Event) {
  event.preventDefault();
  if (!registerSubmitBtn) return;

  const deviceId = registerIdInput ? registerIdInput.value.trim() : "";
  const displayNameInput = registerNameInput ? registerNameInput.value.trim() : "";
  const note = registerNoteInput ? registerNoteInput.value.trim() : "";

  if (!deviceId) {
    setRegisterMessage("デバイスIDを入力してください。", "error");
    registerIdInput?.focus();
    return;
  }

  const payload: any = {
    device_id: deviceId,
    capabilities: [],
    meta: { registered_via: "dashboard" },
    approved: true,
  };

  if (displayNameInput) {
    payload.meta.display_name = displayNameInput;
  }
  if (note) {
    payload.meta.note = note;
  }

  registerSubmitBtn.disabled = true;
  registerSubmitBtn.textContent = "登録中…";
  setRegisterMessage("サーバーへ登録しています…");

  try {
    const { data, unavailable } = await iotAgentRequest("/api/devices/register", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    if (unavailable || data?.status === "unavailable") {
      setRegisterMessage(data?.message || "IoT エージェントに接続できません。", "error");
      return;
    }
    const registeredId = typeof data?.device_id === "string" ? data.device_id : deviceId;
    const registeredDevice = data?.device && typeof data.device === "object" ? data.device : null;
    lastRegisteredDevice = {
      id: registeredId,
      name: registeredDevice ? iotDisplayName(registeredDevice) : displayNameInput || registeredId,
    };
    setRegisterMessage(`デバイス ${lastRegisteredDevice.name} を登録しました。`, "success");
    registerDialog?.close("success");
  } catch (error: any) {
    const message = error instanceof Error ? error.message : String(error);
    setRegisterMessage(`登録に失敗しました: ${message}`, "error");
  } finally {
    registerSubmitBtn.disabled = false;
    registerSubmitBtn.textContent = "登録";
  }
}

function startIotPolling() {
  if (iotState.pollTimer !== null) return;
  iotState.pollTimer = window.setInterval(() => {
    fetchIotDevices({ silent: true });
  }, IOT_FETCH_INTERVAL);
}

export function ensureIotDashboardInitialized({ showLoading = false }: { showLoading?: boolean } = {}) {
  if (!iotState.initialized) {
    iotState.initialized = true;
    fetchIotDevices();
    startIotPolling();
    return;
  }
  if (showLoading) {
    fetchIotDevices();
  }
}

function bindIotEvents() {
  if (registerDeviceBtn && registerDialog) {
    registerDeviceBtn.addEventListener("click", () => {
      resetRegisterDialog();
      registerDialog?.showModal();
      setTimeout(() => registerIdInput?.focus(), 50);
    });
  }

  if (registerCancelBtn && registerDialog) {
    registerCancelBtn.addEventListener("click", () => {
      registerDialog?.close("cancel");
    });
  }

  if (registerForm) {
    registerForm.addEventListener("submit", handleRegisterSubmit);
  }

  if (registerDialog) {
    registerDialog.addEventListener("close", () => {
      if (registerDialog?.returnValue === "success" && lastRegisteredDevice) {
        const label = lastRegisteredDevice.name || lastRegisteredDevice.id;
        const suffix = lastRegisteredDevice.name && lastRegisteredDevice.name !== lastRegisteredDevice.id
          ? ` (ID: ${lastRegisteredDevice.id})`
          : "";
        showIotNotice(`デバイス「${label}」${suffix}を登録しました。エッジデバイスをオンラインにするとジョブの取得を開始できます。`, "success");
        fetchIotDevices({ silent: false });
      }
      lastRegisteredDevice = null;
      resetRegisterDialog();
    });
  }

  if (refreshDevicesBtn) {
    refreshDevicesBtn.addEventListener("click", () => {
      fetchIotDevices();
    });
  }

  if (deviceGrid) {
    deviceGrid.addEventListener("click", async (event) => {
      const target = event.target instanceof Element ? event.target.closest("button[data-action]") : null;
      if (!target) return;
      const action = (target as HTMLElement).dataset.action;
      const deviceId = (target as HTMLElement).dataset.deviceId;
      if (!action || !deviceId) return;
      event.preventDefault();

      if (action === "rename") {
        const device = iotState.devices.find((d) => d.device_id === deviceId);
        const currentName = device?.meta?.display_name && typeof device.meta.display_name === "string"
          ? device.meta.display_name
          : "";
        const promptLabel = currentName || iotDisplayName(device) || deviceId;
        const newName = window.prompt(`「${promptLabel}」の新しい名前を入力してください。`, currentName);
        if (newName === null) return;
        const trimmed = newName.trim();
        if (trimmed === (currentName || "").trim()) return;
        try {
          const updatedDevice = await updateIotDeviceDisplayName(deviceId, trimmed);
          if (updatedDevice) {
            updateLocalDevice(updatedDevice);
            renderIotDevices();
            showIotNotice(`デバイス名を「${iotDisplayName(updatedDevice)}」に更新しました。`, "success");
            fetchIotDevices({ silent: true });
          } else {
            throw new Error("更新後のデバイス情報が取得できませんでした。");
          }
        } catch (error: any) {
          const message = error instanceof Error ? error.message : String(error);
          showIotNotice(`名前の更新に失敗しました: ${message}`, "error");
        }
        return;
      }

      if (action === "delete") {
        const device = iotState.devices.find((d) => d.device_id === deviceId);
        const label = iotDisplayName(device) || deviceId;
        const confirmed = window.confirm(`デバイス「${label}」を削除しますか？\nジョブキューや履歴も失われます。`);
        if (!confirmed) return;
        try {
          await deleteIotDevice(deviceId);
          iotState.devices = iotState.devices.filter((d) => d.device_id !== deviceId);
          renderIotDevices();
          showIotNotice(`デバイス「${label}」を削除しました。`, "success");
          fetchIotDevices({ silent: true });
        } catch (error: any) {
          const message = error instanceof Error ? error.message : String(error);
          showIotNotice(`デバイスの削除に失敗しました: ${message}`, "error");
        }
      }
    });
  }
}

export function initIotDom() {
  deviceGrid = $("#deviceGrid");
  iotNotice = $("#iotNotice");
  registerDeviceBtn = $("#registerDeviceBtn") as HTMLButtonElement | null;
  refreshDevicesBtn = $("#refreshDevicesBtn") as HTMLButtonElement | null;

  registerDialog = $("#iotRegisterDialog") as HTMLDialogElement | null;
  registerForm = $("#iotRegisterForm") as HTMLFormElement | null;
  registerIdInput = $("#iotRegisterId") as HTMLInputElement | null;
  registerNameInput = $("#iotRegisterName") as HTMLInputElement | null;
  registerNoteInput = $("#iotRegisterNote") as HTMLInputElement | null;
  registerMessageEl = $("#iotRegisterMessage");
  registerCancelBtn = $("#iotRegisterCancel") as HTMLButtonElement | null;
  registerSubmitBtn = $("#iotRegisterSubmit") as HTMLButtonElement | null;

  REGISTER_MESSAGE_DEFAULT = registerMessageEl?.textContent?.trim()
    || REGISTER_MESSAGE_DEFAULT;

  bindIotEvents();
}

export function summarizeIotDevices() {
  if (!iotState.devices.length) {
    return "登録済みのデバイスはありません。";
  }
  const summaries = iotState.devices.map((device) => {
    const caps = Array.isArray(device.capabilities)
      ? device.capabilities.map((cap: any) => cap?.name).filter(Boolean)
      : [];
    const capText = caps.length ? `（機能: ${caps.join(", ")}）` : "";
    return `${iotDisplayName(device)}${capText}`;
  });
  return summaries.join(" / ");
}
