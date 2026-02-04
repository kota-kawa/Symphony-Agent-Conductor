import { $ } from "./dom-utils";
import { markAgentAvailable, markAgentUnavailable } from "./agent-status";

let schedulerInline: HTMLElement | null = null;
let schedulerFallback: HTMLElement | null = null;
let schedulerRefreshBtn: HTMLButtonElement | null = null;
let schedulerMonthLabel: HTMLElement | null = null;
let calendarSlot: HTMLElement | null = null;
let inlinePlaceholder: HTMLElement | null = null;
let prevMonthBtn: HTMLButtonElement | null = null;
let nextMonthBtn: HTMLButtonElement | null = null;

let schedulerCalendarPanel: HTMLElement | null = null;
let schedulerDayPanel: HTMLElement | null = null;
let schedulerDayBackBtn: HTMLButtonElement | null = null;
let schedulerDayContent: HTMLElement | null = null;

function sanitizeBase(value: unknown): string {
  return typeof value === "string" ? value.trim().replace(/\/+$/, "") : "";
}

export function resolveSchedulerAgentBase() {
  let queryBase = "";
  try {
    queryBase = new URLSearchParams(window.location.search).get("scheduler_agent_base") || "";
  } catch {
    queryBase = "";
  }

  const sources = [
    sanitizeBase(queryBase),
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

const SCHEDULER_AGENT_BASE = resolveSchedulerAgentBase();

export function buildSchedulerAgentUrl(path = "") {
  if (!path) {
    return SCHEDULER_AGENT_BASE || "/scheduler_agent";
  }
  if (/^https?:/i.test(path)) {
    return path;
  }
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const base = SCHEDULER_AGENT_BASE || "";
  if (!base) return normalizedPath;
  if (/^https?:/i.test(base)) {
    return `${base.replace(/\/+$/, "")}${normalizedPath}`;
  }
  return `${base.replace(/\/+$/, "")}${normalizedPath}` || normalizedPath;
}

export async function schedulerAgentRequest(
  path: string,
  { method = "GET", headers = {}, body, signal }: { method?: string; headers?: Record<string, string>; body?: BodyInit | null; signal?: AbortSignal } = {},
) {
  const url = buildSchedulerAgentUrl(path);
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
    markAgentUnavailable("scheduler", error?.message || "接続に失敗しました。");
    return { data: { status: "unavailable", message: "Scheduler エージェントに接続できません。", error: error?.message }, status: 0, unavailable: true };
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
      markAgentUnavailable("scheduler", message);
      return { data: { status: "unavailable", message: "Scheduler エージェントに接続できません。", error: message }, status: response.status, unavailable: true };
    }
    const error = new Error(message) as any;
    error.status = response.status;
    error.data = data;
    throw error;
  }

  const payload = typeof data === "string" ? { message: data } : data;
  if (payload && payload.status === "unavailable") {
    markAgentUnavailable("scheduler", payload.error || payload.message);
    return { data: payload, status: response.status, unavailable: true };
  }
  markAgentAvailable("scheduler");
  return { data: payload, status: response.status };
}

function showFallback(message?: string) {
  if (!schedulerFallback) return;
  schedulerFallback.textContent = message || schedulerFallback.textContent || "";
  schedulerFallback.hidden = false;
}

function hideFallback() {
  if (!schedulerFallback) return;
  schedulerFallback.hidden = true;
}

function getYearMonth() {
  const now = new Date();
  const year = parseInt(schedulerInline?.dataset.year ?? "", 10);
  const month = parseInt(schedulerInline?.dataset.month ?? "", 10);
  return {
    year: Number.isFinite(year) ? year : now.getFullYear(),
    month: Number.isFinite(month) ? month : now.getMonth() + 1,
  };
}

function setYearMonth(year: number, month: number) {
  if (!schedulerInline) return;
  schedulerInline.dataset.year = String(year);
  schedulerInline.dataset.month = String(month);
  if (schedulerMonthLabel) {
    schedulerMonthLabel.textContent = `${year}年 ${month}月`;
  }
}

function adjustMonth(year: number, month: number, delta: number) {
  const nextMonth = month + delta;
  if (nextMonth > 12) return { year: year + 1, month: 1 };
  if (nextMonth < 1) return { year: year - 1, month: 12 };
  return { year, month: nextMonth };
}

function setLoading(isLoading: boolean) {
  if (schedulerRefreshBtn) {
    schedulerRefreshBtn.disabled = isLoading;
    schedulerRefreshBtn.classList.toggle("is-loading", isLoading);
  }
  if (prevMonthBtn) prevMonthBtn.disabled = isLoading;
  if (nextMonthBtn) nextMonthBtn.disabled = isLoading;
}

async function fetchCalendarPartial(year: number, month: number) {
  const url = `/scheduler-ui/calendar_partial?year=${year}&month=${month}&t=${Date.now()}`;
  const res = await fetch(url, { headers: { "X-Requested-With": "fetch" } });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    const message = text || `HTTP ${res.status}`;
    throw new Error(message);
  }
  return res.text();
}

async function refreshInlineCalendar({ year, month, delta }: { year?: number; month?: number; delta?: number } = {}) {
  if (!schedulerInline || !calendarSlot) return;
  const current = getYearMonth();
  let targetYear = Number.isFinite(year) ? year! : current.year;
  let targetMonth = Number.isFinite(month) ? month! : current.month;

  if (typeof delta === "number") {
    const adjusted = adjustMonth(targetYear, targetMonth, delta);
    targetYear = adjusted.year;
    targetMonth = adjusted.month;
  }

  setLoading(true);
  hideFallback();

  try {
    const html = await fetchCalendarPartial(targetYear, targetMonth);
    const temp = document.createElement("div");
    temp.innerHTML = html;
    const newGrid = temp.querySelector("#calendar-grid");
    if (!newGrid) {
      throw new Error("カレンダーの描画に失敗しました。");
    }
    const currentGrid = calendarSlot.querySelector("#calendar-grid");
    if (currentGrid) {
      currentGrid.replaceWith(newGrid);
    } else {
      calendarSlot.appendChild(newGrid);
    }
    schedulerInline.dataset.hasData = "1";
    if (inlinePlaceholder) {
      inlinePlaceholder.hidden = true;
    }
    setYearMonth(targetYear, targetMonth);
  } catch (error: any) {
    showFallback(error.message || "カレンダーの更新に失敗しました。");
  } finally {
    setLoading(false);
  }
}

function formatWeekday(dateStr: string) {
  const date = new Date(dateStr);
  const weekdays = ["日曜日", "月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日"];
  return weekdays[date.getDay()];
}

function formatDate(dateStr: string) {
  const date = new Date(dateStr);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}.${month}.${day}`;
}

function renderDayViewContent(data: any) {
  if (data?.status === "unavailable") {
    return `
      <div class="scheduler-day-view__error">
        <i class="bi bi-cloud-slash"></i>
        <h4>接続できません</h4>
        <p>${data.message || "Scheduler エージェントに接続できません。"}</p>
        <button class="btn subtle" onclick="window.closeSchedulerDayView()">カレンダーに戻る</button>
      </div>
    `;
  }
  const { date, timeline_items, completion_rate, day_log_content } = data;

  let timelineHtml = "";
  if (timeline_items && timeline_items.length > 0) {
    const timelineItemsHtml = timeline_items.map((item: any) => {
      const isDone = item.is_done || item.log_done;
      const memo = item.log_memo || "";
      const categoryClass = item.step_category ? `badge-${item.step_category.toLowerCase()}` : "badge-other";

      return `
        <div class="scheduler-day-timeline-item ${isDone ? "is-done" : ""}">
          <div class="scheduler-day-timeline-dot"></div>
          <div class="scheduler-day-timeline-time">${item.time}</div>
          <div class="scheduler-day-timeline-card">
            <div class="scheduler-day-timeline-header">
              <div>
                <span class="scheduler-day-badge ${categoryClass}">${item.step_category || "Other"}</span>
                <h5 class="scheduler-day-task-name">${item.step_name}</h5>
                <small class="scheduler-day-routine-name">
                  <i class="bi bi-collection me-1"></i>${item.routine_name}
                </small>
              </div>
              <div class="scheduler-day-status">
                ${isDone
                  ? '<i class="bi bi-check-circle-fill text-success"></i>'
                  : '<i class="bi bi-circle text-muted"></i>'}
              </div>
            </div>
            ${memo ? `<div class="scheduler-day-memo"><i class="bi bi-chat-left-text me-1"></i>${memo}</div>` : ""}
          </div>
        </div>
      `;
    }).join("");

    timelineHtml = `
      <div class="scheduler-day-schedule-card">
        <div class="scheduler-day-schedule-header">
          <div>
            <h6 class="scheduler-day-weekday">${formatWeekday(date)}</h6>
            <h2 class="scheduler-day-date">${formatDate(date)}</h2>
          </div>
          <div class="scheduler-day-completion">
            <div class="scheduler-day-completion-rate">${completion_rate}%</div>
            <small>完了率</small>
          </div>
        </div>
        <hr class="scheduler-day-divider">
        <div class="scheduler-day-timeline">
          ${timelineItemsHtml}
        </div>
      </div>
    `;
  } else {
    timelineHtml = `
      <div class="scheduler-day-schedule-card">
        <div class="scheduler-day-schedule-header">
          <div>
            <h6 class="scheduler-day-weekday">${formatWeekday(date)}</h6>
            <h2 class="scheduler-day-date">${formatDate(date)}</h2>
          </div>
        </div>
        <hr class="scheduler-day-divider">
        <div class="scheduler-day-empty">
          <i class="bi bi-calendar-check"></i>
          <h4>タスクがありません</h4>
          <p>この日にはスケジュールされたタスクがありません。</p>
        </div>
      </div>
    `;
  }

  const logHtml = `
    <div class="scheduler-day-log-card">
      <div class="scheduler-day-log-header">
        <h5><i class="bi bi-journal-text me-2"></i>日報</h5>
        <small>今日の記録・感想</small>
      </div>
      <div class="scheduler-day-log-content">
        ${day_log_content
          ? `<p class="scheduler-day-log-text">${day_log_content.replace(/\n/g, "<br>")}</p>`
          : '<p class="scheduler-day-log-empty">日報は記録されていません。</p>'}
      </div>
    </div>
  `;

  return timelineHtml + logHtml;
}

async function fetchDayViewData(dateStr: string) {
  const url = buildSchedulerAgentUrl(`/api/day/${dateStr}`);
  const res = await fetch(url, { headers: { "X-Requested-With": "fetch" } });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    let errorMessage = `HTTP ${res.status}`;
    try {
      const json = JSON.parse(text);
      if (json.error) errorMessage = json.error;
    } catch {
      if (text) errorMessage = text;
    }
    throw new Error(errorMessage);
  }
  const data = await res.json();
  if (data?.status === "unavailable") {
    markAgentUnavailable("scheduler", data.error || data.message);
  } else {
    markAgentAvailable("scheduler");
  }
  return data;
}

function showDayView() {
  if (schedulerCalendarPanel) schedulerCalendarPanel.hidden = true;
  if (schedulerDayPanel) schedulerDayPanel.hidden = false;
}

function hideDayView() {
  if (schedulerDayPanel) schedulerDayPanel.hidden = true;
  if (schedulerCalendarPanel) schedulerCalendarPanel.hidden = false;
}

function showDayViewLoading() {
  if (schedulerDayContent) {
    schedulerDayContent.innerHTML = `
      <div class="scheduler-day-view__loading">
        <div class="spinner-border" role="status">
          <span class="visually-hidden">読み込み中...</span>
        </div>
        <p>データを読み込んでいます...</p>
      </div>
    `;
  }
}

function showDayViewError(message?: string) {
  if (schedulerDayContent) {
    schedulerDayContent.innerHTML = `
      <div class="scheduler-day-view__error">
        <i class="bi bi-exclamation-triangle"></i>
        <h4>読み込みに失敗しました</h4>
        <p>${message || "データの取得中にエラーが発生しました。"}</p>
        <button class="btn subtle" onclick="window.closeSchedulerDayView()">カレンダーに戻る</button>
      </div>
    `;
  }
}

async function openSchedulerDayView(dateStr: string) {
  showDayView();
  showDayViewLoading();

  try {
    const data = await fetchDayViewData(dateStr);
    if (schedulerDayContent) {
      schedulerDayContent.innerHTML = renderDayViewContent(data);
    }
  } catch (error: any) {
    console.error("Failed to load day view:", error);
    showDayViewError(error.message);
  }
}

function closeSchedulerDayView() {
  hideDayView();
  refreshInlineCalendar();
}

let inlineBound = false;
function bindInlineScheduler() {
  if (inlineBound || !schedulerInline) return;
  inlineBound = true;

  if (prevMonthBtn) {
    prevMonthBtn.addEventListener("click", () => {
      refreshInlineCalendar({ delta: -1 });
    });
  }
  if (nextMonthBtn) {
    nextMonthBtn.addEventListener("click", () => {
      refreshInlineCalendar({ delta: 1 });
    });
  }
  if (schedulerRefreshBtn) {
    schedulerRefreshBtn.addEventListener("click", () => {
      refreshInlineCalendar();
    });
  }
  if (schedulerDayBackBtn) {
    schedulerDayBackBtn.addEventListener("click", closeSchedulerDayView);
  }

  if (schedulerInline.dataset.hasData !== "1") {
    refreshInlineCalendar();
  }
}

export function ensureSchedulerAgentInitialized({ reload = false }: { reload?: boolean } = {}) {
  bindInlineScheduler();
  if (reload) {
    refreshInlineCalendar();
  }
}

export function initSchedulerDom() {
  schedulerInline = $("#schedulerInline");
  schedulerFallback = $("#schedulerCalendarFallback");
  schedulerRefreshBtn = $("#schedulerCalendarRefresh") as HTMLButtonElement | null;
  schedulerMonthLabel = $("#schedulerMonthLabel");
  calendarSlot = schedulerInline?.querySelector<HTMLElement>("[data-calendar-slot]") ?? null;
  inlinePlaceholder = schedulerInline?.querySelector<HTMLElement>(".scheduler-inline__placeholder") ?? null;
  prevMonthBtn = schedulerInline?.querySelector<HTMLButtonElement>("[data-action='prev-month']") ?? null;
  nextMonthBtn = schedulerInline?.querySelector<HTMLButtonElement>("[data-action='next-month']") ?? null;

  schedulerCalendarPanel = $("#schedulerCalendarPanel");
  schedulerDayPanel = $("#schedulerDayPanel");
  schedulerDayBackBtn = $("#schedulerDayBackBtn") as HTMLButtonElement | null;
  schedulerDayContent = $("#schedulerDayContent");

  (window as any).openSchedulerDayView = openSchedulerDayView;
  (window as any).closeSchedulerDayView = closeSchedulerDayView;
}
