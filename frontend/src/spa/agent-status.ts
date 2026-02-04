import { $ } from "./dom-utils";

type AgentKey = "browser" | "lifestyle" | "iot" | "scheduler";

type AgentStatusEntry = {
  available: boolean | null;
  enabled: boolean | null;
  error: string | null;
};

type AgentStatusPayload = {
  agents?: Record<string, { available?: boolean; enabled?: boolean; error?: string }>
  checked_at?: string;
};

const AGENT_LABELS: Record<AgentKey, string> = {
  browser: "ブラウザエージェント",
  lifestyle: "Life-Style エージェント",
  iot: "IoT エージェント",
  scheduler: "Scheduler エージェント",
};

const state: {
  agents: Record<AgentKey, AgentStatusEntry>;
  checkedAt: string | null;
} = {
  agents: {
    browser: { available: null, enabled: true, error: null },
    lifestyle: { available: null, enabled: true, error: null },
    iot: { available: null, enabled: true, error: null },
    scheduler: { available: null, enabled: true, error: null },
  },
  checkedAt: null,
};

function updateBanner() {
  const banner = $("#agentStatusBanner") as HTMLElement | null;
  const settingsBanner = $("#settingsAgentStatusBanner") as HTMLElement | null;
  const entries = Object.entries(state.agents) as [AgentKey, AgentStatusEntry][];
  const disconnected = entries.filter(([, info]) => info.enabled !== false && info.available === false);

  if (!disconnected.length) {
    if (banner) {
      banner.hidden = true;
      banner.textContent = "";
    }
    if (settingsBanner) {
      settingsBanner.hidden = true;
      settingsBanner.textContent = "";
    }
    return;
  }

  const names = disconnected.map(([key]) => AGENT_LABELS[key] || key);
  const message = `未接続: ${names.join(" / ")}。接続できているエージェントの機能のみ利用できます。`;

  if (banner) {
    banner.textContent = message;
    (banner as HTMLElement).dataset.kind = "error";
    banner.hidden = true; // Force hidden
  }

  if (settingsBanner) {
    settingsBanner.textContent = message;
    settingsBanner.hidden = false;
  }
}

function applyStatusPayload(payload: AgentStatusPayload) {
  const agents = payload?.agents && typeof payload.agents === "object" ? payload.agents : {};
  (Object.keys(state.agents) as AgentKey[]).forEach((key) => {
    const entry = agents[key];
    if (!entry || typeof entry !== "object") return;
    state.agents[key] = {
      available: entry.available ?? state.agents[key].available,
      enabled: entry.enabled ?? state.agents[key].enabled,
      error: entry.error ?? state.agents[key].error,
    };
  });
  state.checkedAt = payload?.checked_at || state.checkedAt;
  updateBanner();
}

export async function refreshAgentStatus({ silent = false }: { silent?: boolean } = {}) {
  try {
    const res = await fetch("/api/agent_status", { method: "GET" });
    if (!res.ok) {
      if (!silent) console.warn("Failed to fetch agent status", res.status);
      return null;
    }
    const data = (await res.json()) as AgentStatusPayload;
    applyStatusPayload(data);
    return data;
  } catch (error) {
    if (!silent) console.warn("Failed to fetch agent status", error);
    return null;
  }
}

export function markAgentUnavailable(agent: AgentKey, message?: string) {
  if (!agent || !state.agents[agent]) return;
  state.agents[agent] = {
    ...state.agents[agent],
    available: false,
    error: message || state.agents[agent].error,
  };
  updateBanner();
}

export function markAgentAvailable(agent: AgentKey) {
  if (!agent || !state.agents[agent]) return;
  state.agents[agent] = {
    ...state.agents[agent],
    available: true,
    error: null,
  };
  updateBanner();
}

export function getAgentStatus(agent: AgentKey) {
  if (!agent) return null;
  return state.agents[agent] || null;
}

export function isAgentAvailable(agent: AgentKey) {
  const entry = getAgentStatus(agent);
  if (!entry) return null;
  return entry.available;
}

export function applyAgentStatusPayload(payload: AgentStatusPayload) {
  if (!payload) return;
  applyStatusPayload(payload);
}
