import React, { useEffect } from "react";
import { createRoot } from "react-dom/client";
import {
  activateView,
  getInitialActiveView,
  initLayout,
  registerViewActivationHook,
  requestMainBrowserViewportSync,
} from "./spa/layout";
import {
  ensureChatInitialized,
  ensureBrowserAgentInitialized,
  ensureOrchestratorInitialized,
  ensureIotChatInitialized,
  ensureSchedulerChatInitialized,
  setChatMode,
  initChatDom,
} from "./spa/chat";
import { ensureIotDashboardInitialized, initIotDom } from "./spa/iot";
import { ensureSchedulerAgentInitialized, initSchedulerDom } from "./spa/scheduler";
import { initSettingsModal } from "./spa/settings";
import { refreshAgentStatus } from "./spa/agent-status";

const Bootstrap: React.FC = () => {
  useEffect(() => {
    initLayout();
    initChatDom();
    initIotDom();
    initSchedulerDom();

    let schedulerWarmupScheduled = false;
    const warmupSchedulerResources = () => {
      if (schedulerWarmupScheduled) return;
      schedulerWarmupScheduled = true;

      const run = () => {
        try {
          ensureSchedulerAgentInitialized();
          ensureSchedulerChatInitialized({ forceSidebar: false });
        } catch (error) {
          console.warn("Scheduler warmup failed:", error);
        }
      };

      const idle = (window as any).requestIdleCallback;
      if (typeof idle === "function") {
        idle(run, { timeout: 1500 });
      } else {
        window.setTimeout(run, 600);
      }
    };

    registerViewActivationHook(({ view, isBrowserView, isChatView, isIotView, isGeneralView, isSchedulerView }) => {
      const modeMap: Record<string, string> = {
        browser: "browser",
        iot: "iot",
        general: "orchestrator",
        chat: "general",
        schedule: "scheduler",
      };
      setChatMode(modeMap[view] ?? "general");

      if (isChatView) {
        ensureChatInitialized({ showLoadingSummary: true });
      } else if (!isBrowserView && !isIotView && !isGeneralView) {
        ensureChatInitialized();
      }

      if (isBrowserView) {
        ensureBrowserAgentInitialized({ showLoading: true, forceSidebar: true });
        requestMainBrowserViewportSync({ reloadFallback: true });
      }

      if (isIotView) {
        ensureIotDashboardInitialized({ showLoading: true });
        ensureIotChatInitialized({ forceSidebar: true });
      }

      if (isGeneralView) {
        ensureOrchestratorInitialized({ forceSidebar: true });
      }

      if (isSchedulerView) {
        ensureSchedulerAgentInitialized();
        ensureSchedulerChatInitialized({ forceSidebar: true });
      }
    });

    initSettingsModal();
    activateView(getInitialActiveView());
    warmupSchedulerResources();
    refreshAgentStatus();
    const interval = window.setInterval(() => refreshAgentStatus({ silent: true }), 30000);
    return () => {
      window.clearInterval(interval);
    };
  }, []);

  return null;
};

const mount = document.getElementById("spa-root");
if (mount) {
  createRoot(mount).render(<Bootstrap />);
}
