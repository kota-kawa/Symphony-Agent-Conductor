import React, { useEffect } from "react";
import { createRoot } from "react-dom/client";
import { initSchedulerUi } from "./scheduler/ui";

const SchedulerBootstrap: React.FC = () => {
  useEffect(() => {
    initSchedulerUi();
  }, []);
  return null;
};

const mount = document.getElementById("scheduler-root");
if (mount) {
  createRoot(mount).render(<SchedulerBootstrap />);
}
