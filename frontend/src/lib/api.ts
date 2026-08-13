import type { HyperparamConfig, RunEvent, RunState } from "./types";

const BASE = "/api/runs";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function createRun(prompt: string, autoApprove: boolean): Promise<RunState> {
  const res = await fetch(BASE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, auto_approve: autoApprove }),
  });
  return json<RunState>(res);
}

export async function listRuns(): Promise<RunState[]> {
  const res = await fetch(BASE);
  return json<RunState[]>(res);
}

export async function getRun(runId: string): Promise<RunState> {
  const res = await fetch(`${BASE}/${runId}`);
  return json<RunState>(res);
}

export async function sendMessage(runId: string, text: string): Promise<void> {
  await fetch(`${BASE}/${runId}/message`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

export async function approveRun(runId: string): Promise<void> {
  await fetch(`${BASE}/${runId}/approve`, { method: "POST" });
}

export async function patchHyperparams(
  runId: string,
  updates: Partial<HyperparamConfig>
): Promise<RunState> {
  const res = await fetch(`${BASE}/${runId}/hyperparams`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
  return json<RunState>(res);
}

export async function cancelRun(runId: string): Promise<void> {
  await fetch(`${BASE}/${runId}/cancel`, { method: "POST" });
}

export function subscribeEvents(runId: string, onEvent: (e: RunEvent) => void): () => void {
  const es = new EventSource(`${BASE}/${runId}/events`);
  es.addEventListener("log", (ev) => {
    try {
      onEvent(JSON.parse((ev as MessageEvent).data));
    } catch {
      /* ignore malformed event */
    }
  });
  return () => es.close();
}

export async function getHealth(): Promise<{
  status: string;
  tinker_configured: boolean;
  ram_available_gb: number;
  ram_total_gb: number;
  mps: boolean;
  cuda: boolean;
}> {
  const res = await fetch("/api/health");
  return json(res);
}
