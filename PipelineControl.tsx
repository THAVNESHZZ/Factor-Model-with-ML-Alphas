"use client";

import { useState } from "react";
import { api } from "@/lib/api";

export function PipelineControl({ onReady }: { onReady: () => void }) {
  const [status, setStatus] = useState<"idle" | "running" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setStatus("running");
    setError(null);
    try {
      await api.runPipeline();
      setStatus("ready");
      onReady();
    } catch (e) {
      setStatus("error");
      setError(e instanceof Error ? e.message : "Unknown error");
    }
  }

  return (
    <div className="flex items-center gap-4 rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <button
        onClick={handleRun}
        disabled={status === "running"}
        className="rounded-md bg-emerald-500 px-4 py-2 text-sm font-medium text-black transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {status === "running" ? "Running pipeline..." : "Run pipeline"}
      </button>
      <span className="text-sm text-neutral-400">
        {status === "idle" && "Ingestion → features → factor model → ML alpha → Alphalens"}
        {status === "running" && "Ingesting data, fitting factors, walk-forward training..."}
        {status === "ready" && "Pipeline ready. Data below is live."}
        {status === "error" && `Failed: ${error}`}
      </span>
    </div>
  );
}
