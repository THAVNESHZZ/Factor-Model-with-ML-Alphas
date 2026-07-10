"use client";

import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { api, ICDecayResponse } from "@/lib/api";

export function AnalysisCharts({ refreshKey }: { refreshKey: number }) {
  const [ic, setIc] = useState<ICDecayResponse | null>(null);
  const [importance, setImportance] = useState<Record<string, number> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (refreshKey === 0) return;
    Promise.all([api.icDecay(), api.featureImportance()])
      .then(([icRes, impRes]) => {
        setIc(icRes);
        setImportance(impRes);
        setError(null);
      })
      .catch((e) => setError(e.message));
  }, [refreshKey]);

  if (error) return <p className="text-sm text-red-400">{error}</p>;
  if (!ic || !importance) return null;

  const periods = Object.keys(ic.classical.ic_summary);
  const icData = periods.map((p) => ({
    period: p.replace("period_", "") + "D",
    Classical: ic.classical.ic_summary[p].mean_ic,
    ML: ic.ml.ic_summary[p].mean_ic,
  }));

  const importanceData = Object.entries(importance)
    .sort((a, b) => b[1] - a[1])
    .map(([feature, value]) => ({ feature, importance: value }));

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
        <h3 className="mb-1 text-sm font-medium text-neutral-200">Mean IC by horizon (decay)</h3>
        <p className="mb-3 text-xs text-neutral-500">
          Classical turnover: {(ic.classical.mean_quantile_turnover ?? 0).toFixed(2)} · ML turnover:{" "}
          {(ic.ml.mean_quantile_turnover ?? 0).toFixed(2)}
        </p>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={icData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
            <XAxis dataKey="period" stroke="#a3a3a3" fontSize={12} />
            <YAxis stroke="#a3a3a3" fontSize={12} />
            <Tooltip contentStyle={{ background: "#171717", border: "1px solid #404040" }} />
            <Legend />
            <Bar dataKey="Classical" fill="#38bdf8" />
            <Bar dataKey="ML" fill="#34d399" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
        <h3 className="mb-1 text-sm font-medium text-neutral-200">ML feature importance</h3>
        <p className="mb-3 text-xs text-neutral-500">Normalized LightGBM split importance</p>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={importanceData} layout="vertical" margin={{ left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
            <XAxis type="number" stroke="#a3a3a3" fontSize={12} />
            <YAxis dataKey="feature" type="category" stroke="#a3a3a3" fontSize={12} width={100} />
            <Tooltip contentStyle={{ background: "#171717", border: "1px solid #404040" }} />
            <Bar dataKey="importance" fill="#a78bfa" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
