"use client";

import { useEffect, useState } from "react";
import { api, RankedUniverseRow } from "@/lib/api";

export function RankedUniverseTable({ refreshKey }: { refreshKey: number }) {
  const [rows, setRows] = useState<RankedUniverseRow[]>([]);
  const [asOf, setAsOf] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (refreshKey === 0) return;
    api
      .rankedUniverse(25)
      .then((res) => {
        setRows(res.universe);
        setAsOf(res.as_of);
        setError(null);
      })
      .catch((e) => setError(e.message));
  }, [refreshKey]);

  if (error) return <p className="text-sm text-red-400">{error}</p>;
  if (!rows.length) return <p className="text-sm text-neutral-500">Run the pipeline to see the ranked universe.</p>;

  return (
    <div className="overflow-x-auto rounded-lg border border-neutral-800">
      <div className="border-b border-neutral-800 bg-neutral-900 px-4 py-2 text-xs text-neutral-400">
        Ranked as of {asOf}
      </div>
      <table className="w-full text-left text-sm">
        <thead className="bg-neutral-900 text-neutral-400">
          <tr>
            <th className="px-3 py-2">Rank</th>
            <th className="px-3 py-2">Ticker</th>
            <th className="px-3 py-2">Combined score</th>
            <th className="px-3 py-2">Classical α</th>
            <th className="px-3 py-2">ML α</th>
            <th className="px-3 py-2 w-48">Attribution (classical / ML)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.ticker} className="border-t border-neutral-800 hover:bg-neutral-900/50">
              <td className="px-3 py-2 text-neutral-400">{r.rank}</td>
              <td className="px-3 py-2 font-medium">{r.ticker}</td>
              <td className="px-3 py-2">{r.combined_score.toFixed(3)}</td>
              <td className="px-3 py-2">{(r.classical_alpha * 100).toFixed(2)}%</td>
              <td className="px-3 py-2">{(r.ml_alpha * 100).toFixed(2)}%</td>
              <td className="px-3 py-2">
                <div className="flex h-2 w-full overflow-hidden rounded-full bg-neutral-800">
                  <div
                    className="h-full bg-sky-500"
                    style={{ width: `${r.classical_attribution_pct}%` }}
                    title={`Classical ${r.classical_attribution_pct}%`}
                  />
                  <div
                    className="h-full bg-emerald-500"
                    style={{ width: `${r.ml_attribution_pct}%` }}
                    title={`ML ${r.ml_attribution_pct}%`}
                  />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
