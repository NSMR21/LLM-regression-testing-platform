import type { ReactNode } from "react";
import { ACTIVE, type RunStatus, type Verdict } from "../api";

export const pct = (x: number | null | undefined, digits = 1) => (x === null || x === undefined ? "–" : `${(x * 100).toFixed(digits)}%`);
export const pts = (x: number | null | undefined, digits = 1) =>
  x === null || x === undefined ? "–" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(digits)}`;
export const shortId = (id: string) => id.replace(/^run_/, "").slice(0, 8);
export const when = (iso: string | null | undefined) => {
  if (!iso) return "–";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
};

export function VerdictBadge({ verdict, big }: { verdict: Verdict | null | undefined; big?: boolean }) {
  if (!verdict) return <span className="badge b-neutral">–</span>;
  const icon = { pass: "✓", fail: "✕", inconclusive: "?", error: "!" }[verdict];
  return (
    <span className={`badge b-${verdict}${big ? " verdict-big" : ""}`}>
      {icon} {verdict.toUpperCase()}
    </span>
  );
}

export function StatusBadge({ status }: { status: RunStatus }) {
  if (status === "completed") return <span className="badge b-neutral">completed</span>;
  if (status === "errored") return <span className="badge b-error">errored</span>;
  if (status === "cancelled") return <span className="badge b-neutral">cancelled</span>;
  return <span className="badge b-active">{ACTIVE.includes(status) ? "● " : ""}{status}</span>;
}

export function Progress({ done, total }: { done: number; total: number }) {
  const width = total ? Math.round((100 * done) / total) : 0;
  return (
    <div className="progress" title={`${done}/${total}`}>
      <div style={{ width: `${width}%` }} />
    </div>
  );
}

export function Kpi({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="kpi">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {hint ? <div className="hint">{hint}</div> : null}
    </div>
  );
}

export function Loading() {
  return <div className="empty">Loading…</div>;
}

export function ErrorBox({ error }: { error: unknown }) {
  return <div className="error-box">{error instanceof Error ? error.message : String(error)}</div>;
}

export function Dots({ outcomes, errors }: { outcomes: (boolean | null)[]; errors?: boolean[] }) {
  return (
    <span className="grid-dots">
      {outcomes.map((ok, i) => (
        <span key={i} className={`dot ${errors?.[i] ? "err" : ok ? "ok" : "no"}`} title={errors?.[i] ? "error" : ok ? "pass" : "fail"} />
      ))}
    </span>
  );
}

const CATEGORY: Record<string, string> = {
  broken: "b-fail",
  fixed: "b-pass",
  flaky: "b-inconclusive",
  stable_pass: "b-neutral",
  stable_fail: "b-neutral",
  changed: "b-neutral",
};

export function CategoryBadge({ category }: { category: string }) {
  return <span className={`badge ${CATEGORY[category] ?? "b-neutral"}`}>{category.replace("_", " ")}</span>;
}
