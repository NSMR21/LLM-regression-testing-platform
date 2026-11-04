import type { MetricResult } from "../api";
import { pts } from "./ui";

// Δ ± CI per metric, with the zero line and the tolerated-drop margin.
export function ForestPlot({ metrics, margin }: { metrics: MetricResult[]; margin: number }) {
  if (!metrics.length) return null;
  const values = metrics.flatMap((m) => [m.ci_low ?? m.delta, m.ci_high ?? m.delta, m.delta]);
  const extent = Math.max(0.05, margin * 1.5, ...values.map((v) => Math.abs(v))) * 1.1;
  // Drawn at roughly its on-screen size so labels stay at body-text size; shrinks on narrow screens.
  const width = 860;
  const label = 150;
  const plot = width - label - 90;
  const rowH = 28;
  const height = metrics.length * rowH + 28;
  const x = (v: number) => label + ((v + extent) / (2 * extent)) * plot;
  const colour = (m: MetricResult) =>
    m.status === "fail" || m.status === "worse" ? "var(--fail)" : m.status === "pass" || m.status === "better" ? "var(--pass)" : m.status === "inconclusive" ? "var(--warn)" : "var(--muted)";
  const ticks = [-extent, -extent / 2, 0, extent / 2, extent];
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" style={{ maxWidth: width, display: "block" }} role="img" aria-label="Change per metric with 95% confidence intervals">
      <rect x={x(-margin)} y={0} width={x(0) - x(-margin)} height={height - 22} fill="var(--warn-soft)" opacity={0.6} />
      <line x1={x(0)} x2={x(0)} y1={0} y2={height - 22} stroke="var(--border)" strokeWidth={1.5} />
      <line x1={x(-margin)} x2={x(-margin)} y1={0} y2={height - 22} stroke="var(--warn)" strokeDasharray="3 3" />
      {metrics.map((m, i) => {
        const y = i * rowH + rowH / 2;
        const lo = m.ci_low ?? m.delta;
        const hi = m.ci_high ?? m.delta;
        return (
          <g key={m.name}>
            <text x={label - 8} y={y + 4} textAnchor="end" fontSize={12} fill="var(--text)" fontWeight={m.gated ? 650 : 400}>
              {m.name}
            </text>
            <line x1={x(lo)} x2={x(hi)} y1={y} y2={y} stroke={colour(m)} strokeWidth={2.5} strokeLinecap="round" />
            <circle cx={x(m.delta)} cy={y} r={4.5} fill={colour(m)} />
            <text x={width - 4} y={y + 4} textAnchor="end" fontSize={12} fill="var(--muted)">
              {pts(m.delta)} pts
            </text>
          </g>
        );
      })}
      {ticks.map((t) => (
        <text key={t} x={x(t)} y={height - 6} textAnchor="middle" fontSize={11} fill="var(--muted)">
          {pts(t, 0)}
        </text>
      ))}
    </svg>
  );
}
