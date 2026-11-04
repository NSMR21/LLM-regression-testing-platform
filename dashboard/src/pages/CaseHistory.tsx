import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { Dots, ErrorBox, Kpi, Loading, pct, VerdictBadge, when } from "../components/ui";

export function CaseHistoryPage() {
  const { suite = "", caseKey = "" } = useParams();
  const history = useQuery({ queryKey: ["case", suite, caseKey], queryFn: () => api.caseHistory(suite, caseKey) });
  if (history.isLoading) return <Loading />;
  if (history.error || !history.data) return <ErrorBox error={history.error} />;
  const points = [...history.data.points].reverse();
  const rates = history.data.points.map((p) => p.pass_rate);
  return (
    <>
      <div className="header">
        <div>
          <div className="sub">
            <Link to="/suites">Suites</Link> / <Link to={`/suites/${encodeURIComponent(suite)}`}>{suite}</Link>
          </div>
          <h1 className="mono">{caseKey}</h1>
          <div className="sub">How this case behaved across runs, newest first.</div>
        </div>
      </div>
      <div className="kpis" style={{ marginBottom: 16 }}>
        <Kpi label="Runs" value={history.data.points.length} />
        <Kpi label="Mean pass rate" value={pct(rates.length ? rates.reduce((a, b) => a + b, 0) / rates.length : null)} />
        <Kpi label="Flaky in" value={pct(history.data.flaky_fraction, 0)} hint="of runs (mixed results across repeats)" />
      </div>
      <div className="card">
        <table>
          <thead>
            <tr><th>Run</th><th>When</th><th>Prompt</th><th>Model</th><th className="num">Pass rate</th><th>Repeats</th><th>Verdict</th><th>Sample output</th></tr>
          </thead>
          <tbody>
            {points.map((p) => (
              <tr key={p.run_id}>
                <td className="mono"><Link to={`/runs/${p.run_id}`}>{p.run_id.slice(4, 12)}</Link></td>
                <td className="small muted">{when(p.created_at)}</td>
                <td>{p.prompt}</td>
                <td>{p.model}</td>
                <td className="num">{pct(p.pass_rate, 0)}</td>
                <td><Dots outcomes={p.outcomes.map((o) => o === 1)} /></td>
                <td><VerdictBadge verdict={p.verdict} /></td>
                <td className="small" style={{ maxWidth: 320 }}><code>{(p.sample_output ?? "").slice(0, 140)}</code></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
