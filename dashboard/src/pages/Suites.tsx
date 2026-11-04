import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";
import { ErrorBox, Loading, when } from "../components/ui";

export function SuitesPage() {
  const suites = useQuery({ queryKey: ["suites"], queryFn: api.suites });
  const baselines = useQuery({ queryKey: ["baselines"], queryFn: () => api.baselines() });
  return (
    <>
      <div className="header">
        <div>
          <h1>Suites</h1>
          <div className="sub">Trends, baselines and per-case history for each test suite.</div>
        </div>
      </div>
      <div className="card">
        {suites.isLoading ? <Loading /> : suites.error ? <ErrorBox error={suites.error} /> : null}
        {suites.data?.length === 0 ? <div className="empty">No suites yet: they appear after their first run.</div> : null}
        {suites.data?.length ? (
          <table>
            <thead>
              <tr>
                <th>Suite</th><th className="num">Cases</th><th className="num">Runs</th><th>Graders</th><th>Baselines</th><th>Last run</th>
              </tr>
            </thead>
            <tbody>
              {suites.data.map((s) => (
                <tr key={s.name}>
                  <td><Link to={`/suites/${encodeURIComponent(s.name)}`}>{s.name}</Link></td>
                  <td className="num">{s.cases ?? "–"}</td>
                  <td className="num">{s.runs}</td>
                  <td className="small">{s.graders.map((g) => `${g.id} (${g.type})`).join(", ")}</td>
                  <td className="small">
                    {baselines.data?.filter((b) => b.suite === s.name).map((b) => (
                      <span key={b.name} style={{ marginRight: 8 }}>
                        {b.name} → <Link className="mono" to={`/runs/${b.run_id}`}>{b.run_id.slice(4, 12)}</Link>
                      </span>
                    ))}
                  </td>
                  <td className="small muted">{when(s.last_run_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </div>
    </>
  );
}
