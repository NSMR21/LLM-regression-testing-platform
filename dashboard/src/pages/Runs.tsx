import { useQuery } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ACTIVE, api } from "../api";
import { ErrorBox, Loading, pct, Progress, shortId, StatusBadge, VerdictBadge, when } from "../components/ui";

export function RunsPage() {
  const [params, setParams] = useSearchParams();
  const suite = params.get("suite") ?? "";
  const navigate = useNavigate();
  const suites = useQuery({ queryKey: ["suites"], queryFn: api.suites });
  const runs = useQuery({
    queryKey: ["runs", suite],
    queryFn: () => api.runs({ suite: suite || undefined, limit: 100 }),
    refetchInterval: (query) => (query.state.data?.some((r) => ACTIVE.includes(r.status)) ? 3000 : 15000),
  });

  return (
    <>
      <div className="header">
        <div>
          <h1>Runs</h1>
          <div className="sub">Every run keeps its frozen configuration, outputs, scores and verdict.</div>
        </div>
        <div className="toolbar">
          <select value={suite} onChange={(e) => setParams(e.target.value ? { suite: e.target.value } : {})}>
            <option value="">All suites</option>
            {suites.data?.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="card">
        {runs.isLoading ? <Loading /> : runs.error ? <ErrorBox error={runs.error} /> : null}
        {runs.data && runs.data.length === 0 ? (
          <div className="empty">
            No runs yet. Start one with <code>rlens run --url … -s suite.yaml -p prompt.yaml -m model.yaml</code>.
          </div>
        ) : null}
        {runs.data && runs.data.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Verdict</th>
                  <th>Run</th>
                  <th>Suite</th>
                  <th>Prompt</th>
                  <th>Model</th>
                  <th className="num">Pass rate</th>
                  <th>Status</th>
                  <th>Branch / PR</th>
                  <th>Started</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.map((r) => (
                  <tr key={r.run_id} className="clickable" onClick={() => navigate(`/runs/${r.run_id}`)}>
                    <td><VerdictBadge verdict={r.verdict} /></td>
                    <td className="mono">
                      {shortId(r.run_id)}
                      {r.kind !== "run" ? <span className="badge b-neutral" style={{ marginLeft: 6 }}>{r.kind.replace("_", " ")}</span> : null}
                    </td>
                    <td>{r.suite}</td>
                    <td>{r.prompt}</td>
                    <td>{r.model}</td>
                    <td className="num">{pct(r.metrics.pass_rate)}</td>
                    <td style={{ minWidth: 120 }}>
                      <StatusBadge status={r.status} />
                      {ACTIVE.includes(r.status) ? (
                        <div style={{ marginTop: 4 }}>
                          <Progress done={r.progress.executed + r.progress.graded} total={2 * r.progress.total} />
                        </div>
                      ) : null}
                    </td>
                    <td className="small">
                      {String(r.ci_meta.branch ?? "")}
                      {r.ci_meta.pr ? ` #${r.ci_meta.pr}` : ""}
                    </td>
                    <td className="small muted">{when(r.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
    </>
  );
}
