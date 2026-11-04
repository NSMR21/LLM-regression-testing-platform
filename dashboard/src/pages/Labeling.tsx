import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, type Agreement } from "../api";
import { ErrorBox, Kpi, Loading, pct } from "../components/ui";

const LABELER_KEY = "rlens.labeler";

function readLabeler(): string {
  try {
    return localStorage.getItem(LABELER_KEY) ?? "";
  } catch {
    return "";
  }
}

function AgreementPanel({ report, version }: { report: Agreement | null; version: string | null }) {
  if (!report) return <div className="empty">No labels yet for this grader.</div>;
  const trustClass = report.trust === "calibrated" ? "b-pass" : report.trust === "weak" ? "b-inconclusive" : "b-neutral";
  const c = report.confusion;
  return (
    <>
      <div className="sub" style={{ marginBottom: 8 }}>
        Grader version <code>{version}</code> · <span className={`badge ${trustClass}`}>{report.trust}</span>
      </div>
      <div className="kpis">
        <Kpi label="Labels" value={report.n} hint="30+ needed to calibrate" />
        <Kpi label="Cohen's κ" value={report.kappa === null ? "–" : report.kappa.toFixed(2)} hint={report.kappa_ci[0] !== null ? `95% CI ${report.kappa_ci[0]?.toFixed(2)}–${report.kappa_ci[1]?.toFixed(2)}` : undefined} />
        <Kpi label="Agreement" value={pct(report.accuracy)} />
        <Kpi label="False pass" value={pct(report.false_pass_rate)} hint="grader passed, human failed" />
        <Kpi label="False fail" value={pct(report.false_fail_rate)} hint="grader failed, human passed" />
      </div>
      <table style={{ marginTop: 12, maxWidth: 360 }}>
        <thead><tr><th></th><th className="num">human pass</th><th className="num">human fail</th></tr></thead>
        <tbody>
          <tr><td>grader pass</td><td className="num">{c.grader_pass_human_pass ?? 0}</td><td className="num">{c.grader_pass_human_fail ?? 0}</td></tr>
          <tr><td>grader fail</td><td className="num">{c.grader_fail_human_pass ?? 0}</td><td className="num">{c.grader_fail_human_fail ?? 0}</td></tr>
        </tbody>
      </table>
    </>
  );
}

export function LabelingPage() {
  const [params, setParams] = useSearchParams();
  const suite = params.get("suite") ?? "";
  const grader = params.get("grader") ?? "";
  const [labeler, setLabeler] = useState(readLabeler);
  const [index, setIndex] = useState(0);
  const client = useQueryClient();
  const suites = useQuery({ queryKey: ["suites"], queryFn: api.suites });
  const judgeGraders = suites.data?.flatMap((s) => s.graders.filter((g) => g.type === "llm_judge").map((g) => ({ suite: s.name, grader: g.id }))) ?? [];
  const ready = !!suite && !!grader && !!labeler;
  const queue = useQuery({ queryKey: ["label-queue", suite, grader, labeler], queryFn: () => api.labelQueue(suite, grader, labeler, 20), enabled: ready });
  const calibration = useQuery({ queryKey: ["calibration", suite, grader], queryFn: () => api.calibration(suite, grader), enabled: !!suite && !!grader });
  const item = queue.data?.[index];
  const label = useMutation({
    mutationFn: (value: number) => api.addLabel({ trial_id: item!.trial_id, grader_id: grader, label: value, labeler }),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["calibration", suite, grader] });
      if (queue.data && index + 1 >= queue.data.length) {
        setIndex(0);
        client.invalidateQueries({ queryKey: ["label-queue", suite, grader, labeler] });
      } else setIndex(index + 1);
    },
  });

  useEffect(() => {
    try {
      localStorage.setItem(LABELER_KEY, labeler);
    } catch {
      /* ignore */
    }
  }, [labeler]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!item || label.isPending || (e.target as HTMLElement)?.tagName === "INPUT") return;
      if (e.key === "p") label.mutate(1);
      if (e.key === "f") label.mutate(0);
      if (e.key === "s") setIndex((i) => i + 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [item, label]);

  return (
    <>
      <div className="header">
        <div>
          <h1>Label graders</h1>
          <div className="sub">Judge outputs blind. Your labels measure how far each model-based grader can be trusted.</div>
        </div>
        <div className="toolbar">
          <select
            value={suite && grader ? `${suite}::${grader}` : ""}
            onChange={(e) => {
              const [s, g] = e.target.value.split("::");
              setIndex(0);
              setParams(s ? { suite: s, grader: g } : {});
            }}
          >
            <option value="">Choose a model-based grader…</option>
            {judgeGraders.map((j) => (
              <option key={`${j.suite}::${j.grader}`} value={`${j.suite}::${j.grader}`}>{j.suite} / {j.grader}</option>
            ))}
          </select>
          <input placeholder="your name" value={labeler} onChange={(e) => setLabeler(e.target.value)} style={{ width: 140 }} />
        </div>
      </div>

      {suites.data && judgeGraders.length === 0 ? <div className="card empty">No suite uses an <code>llm_judge</code> grader yet.</div> : null}

      {suite && grader ? (
        <div className="card">
          <h2>Agreement with human labels</h2>
          {calibration.isLoading ? <Loading /> : calibration.error ? <ErrorBox error={calibration.error} /> : <AgreementPanel report={calibration.data?.current ?? null} version={calibration.data?.current_version ?? null} />}
        </div>
      ) : null}

      {suite && grader && !labeler ? <div className="card empty">Enter your name to start labeling.</div> : null}
      {ready ? (
        <div className="card">
          {queue.isLoading ? <Loading /> : queue.error ? <ErrorBox error={queue.error} /> : null}
          {queue.data && !item ? <div className="empty">Nothing left to label. Run the suite again for more outputs.</div> : null}
          {item ? (
            <>
              <div className="header" style={{ marginBottom: 8 }}>
                <h2 style={{ margin: 0 }}>
                  {index + 1} / {queue.data?.length} · <span className="mono">{item.case_key}</span>
                </h2>
                <div className="toolbar">
                  <button className="pass" onClick={() => label.mutate(1)} disabled={label.isPending}>Pass (p)</button>
                  <button className="fail" onClick={() => label.mutate(0)} disabled={label.isPending}>Fail (f)</button>
                  <button onClick={() => setIndex(index + 1)}>Skip (s)</button>
                </div>
              </div>
              {label.error ? <ErrorBox error={label.error} /> : null}
              {item.rubric ? (<><h3>Rubric</h3><pre className="output">{item.rubric}</pre></>) : null}
              <div className="split" style={{ marginTop: 12 }}>
                <div>
                  <h3>Task</h3>
                  {item.messages.map((m, i) => (
                    <pre key={i} className="output"><b>{m.role}</b>{"\n"}{m.content}</pre>
                  ))}
                  {item.expected !== null && item.expected !== undefined ? (<><h3 style={{ marginTop: 12 }}>Reference</h3><pre className="output">{typeof item.expected === "string" ? item.expected : JSON.stringify(item.expected, null, 2)}</pre></>) : null}
                </div>
                <div>
                  <h3>Model output</h3>
                  <pre className="output" style={{ maxHeight: 520 }}>{item.output}</pre>
                </div>
              </div>
            </>
          ) : null}
        </div>
      ) : null}
    </>
  );
}
