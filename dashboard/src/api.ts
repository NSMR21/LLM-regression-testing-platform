// Typed client for the RegressionLens API.

export type Verdict = "pass" | "fail" | "inconclusive" | "error";
export type RunStatus = "queued" | "executing" | "grading" | "analyzing" | "completed" | "errored" | "cancelled";

export interface RunBrief {
  run_id: string;
  kind: string;
  suite: string;
  status: RunStatus;
  verdict: Verdict | null;
  prompt: string | null;
  model: string | null;
  fingerprint: string;
  baseline_run_id: string | null;
  baseline_name: string | null;
  source_run_id: string | null;
  progress: { total: number; executed: number; graded: number };
  metrics: Record<string, number>;
  ci_meta: Record<string, string | boolean>;
  error: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface MetricResult {
  name: string;
  gated: boolean;
  baseline: number;
  candidate: number;
  delta: number;
  ci_low: number | null;
  ci_high: number | null;
  p_value: number;
  p_adjusted: number;
  n_cases: number;
  mde: number | null;
  status: string;
}

export interface GateResult {
  metric: string;
  value: number | null;
  min: number | null;
  max: number | null;
  passed: boolean;
}

export interface CaseDiff {
  case_key: string;
  baseline: number;
  candidate: number;
  delta: number;
  category: "broken" | "fixed" | "flaky" | "stable_pass" | "stable_fail" | "changed";
}

export interface VerdictReport {
  verdict: Verdict;
  reasons: string[];
  notes: string[];
  metrics: MetricResult[];
  gates: GateResult[];
  critical_failures: string[];
  case_diff: CaseDiff[];
  excluded_cases: Record<string, string>;
  config_diff: Record<string, { baseline: unknown; candidate: unknown }>;
  flaky_share_of_drop: number | null;
  baseline_run_id: string | null;
  policy: { non_inferiority_margin: number; alpha: number; primary_metric: string };
  graders?: Record<string, { trust: string; kappa?: number; n: number }>;
}

export interface RunSummary {
  n_cases: number;
  n_trials: number;
  repeats: number;
  metrics: Record<string, number>;
  metric_ci: Record<string, [number, number]>;
  per_tag: Record<string, Record<string, number>>;
  error_rate: number;
  grader_error_rate: number;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  flaky_cases: { case_key: string; pass_rate: number; score: number; repeats: number }[];
  flaky_rate: number;
  mde: number | null;
  notes: string[];
}

export interface SuiteSpec {
  name: string;
  cases: { key: string; input: Record<string, unknown>; expected: unknown; tags: string[]; critical: boolean }[];
  verdict: { non_inferiority_margin: number; primary_metric: string };
}

export interface RunDetail extends RunBrief {
  summary: RunSummary | null;
  verdict_report: VerdictReport | null;
  snapshot: Record<string, any>;
  environment: Record<string, unknown>;
  execution: Record<string, unknown>;
  run_seed: number;
  suite_spec: SuiteSpec;
  prompt_spec: { name: string; system: string; template: string };
  model_spec: Record<string, unknown>;
}

export interface Score {
  value: number;
  passed: boolean;
  rationale: string;
  error: boolean;
}

export interface Trial {
  trial_id: number;
  status: string;
  case_key: string;
  repeat_idx: number;
  seed: number | null;
  messages: { role: string; content: string }[];
  output: string | null;
  error: string | null;
  latency_ms: number | null;
  tokens_out: number | null;
  scores: Record<string, Score>;
  passed: boolean | null;
}

export interface TrendPoint {
  run_id: string;
  created_at: string;
  value: number;
  ci: [number, number] | null;
  verdict: Verdict | null;
  prompt: string;
  model: string;
  fingerprint: string;
  config_changed: boolean;
  branch: string | null;
}

export interface Trends {
  suite: string;
  metric: string;
  points: TrendPoint[];
  baseline_events: { name: string; run_id: string; created_at: string }[];
}

export interface CaseHistory {
  suite: string;
  case_key: string;
  flaky_fraction: number | null;
  points: {
    run_id: string;
    created_at: string;
    pass_rate: number;
    repeats: number;
    flaky: boolean;
    outcomes: number[];
    prompt: string;
    model: string;
    sample_output: string | null;
    verdict: Verdict | null;
  }[];
}

export interface SuiteInfo {
  name: string;
  runs: number;
  last_run_at: string;
  cases: number | null;
  graders: { id: string; type: string }[];
}

export interface Baseline {
  suite: string;
  name: string;
  run_id: string;
  updated_at: string;
}

export interface LabelItem {
  trial_id: number;
  run_id: string;
  case_key: string;
  messages: { role: string; content: string }[];
  output: string;
  expected: unknown;
  rubric: string | null;
  grader_version: string;
}

export interface Agreement {
  n: number;
  accuracy: number | null;
  kappa: number | null;
  kappa_ci: [number | null, number | null];
  accuracy_ci: [number | null, number | null];
  false_pass_rate: number | null;
  false_fail_rate: number | null;
  confusion: Record<string, number>;
  trust: "calibrated" | "weak" | "uncalibrated";
}

export interface Calibration {
  suite: string;
  grader_id: string;
  current_version: string | null;
  current: Agreement | null;
  versions: Record<string, Agreement>;
}

const TOKEN_KEY = "rlens.token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable: token lives for this page only */
  }
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

const q = (params: Record<string, string | number | undefined | null>) => {
  const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "");
  return entries.length ? "?" + new URLSearchParams(entries.map(([k, v]) => [k, String(v)])).toString() : "";
};

export const api = {
  me: () => request<{ project: string; scopes: string[] }>("GET", "/api/v1/me"),
  runs: (params: { suite?: string; status?: string; limit?: number } = {}) => request<RunBrief[]>("GET", `/api/v1/runs${q(params)}`),
  run: (id: string) => request<RunDetail>("GET", `/api/v1/runs/${id}`),
  trials: (id: string, caseKey?: string) => request<Trial[]>("GET", `/api/v1/runs/${id}/trials${q({ case_key: caseKey, limit: 2000 })}`),
  suites: () => request<SuiteInfo[]>("GET", "/api/v1/suites"),
  baselines: (suite?: string) => request<Baseline[]>("GET", `/api/v1/baselines${q({ suite })}`),
  trends: (suite: string, metric: string) => request<Trends>("GET", `/api/v1/suites/${encodeURIComponent(suite)}/trends${q({ metric })}`),
  caseHistory: (suite: string, key: string) =>
    request<CaseHistory>("GET", `/api/v1/suites/${encodeURIComponent(suite)}/cases/${encodeURIComponent(key)}/history`),
  promote: (runId: string, name: string) => request<Baseline>("PUT", `/api/v1/baselines/${encodeURIComponent(name)}`, { run_id: runId }),
  rerun: (runId: string) => request<RunBrief>("POST", `/api/v1/runs/${runId}/rerun`, {}),
  rescore: (runId: string) => request<RunBrief>("POST", `/api/v1/runs/${runId}/rescore`, {}),
  cancel: (runId: string) => request<RunBrief>("POST", `/api/v1/runs/${runId}/cancel`, {}),
  labelQueue: (suite: string, grader: string, labeler: string, n = 20) =>
    request<LabelItem[]>("GET", `/api/v1/suites/${encodeURIComponent(suite)}/graders/${encodeURIComponent(grader)}/label-queue${q({ labeler, n })}`),
  addLabel: (body: { trial_id: number; grader_id: string; label: number; labeler: string }) => request<unknown>("POST", "/api/v1/labels", body),
  calibration: (suite: string, grader: string) =>
    request<Calibration>("GET", `/api/v1/suites/${encodeURIComponent(suite)}/graders/${encodeURIComponent(grader)}/calibration`),
};

export const ACTIVE: RunStatus[] = ["queued", "executing", "grading", "analyzing"];
