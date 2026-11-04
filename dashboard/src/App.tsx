import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { api, ApiError, getToken, setToken } from "./api";
import { ErrorBox, Loading } from "./components/ui";
import { CaseHistoryPage } from "./pages/CaseHistory";
import { LabelingPage } from "./pages/Labeling";
import { RunDetailPage } from "./pages/RunDetail";
import { RunsPage } from "./pages/Runs";
import { SuitePage } from "./pages/Suite";
import { SuitesPage } from "./pages/Suites";

function Logo() {
  return (
    <svg width="20" height="20" viewBox="0 0 32 32" aria-hidden>
      <circle cx="14" cy="14" r="9" fill="none" stroke="var(--accent)" strokeWidth="3" />
      <path d="M21 21l7 7" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" />
      <path d="M9 15l3 3 6-7" fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Login({ error }: { error?: string }) {
  const [value, setValue] = useState("");
  const client = useQueryClient();
  return (
    <div style={{ maxWidth: 420, margin: "12vh auto", padding: 16 }}>
      <div className="card">
        <div className="brand" style={{ padding: 0, marginBottom: 12 }}>
          <Logo /> RegressionLens
        </div>
        <p className="sub">Paste an API token. Create one with <code>rlens token create</code>, or use the server's <code>RLENS_ADMIN_TOKEN</code>.</p>
        {error ? <ErrorBox error={error} /> : null}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            setToken(value.trim());
            client.invalidateQueries();
          }}
          style={{ display: "flex", gap: 8, marginTop: 12 }}
        >
          <input className="grow" type="password" placeholder="API token" value={value} onChange={(e) => setValue(e.target.value)} autoFocus />
          <button className="primary" type="submit" disabled={!value.trim()}>
            Sign in
          </button>
        </form>
      </div>
    </div>
  );
}

export function App() {
  const me = useQuery({ queryKey: ["me", getToken()], queryFn: api.me });
  const client = useQueryClient();
  if (me.isLoading) return <Loading />;
  if (me.error) {
    const auth = me.error instanceof ApiError && (me.error.status === 401 || me.error.status === 403);
    return auth ? <Login error={getToken() ? me.error.message : undefined} /> : <div className="main"><ErrorBox error={me.error} /></div>;
  }
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <Logo /> RegressionLens
        </div>
        <nav className="nav">
          <NavLink to="/" end>Runs</NavLink>
          <NavLink to="/suites">Suites</NavLink>
          <NavLink to="/label">Label graders</NavLink>
        </nav>
        <div className="foot">
          <div>project <b>{me.data?.project}</b></div>
          {getToken() ? (
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault();
                setToken(null);
                client.invalidateQueries();
              }}
            >
              sign out
            </a>
          ) : null}
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<RunsPage />} />
          <Route path="/runs/:runId" element={<RunDetailPage />} />
          <Route path="/suites" element={<SuitesPage />} />
          <Route path="/suites/:suite" element={<SuitePage />} />
          <Route path="/suites/:suite/cases/:caseKey" element={<CaseHistoryPage />} />
          <Route path="/label" element={<LabelingPage />} />
          <Route path="*" element={<div className="empty">Not found</div>} />
        </Routes>
      </main>
    </div>
  );
}
