import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EngineeringHub")


# ============================================================
# CONFIG
# ============================================================

def load_dotenv_file(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class AppConfig:
    jira_base_url: str
    jira_email: str
    jira_api_token: str
    gitlab_base_url: str
    gitlab_private_token: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str

    @classmethod
    def load_from_env_or_secrets(cls) -> "AppConfig":
        load_dotenv_file()

        def fetch(section: str, key: str, fallback: str = "") -> str:
            try:
                return st.secrets[section][key]
            except Exception:
                return os.getenv(f"{section.upper()}_{key.upper()}", fallback)

        return cls(
            jira_base_url=os.getenv("JIRA_BASE_URL", fetch("Jira", "BASE_URL")),
            jira_email=os.getenv("JIRA_EMAIL", fetch("Jira", "EMAIL")),
            jira_api_token=os.getenv("JIRA_API_TOKEN", fetch("Jira", "API_TOKEN")),
            gitlab_base_url=os.getenv("GITLAB_BASE_URL", fetch("GitLab", "BASE_URL")),
            gitlab_private_token=os.getenv("GITLAB_PRIVATE_TOKEN", fetch("GitLab", "PRIVATE_TOKEN")),
            llm_base_url=os.getenv("LLM_BASE_URL", fetch("LLM", "BASE_URL", "https://api.openai.com/v1")),
            llm_api_key=os.getenv("LLM_API_KEY", fetch("LLM", "API_KEY")),
            llm_model=os.getenv("LLM_MODEL", fetch("LLM", "MODEL", "gpt-4o")),
        )


@dataclass
class JiraIssueDTO:
    key: str
    summary: str
    status: str
    status_category: str
    assignee: str
    reporter: str
    priority: str
    issue_type: str
    project: str
    sprint: str
    labels: str
    created: str
    updated: str
    due_date: str
    description: str
    time_spent_hours: float
    story_points: float
    url: str


@dataclass
class GitLabMRDTO:
    mr_id: int
    title: str
    state: str
    author: str
    assignee: Optional[str]
    created_at: str
    updated_at: str
    merged_at: Optional[str]
    source_branch: str
    target_branch: str
    description: str


@dataclass
class GitLabCommitDTO:
    commit_id: str
    title: str
    author: str
    created_at: str
    message: str


# ============================================================
# SERVICES
# ============================================================

class RobustHTTPClient:
    @staticmethod
    def create_session(retries: int = 3, backoff_factor: float = 0.5):
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session


class JiraService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.session = RobustHTTPClient.create_session()
        self.auth = (config.jira_email, config.jira_api_token)
        self.headers = {"Accept": "application/json", "Content-Type": "application/json"}

    def fetch_issues(self, project_key: str, jql_filter: str = "") -> List[JiraIssueDTO]:
        url = f"{self.config.jira_base_url.rstrip('/')}/rest/api/3/search"
        query = f"project = '{project_key}'"
        if jql_filter:
            query += f" AND ({jql_filter})"

        fields = ",".join([
            "summary", "status", "assignee", "reporter", "priority", "issuetype",
            "project", "created", "updated", "duedate", "description", "timespent",
            "labels", "customfield_10020", "customfield_10016"
        ])

        response = self.session.get(
            url, headers=self.headers, auth=self.auth,
            params={"jql": query, "maxResults": 100, "fields": fields}, timeout=30
        )
        response.raise_for_status()

        result = []
        for raw in response.json().get("issues", []):
            f = raw.get("fields", {})
            desc = f.get("description") or ""
            if isinstance(desc, dict):
                desc = "Rich Text / ADF description"
            elif isinstance(desc, str) and len(desc) > 500:
                desc = desc[:500] + "... [Truncated]"

            status_obj = f.get("status") or {}
            assignee = f.get("assignee") or {}
            reporter = f.get("reporter") or {}
            priority = f.get("priority") or {}
            issue_type = f.get("issuetype") or {}
            project = f.get("project") or {}

            sprint = f.get("customfield_10020")
            if isinstance(sprint, list):
                names = []
                for x in sprint:
                    names.append(x.get("name", "") if isinstance(x, dict) else str(x))
                sprint = ", ".join(x for x in names if x) or "No Sprint"
            elif not sprint:
                sprint = "No Sprint"
            else:
                sprint = str(sprint)

            try:
                points = float(f.get("customfield_10016") or 0)
            except (TypeError, ValueError):
                points = 0.0

            result.append(JiraIssueDTO(
                key=raw.get("key", ""),
                summary=f.get("summary", ""),
                status=status_obj.get("name", "Unknown"),
                status_category=(status_obj.get("statusCategory") or {}).get("name", "Unknown"),
                assignee=assignee.get("displayName", "Unassigned"),
                reporter=reporter.get("displayName", "Unknown"),
                priority=priority.get("name", "None"),
                issue_type=issue_type.get("name", "Unknown"),
                project=project.get("key", project_key),
                sprint=sprint,
                labels=", ".join(f.get("labels") or []),
                created=(f.get("created") or "")[:10],
                updated=(f.get("updated") or "")[:10],
                due_date=f.get("duedate") or "",
                description=desc,
                time_spent_hours=round((f.get("timespent") or 0) / 3600, 2),
                story_points=points,
                url=f"{self.config.jira_base_url.rstrip('/')}/browse/{raw.get('key', '')}",
            ))
        return result


class GitLabService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.session = RobustHTTPClient.create_session()
        self.headers = {"PRIVATE-TOKEN": config.gitlab_private_token}

    def fetch_projects(self) -> List[Dict[str, Any]]:
        url = f"{self.config.gitlab_base_url.rstrip('/')}/api/v4/projects"
        r = self.session.get(
            url, headers=self.headers,
            params={"membership": "true", "per_page": 50, "simple": "true", "order_by": "last_activity_at"},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()

    def fetch_deep_telemetry(self, project_id: int) -> Dict[str, List[Any]]:
        base = self.config.gitlab_base_url.rstrip("/")

        def fetch_mrs():
            r = self.session.get(
                f"{base}/api/v4/projects/{project_id}/merge_requests",
                params={"per_page": 50, "state": "all", "order_by": "updated_at"},
                headers=self.headers, timeout=20,
            )
            if r.status_code != 200:
                return []
            return [
                GitLabMRDTO(
                    mr_id=x.get("iid"), title=x.get("title", ""), state=x.get("state", ""),
                    author=(x.get("author") or {}).get("username", "unknown"),
                    assignee=(x.get("assignee") or {}).get("username"),
                    created_at=(x.get("created_at") or "")[:10],
                    updated_at=(x.get("updated_at") or "")[:10],
                    merged_at=(x.get("merged_at") or "")[:10] if x.get("merged_at") else None,
                    source_branch=x.get("source_branch", ""), target_branch=x.get("target_branch", ""),
                    description=(x.get("description") or "")[:300],
                ) for x in r.json()
            ]

        def fetch_commits():
            r = self.session.get(
                f"{base}/api/v4/projects/{project_id}/repository/commits",
                params={"per_page": 50}, headers=self.headers, timeout=20,
            )
            if r.status_code != 200:
                return []
            return [
                GitLabCommitDTO(
                    commit_id=x.get("short_id", ""), title=x.get("title", ""),
                    author=x.get("author_name", ""), created_at=(x.get("created_at") or "")[:10],
                    message=(x.get("message") or "").strip(),
                ) for x in r.json()
            ]

        with ThreadPoolExecutor(max_workers=2) as ex:
            return {"merge_requests": ex.submit(fetch_mrs).result(),
                    "commits": ex.submit(fetch_commits).result()}


class LLMAssistantService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.session = RobustHTTPClient.create_session()

    def generate_analysis(self, prompt: str, context_payload: str) -> str:
        if not self.config.llm_api_key:
            raise ValueError("LLM_API_KEY is not configured.")
        r = self.session.post(
            f"{self.config.llm_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.config.llm_api_key}", "Content-Type": "application/json"},
            json={
                "model": self.config.llm_model,
                "messages": [
                    {"role": "system", "content":
                     "You are an Engineering Manager's technical program assistant. "
                     "Analyze Jira and GitLab telemetry. Focus on observable delivery signals, "
                     "workload, aging, blockers, review queues, and trends. Do not infer personal "
                     "traits or judge individual employee performance. Clearly separate facts from "
                     "possible explanations. Return concise markdown with actionable follow-ups."},
                    {"role": "user", "content": f"### TELEMETRY\n{context_payload}\n\n### TASK\n{prompt}"},
                ],
                "temperature": 0.2,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ============================================================
# ANALYTICS
# ============================================================

def prepare_jira_dataframe(issues: List[Any]) -> pd.DataFrame:
    if not issues:
        return pd.DataFrame()
    rows = [asdict(x) if hasattr(x, "__dataclass_fields__") else x for x in issues]
    df = pd.DataFrame(rows)
    for col in ["created", "updated", "due_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    today = pd.Timestamp.today().normalize()
    df["age_days"] = (today - df["created"]).dt.days.fillna(0).clip(lower=0)
    df["days_since_update"] = (today - df["updated"]).dt.days.fillna(0).clip(lower=0)
    status = df["status"].fillna("").str.lower()

    df["overdue"] = (
        df["due_date"].notna() & (df["due_date"] < today)
        & ~status.isin(["done", "closed", "resolved"])
    )
    df["is_done"] = status.isin(["done", "closed", "resolved", "complete", "completed"])
    df["is_blocked"] = (
        status.str.contains("block", na=False)
        | df["labels"].fillna("").str.lower().str.contains("blocked", na=False)
    )
    df["is_stale"] = (df["days_since_update"] >= 5) & ~df["is_done"]
    return df


def filtered_jira_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    filters = {
        "project": st.session_state.get("filter_project", "All"),
        "assignee": st.session_state.get("filter_assignee", "All"),
        "status": st.session_state.get("filter_status", "All"),
        "priority": st.session_state.get("filter_priority", "All"),
    }
    for col, value in filters.items():
        if value != "All":
            result = result[result[col] == value]
    return result


# ============================================================
# GLOSSY GREEN DESIGN SYSTEM
# ============================================================

THEMES = {
    "Dark": {
        "bg": "#06100B", "bg2": "#0A1710", "surface": "rgba(12, 29, 19, .78)",
        "surface2": "#0D2115", "border": "rgba(74, 222, 128, .16)",
        "text": "#F0FDF4", "muted": "#86A893", "accent": "#4ADE80",
        "accent2": "#22C55E", "soft": "#BBF7D0", "shadow": "rgba(0,0,0,.38)",
        "grid": "rgba(134,168,147,.12)", "danger": "#FB7185", "warning": "#FBBF24",
    },
    "Light": {
        "bg": "#F3FAF5", "bg2": "#FFFFFF", "surface": "rgba(255,255,255,.82)",
        "surface2": "#FFFFFF", "border": "rgba(22, 101, 52, .13)",
        "text": "#102117", "muted": "#5E7465", "accent": "#16A34A",
        "accent2": "#15803D", "soft": "#166534", "shadow": "rgba(25,80,43,.10)",
        "grid": "rgba(30,75,45,.10)", "danger": "#E11D48", "warning": "#D97706",
    },
}


def inject_custom_styles(theme_name: str):
    t = THEMES[theme_name]
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

    :root {{
        --bg: {t["bg"]}; --bg2: {t["bg2"]}; --surface: {t["surface"]};
        --surface2: {t["surface2"]}; --border: {t["border"]};
        --text: {t["text"]}; --muted: {t["muted"]}; --accent: {t["accent"]};
        --accent2: {t["accent2"]}; --soft: {t["soft"]}; --shadow: {t["shadow"]};
        --danger: {t["danger"]}; --warning: {t["warning"]};
    }}

    html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
    .stApp {{
        background:
          radial-gradient(circle at 4% 0%, rgba(34,197,94,.15), transparent 27%),
          radial-gradient(circle at 94% 8%, rgba(74,222,128,.09), transparent 24%),
          linear-gradient(135deg, var(--bg), var(--bg2));
        color: var(--text);
    }}
    .main .block-container {{ max-width: 1540px; padding: 1.4rem 2rem 3rem; }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--surface2), var(--bg)) !important;
        border-right: 1px solid var(--border) !important;
        backdrop-filter: blur(18px);
    }}
    section[data-testid="stSidebar"] .block-container {{ padding: 1.1rem .9rem; }}

    .brand {{
        display:flex; align-items:center; gap:.75rem; padding:.45rem .45rem 1.25rem;
    }}
    .brand-mark {{
        width:42px;height:42px;border-radius:13px;display:flex;align-items:center;justify-content:center;
        color:#052e16;font-weight:900;font-size:1.2rem;
        background:linear-gradient(145deg,#86efac,#22c55e);
        box-shadow:0 8px 25px rgba(34,197,94,.28);
    }}
    .brand-title {{ font-family:'Manrope'; font-size:1.05rem; font-weight:800; color:var(--text); }}
    .brand-sub {{ color:var(--accent); font-size:.62rem; font-weight:800; letter-spacing:.13em; }}

    .hero {{
        position:relative; overflow:hidden; margin-bottom:1.2rem; padding:1.65rem 1.7rem;
        border:1px solid var(--border); border-radius:24px;
        background:
          radial-gradient(circle at 90% 15%, rgba(74,222,128,.16), transparent 28%),
          linear-gradient(135deg, var(--surface), rgba(34,197,94,.04));
        box-shadow:0 20px 60px var(--shadow);
        backdrop-filter:blur(20px);
    }}
    .hero:after {{
        content:""; position:absolute; width:170px;height:170px;right:-75px;bottom:-90px;
        border-radius:50%; background:rgba(74,222,128,.08); filter:blur(2px);
    }}
    .hero-kicker {{ color:var(--accent); font-size:.68rem; font-weight:800; letter-spacing:.16em; }}
    .hero h1 {{ margin:.25rem 0 .35rem; color:var(--text); font-family:'Manrope'; font-size:2.05rem; letter-spacing:-.045em; }}
    .hero p {{ margin:0; color:var(--muted); font-size:.9rem; }}

    .metric-card {{
        min-height:122px; padding:1.05rem 1.15rem; border-radius:18px;
        border:1px solid var(--border);
        background:linear-gradient(145deg, var(--surface), rgba(34,197,94,.035));
        box-shadow:0 12px 34px var(--shadow); backdrop-filter:blur(14px);
        position:relative; overflow:hidden;
    }}
    .metric-card:before {{
        content:""; position:absolute; top:0; left:0; right:0; height:2px;
        background:linear-gradient(90deg,transparent,var(--accent),transparent); opacity:.75;
    }}
    .metric-title {{ color:var(--muted); text-transform:uppercase; font-size:.64rem; font-weight:800; letter-spacing:.1em; }}
    .metric-value {{ color:var(--accent) !important; font-family:'Manrope'; font-size:1.9rem; font-weight:800; line-height:1.15; margin-top:.35rem; }}
    .metric-subtitle {{ color:var(--muted); font-size:.7rem; margin-top:.3rem; }}

    .panel {{
        padding:1.1rem 1.15rem; margin-bottom:1rem; border-radius:19px;
        border:1px solid var(--border); background:var(--surface);
        box-shadow:0 12px 38px var(--shadow); backdrop-filter:blur(16px);
    }}
    .panel-title {{ color:var(--text); font-family:'Manrope'; font-size:.98rem; font-weight:800; }}
    .panel-subtitle {{ color:var(--muted); font-size:.71rem; margin:.15rem 0 .75rem; }}
    .section-title {{ color:var(--text); font-family:'Manrope'; font-size:1.05rem; font-weight:800; margin:1.25rem 0 .7rem; }}

    .alert-row {{
        border:1px solid var(--border); border-left:3px solid var(--accent);
        border-radius:13px; padding:.7rem .8rem; margin:.45rem 0;
        background:rgba(34,197,94,.035);
    }}
    .alert-title {{ color:var(--text); font-size:.8rem; font-weight:700; }}
    .alert-meta {{ color:var(--muted); font-size:.68rem; margin-top:.15rem; }}

    .status-pill {{ display:inline-block; padding:.22rem .5rem; border-radius:999px; font-size:.62rem; font-weight:800; }}
    .pill-success {{ color:#166534; background:#DCFCE7; }}
    .pill-warning {{ color:#92400E; background:#FEF3C7; }}
    .pill-danger {{ color:#9F1239; background:#FFE4E6; }}
    .pill-info {{ color:#166534; background:#DCFCE7; }}
    .pill-neutral {{ color:var(--muted); background:rgba(134,168,147,.12); }}

    .insight {{
        border:1px solid var(--border); border-radius:15px; padding:.9rem 1rem;
        background:linear-gradient(120deg,rgba(34,197,94,.07),transparent);
        margin-bottom:.6rem;
    }}
    .insight b {{ color:var(--text); }} .insight span {{ color:var(--muted); font-size:.73rem; }}

    .stButton > button {{
        border:1px solid var(--border); border-radius:11px; font-weight:800;
        background:linear-gradient(135deg,rgba(34,197,94,.10),rgba(34,197,94,.03));
        color:var(--text); transition:.2s ease;
    }}
    .stButton > button:hover {{
        border-color:var(--accent); color:var(--accent);
        box-shadow:0 8px 25px rgba(34,197,94,.12); transform:translateY(-1px);
    }}
    div[data-testid="stDataFrame"] {{
        border:1px solid var(--border); border-radius:14px; overflow:hidden;
    }}
    div[data-baseweb="select"] > div {{
        border-radius:11px !important; border-color:var(--border) !important;
        background:var(--surface2) !important;
    }}
    .stTextInput input, .stTextArea textarea {{
        border-radius:11px !important; border-color:var(--border) !important;
        background:var(--surface2) !important; color:var(--text) !important;
    }}
    hr {{ border-color:var(--border); }}
    .small-muted {{ color:var(--muted); font-size:.7rem; }}
    .health-dot {{
        display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--accent);
        box-shadow:0 0 0 5px rgba(34,197,94,.10);margin-right:7px;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_hero(title, subtitle, kicker="ENGINEERING INTELLIGENCE"):
    st.markdown(f"""
    <div class="hero">
      <div class="hero-kicker">{kicker}</div>
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_kpi(title, value, subtitle="", tone="green"):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-title">{title}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_alert(title, meta, kind="info"):
    accent = {"danger": "var(--danger)", "warning": "var(--warning)", "info": "var(--accent)"}.get(kind, "var(--accent)")
    st.markdown(f"""
    <div class="alert-row" style="border-left-color:{accent}">
      <div class="alert-title">{title}</div>
      <div class="alert-meta">{meta}</div>
    </div>
    """, unsafe_allow_html=True)


def status_kind(status):
    s = str(status).lower()
    if any(x in s for x in ["done", "closed", "resolved", "complete"]): return "success"
    if any(x in s for x in ["block", "cancel"]): return "danger"
    if any(x in s for x in ["progress", "review", "testing"]): return "warning"
    return "info"


def render_filters(df):
    if df.empty:
        return
    st.markdown('<div class="section-title">Live filters</div>', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    with a:
        opts=["All"]+sorted(df["project"].dropna().astype(str).unique())
        st.session_state["filter_project"]=st.selectbox("Project", opts, key="filter_project_widget")
    with b:
        opts=["All"]+sorted(df["assignee"].dropna().astype(str).unique())
        st.session_state["filter_assignee"]=st.selectbox("Assignee", opts, key="filter_assignee_widget")
    with c:
        opts=["All"]+sorted(df["status"].dropna().astype(str).unique())
        st.session_state["filter_status"]=st.selectbox("Status", opts, key="filter_status_widget")
    with d:
        opts=["All"]+sorted(df["priority"].dropna().astype(str).unique())
        st.session_state["filter_priority"]=st.selectbox("Priority", opts, key="filter_priority_widget")


# ============================================================
# PAGES
# ============================================================

def overview_page(df, gitlab):
    render_hero("Engineering Overview",
                "A polished command center for delivery, workload, risk and engineering activity.")

    if df.empty:
        render_empty_state("No Jira telemetry loaded", "Open Data Sources in the sidebar and load a Jira project.")
        return

    d=filtered_jira_df(df)
    total=len(d); done=int(d["is_done"].sum())
    progress=int(d["status"].fillna("").str.lower().str.contains("progress|review|testing",regex=True).sum())
    blocked=int(d["is_blocked"].sum()); overdue=int(d["overdue"].sum()); stale=int(d["is_stale"].sum())
    pct=round(done*100/total) if total else 0

    k1,k2,k3,k4,k5=st.columns(5)
    with k1: render_kpi("Total work",total,"Filtered Jira issues")
    with k2: render_kpi("Completed",done,f"{pct}% of visible work")
    with k3: render_kpi("In progress",progress,"Active execution")
    with k4: render_kpi("Blocked",blocked,"Needs attention")
    with k5: render_kpi("Overdue",overdue,f"{stale} stale")

    st.markdown('<div class="section-title">Delivery intelligence</div>', unsafe_allow_html=True)
    a,b=st.columns([1.25,.9])
    with a:
        st.markdown('<div class="panel"><div class="panel-title">Work by status</div><div class="panel-subtitle">Current distribution of visible Jira work</div>',unsafe_allow_html=True)
        st.bar_chart(d["status"].value_counts(),height=285,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="panel-title">Attention radar</div><div class="panel-subtitle">Observable signals from Jira fields</div>',unsafe_allow_html=True)
        if overdue: render_alert(f"{overdue} overdue issue(s)","Due date passed while work is not completed.","danger")
        if blocked: render_alert(f"{blocked} blocked issue(s)","Status or labels indicate blocked work.","danger")
        if stale: render_alert(f"{stale} stale issue(s)","No Jira update for five or more days.","warning")
        unassigned=int((d["assignee"]=="Unassigned").sum())
        if unassigned: render_alert(f"{unassigned} unassigned issue(s)","No current assignee is recorded.","warning")
        if not any([overdue,blocked,stale,unassigned]): render_alert("No major attention signals","Based on currently loaded Jira fields.","info")
        st.markdown('</div>',unsafe_allow_html=True)

    workload=d.groupby("assignee").agg(Open=("key","count"),Completed=("is_done","sum"),
        Blocked=("is_blocked","sum"),Overdue=("overdue","sum"),Logged_Hours=("time_spent_hours","sum")).sort_values("Open",ascending=False)
    st.markdown('<div class="section-title">Team workload</div>',unsafe_allow_html=True)
    x,y=st.columns([1.15,.85])
    with x:
        display=workload.reset_index(); display["Logged_Hours"]=display["Logged_Hours"].round(1)
        st.dataframe(display,use_container_width=True,hide_index=True)
    with y:
        st.markdown('<div class="panel"><div class="panel-title">Open work by assignee</div><div class="panel-subtitle">Relative visible workload</div>',unsafe_allow_html=True)
        st.bar_chart(workload["Open"].sort_values(),height=300,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)


def delivery_page(df):
    render_hero("Delivery Dashboard","Backlog composition, aging, throughput and delivery pressure.","DELIVERY INTELLIGENCE")
    if df.empty:
        render_empty_state("No delivery data","Load Jira telemetry first.")
        return
    d=filtered_jira_df(df)
    backlog=int((~d["is_done"]).sum()); completed=int(d["is_done"].sum())
    avg_age=float(d["age_days"].mean()) if len(d) else 0
    hours=float(d["time_spent_hours"].sum())
    a,b,c,e=st.columns(4)
    with a: render_kpi("Backlog",backlog,"Not completed")
    with b: render_kpi("Completed",completed,"Completed issues")
    with c: render_kpi("Average age",f"{avg_age:.1f}d","Across visible issues")
    with e: render_kpi("Logged time",f"{hours:.1f}h","Recorded work")

    a,b=st.columns(2)
    with a:
        st.markdown('<div class="panel"><div class="panel-title">Priority mix</div><div class="panel-subtitle">Current work by Jira priority</div>',unsafe_allow_html=True)
        st.bar_chart(d["priority"].value_counts(),height=260,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="panel-title">Issue age distribution</div><div class="panel-subtitle">How long visible issues have existed</div>',unsafe_allow_html=True)
        age_bins=pd.cut(d["age_days"],bins=[-1,7,14,30,60,10**9],labels=["0–7d","8–14d","15–30d","31–60d","60d+"])
        st.bar_chart(age_bins.value_counts().sort_index(),height=260,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="section-title">Delivery table</div>',unsafe_allow_html=True)
    cols=["key","summary","status","priority","assignee","age_days","days_since_update","overdue","story_points"]
    show=d[[c for c in cols if c in d.columns]].copy()
    st.dataframe(show,use_container_width=True,hide_index=True)


def data_sources_page():
    render_hero("Data Sources","Connect Jira and GitLab telemetry without changing your existing service layer.","CONNECT & OBSERVE")
    cfg=st.session_state.config
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="panel"><div class="panel-title"><span class="health-dot"></span>Jira</div><div class="panel-subtitle">REST API · issue telemetry</div>',unsafe_allow_html=True)
        project=st.text_input("Project key",value=st.session_state.get("jira_project",""),placeholder="e.g. ENG")
        jql=st.text_input("Optional JQL filter",value="",placeholder="status != Done")
        if st.button("Load Jira telemetry",use_container_width=True):
            if not (cfg.jira_base_url and cfg.jira_email and cfg.jira_api_token and project):
                st.error("Provide Jira credentials and a project key in your environment/secrets.")
            else:
                with st.spinner("Fetching Jira telemetry…"):
                    issues=JiraService(cfg).fetch_issues(project,jql)
                    st.session_state.jira_df=prepare_jira_dataframe(issues)
                    st.session_state.jira_project=project
                st.success(f"Loaded {len(issues)} Jira issues.")
        st.markdown('</div>',unsafe_allow_html=True)

    with b:
        st.markdown('<div class="panel"><div class="panel-title"><span class="health-dot"></span>GitLab</div><div class="panel-subtitle">Projects · merge requests · commits</div>',unsafe_allow_html=True)
        if st.button("Discover GitLab projects",use_container_width=True):
            if not (cfg.gitlab_base_url and cfg.gitlab_private_token):
                st.error("Provide GitLab credentials in your environment/secrets.")
            else:
                with st.spinner("Fetching projects…"):
                    try:
                        st.session_state.gitlab_projects=GitLabService(cfg).fetch_projects()
                        st.success(f"Found {len(st.session_state.gitlab_projects)} projects.")
                    except Exception as ex:
                        st.error(str(ex))
        projects=st.session_state.get("gitlab_projects",[])
        if projects:
            labels={f'{x.get("name","")} · {x.get("path_with_namespace","")}':x.get("id") for x in projects}
            selected=st.selectbox("Project",list(labels))
            if st.button("Load GitLab telemetry",use_container_width=True):
                with st.spinner("Fetching merge requests and commits…"):
                    st.session_state.gitlab=GitLabService(cfg).fetch_deep_telemetry(labels[selected])
                st.success("GitLab telemetry loaded.")
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="section-title">Connection status</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: render_kpi("Jira", "Connected" if st.session_state.get("jira_df") is not None and not st.session_state.jira_df.empty else "Not loaded","Telemetry state")
    with c2: render_kpi("GitLab", "Connected" if st.session_state.get("gitlab") else "Not loaded","Telemetry state")
    with c3: render_kpi("LLM", "Configured" if cfg.llm_api_key else "Not configured","Analysis assistant")


def gitlab_page(gitlab):
    render_hero("GitLab Activity","Code review flow and repository activity from the selected project.","SOURCE CONTROL")
    if not gitlab:
        render_empty_state("No GitLab telemetry","Open Data Sources and load a project.")
        return
    mrs=gitlab.get("merge_requests",[]); commits=gitlab.get("commits",[])
    a,b,c=st.columns(3)
    with a: render_kpi("Merge requests",len(mrs),"Loaded")
    with b: render_kpi("Commits",len(commits),"Loaded")
    with c: render_kpi("Merged",sum(x.state=="merged" for x in mrs),"Merge requests")

    x,y=st.columns(2)
    with x:
        st.markdown('<div class="panel"><div class="panel-title">Merge request states</div><div class="panel-subtitle">Review pipeline snapshot</div>',unsafe_allow_html=True)
        st.bar_chart(pd.Series([x.state for x in mrs]).value_counts(),height=250,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with y:
        st.markdown('<div class="panel"><div class="panel-title">Recent commits</div><div class="panel-subtitle">Latest repository activity</div>',unsafe_allow_html=True)
        cdf=pd.DataFrame([asdict(x) for x in commits]).head(12)
        if not cdf.empty: st.dataframe(cdf[["commit_id","title","author","created_at"]],use_container_width=True,hide_index=True)
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="section-title">Merge requests</div>',unsafe_allow_html=True)
    mdf=pd.DataFrame([asdict(x) for x in mrs])
    if not mdf.empty: st.dataframe(mdf,use_container_width=True,hide_index=True)


def ai_page(df,gitlab):
    render_hero("AI Engineering Copilot","Ask grounded questions about the Jira and GitLab telemetry currently loaded.","AI ASSIST")
    cfg=st.session_state.config
    if not cfg.llm_api_key:
        render_empty_state("LLM is not configured","Set LLM_API_KEY and LLM_MODEL in your environment or Streamlit secrets.")
        return
    if df.empty and not gitlab:
        render_empty_state("No telemetry available","Load Jira or GitLab data before asking for analysis.")
        return
    prompt=st.text_area("What should the copilot analyze?",value="Summarize the most important delivery risks and suggest observable follow-up checks.",height=110)
    if st.button("Generate analysis",use_container_width=True):
        context={
            "jira": df.to_dict("records") if not df.empty else [],
            "gitlab": {
                "merge_requests":[asdict(x) for x in gitlab.get("merge_requests",[])] if gitlab else [],
                "commits":[asdict(x) for x in gitlab.get("commits",[])] if gitlab else [],
            }
        }
        with st.spinner("Analyzing telemetry…"):
            try:
                result=LLMAssistantService(cfg).generate_analysis(prompt,json.dumps(context,default=str)[:50000])
                st.markdown('<div class="panel"><div class="panel-title">Analysis</div>',unsafe_allow_html=True)
                st.markdown(result)
                st.markdown('</div>',unsafe_allow_html=True)
            except Exception as ex:
                st.error(str(ex))


def render_empty_state(title, message):
    st.markdown(f"""
    <div class="panel" style="padding:2rem;text-align:center;">
      <div style="font-size:2rem;margin-bottom:.5rem;">✦</div>
      <div style="font-family:Manrope;font-size:1.1rem;font-weight:800;color:var(--text)">{title}</div>
      <div style="color:var(--muted);font-size:.78rem;margin-top:.35rem">{message}</div>
    </div>
    """,unsafe_allow_html=True)


# ============================================================
# APP
# ============================================================

def main():
    st.set_page_config(page_title="Engineering Hub",page_icon="✦",layout="wide",initial_sidebar_state="expanded")

    if "theme" not in st.session_state: st.session_state.theme="Dark"
    if "config" not in st.session_state: st.session_state.config=AppConfig.load_from_env_or_secrets()
    if "jira_df" not in st.session_state: st.session_state.jira_df=pd.DataFrame()
    if "gitlab" not in st.session_state: st.session_state.gitlab=None
    if "gitlab_projects" not in st.session_state: st.session_state.gitlab_projects=[]

    with st.sidebar:
        st.markdown("""
        <div class="brand">
          <div class="brand-mark">✦</div>
          <div><div class="brand-title">Engineering Hub</div><div class="brand-sub">DELIVERY INTELLIGENCE</div></div>
        </div>
        """,unsafe_allow_html=True)
        theme=st.radio("Appearance",["Dark","Light"],index=0 if st.session_state.theme=="Dark" else 1,horizontal=True,key="theme_picker")
        st.session_state.theme=theme
        st.divider()
        page=st.radio("Workspace",[
            "Executive Overview","Delivery Dashboard","GitLab Activity","AI Copilot","Data Sources"
        ],format_func=lambda x: {
            "Executive Overview":"◈  Overview",
            "Delivery Dashboard":"◫  Delivery",
            "GitLab Activity":"⌘  GitLab",
            "AI Copilot":"✦  AI Copilot",
            "Data Sources":"◉  Data Sources",
        }[x])
        st.divider()
        df=st.session_state.jira_df
        st.markdown(f'<div class="small-muted">Jira <b>{len(df) if not df.empty else 0}</b> issues loaded</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="small-muted">GitLab <b>{len(st.session_state.gitlab.get("commits",[])) if st.session_state.gitlab else 0}</b> commits loaded</div>',unsafe_allow_html=True)
        st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
        if st.button("↻ Refresh page",use_container_width=True):
            st.rerun()

    inject_custom_styles(st.session_state.theme)

    df=st.session_state.jira_df
    gitlab=st.session_state.gitlab

    if page in ["Executive Overview","Delivery Dashboard"] and not df.empty:
        render_filters(df)

    if page=="Executive Overview": overview_page(df,gitlab)
    elif page=="Delivery Dashboard": delivery_page(df)
    elif page=="GitLab Activity": gitlab_page(gitlab)
    elif page=="AI Copilot": ai_page(df,gitlab)
    elif page=="Data Sources": data_sources_page()


if __name__=="__main__":
    main()
