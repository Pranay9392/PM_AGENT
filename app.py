import html
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("EngineeringHub")


def load_dotenv_file(path: str = ".env") -> None:
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


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
            gitlab_private_token=os.getenv(
                "GITLAB_PRIVATE_TOKEN", fetch("GitLab", "PRIVATE_TOKEN")
            ),
            llm_base_url=os.getenv(
                "LLM_BASE_URL",
                fetch("LLM", "BASE_URL", "https://api.openai.com/v1"),
            ),
            llm_api_key=os.getenv("LLM_API_KEY", fetch("LLM", "API_KEY")),
            llm_model=os.getenv("LLM_MODEL", fetch("LLM", "MODEL", "gpt-4o")),
        )


# =====================================================================
# 2. DATA MODELS
# =====================================================================

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


# =====================================================================
# 3. HTTP CLIENT
# =====================================================================

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


# =====================================================================
# 4. JIRA SERVICE
# =====================================================================

class JiraService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.session = RobustHTTPClient.create_session()
        self.auth = (config.jira_email, config.jira_api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def fetch_issues(self, project_key: str, jql_filter: str = "") -> List[JiraIssueDTO]:
        url = f"{self.config.jira_base_url.rstrip('/')}/rest/api/3/search"

        query = f"project = '{project_key}'"
        if jql_filter:
            query += f" AND ({jql_filter})"

        fields = ",".join(
            [
                "summary",
                "status",
                "assignee",
                "reporter",
                "priority",
                "issuetype",
                "project",
                "created",
                "updated",
                "duedate",
                "description",
                "timespent",
                "labels",
                "customfield_10020",  # Common Jira Cloud sprint field
                "customfield_10016",  # Common Jira Cloud story-points field
            ]
        )

        params = {
            "jql": query,
            "maxResults": 100,
            "fields": fields,
        }

        response = self.session.get(
            url,
            headers=self.headers,
            auth=self.auth,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        raw_issues = response.json().get("issues", [])

        dtos = []

        for raw in raw_issues:
            fields_data = raw.get("fields", {})

            desc = fields_data.get("description") or ""
            if isinstance(desc, dict):
                desc = "Rich Text / ADF description"
            elif isinstance(desc, str) and len(desc) > 500:
                desc = desc[:500] + "... [Truncated]"

            status_obj = fields_data.get("status") or {}
            status = status_obj.get("name", "Unknown")
            status_category = (
                status_obj.get("statusCategory", {}).get("name", "Unknown")
            )

            assignee = fields_data.get("assignee") or {}
            reporter = fields_data.get("reporter") or {}
            priority = fields_data.get("priority") or {}
            issue_type = fields_data.get("issuetype") or {}
            project = fields_data.get("project") or {}

            sprint_value = fields_data.get("customfield_10020")
            sprint_name = self._extract_sprint(sprint_value)

            story_points = fields_data.get("customfield_10016")
            try:
                story_points = float(story_points or 0)
            except (TypeError, ValueError):
                story_points = 0.0

            base_url = self.config.jira_base_url.rstrip("/")

            dtos.append(
                JiraIssueDTO(
                    key=raw.get("key", ""),
                    summary=fields_data.get("summary", ""),
                    status=status,
                    status_category=status_category,
                    assignee=assignee.get("displayName", "Unassigned"),
                    reporter=reporter.get("displayName", "Unknown"),
                    priority=priority.get("name", "None"),
                    issue_type=issue_type.get("name", "Unknown"),
                    project=project.get("key", project_key),
                    sprint=sprint_name,
                    labels=", ".join(fields_data.get("labels") or []),
                    created=(fields_data.get("created") or "")[:10],
                    updated=(fields_data.get("updated") or "")[:10],
                    due_date=(fields_data.get("duedate") or ""),
                    description=desc,
                    time_spent_hours=round(
                        (fields_data.get("timespent") or 0) / 3600, 2
                    ),
                    story_points=story_points,
                    url=f"{base_url}/browse/{raw.get('key', '')}",
                )
            )

        return dtos

    @staticmethod
    def _extract_sprint(value: Any) -> str:
        if not value:
            return "No Sprint"

        if isinstance(value, list):
            names = []
            for item in value:
                if isinstance(item, dict):
                    names.append(str(item.get("name", "")))
                elif isinstance(item, str):
                    names.append(item)
            return ", ".join([x for x in names if x]) or "No Sprint"

        return str(value)


# =====================================================================
# 5. GITLAB SERVICE
# =====================================================================

class GitLabService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.session = RobustHTTPClient.create_session()
        self.headers = {"PRIVATE-TOKEN": config.gitlab_private_token}

    def fetch_projects(self) -> List[Dict[str, Any]]:
        url = f"{self.config.gitlab_base_url.rstrip('/')}/api/v4/projects"
        params = {
            "membership": "true",
            "per_page": 50,
            "simple": "true",
            "order_by": "last_activity_at",
        }
        response = self.session.get(
            url, headers=self.headers, params=params, timeout=20
        )
        response.raise_for_status()
        return response.json()

    def fetch_deep_telemetry(self, project_id: int) -> Dict[str, List[Any]]:
        base = self.config.gitlab_base_url.rstrip("/")

        def fetch_mrs():
            res = self.session.get(
                f"{base}/api/v4/projects/{project_id}/merge_requests",
                params={"per_page": 50, "state": "all", "order_by": "updated_at"},
                headers=self.headers,
                timeout=20,
            )
            if res.status_code != 200:
                return []

            return [
                GitLabMRDTO(
                    mr_id=mr.get("iid"),
                    title=mr.get("title", ""),
                    state=mr.get("state", ""),
                    author=(mr.get("author") or {}).get("username", "unknown"),
                    assignee=(mr.get("assignee") or {}).get("username"),
                    created_at=(mr.get("created_at") or "")[:10],
                    updated_at=(mr.get("updated_at") or "")[:10],
                    merged_at=(mr.get("merged_at") or "")[:10]
                    if mr.get("merged_at")
                    else None,
                    source_branch=mr.get("source_branch", ""),
                    target_branch=mr.get("target_branch", ""),
                    description=(mr.get("description") or "")[:300],
                )
                for mr in res.json()
            ]

        def fetch_commits():
            res = self.session.get(
                f"{base}/api/v4/projects/{project_id}/repository/commits",
                params={"per_page": 50},
                headers=self.headers,
                timeout=20,
            )
            if res.status_code != 200:
                return []

            return [
                GitLabCommitDTO(
                    commit_id=c.get("short_id", ""),
                    title=c.get("title", ""),
                    author=c.get("author_name", ""),
                    created_at=(c.get("created_at") or "")[:10],
                    message=(c.get("message") or "").strip(),
                )
                for c in res.json()
            ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_mrs = executor.submit(fetch_mrs)
            future_commits = executor.submit(fetch_commits)

        return {
            "merge_requests": future_mrs.result(),
            "commits": future_commits.result(),
        }


# =====================================================================
# 6. LLM SERVICE
# =====================================================================

class LLMAssistantService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.session = RobustHTTPClient.create_session()

    def generate_analysis(self, prompt: str, context_payload: str) -> str:
        if not self.config.llm_api_key:
            raise ValueError("LLM_API_KEY is not configured.")

        url = f"{self.config.llm_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.llm_api_key}",
            "Content-Type": "application/json",
        }

        system_prompt = (
            "You are an Engineering Manager's technical program assistant. "
            "Analyze Jira and GitLab telemetry. Focus on observable delivery signals, "
            "workload, aging, blockers, review queues, and trends. Do not infer personal "
            "traits or judge individual employee performance. Clearly separate facts from "
            "possible explanations. Return concise markdown with actionable follow-ups."
        )

        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"### TELEMETRY\n{context_payload}\n\n"
                        f"### TASK\n{prompt}"
                    ),
                },
            ],
            "temperature": 0.2,
        }

        response = self.session.post(
            url, headers=headers, json=payload, timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


# =====================================================================
# 7. THEME TOKENS (green, light + dark)
# =====================================================================

DISPLAY_FONT = "Sora"
BODY_FONT = "Manrope"

DARK_THEME: Dict[str, Any] = {
    "name": "dark",
    "scheme": "dark",
    "bg": "#04100B",
    "bg_a": "rgba(16,185,129,.20)",
    "bg_b": "rgba(45,212,191,.11)",
    "surface": "rgba(14,38,30,.66)",
    "surface_solid": "#0E261E",
    "surface2": "rgba(255,255,255,.035)",
    "border": "rgba(110,231,183,.17)",
    "text": "#E8F7F0",
    "muted": "#8FB8A6",
    "faint": "#5E8474",
    "accent": "#34D399",
    "accent_strong": "#10B981",
    "accent_soft": "rgba(52,211,153,.13)",
    "grid": "rgba(143,184,166,.15)",
    "track": "rgba(255,255,255,.09)",
    "input_bg": "rgba(6,22,16,.85)",
    "sidebar": "linear-gradient(180deg,#07180F 0%,#040F0A 100%)",
    "shadow": "0 14px 44px rgba(0,0,0,.38)",
    "gloss": "rgba(255,255,255,.11)",
    "gloss_soft": "rgba(255,255,255,.055)",
    "btn_top": "rgba(255,255,255,.09)",
    "btn_bot": "rgba(255,255,255,.03)",
    "th": "rgba(10,31,24,.96)",
    "row_line": "rgba(143,184,166,.10)",
    "tones": {
        "green": "#34D399",
        "teal": "#2DD4BF",
        "lime": "#A3E635",
        "amber": "#FBBF24",
        "red": "#FB7185",
        "sky": "#38BDF8",
    },
    "pills": {
        "success": ("rgba(52,211,153,.16)", "#6EE7B7"),
        "warning": ("rgba(251,191,36,.16)", "#FCD34D"),
        "danger": ("rgba(251,113,133,.16)", "#FDA4AF"),
        "info": ("rgba(56,189,248,.16)", "#7DD3FC"),
        "neutral": ("rgba(148,163,184,.16)", "#CBD5E1"),
    },
}

LIGHT_THEME: Dict[str, Any] = {
    "name": "light",
    "scheme": "light",
    "bg": "#EDF7F1",
    "bg_a": "rgba(16,185,129,.22)",
    "bg_b": "rgba(45,212,191,.18)",
    "surface": "rgba(255,255,255,.74)",
    "surface_solid": "#FFFFFF",
    "surface2": "rgba(5,150,105,.045)",
    "border": "rgba(5,150,105,.17)",
    "text": "#092A1F",
    "muted": "#4A6C5D",
    "faint": "#7D9C8C",
    "accent": "#059669",
    "accent_strong": "#047857",
    "accent_soft": "rgba(5,150,105,.09)",
    "grid": "rgba(9,42,31,.10)",
    "track": "rgba(5,150,105,.13)",
    "input_bg": "rgba(255,255,255,.92)",
    "sidebar": "linear-gradient(180deg,#FFFFFF 0%,#E7F4EC 100%)",
    "shadow": "0 12px 36px rgba(6,78,59,.11)",
    "gloss": "rgba(255,255,255,.95)",
    "gloss_soft": "rgba(255,255,255,.65)",
    "btn_top": "rgba(255,255,255,.98)",
    "btn_bot": "rgba(226,243,234,.9)",
    "th": "rgba(236,247,241,.97)",
    "row_line": "rgba(9,42,31,.07)",
    "tones": {
        "green": "#059669",
        "teal": "#0D9488",
        "lime": "#65A30D",
        "amber": "#D97706",
        "red": "#E11D48",
        "sky": "#0284C7",
    },
    "pills": {
        "success": ("#D1FAE5", "#047857"),
        "warning": ("#FEF3C7", "#B45309"),
        "danger": ("#FFE4E6", "#BE123C"),
        "info": ("#E0F2FE", "#0369A1"),
        "neutral": ("#E4EEE9", "#3F5B4E"),
    },
}


def T() -> Dict[str, Any]:
    return DARK_THEME if st.session_state.get("dark_mode", True) else LIGHT_THEME


def palette() -> List[str]:
    tones = T()["tones"]
    return [tones[k] for k in ["green", "teal", "lime", "amber", "sky", "red"]]


# =====================================================================
# 8. ANALYTICS ENGINE
# =====================================================================

ACTIVE_PATTERN = "progress|review|testing"


def prepare_jira_dataframe(issues: List[Dict[str, Any]]) -> pd.DataFrame:
    if not issues:
        return pd.DataFrame()

    df = pd.DataFrame(issues)

    for col in ["created", "updated", "due_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    today = pd.Timestamp.today().normalize()

    df["age_days"] = (
        (today - df["created"]).dt.days.fillna(0).clip(lower=0)
        if "created" in df
        else 0
    )

    df["days_since_update"] = (
        (today - df["updated"]).dt.days.fillna(0).clip(lower=0)
        if "updated" in df
        else 0
    )

    if "due_date" in df:
        df["overdue"] = (
            df["due_date"].notna()
            & (df["due_date"] < today)
            & ~df["status"].fillna("").str.lower().isin(["done", "closed", "resolved"])
        )
    else:
        df["overdue"] = False

    status_lower = df["status"].fillna("").str.lower()

    df["is_done"] = status_lower.isin(
        ["done", "closed", "resolved", "complete", "completed"]
    )
    df["is_active"] = status_lower.str.contains(ACTIVE_PATTERN, regex=True, na=False)
    df["is_blocked"] = (
        status_lower.str.contains("block", na=False)
        | df.get("labels", pd.Series("", index=df.index))
        .fillna("")
        .str.lower()
        .str.contains("blocked", na=False)
    )
    df["is_stale"] = (df["days_since_update"] >= 5) & ~df["is_done"]

    return df


def filtered_jira_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    project = st.session_state.get("filter_project", "All")
    assignee = st.session_state.get("filter_assignee", "All")
    status = st.session_state.get("filter_status", "All")
    priority = st.session_state.get("filter_priority", "All")

    result = df.copy()

    if project != "All" and "project" in result:
        result = result[result["project"] == project]

    if assignee != "All" and "assignee" in result:
        result = result[result["assignee"] == assignee]

    if status != "All" and "status" in result:
        result = result[result["status"] == status]

    if priority != "All" and "priority" in result:
        result = result[result["priority"] == priority]

    return result


def weekly_trend(d: pd.DataFrame, weeks: int = 16) -> pd.DataFrame:
    """Created vs completed issues per week. Completion uses the last-update date."""

    def _weekly(series: pd.Series) -> pd.Series:
        s = series.dropna()
        if s.empty:
            return pd.Series(dtype="int64")
        return s.dt.to_period("W").dt.start_time.value_counts()

    created = _weekly(d["created"])
    completed = _weekly(d.loc[d["is_done"], "updated"])
    all_weeks = created.index.union(completed.index)
    if len(all_weeks) == 0:
        return pd.DataFrame(columns=["week", "Series", "Issues"])

    full = pd.date_range(all_weeks.min(), all_weeks.max(), freq="7D")[-weeks:]
    rows = []
    for name, series in [("Created", created), ("Completed", completed)]:
        aligned = series.reindex(full, fill_value=0)
        rows += [
            {"week": wk, "Series": name, "Issues": int(v)} for wk, v in aligned.items()
        ]
    return pd.DataFrame(rows)


def workload_long(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for person, g in d.groupby("assignee"):
        done = int(g["is_done"].sum())
        blocked = int((g["is_blocked"] & ~g["is_done"]).sum())
        open_other = int(len(g) - done - blocked)
        rows += [
            {"assignee": person, "Kind": "Completed", "Issues": done},
            {"assignee": person, "Kind": "Open", "Issues": open_other},
            {"assignee": person, "Kind": "Blocked", "Issues": blocked},
        ]
    return pd.DataFrame(rows)


# =====================================================================
# 9. CHART HELPERS (Altair, theme aware)
# =====================================================================

def _finish(chart, height: int):
    t = T()
    return (
        chart.properties(
            height=height,
            background="transparent",
            padding={"left": 4, "right": 22, "top": 10, "bottom": 4},
        )
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=t["muted"],
            titleColor=t["muted"],
            gridColor=t["grid"],
            domainColor=t["grid"],
            tickColor=t["grid"],
            labelFont=BODY_FONT,
            titleFont=BODY_FONT,
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_legend(
            orient="bottom",
            labelColor=t["text"],
            labelFont=BODY_FONT,
            labelFontSize=11,
            symbolType="circle",
            symbolSize=90,
            padding=6,
        )
    )


def show_chart(chart) -> None:
    if chart is None:
        return
    try:
        st.altair_chart(chart, theme=None, width="stretch")
    except TypeError:
        st.altair_chart(chart, theme=None, use_container_width=True)


def _gradient(vertical: bool = False):
    t = T()
    stops = [
        alt.GradientStop(color=t["accent_strong"], offset=0),
        alt.GradientStop(color=t["tones"]["teal"], offset=1),
    ]
    if vertical:
        return alt.Gradient(gradient="linear", stops=stops, x1=0, y1=1, x2=0, y2=0)
    return alt.Gradient(gradient="linear", stops=stops, x1=0, y1=0, x2=1, y2=0)


def donut_chart(
    df: pd.DataFrame,
    cat: str,
    val: str,
    center: Optional[str] = None,
    height: int = 250,
    colors: Optional[Dict[str, str]] = None,
):
    if df is None or df.empty:
        return None
    t = T()
    df = df.copy()
    df[cat] = df[cat].astype(str)
    cats = df[cat].tolist()
    pal = palette()
    rng = [(colors or {}).get(c, pal[i % len(pal)]) for i, c in enumerate(cats)]

    arc = (
        alt.Chart(df)
        .mark_arc(innerRadius=64, outerRadius=98, cornerRadius=5, padAngle=0.025)
        .encode(
            theta=alt.Theta(f"{val}:Q"),
            color=alt.Color(
                f"{cat}:N",
                scale=alt.Scale(domain=cats, range=rng),
                legend=alt.Legend(title=None, orient="right"),
            ),
            order=alt.Order(f"{val}:Q", sort="descending"),
            tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{val}:Q")],
        )
    )
    layers = [arc]
    if center:
        layers.append(
            alt.Chart(pd.DataFrame({"t": [center]}))
            .mark_text(
                fontSize=28, fontWeight=700, color=t["text"], font=DISPLAY_FONT
            )
            .encode(text="t:N")
        )
    return _finish(alt.layer(*layers), height)


def hbar_chart(
    df: pd.DataFrame,
    cat: str,
    val: str,
    height: Optional[int] = None,
    order: Optional[List[str]] = None,
    color_map: Optional[Dict[str, str]] = None,
):
    if df is None or df.empty:
        return None
    t = T()
    df = df.copy()
    df[cat] = df[cat].astype(str)
    height = height or max(180, 34 * len(df) + 40)

    base = alt.Chart(df).encode(
        y=alt.Y(
            f"{cat}:N",
            sort=order if order else "-x",
            title=None,
            axis=alt.Axis(labelLimit=170, ticks=False, domain=False),
        ),
        x=alt.X(f"{val}:Q", title=None, axis=None),
        tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{val}:Q")],
    )
    if color_map:
        cats = df[cat].tolist()
        pal = palette()
        rng = [color_map.get(c, pal[i % len(pal)]) for i, c in enumerate(cats)]
        bars = base.mark_bar(size=18, cornerRadiusEnd=8).encode(
            color=alt.Color(
                f"{cat}:N", scale=alt.Scale(domain=cats, range=rng), legend=None
            )
        )
    else:
        bars = base.mark_bar(size=18, cornerRadiusEnd=8, color=_gradient())
    labels = base.mark_text(
        align="left", dx=7, color=t["muted"], fontSize=11, fontWeight=700
    ).encode(text=f"{val}:Q")
    return _finish(alt.layer(bars, labels), height)


def vbar_chart(df: pd.DataFrame, cat: str, val: str, order: List[str], height: int = 240):
    if df is None or df.empty:
        return None
    t = T()
    df = df.copy()
    df[cat] = df[cat].astype(str)
    base = alt.Chart(df).encode(
        x=alt.X(
            f"{cat}:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelAngle=0, ticks=False, domain=False),
        ),
        y=alt.Y(f"{val}:Q", title=None, axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{val}:Q")],
    )
    bars = base.mark_bar(
        size=34,
        cornerRadiusTopLeft=8,
        cornerRadiusTopRight=8,
        color=_gradient(vertical=True),
    )
    labels = base.mark_text(
        dy=-8, color=t["muted"], fontSize=11, fontWeight=700, baseline="bottom"
    ).encode(text=f"{val}:Q")
    return _finish(alt.layer(bars, labels), height)


def stacked_hbar(
    df: pd.DataFrame,
    cat: str,
    kind: str,
    val: str,
    colors: Dict[str, str],
    height: Optional[int] = None,
):
    if df is None or df.empty:
        return None
    df = df.copy()
    df[cat] = df[cat].astype(str)
    order = (
        df.groupby(cat)[val].sum().sort_values(ascending=False).index.tolist()
    )
    height = height or max(200, 36 * len(order) + 60)
    kinds = list(colors.keys())
    chart = (
        alt.Chart(df)
        .mark_bar(size=20, cornerRadius=4)
        .encode(
            y=alt.Y(
                f"{cat}:N",
                sort=order,
                title=None,
                axis=alt.Axis(labelLimit=170, ticks=False, domain=False),
            ),
            x=alt.X(f"sum({val}):Q", title=None, stack="zero"),
            color=alt.Color(
                f"{kind}:N",
                scale=alt.Scale(domain=kinds, range=[colors[k] for k in kinds]),
                legend=alt.Legend(title=None),
            ),
            order=alt.Order("kind_order:Q"),
            tooltip=[
                alt.Tooltip(f"{cat}:N"),
                alt.Tooltip(f"{kind}:N"),
                alt.Tooltip(f"{val}:Q"),
            ],
        )
        .transform_calculate(
            kind_order=f"indexof({json.dumps(kinds)}, datum.{kind})"
        )
    )
    return _finish(chart, height)


def trend_chart(df: pd.DataFrame, x: str, y: str, series: str, height: int = 260):
    if df is None or df.empty:
        return None
    df = df.copy()
    kinds = df[series].unique().tolist()
    pal = palette()
    scale = alt.Scale(domain=kinds, range=[pal[0], pal[3]][: len(kinds)])
    base = alt.Chart(df).encode(
        x=alt.X(
            f"{x}:T",
            title=None,
            axis=alt.Axis(format="%b %d", labelOverlap=True, grid=False),
        ),
        y=alt.Y(f"{y}:Q", title=None, axis=alt.Axis(tickMinStep=1)),
        color=alt.Color(f"{series}:N", scale=scale, legend=alt.Legend(title=None)),
    )
    area = base.mark_area(opacity=0.13, interpolate="monotone")
    line = base.mark_line(strokeWidth=3, interpolate="monotone")
    pts = base.mark_point(filled=True, size=55, opacity=1).encode(
        tooltip=[
            alt.Tooltip(f"{x}:T", title="Week of", format="%d %b %Y"),
            alt.Tooltip(f"{series}:N"),
            alt.Tooltip(f"{y}:Q"),
        ]
    )
    return _finish(alt.layer(area, line, pts), height)


def daily_area_chart(df: pd.DataFrame, x: str, y: str, height: int = 240):
    if df is None or df.empty:
        return None
    t = T()
    grad = alt.Gradient(
        gradient="linear",
        stops=[
            alt.GradientStop(color=t["accent"], offset=0),
            alt.GradientStop(color="rgba(0,0,0,0)", offset=1),
        ],
        x1=1, y1=0, x2=1, y2=1,
    )
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:T", title=None, axis=alt.Axis(format="%b %d", grid=False)),
        y=alt.Y(f"{y}:Q", title=None, axis=alt.Axis(tickMinStep=1)),
    )
    area = base.mark_area(interpolate="monotone", opacity=0.35, color=grad)
    line = base.mark_line(interpolate="monotone", strokeWidth=3, color=t["accent"])
    pts = base.mark_point(filled=True, size=50, color=t["accent"]).encode(
        tooltip=[
            alt.Tooltip(f"{x}:T", format="%d %b %Y"),
            alt.Tooltip(f"{y}:Q"),
        ]
    )
    return _finish(alt.layer(area, line, pts), height)


def scatter_chart(d: pd.DataFrame, height: int = 300):
    if d is None or d.empty:
        return None
    t = T()
    data = d[
        ["key", "summary", "assignee", "priority", "age_days", "days_since_update"]
    ].copy()
    data["summary"] = data["summary"].astype(str).str.slice(0, 70)
    data["priority"] = data["priority"].astype(str)
    cats = sorted(data["priority"].unique().tolist())
    pal = palette()
    points = (
        alt.Chart(data)
        .mark_circle(size=110, opacity=0.82, stroke=t["surface_solid"], strokeWidth=1.5)
        .encode(
            x=alt.X("age_days:Q", title="Age in days"),
            y=alt.Y("days_since_update:Q", title="Days since last update"),
            color=alt.Color(
                "priority:N",
                scale=alt.Scale(
                    domain=cats, range=[pal[i % len(pal)] for i in range(len(cats))]
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=["key", "summary", "assignee", "priority", "age_days", "days_since_update"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [5]}))
        .mark_rule(strokeDash=[5, 5], color=t["tones"]["red"], opacity=0.7)
        .encode(y="y:Q")
    )
    return _finish(alt.layer(points, rule), height)


# =====================================================================
# 10. HTML / UI COMPONENTS
# =====================================================================

_PANEL_N = [0]


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value)).replace("$", "&#36;")


@contextmanager
def panel(title: str = "", subtitle: str = ""):
    _PANEL_N[0] += 1
    try:
        box = st.container(key=f"panel_{_PANEL_N[0]}")
    except TypeError:  # older Streamlit without container keys
        box = st.container(border=True)
    with box:
        if title:
            sub = f'<div class="panel-subtitle">{esc(subtitle)}</div>' if subtitle else ""
            st.markdown(
                f'<div class="panel-title">{esc(title)}</div>{sub}',
                unsafe_allow_html=True,
            )
        yield


def render_kpi(title: str, value: Any, subtitle: str = "", tone: str = "green", icon: str = "●"):
    color = T()["tones"].get(tone, T()["tones"]["green"])
    st.markdown(
        f'<div class="kpi" style="--tone:{color}">'
        f'<div class="kpi-top"><span class="kpi-title">{esc(title)}</span>'
        f'<span class="kpi-icon">{icon}</span></div>'
        f'<div class="kpi-value">{esc(value)}</div>'
        f'<div class="kpi-sub">{esc(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "neutral") -> str:
    return f'<span class="pill pill-{kind}">{esc(text)}</span>'


def status_kind(status: str) -> str:
    s = str(status).lower()
    if any(x in s for x in ["done", "closed", "resolved", "complete"]):
        return "success"
    if any(x in s for x in ["block", "cancel"]):
        return "danger"
    if any(x in s for x in ["progress", "review", "testing"]):
        return "warning"
    return "info"


def priority_kind(priority: str) -> str:
    p = str(priority).lower()
    if any(x in p for x in ["highest", "critical", "blocker"]):
        return "danger"
    if "high" in p:
        return "warning"
    if "medium" in p:
        return "info"
    return "neutral"


def mr_state_kind(state: str) -> str:
    s = str(state).lower()
    if s == "merged":
        return "success"
    if s == "opened":
        return "warning"
    if s == "closed":
        return "danger"
    return "neutral"


PILL_RULES = {
    "status": status_kind,
    "priority": priority_kind,
    "state": mr_state_kind,
}

COLUMN_LABELS = {
    "key": "Key",
    "mr_id": "MR",
    "commit_id": "Commit",
    "created_at": "Created",
    "updated_at": "Updated",
    "merged_at": "Merged",
    "issue_type": "Type",
    "days_since_update": "Days idle",
}


def html_table(df: pd.DataFrame, max_rows: int = 200, height: int = 380, empty: str = "Nothing to show."):
    if df is None or df.empty:
        st.caption(empty)
        return

    view = df.head(max_rows)
    head = "".join(
        f"<th>{esc(COLUMN_LABELS.get(c, str(c).replace('_', ' ').title()))}</th>"
        for c in view.columns
    )
    body = []
    for _, row in view.iterrows():
        cells = []
        for col in view.columns:
            v = row[col]
            if col in PILL_RULES and isinstance(v, str) and v:
                cell = pill(v, PILL_RULES[col](v))
                title = ""
            elif isinstance(v, (bool,)) or str(type(v)).endswith("bool_'>"):
                cell = pill("Yes", "warning") if v else '<span class="dash">–</span>'
                title = ""
            elif isinstance(v, pd.Timestamp):
                cell = esc(v.strftime("%Y-%m-%d")) if pd.notna(v) else '<span class="dash">–</span>'
                title = ""
            elif isinstance(v, float):
                cell = "–" if pd.isna(v) else esc(f"{v:.1f}".rstrip("0").rstrip("."))
                title = ""
            elif v is None or (not isinstance(v, str) and pd.isna(v)) or v == "":
                cell = '<span class="dash">–</span>'
                title = ""
            else:
                cell = esc(v)
                title = f' title="{esc(v)}"'
            cells.append(f"<td{title}>{cell}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")

    st.markdown(
        f'<div class="tbl-wrap" style="max-height:{height}px">'
        f'<table class="tbl"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, chip: str = ""):
    chip_html = f'<span class="hero-chip"><i></i>{esc(chip)}</span>' if chip else ""
    st.markdown(
        f'<div class="hero"><div class="hero-body">{chip_html}'
        f"<h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div></div>",
        unsafe_allow_html=True,
    )


def render_empty(title: str, body: str):
    st.markdown(
        f'<div class="empty"><div class="empty-title">{esc(title)}</div>'
        f"<div>{esc(body)}</div></div>",
        unsafe_allow_html=True,
    )


def render_alert(title: str, meta: str, kind: str):
    tones = T()["tones"]
    color = {"danger": tones["red"], "warning": tones["amber"], "info": tones["sky"]}.get(
        kind, tones["sky"]
    )
    st.markdown(
        f'<div class="alert" style="--tone:{color}">'
        f'<div class="alert-title">{esc(title)}</div>'
        f'<div class="alert-meta">{esc(meta)}</div></div>',
        unsafe_allow_html=True,
    )


def render_ring(pct: int, label: str):
    st.markdown(
        f'<div class="ring-wrap"><div class="ring" style="--p:{int(pct)}">'
        f'<div class="ring-inner"><span>{int(pct)}%</span><small>{esc(label)}</small></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


def stat_rows(rows: List[tuple]):
    inner = "".join(
        f'<div class="stat-row"><span>{esc(k)}</span><strong>{esc(v)}</strong></div>'
        for k, v in rows
    )
    st.markdown(f'<div class="stat-list">{inner}</div>', unsafe_allow_html=True)


def get_selected_issue(df: pd.DataFrame, key: str):
    if df.empty:
        return None
    rows = df[df["key"] == key]
    return rows.iloc[0] if not rows.empty else None


# =====================================================================
# 11. STYLES
# =====================================================================

STATIC_CSS = """
html, body, .stApp, [class*="css"] { font-family: 'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif; }

.stApp {
  background:
    radial-gradient(1100px 540px at 6% -10%, var(--bg-a), transparent 62%),
    radial-gradient(900px 480px at 100% 0%, var(--bg-b), transparent 58%),
    var(--bg);
  color: var(--text);
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
[data-testid="stMainBlockContainer"], .main .block-container {
  padding: 1.8rem 2.4rem 3.5rem; max-width: 1560px;
}

:where(.stApp) :where(h1, h2, h3, h4, h5, h6) {
  font-family: 'Sora', 'Manrope', sans-serif; color: var(--text); letter-spacing: -.02em;
}
:where(.stApp) :where(p, li, label, [data-testid="stMarkdownContainer"]) { color: var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); }
[data-testid="stWidgetLabel"] p { color: var(--muted); font-weight: 600; font-size: .82rem; }
hr { border-color: var(--border); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--border); }
[data-testid="stSidebarNav"] a, a[data-testid="stSidebarNavLink"] {
  border-radius: 12px; color: var(--muted) !important; font-weight: 600; margin: 2px 0;
}
[data-testid="stSidebarNav"] a span, a[data-testid="stSidebarNavLink"] span { color: inherit !important; }
[data-testid="stSidebarNav"] a:hover, a[data-testid="stSidebarNavLink"]:hover { background: var(--accent-soft); }
[data-testid="stSidebarNav"] a[aria-current="page"], a[data-testid="stSidebarNavLink"][aria-current="page"] {
  background: linear-gradient(135deg, var(--accent-soft), transparent);
  color: var(--text) !important; box-shadow: inset 0 0 0 1px var(--border), inset 0 1px 0 var(--gloss);
}
.brand { display: flex; align-items: center; gap: .7rem; padding: .3rem 0 .4rem; }
.brand-mark {
  width: 38px; height: 38px; border-radius: 12px; display: grid; place-items: center; font-size: 1.1rem;
  background: linear-gradient(160deg, #34D399, #047857); color: #fff;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.5), 0 6px 18px rgba(4,120,87,.45);
}
.brand-name { font-family: 'Sora', sans-serif; font-weight: 700; color: var(--text); font-size: 1rem; line-height: 1.1; }
.brand-sub { color: var(--muted); font-size: .72rem; }
.side-stats { color: var(--muted); font-size: .78rem; line-height: 1.8; }
.side-stats strong { color: var(--text); }

/* ---------- hero ---------- */
.hero {
  position: relative; overflow: hidden; border-radius: 24px; margin: 0 0 1.3rem; padding: 1.7rem 1.9rem;
  background: linear-gradient(118deg, #064E3B 0%, #047857 48%, #0F766E 100%);
  border: 1px solid rgba(255,255,255,.18);
  box-shadow: 0 18px 50px rgba(4,120,87,.30), inset 0 1px 0 rgba(255,255,255,.35);
}
.hero::before {
  content: ""; position: absolute; inset: 0;
  background: radial-gradient(520px 220px at 12% -30%, rgba(255,255,255,.38), transparent 70%);
}
.hero::after {
  content: ""; position: absolute; top: -60%; right: -8%; width: 46%; height: 240%;
  transform: rotate(18deg);
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.10), transparent);
}
.hero-body { position: relative; z-index: 1; }
.hero h1 { margin: .55rem 0 .3rem; font-size: 2.05rem; color: #fff !important; letter-spacing: -.035em; }
.hero p { margin: 0; color: rgba(236,253,245,.86) !important; font-size: .94rem; max-width: 62ch; }
.hero-chip {
  display: inline-flex; align-items: center; gap: .45rem; font-size: .74rem; font-weight: 700;
  color: #ECFDF5; padding: .28rem .7rem; border-radius: 999px;
  background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22);
}
.hero-chip i { width: 7px; height: 7px; border-radius: 50%; background: #A7F3D0; box-shadow: 0 0 10px #6EE7B7; }

/* ---------- KPI cards ---------- */
.kpi {
  position: relative; isolation: isolate; overflow: hidden; min-height: 122px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
  padding: 1rem 1.1rem .95rem; backdrop-filter: blur(14px) saturate(140%);
  box-shadow: var(--shadow), inset 0 1px 0 var(--gloss);
}
.kpi::before {
  content: ""; position: absolute; inset: 0 0 45% 0; z-index: -1;
  background: linear-gradient(180deg, var(--gloss-soft), transparent);
}
.kpi::after {
  content: ""; position: absolute; left: 1.1rem; right: 1.1rem; bottom: 0; height: 3px; border-radius: 3px 3px 0 0;
  background: linear-gradient(90deg, var(--tone), transparent);
}
.kpi-top { display: flex; justify-content: space-between; align-items: center; }
.kpi-title { color: var(--muted); font-size: .8rem; font-weight: 700; }
.kpi-icon {
  width: 30px; height: 30px; border-radius: 10px; display: grid; place-items: center; font-size: .9rem;
  color: var(--tone); background: color-mix(in srgb, var(--tone) 16%, transparent);
  box-shadow: inset 0 1px 0 var(--gloss);
}
.kpi-value {
  font-family: 'Sora', sans-serif; font-weight: 700; font-size: 2rem; line-height: 1.15;
  margin-top: .45rem; color: var(--text); letter-spacing: -.03em;
}
.kpi-sub { color: var(--faint); font-size: .74rem; margin-top: .2rem; }

/* ---------- panels ---------- */
[class*="st-key-panel_"] {
  position: relative; isolation: isolate; overflow: hidden;
  background: var(--surface); border: 1px solid var(--border); border-radius: 18px;
  padding: 1.1rem 1.3rem 1.05rem; margin-bottom: .4rem;
  backdrop-filter: blur(14px) saturate(140%);
  box-shadow: var(--shadow), inset 0 1px 0 var(--gloss);
}
[class*="st-key-panel_"]::before {
  content: ""; position: absolute; inset: 0 0 60% 0; z-index: -1; pointer-events: none;
  background: linear-gradient(180deg, var(--gloss-soft), transparent);
}
.panel-title { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1rem; color: var(--text); }
.panel-subtitle { color: var(--muted); font-size: .78rem; margin: .1rem 0 .7rem; }
.section-title { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.15rem; color: var(--text); margin: 1.5rem 0 .8rem; }

/* ---------- alerts / pills / stats ---------- */
.alert {
  border: 1px solid var(--border); border-left: 3px solid var(--tone); border-radius: 12px;
  padding: .7rem .9rem; margin: .5rem 0; background: var(--surface2);
}
.alert-title { font-weight: 700; font-size: .86rem; color: var(--text); }
.alert-meta { color: var(--muted); font-size: .76rem; margin-top: .15rem; }
.pill { display: inline-block; padding: .16rem .6rem; border-radius: 999px; font-size: .72rem; font-weight: 700; white-space: nowrap; }
.stat-list { margin-top: .4rem; }
.stat-row { display: flex; justify-content: space-between; padding: .5rem 0; border-bottom: 1px solid var(--row-line); font-size: .84rem; color: var(--muted); }
.stat-row:last-child { border-bottom: 0; }
.stat-row strong { color: var(--text); }
.empty {
  border: 1.5px dashed var(--border); border-radius: 18px; padding: 2.2rem 1.5rem; text-align: center;
  color: var(--muted); background: var(--surface2);
}
.empty-title { font-family: 'Sora', sans-serif; font-weight: 700; color: var(--text); font-size: 1.05rem; margin-bottom: .3rem; }

/* ---------- progress ring ---------- */
.ring-wrap { display: flex; justify-content: center; padding: .4rem 0 .2rem; }
.ring {
  width: 148px; height: 148px; border-radius: 50%; display: grid; place-items: center;
  background: conic-gradient(var(--accent-strong) 0, var(--accent) calc(var(--p) * 1%), var(--track) 0);
  box-shadow: 0 0 34px color-mix(in srgb, var(--accent) 28%, transparent), inset 0 1px 0 var(--gloss);
}
.ring-inner {
  width: 112px; height: 112px; border-radius: 50%; display: grid; place-content: center; text-align: center;
  background: var(--surface-solid); box-shadow: inset 0 2px 10px rgba(0,0,0,.12);
}
.ring-inner span { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.8rem; color: var(--text); line-height: 1; }
.ring-inner small { color: var(--muted); font-size: .72rem; margin-top: .25rem; }

/* ---------- tables ---------- */
.tbl-wrap { overflow: auto; border: 1px solid var(--border); border-radius: 14px; background: var(--surface2); }
.tbl { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .82rem; }
.tbl th {
  position: sticky; top: 0; z-index: 1; background: var(--th); color: var(--muted); text-align: left;
  font-weight: 700; padding: .7rem .9rem; border-bottom: 1px solid var(--border); white-space: nowrap;
}
.tbl td {
  padding: .62rem .9rem; border-bottom: 1px solid var(--row-line); color: var(--text);
  max-width: 340px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.tbl tbody tr:hover td { background: var(--accent-soft); }
.tbl tbody tr:last-child td { border-bottom: 0; }
.dash { color: var(--faint); }

/* ---------- native widgets ---------- */
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="textarea"],
div[data-baseweb="select"] > div {
  background: var(--input-bg) !important; border-color: var(--border) !important; border-radius: 12px !important;
}
div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea, div[data-baseweb="base-input"] input {
  color: var(--text) !important; -webkit-text-fill-color: var(--text) !important;
}
div[data-baseweb="select"] * { color: var(--text) !important; }
div[data-baseweb="select"] svg { fill: var(--muted) !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] [data-baseweb="menu"] {
  background: var(--surface-solid) !important; border: 1px solid var(--border); border-radius: 12px;
}
div[data-baseweb="popover"] li { color: var(--text) !important; }
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] { background: var(--accent-soft) !important; }

.stButton > button, .stDownloadButton > button, [data-testid="stLinkButton"] a,
button[data-testid="stBaseButton-secondary"] {
  border-radius: 12px; font-weight: 700; border: 1px solid var(--border);
  background: linear-gradient(180deg, var(--btn-top), var(--btn-bot)); color: var(--text);
  box-shadow: inset 0 1px 0 var(--gloss), 0 4px 14px rgba(0,0,0,.10);
  transition: border-color .15s ease, transform .1s ease;
}
.stButton > button:hover, [data-testid="stLinkButton"] a:hover, button[data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--accent); color: var(--text);
}
button[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] {
  background: linear-gradient(180deg, #10B981 0%, #047857 100%) !important; color: #fff !important;
  border: 1px solid rgba(255,255,255,.22) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.45), 0 8px 22px rgba(4,120,87,.35) !important;
}
button[data-testid^="stBaseButton"] p, [data-testid="stLinkButton"] a p, .stButton > button p { color: inherit !important; }

button[data-baseweb="tab"] { color: var(--muted); font-weight: 700; }
button[data-baseweb="tab"] p { color: inherit !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent); }
div[data-baseweb="tab-highlight"] { background: var(--accent) !important; height: 3px; border-radius: 3px; }
div[data-baseweb="tab-border"] { background: var(--border) !important; }

[data-testid="stAlert"] {
  background: var(--surface); border: 1px solid var(--border); border-radius: 14px; color: var(--text);
}
[data-testid="stAlert"] p { color: var(--text); }

@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"], .main .block-container { padding: 1.1rem 1rem 2.5rem; }
  .hero { padding: 1.3rem 1.2rem; border-radius: 20px; }
  .hero h1 { font-size: 1.55rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""


def inject_custom_styles():
    t = T()
    pills = "".join(
        f".pill-{kind} {{ background: {bg}; color: {fg}; }}\n"
        for kind, (bg, fg) in t["pills"].items()
    )
    root = f"""
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@500;600;700&display=swap');
    :root {{
      color-scheme: {t['scheme']};
      --bg: {t['bg']}; --bg-a: {t['bg_a']}; --bg-b: {t['bg_b']};
      --surface: {t['surface']}; --surface-solid: {t['surface_solid']}; --surface2: {t['surface2']};
      --border: {t['border']}; --text: {t['text']}; --muted: {t['muted']}; --faint: {t['faint']};
      --accent: {t['accent']}; --accent-strong: {t['accent_strong']}; --accent-soft: {t['accent_soft']};
      --track: {t['track']}; --input-bg: {t['input_bg']}; --sidebar: {t['sidebar']};
      --shadow: {t['shadow']}; --gloss: {t['gloss']}; --gloss-soft: {t['gloss_soft']};
      --btn-top: {t['btn_top']}; --btn-bot: {t['btn_bot']}; --th: {t['th']}; --row-line: {t['row_line']};
    }}
    {pills}
    """
    st.markdown(f"<style>{root}{STATIC_CSS}</style>", unsafe_allow_html=True)


# =====================================================================
# 12. COMMON COMPONENTS
# =====================================================================

def render_global_filters(df: pd.DataFrame):
    if df.empty:
        return

    with panel("Filters", "Applied to every Jira view"):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            projects = ["All"] + sorted(df["project"].dropna().astype(str).unique().tolist())
            st.session_state["filter_project"] = st.selectbox(
                "Project", projects, key="global_project"
            )

        with c2:
            people = ["All"] + sorted(df["assignee"].dropna().astype(str).unique().tolist())
            st.session_state["filter_assignee"] = st.selectbox(
                "Assignee", people, key="global_assignee"
            )

        with c3:
            statuses = ["All"] + sorted(df["status"].dropna().astype(str).unique().tolist())
            st.session_state["filter_status"] = st.selectbox(
                "Status", statuses, key="global_status"
            )

        with c4:
            priorities = ["All"] + sorted(df["priority"].dropna().astype(str).unique().tolist())
            st.session_state["filter_priority"] = st.selectbox(
                "Priority", priorities, key="global_priority"
            )


def section(title: str):
    st.markdown(f'<div class="section-title">{esc(title)}</div>', unsafe_allow_html=True)


# =====================================================================
# 13. EXECUTIVE OVERVIEW
# =====================================================================

def render_executive_overview(df: pd.DataFrame, gitlab: Dict[str, Any]):
    render_hero(
        "Engineering overview",
        "Delivery progress, workload and risk across your Jira and GitLab activity.",
        f"Updated {datetime.now().strftime('%d %b %Y')}",
    )

    if df.empty:
        render_empty(
            "No Jira data loaded yet",
            "Open Data Sources, enter a project key and select Fetch Jira to fill this dashboard.",
        )
        return

    d = filtered_jira_df(df)
    if d.empty:
        st.warning("No issues match the current filters.")
        return

    total = len(d)
    done = int(d["is_done"].sum())
    in_progress = int(d["is_active"].sum())
    blocked = int(d["is_blocked"].sum())
    overdue = int(d["overdue"].sum())
    stale = int(d["is_stale"].sum())
    unassigned = int((d["assignee"] == "Unassigned").sum())
    progress_pct = round((done / total) * 100) if total else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi("Total work", total, "Filtered Jira issues", "sky", "▦")
    with k2:
        render_kpi("Completed", done, f"{progress_pct}% of visible work", "green", "✓")
    with k3:
        render_kpi("In progress", in_progress, "Active execution", "amber", "↻")
    with k4:
        render_kpi("Blocked", blocked, "Needs attention", "red", "!")
    with k5:
        render_kpi("Overdue", overdue, f"{stale} stale", "lime", "⏱")

    st.write("")
    a, b, c = st.columns([0.8, 1.25, 1.05])

    with a:
        with panel("Delivery progress", "Share of visible issues completed"):
            render_ring(progress_pct, "complete")
            sp_total = d["story_points"].sum()
            sp_done = d.loc[d["is_done"], "story_points"].sum()
            stat_rows(
                [
                    ("Issues done", f"{done} of {total}"),
                    ("Story points", f"{sp_done:g} of {sp_total:g}"),
                    ("Logged time", f"{d['time_spent_hours'].sum():.1f} h"),
                ]
            )

    with b:
        with panel("Work by status", "Current distribution of visible Jira work"):
            counts = d["status"].value_counts().rename_axis("Status").reset_index(name="Issues")
            show_chart(donut_chart(counts, "Status", "Issues", center=str(total), height=270))

    with c:
        with panel("Needs attention", "Signals worth raising at stand-up"):
            if overdue:
                render_alert(f"{overdue} overdue issue(s)", "Past the due date and not completed.", "danger")
            if blocked:
                render_alert(f"{blocked} blocked issue(s)", "Status or labels indicate blocked work.", "danger")
            if stale:
                render_alert(f"{stale} stale issue(s)", "No Jira update for five or more days.", "warning")
            if unassigned:
                render_alert(f"{unassigned} unassigned issue(s)", "Work has no current assignee.", "warning")
            if not any([overdue, blocked, stale, unassigned]):
                render_alert("Nothing needs attention", "Based on the Jira fields currently loaded.", "info")

    left, right = st.columns(2)
    with left:
        with panel("Created vs completed", "Weekly flow. Completion uses each issue's last update date."):
            show_chart(trend_chart(weekly_trend(d), "week", "Issues", "Series"))

    with right:
        with panel("Team workload", "Completed, open and blocked issues per assignee"):
            show_chart(
                stacked_hbar(
                    workload_long(d),
                    "assignee",
                    "Kind",
                    "Issues",
                    {
                        "Completed": T()["tones"]["green"],
                        "Open": T()["tones"]["sky"],
                        "Blocked": T()["tones"]["red"],
                    },
                    height=260,
                )
            )

    section("Recent Jira activity")
    recent = d.sort_values("updated", ascending=False).head(8)
    html_table(
        recent[["key", "summary", "status", "assignee", "priority", "updated"]],
        height=360,
    )


# =====================================================================
# 14. DELIVERY DASHBOARD
# =====================================================================

PRIORITY_ORDER = ["Highest", "Critical", "Blocker", "High", "Medium", "Low", "Lowest", "None"]


def render_delivery_dashboard(df: pd.DataFrame):
    render_hero(
        "Delivery dashboard",
        "Backlog composition, aging and delivery pressure at a glance.",
        "Jira delivery",
    )

    if df.empty:
        render_empty("No Jira data loaded yet", "Load a project from Data Sources first.")
        return

    d = filtered_jira_df(df)
    if d.empty:
        st.warning("No issues match the current filters.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("Backlog", int((~d["is_done"]).sum()), "Not completed", "sky", "▦")
    with c2:
        render_kpi("Completed", int(d["is_done"].sum()), "Completed issues", "green", "✓")
    with c3:
        render_kpi("Average age", f'{d["age_days"].mean():.1f}d', "Across visible issues", "amber", "◷")
    with c4:
        render_kpi("Logged time", f'{d["time_spent_hours"].sum():.1f}h', "Recorded work", "lime", "⏲")

    st.write("")
    a, b = st.columns(2)

    with a:
        with panel("Priority mix", "Current work by Jira priority"):
            pr = d["priority"].value_counts().rename_axis("Priority").reset_index(name="Issues")
            known = [p for p in PRIORITY_ORDER if p in pr["Priority"].values]
            extra = [p for p in pr["Priority"] if p not in known]
            tones = T()["tones"]
            cmap = {
                "Highest": tones["red"], "Critical": tones["red"], "Blocker": tones["red"],
                "High": tones["amber"], "Medium": tones["sky"], "Low": tones["teal"],
                "Lowest": tones["green"],
            }
            show_chart(hbar_chart(pr, "Priority", "Issues", order=known + extra, color_map=cmap))

    with b:
        with panel("Issue types", "Where work is concentrated"):
            ty = d["issue_type"].value_counts().rename_axis("Type").reset_index(name="Issues")
            show_chart(donut_chart(ty, "Type", "Issues", center=str(len(d)), height=250))

    a, b = st.columns(2)

    with a:
        with panel("Issue aging", "How long visible work has existed"):
            labels = ["0–2d", "3–7d", "8–14d", "15–30d", "30d+"]
            bins = pd.cut(d["age_days"], bins=[-1, 2, 7, 14, 30, float("inf")], labels=labels)
            aging = bins.value_counts().reindex(labels).fillna(0).astype(int)
            aging = aging.rename_axis("Age").reset_index(name="Issues")
            show_chart(vbar_chart(aging, "Age", "Issues", labels))

    with b:
        with panel("Update freshness", "Time since each issue was last touched"):
            labels = ["0–1d", "2–3d", "4–5d", "6–10d", "10d+"]
            fr = pd.cut(d["days_since_update"], bins=[-1, 1, 3, 5, 10, float("inf")], labels=labels)
            fresh = fr.value_counts().reindex(labels).fillna(0).astype(int)
            fresh = fresh.rename_axis("Idle").reset_index(name="Issues")
            show_chart(vbar_chart(fresh, "Idle", "Issues", labels))

    with panel("Age vs. idle time", "Open issues. Points above the dashed line have been idle for five days or more."):
        show_chart(scatter_chart(d[~d["is_done"]], height=320))

    section("Work needing review")
    attention = d[(d["overdue"]) | (d["is_blocked"]) | (d["is_stale"])].copy()
    if attention.empty:
        st.success("No overdue, blocked or stale issues in the current filter.")
    else:
        attention["reason"] = attention.apply(
            lambda r: ", ".join(
                [
                    x
                    for x, flag in [
                        ("Overdue", r["overdue"]),
                        ("Blocked", r["is_blocked"]),
                        ("Stale", r["is_stale"]),
                    ]
                    if flag
                ]
            ),
            axis=1,
        )
        html_table(
            attention[
                ["key", "summary", "assignee", "status", "priority", "reason", "days_since_update"]
            ].sort_values("days_since_update", ascending=False)
        )


# =====================================================================
# 15. TEAM DASHBOARD
# =====================================================================

def render_team_dashboard(df: pd.DataFrame):
    render_hero(
        "Team dashboard",
        "Workload and delivery signals for team-level coordination.",
        "Team view",
    )

    if df.empty:
        render_empty("No Jira data loaded yet", "Load a project from Data Sources first.")
        return

    d = filtered_jira_df(df)
    if d.empty:
        st.warning("No issues match the current filters.")
        return

    team = (
        d.groupby("assignee")
        .agg(
            Issues=("key", "count"),
            Completed=("is_done", "sum"),
            In_Progress=("is_active", "sum"),
            Blocked=("is_blocked", "sum"),
            Overdue=("overdue", "sum"),
            Stale=("is_stale", "sum"),
            Logged_Hours=("time_spent_hours", "sum"),
        )
        .reset_index()
        .sort_values(["Blocked", "Issues"], ascending=[False, False])
    )
    team["Logged_Hours"] = team["Logged_Hours"].round(1)

    left, right = st.columns([1.25, 1])
    with left:
        with panel("Team overview", "One row per assignee"):
            html_table(team, height=340)

    with right:
        with panel("Workload split", "Completed, open and blocked issues"):
            show_chart(
                stacked_hbar(
                    workload_long(d),
                    "assignee",
                    "Kind",
                    "Issues",
                    {
                        "Completed": T()["tones"]["green"],
                        "Open": T()["tones"]["sky"],
                        "Blocked": T()["tones"]["red"],
                    },
                    height=300,
                )
            )

    a, b = st.columns(2)
    with a:
        with panel("Logged hours", "Recorded time per person"):
            hrs = team[["assignee", "Logged_Hours"]].rename(columns={"Logged_Hours": "Hours"})
            show_chart(hbar_chart(hrs, "assignee", "Hours"))
    with b:
        with panel("Attention signals", "Overdue, blocked and stale open work"):
            sig = team.melt(
                id_vars="assignee",
                value_vars=["Blocked", "Overdue", "Stale"],
                var_name="Kind",
                value_name="Issues",
            )
            tones = T()["tones"]
            show_chart(
                stacked_hbar(
                    sig, "assignee", "Kind", "Issues",
                    {"Blocked": tones["red"], "Overdue": tones["amber"], "Stale": tones["lime"]},
                )
            )

    section("Individual drill-down")
    selected_person = st.selectbox(
        "Select team member",
        team["assignee"].tolist(),
        key="team_person_selector",
    )
    render_person_detail(d, selected_person)


def render_person_detail(df: pd.DataFrame, person: str):
    p = df[df["assignee"] == person].copy()

    if p.empty:
        st.warning("No issues found for this team member.")
        return

    total = len(p)
    done = int(p["is_done"].sum())
    active = int(p["is_active"].sum())
    blocked = int(p["is_blocked"].sum())
    overdue = int(p["overdue"].sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Issues", total, f"Assigned to {person}", "sky", "▦")
    with k2:
        render_kpi("Completed", done, "Completed", "green", "✓")
    with k3:
        render_kpi("Active", active, "In progress or review", "amber", "↻")
    with k4:
        render_kpi("Attention", blocked + overdue, f"{blocked} blocked, {overdue} overdue", "red", "!")

    st.write("")
    left, right = st.columns([1.6, 1])

    with left:
        with panel("Current work", "Open issues, most urgent first"):
            current = p[~p["is_done"]].sort_values(
                ["is_blocked", "overdue", "updated"], ascending=[False, False, False]
            )
            if current.empty:
                st.success("No open work in the current filter.")
            else:
                html_table(
                    current[["key", "summary", "status", "priority", "updated", "days_since_update"]],
                    height=340,
                )

    with right:
        with panel("Work profile", "Issues by status"):
            prof = p["status"].value_counts().rename_axis("Status").reset_index(name="Issues")
            show_chart(donut_chart(prof, "Status", "Issues", center=str(total), height=260))


# =====================================================================
# 16. RISKS & ATTENTION
# =====================================================================

def render_risks(df: pd.DataFrame):
    render_hero(
        "Risks & attention",
        "A focused queue of issues that may need manager follow-up.",
        "Delivery risk",
    )

    if df.empty:
        render_empty("No Jira data loaded yet", "Load a project from Data Sources first.")
        return

    d = filtered_jira_df(df)
    if d.empty:
        st.warning("No issues match the current filters.")
        return

    categories = {
        "Overdue": d[d["overdue"]],
        "Blocked": d[d["is_blocked"]],
        "Stale": d[d["is_stale"]],
        "Unassigned": d[d["assignee"] == "Unassigned"],
    }
    tones = {"Overdue": "red", "Blocked": "red", "Stale": "amber", "Unassigned": "lime"}
    icons = {"Overdue": "⏱", "Blocked": "⛔", "Stale": "◷", "Unassigned": "?"}

    cols = st.columns(4)
    for i, (name, subset) in enumerate(categories.items()):
        with cols[i]:
            render_kpi(name, len(subset), "Requires review", tones[name], icons[name])

    st.write("")
    left, right = st.columns([1, 1.3])

    with left:
        with panel("Risk mix", "Share of flagged issues by signal"):
            mix = pd.DataFrame(
                {"Signal": list(categories.keys()), "Issues": [len(s) for s in categories.values()]}
            )
            mix = mix[mix["Issues"] > 0]
            if mix.empty:
                st.success("No risk signals in the current filter.")
            else:
                tn = T()["tones"]
                show_chart(
                    donut_chart(
                        mix, "Signal", "Issues", center=str(int(mix["Issues"].sum())),
                        colors={"Overdue": tn["red"], "Blocked": tn["amber"], "Stale": tn["lime"], "Unassigned": tn["sky"]},
                    )
                )

    with right:
        with panel("Risk by assignee", "Where flagged work sits"):
            rows = []
            for person, g in d.groupby("assignee"):
                rows += [
                    {"assignee": person, "Signal": "Overdue", "Issues": int(g["overdue"].sum())},
                    {"assignee": person, "Signal": "Blocked", "Issues": int(g["is_blocked"].sum())},
                    {"assignee": person, "Signal": "Stale", "Issues": int(g["is_stale"].sum())},
                ]
            rdf = pd.DataFrame(rows)
            if rdf.empty or rdf["Issues"].sum() == 0:
                st.success("No flagged work per assignee.")
            else:
                tn = T()["tones"]
                show_chart(
                    stacked_hbar(
                        rdf, "assignee", "Signal", "Issues",
                        {"Overdue": tn["red"], "Blocked": tn["amber"], "Stale": tn["lime"]},
                        height=260,
                    )
                )

    section("Issue queues")
    tabs = st.tabs([f"{name} ({len(subset)})" for name, subset in categories.items()])
    for tab, (name, subset) in zip(tabs, categories.items()):
        with tab:
            if subset.empty:
                st.success(f"No {name.lower()} issues in the current filter.")
            else:
                html_table(
                    subset[["key", "summary", "assignee", "status", "priority", "updated"]]
                )


# =====================================================================
# 17. ISSUE EXPLORER
# =====================================================================

def render_issue_explorer(df: pd.DataFrame):
    render_hero(
        "Jira issue explorer",
        "Search, inspect and drill into individual Jira work items.",
        "Jira workspace",
    )

    if df.empty:
        render_empty("No Jira data loaded yet", "Load a project from Data Sources first.")
        return

    d = filtered_jira_df(df)

    search = st.text_input(
        "Search issues",
        placeholder="Search by key, summary, assignee or description",
    )

    result = d.copy()

    if search.strip():
        q = search.strip().lower()
        mask = (
            result["key"].astype(str).str.lower().str.contains(q, na=False, regex=False)
            | result["summary"].astype(str).str.lower().str.contains(q, na=False, regex=False)
            | result["assignee"].astype(str).str.lower().str.contains(q, na=False, regex=False)
            | result["description"].astype(str).str.lower().str.contains(q, na=False, regex=False)
        )
        result = result[mask]

    st.caption(f"{len(result)} issue(s)")

    if result.empty:
        st.warning("No matching issues.")
        return

    with panel("Results", "Sorted by most recently updated"):
        html_table(
            result.sort_values("updated", ascending=False)[
                ["key", "summary", "status", "assignee", "priority", "issue_type", "updated", "overdue"]
            ],
            height=360,
        )

    selected_key = st.selectbox(
        "Open issue",
        result["key"].tolist(),
        key="issue_detail_selector",
    )

    issue = get_selected_issue(result, selected_key)
    if issue is None:
        return

    st.markdown(
        f'<div class="hero"><div class="hero-body">'
        f'<span class="hero-chip"><i></i>{esc(issue["key"])} · {esc(issue["issue_type"])}</span>'
        f'<h1>{esc(issue["summary"])}</h1>'
        f'<p>{esc(issue["assignee"])} · {esc(issue["priority"])} · {esc(issue["status"])}</p>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Age", f"{int(issue['age_days'])}d", "Since creation", "sky", "◷")
    with k2:
        render_kpi("Logged", f"{issue['time_spent_hours']:.1f}h", "Recorded time", "lime", "⏲")
    with k3:
        render_kpi("Freshness", f"{int(issue['days_since_update'])}d", "Since last update", "amber", "↻")
    with k4:
        render_kpi("Story points", f"{issue['story_points']:g}", "If configured", "green", "★")

    st.write("")
    left, right = st.columns([1.5, 1])

    with left:
        with panel("Description"):
            st.write(issue["description"] or "No description available.")

    with right:
        with panel("Issue context"):
            stat_rows(
                [
                    ("Project", issue["project"]),
                    ("Sprint", issue["sprint"]),
                    ("Reporter", issue["reporter"]),
                    ("Due date", issue["due_date"].strftime("%Y-%m-%d") if pd.notna(issue["due_date"]) else "Not set"),
                    ("Labels", issue["labels"] or "None"),
                ]
            )

    if issue["url"]:
        st.link_button("Open in Jira", issue["url"])


# =====================================================================
# 18. ENGINEERING / GITLAB
# =====================================================================

def render_engineering(gitlab_projects: List[Dict[str, Any]], telemetry: Dict[str, Any], config):
    render_hero(
        "Engineering activity",
        "Connect delivery work with merge requests and repository activity.",
        "Jira + GitLab",
    )

    if not gitlab_projects:
        render_empty(
            "No repositories scanned yet",
            "Open Data Sources and select Scan accessible GitLab projects.",
        )
        return

    options = {
        f"{p.get('name_with_namespace', p.get('name', 'Repository'))} (ID {p.get('id')})": p["id"]
        for p in gitlab_projects
    }

    c1, c2 = st.columns([3, 1])
    with c1:
        selected = st.selectbox("Repository", list(options.keys()), key="engineering_repo")
    with c2:
        st.write("")
        st.write("")
        sync = st.button("Sync repository telemetry", type="primary")

    if sync:
        with st.spinner("Fetching merge requests and commits..."):
            try:
                service = GitLabService(config)
                tel = service.fetch_deep_telemetry(options[selected])
                st.session_state["gitlab_telemetry"] = {
                    "merge_requests": [asdict(x) for x in tel["merge_requests"]],
                    "commits": [asdict(x) for x in tel["commits"]],
                }
                telemetry = st.session_state["gitlab_telemetry"]
                st.success("Repository telemetry synced.")
            except Exception as exc:
                st.error(f"GitLab error: {exc}")

    mr = pd.DataFrame(telemetry.get("merge_requests", []))
    commits = pd.DataFrame(telemetry.get("commits", []))

    open_mr = int((mr["state"] == "opened").sum()) if not mr.empty else 0
    merged_mr = int((mr["state"] == "merged").sum()) if not mr.empty else 0

    avg_merge = "–"
    if not mr.empty:
        merged = mr[mr["state"] == "merged"].copy()
        if not merged.empty:
            days = (
                pd.to_datetime(merged["merged_at"], errors="coerce")
                - pd.to_datetime(merged["created_at"], errors="coerce")
            ).dt.days
            if days.notna().any():
                avg_merge = f"{days.mean():.1f}d"

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi("Merge requests", len(mr), "Loaded", "sky", "⑂")
    with k2:
        render_kpi("Open", open_mr, "Review queue", "amber", "◔")
    with k3:
        render_kpi("Merged", merged_mr, "Completed reviews", "green", "✓")
    with k4:
        render_kpi("Avg time to merge", avg_merge, "Created to merged", "teal", "⏱")
    with k5:
        render_kpi("Commits", len(commits), "Recent activity", "lime", "●")

    st.write("")
    a, b, c = st.columns([1, 1, 1.2])

    with a:
        with panel("Merge request states"):
            if mr.empty:
                st.caption("No merge requests loaded.")
            else:
                st_counts = mr["state"].value_counts().rename_axis("State").reset_index(name="MRs")
                tn = T()["tones"]
                show_chart(
                    donut_chart(
                        st_counts, "State", "MRs", center=str(len(mr)),
                        colors={"merged": tn["green"], "opened": tn["amber"], "closed": tn["red"]},
                    )
                )

    with b:
        with panel("Top authors", "Merge requests per author"):
            if mr.empty:
                st.caption("No merge requests loaded.")
            else:
                au = mr["author"].value_counts().head(8).rename_axis("Author").reset_index(name="MRs")
                show_chart(hbar_chart(au, "Author", "MRs"))

    with c:
        with panel("Commit activity", "Commits per day"):
            if commits.empty:
                st.caption("No commits loaded.")
            else:
                cd = commits.copy()
                cd["day"] = pd.to_datetime(cd["created_at"], errors="coerce")
                daily = cd.dropna(subset=["day"]).groupby("day").size().reset_index(name="Commits")
                show_chart(daily_area_chart(daily, "day", "Commits"))

    section("Details")
    t1, t2 = st.tabs(["Merge requests", "Commits"])

    with t1:
        if mr.empty:
            st.caption("No merge requests loaded.")
        else:
            html_table(
                mr[
                    [
                        "mr_id", "title", "state", "author", "assignee",
                        "created_at", "updated_at", "merged_at",
                        "source_branch", "target_branch",
                    ]
                ]
            )

    with t2:
        if commits.empty:
            st.caption("No commits loaded.")
        else:
            html_table(commits[["commit_id", "title", "author", "created_at"]])


# =====================================================================
# 19. AI INSIGHTS
# =====================================================================

def render_ai_insights(df: pd.DataFrame, telemetry: Dict[str, Any], config: AppConfig):
    render_hero(
        "AI engineering insights",
        "Turn Jira and GitLab telemetry into concise management briefings.",
        "AI assistant",
    )

    if df.empty and not telemetry:
        render_empty(
            "No telemetry loaded yet",
            "Load Jira issues or GitLab activity from Data Sources, then come back to generate a briefing.",
        )
        return

    templates = {
        "Daily manager briefing": (
            "Create a concise manager briefing. Cover delivery status, workload "
            "distribution, blockers, stale work, overdue work, and code review activity."
        ),
        "Sprint risk review": (
            "Identify delivery risks visible in the telemetry. Separate direct facts "
            "from possible explanations and list the issues that deserve follow-up."
        ),
        "Stand-up summary": (
            "Create a short stand-up summary organized as: completed, in progress, "
            "blocked, stale/attention, and review queue."
        ),
        "Jira + GitLab correlation": (
            "Explain useful relationships between Jira delivery work and GitLab activity. "
            "Highlight where issue status and engineering activity may be out of sync."
        ),
    }

    with panel("Briefing setup", "Choose a template and add optional focus"):
        selected = st.selectbox("Analysis template", list(templates.keys()))
        custom = st.text_area(
            "Additional instruction",
            placeholder="Optional: focus on a specific sprint, team or delivery concern.",
            height=100,
        )
        generate = st.button("Generate management briefing", type="primary")

    if generate:
        blocks = []

        if not df.empty:
            compact = df[
                [
                    "key", "summary", "status", "assignee", "priority", "created",
                    "updated", "due_date", "time_spent_hours", "is_blocked",
                    "is_stale", "overdue",
                ]
            ].copy()
            compact["created"] = compact["created"].astype(str)
            compact["updated"] = compact["updated"].astype(str)
            compact["due_date"] = compact["due_date"].astype(str)
            blocks.append("=== JIRA ===\n" + compact.to_json(orient="records"))

        if telemetry:
            blocks.append("=== GITLAB ===\n" + json.dumps(telemetry, default=str))

        prompt = templates[selected]
        if custom.strip():
            prompt += "\nAdditional instruction: " + custom.strip()

        try:
            with st.spinner("Analyzing engineering telemetry..."):
                service = LLMAssistantService(config)
                result = service.generate_analysis(prompt, "\n\n".join(blocks))

            with panel("Management briefing", selected):
                st.markdown(result)

        except Exception as exc:
            st.error(f"AI analysis failed: {exc}")


# =====================================================================
# 20. DATA SOURCES
# =====================================================================

def render_data_sources(config: AppConfig):
    render_hero(
        "Data sources",
        "Load the Jira and GitLab data that powers every dashboard.",
        "Integrations",
    )

    def conn(label: str, ok: bool) -> str:
        return pill(f"{label} connected" if ok else f"{label} not configured", "success" if ok else "danger")

    st.markdown(
        conn("Jira", bool(config.jira_api_token and config.jira_base_url))
        + " "
        + conn("GitLab", bool(config.gitlab_private_token and config.gitlab_base_url))
        + " "
        + conn("LLM", bool(config.llm_api_key)),
        unsafe_allow_html=True,
    )
    st.write("")

    with panel("Jira", "Fetch up to 100 issues from a project"):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            project_key = st.text_input(
                "Project key", placeholder="e.g. CORE", key="jira_project_key"
            ).strip().upper()

        with c2:
            jql = st.text_input(
                "Additional JQL",
                placeholder="status != Done AND priority = High",
                key="jira_jql",
            )

        with c3:
            st.write("")
            st.write("")
            fetch = st.button("Fetch Jira", type="primary")

        if fetch:
            if not project_key:
                st.warning("Enter a Jira project key.")
            else:
                try:
                    with st.spinner("Fetching Jira issues..."):
                        service = JiraService(config)
                        issues = service.fetch_issues(project_key, jql)
                        st.session_state["jira_issues"] = [asdict(issue) for issue in issues]
                        st.session_state["jira_project"] = project_key
                    st.success(f"Loaded {len(issues)} Jira issues.")
                except Exception as exc:
                    st.error(f"Jira fetch failed: {exc}")

    with panel("GitLab", "Scan repositories you are a member of"):
        if st.button("Scan accessible GitLab projects"):
            try:
                with st.spinner("Scanning GitLab projects..."):
                    service = GitLabService(config)
                    projects = service.fetch_projects()
                    st.session_state["gitlab_projects"] = projects
                st.success(f"Found {len(projects)} repositories.")
            except Exception as exc:
                st.error(f"GitLab scan failed: {exc}")

        if st.session_state.get("gitlab_projects"):
            html_table(
                pd.DataFrame(
                    [
                        {
                            "Project": p.get("name_with_namespace", p.get("name")),
                            "ID": p.get("id"),
                            "Default branch": p.get("default_branch"),
                            "Visibility": p.get("visibility"),
                        }
                        for p in st.session_state["gitlab_projects"]
                    ]
                ),
                height=300,
            )

    if st.button("Clear loaded telemetry"):
        st.session_state["jira_issues"] = []
        st.session_state["gitlab_projects"] = []
        st.session_state["gitlab_telemetry"] = {}
        st.success("Loaded telemetry cleared.")
        st.rerun()


# =====================================================================
# 21. SETTINGS
# =====================================================================

def render_settings(config: AppConfig):
    render_hero(
        "System settings",
        "Connection status and runtime controls.",
        "Platform",
    )

    a, b = st.columns(2)

    with a:
        with panel("Jira connection"):
            st.text_input("Base URL", value=config.jira_base_url, disabled=True, key="settings_jira_base_url")
            st.text_input("Account", value=config.jira_email, disabled=True, key="settings_jira_email")
            st.text_input(
                "API token",
                value="Configured" if config.jira_api_token else "Not configured",
                disabled=True,
                key="settings_jira_api_token",
            )

        with panel("GitLab connection"):
            st.text_input("Base URL", value=config.gitlab_base_url, disabled=True, key="settings_gitlab_base_url")
            st.text_input(
                "Private token",
                value="Configured" if config.gitlab_private_token else "Not configured",
                disabled=True,
                key="settings_gitlab_private_token",
            )

    with b:
        with panel("LLM"):
            st.text_input("Endpoint", value=config.llm_base_url, disabled=True, key="settings_llm_endpoint")
            st.text_input("Model", value=config.llm_model, disabled=True, key="settings_llm_model")
            st.text_input(
                "API key",
                value="Configured" if config.llm_api_key else "Not configured",
                disabled=True,
                key="settings_llm_api_key",
            )

        with panel("Runtime", "Reset everything loaded in this browser session"):
            if st.button("Clear session state", key="settings_clear_session"):
                st.session_state.clear()
                st.success("Runtime state cleared.")
                st.rerun()


# =====================================================================
# 22. MAIN APPLICATION
# =====================================================================

def main():
    st.set_page_config(
        page_title="Engineering Intelligence Hub",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _PANEL_N[0] = 0
    st.session_state.setdefault("dark_mode", True)

    inject_custom_styles()

    config = AppConfig.load_from_env_or_secrets()

    st.session_state.setdefault("jira_issues", [])
    st.session_state.setdefault("gitlab_projects", [])
    st.session_state.setdefault("gitlab_telemetry", {})
    st.session_state.setdefault("filter_project", "All")
    st.session_state.setdefault("filter_assignee", "All")
    st.session_state.setdefault("filter_status", "All")
    st.session_state.setdefault("filter_priority", "All")

    jira_df = prepare_jira_dataframe(st.session_state["jira_issues"])

    def render_executive_page():
        render_global_filters(jira_df)
        render_executive_overview(jira_df, st.session_state["gitlab_telemetry"])

    def render_delivery_page():
        render_global_filters(jira_df)
        render_delivery_dashboard(jira_df)

    def render_team_page():
        render_global_filters(jira_df)
        render_team_dashboard(jira_df)

    def render_risks_page():
        render_global_filters(jira_df)
        render_risks(jira_df)

    def render_issue_explorer_page():
        render_global_filters(jira_df)
        render_issue_explorer(jira_df)

    def render_activity_page():
        render_engineering(
            st.session_state["gitlab_projects"],
            st.session_state["gitlab_telemetry"],
            config,
        )

    def render_ai_page():
        render_ai_insights(jira_df, st.session_state["gitlab_telemetry"], config)

    def render_data_sources_page():
        render_data_sources(config)

    def render_settings_page():
        render_settings(config)

    pages = [
        st.Page(render_executive_page, title="Executive Overview", icon="🏠", url_path="overview", default=True),
        st.Page(render_delivery_page, title="Delivery Dashboard", icon="📊", url_path="delivery"),
        st.Page(render_team_page, title="Team Dashboard", icon="👥", url_path="team"),
        st.Page(render_risks_page, title="Risks & Attention", icon="⚠️", url_path="risks"),
        st.Page(render_issue_explorer_page, title="Jira Issue Explorer", icon="🔎", url_path="issues"),
        st.Page(render_activity_page, title="Engineering Activity", icon="🔀", url_path="activity"),
        st.Page(render_ai_page, title="AI Insights", icon="🤖", url_path="ai"),
        st.Page(render_data_sources_page, title="Data Sources", icon="🔌", url_path="sources"),
        st.Page(render_settings_page, title="Settings", icon="⚙️", url_path="settings"),
    ]

    navigation = st.navigation(pages, position="sidebar")

    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">⚡</div>'
            '<div><div class="brand-name">Engineering Hub</div>'
            '<div class="brand-sub">Delivery intelligence</div></div></div>',
            unsafe_allow_html=True,
        )
        st.toggle("Dark mode", key="dark_mode")
        st.markdown("---")
        st.markdown(
            f'<div class="side-stats"><strong>Loaded data</strong><br>'
            f'Jira issues: <strong>{len(st.session_state["jira_issues"])}</strong><br>'
            f'GitLab MRs: <strong>{len(st.session_state["gitlab_telemetry"].get("merge_requests", []))}</strong><br>'
            f'GitLab commits: <strong>{len(st.session_state["gitlab_telemetry"].get("commits", []))}</strong></div>',
            unsafe_allow_html=True,
        )

    navigation.run()


if __name__ == "__main__":
    main()

