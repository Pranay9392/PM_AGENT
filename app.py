"""
Engineering Intelligence Hub
Jira + GitLab analytics for Scrum Masters, Project / Program Managers,
Engineering Managers and Delivery Leads.

Run:  streamlit run engineering_hub.py
"""
import html
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
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
    jira_sprint_field: str
    jira_story_points_field: str
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
            jira_sprint_field=os.getenv(
                "JIRA_SPRINT_FIELD", fetch("Jira", "SPRINT_FIELD", "customfield_10020")
            ),
            jira_story_points_field=os.getenv(
                "JIRA_STORY_POINTS_FIELD",
                fetch("Jira", "STORY_POINTS_FIELD", "customfield_10016"),
            ),
            gitlab_base_url=os.getenv("GITLAB_BASE_URL", fetch("GitLab", "BASE_URL")),
            gitlab_private_token=os.getenv(
                "GITLAB_PRIVATE_TOKEN", fetch("GitLab", "PRIVATE_TOKEN")
            ),
            llm_base_url=os.getenv(
                "LLM_BASE_URL", fetch("LLM", "BASE_URL", "https://api.openai.com/v1")
            ),
            llm_api_key=os.getenv("LLM_API_KEY", fetch("LLM", "API_KEY")),
            llm_model=os.getenv("LLM_MODEL", fetch("LLM", "MODEL", "gpt-4o")),
        )


# =====================================================================
# 2. SHARED HELPERS
# =====================================================================

DONE_NAMES = {"done", "closed", "resolved", "complete", "completed"}
ACTIVE_PATTERN = "progress|review|testing"
STARTED_RE = re.compile(r"progress|review|testing|develop|\bqa\b|verif", re.I)
STRICT_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d+\b")
LOOSE_KEY_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]{1,9})-(\d+)(?![A-Za-z0-9])")


def seed(key: str, value: Any) -> None:
    """Initialise a widget key once (widget state is dropped when a page is left)."""
    if key not in st.session_state:
        st.session_state[key] = value


def pct(n: float, d: float) -> int:
    return int(round(100 * n / d)) if d else 0


def fmt_hours(h: Any) -> str:
    if h is None or pd.isna(h):
        return "–"
    return f"{h:.1f}h" if h < 48 else f"{h / 24:.1f}d"


def fmt_days(x: Any) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:.1f}d"


def utc_now_naive() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def extract_issue_keys(text: str, projects: Optional[set] = None) -> List[str]:
    if not text:
        return []
    if projects:
        out: List[str] = []
        for m in LOOSE_KEY_RE.finditer(text):
            prefix = m.group(1).upper()
            key = f"{prefix}-{m.group(2)}"
            if prefix in projects and key not in out:
                out.append(key)
        return out
    return list(dict.fromkeys(STRICT_KEY_RE.findall(text)))


def adf_to_text(node: Any) -> str:
    """Flatten an Atlassian Document Format tree into readable plain text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(adf_to_text(n) for n in node)
    if isinstance(node, dict):
        kind = node.get("type")
        if kind == "text":
            return node.get("text", "")
        if kind == "hardBreak":
            return "\n"
        if kind == "mention":
            return (node.get("attrs") or {}).get("text", "@user")
        inner = adf_to_text(node.get("content", []))
        if kind in ("paragraph", "heading", "blockquote", "codeBlock"):
            return inner + "\n"
        if kind == "listItem":
            return "• " + inner.strip() + "\n"
        return inner
    return ""


def parse_sprints(value: Any) -> List[Dict[str, str]]:
    """Normalise Jira sprint field values (Cloud dicts or legacy strings)."""
    out: List[Dict[str, str]] = []
    if not value:
        return out
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, dict):
            out.append(
                {
                    "id": str(item.get("id", "")),
                    "name": str(item.get("name", "")),
                    "state": str(item.get("state", "")).lower(),
                    "start": str(item.get("startDate") or "")[:10],
                    "end": str(item.get("endDate") or item.get("completeDate") or "")[:10],
                    "goal": str(item.get("goal") or ""),
                }
            )
        elif isinstance(item, str):

            def g(k: str) -> str:
                m = re.search(rf"{k}=([^,\]]*)", item)
                return "" if not m or m.group(1) == "<null>" else m.group(1)

            out.append(
                {
                    "id": g("id"),
                    "name": g("name"),
                    "state": g("state").lower(),
                    "start": g("startDate")[:10],
                    "end": (g("endDate") or g("completeDate"))[:10],
                    "goal": g("goal"),
                }
            )
    return [s for s in out if s["name"]]


def primary_sprint(sprints: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Active sprint wins, then the newest future sprint, then the newest closed sprint."""
    if not sprints:
        return None

    def order(s: Dict[str, str]):
        sid = int(s["id"]) if s["id"].isdigit() else 0
        return (s["end"] or "", sid)

    for state in ("active", "future"):
        pool = [s for s in sprints if s["state"] == state]
        if pool:
            return sorted(pool, key=order)[-1]
    return sorted(sprints, key=order)[-1]


def analyse_changelog(raw: Dict[str, Any]) -> Tuple[str, str, int]:
    """Return (first_started, status_since, reopen_count) from the expanded changelog."""
    histories = (raw.get("changelog") or {}).get("histories") or []
    first_started, status_since, reopen = "", "", 0
    for h in sorted(histories, key=lambda x: x.get("created", "")):
        ts = (h.get("created") or "")[:19]
        for item in h.get("items", []):
            if item.get("field") != "status":
                continue
            to = item.get("toString") or ""
            frm = item.get("fromString") or ""
            status_since = ts
            if not first_started and STARTED_RE.search(to):
                first_started = ts
            if frm.lower() in DONE_NAMES and to.lower() not in DONE_NAMES:
                reopen += 1
    return first_started, status_since, reopen


# =====================================================================
# 3. DATA MODELS
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
    sprint_state: str
    sprint_start: str
    sprint_end: str
    sprint_goal: str
    sprint_count: int
    sprint_data: str
    labels: str
    components: str
    fix_versions: str
    parent_key: str
    parent_summary: str
    created: str
    updated: str
    resolved: str
    first_started: str
    status_since: str
    due_date: str
    description: str
    time_spent_hours: float
    original_estimate_hours: float
    remaining_hours: float
    story_points: float
    reopen_count: int
    url: str


@dataclass
class GitLabMRDTO:
    mr_id: int
    title: str
    state: str
    author: str
    assignee: Optional[str]
    reviewers: str
    created_at: str
    updated_at: str
    merged_at: Optional[str]
    closed_at: Optional[str]
    source_branch: str
    target_branch: str
    description: str
    labels: str
    draft: bool
    has_conflicts: bool
    notes: int
    web_url: str


@dataclass
class GitLabCommitDTO:
    commit_id: str
    title: str
    author: str
    created_at: str
    message: str
    additions: int
    deletions: int
    web_url: str


@dataclass
class GitLabPipelineDTO:
    pipeline_id: int
    status: str
    ref: str
    source: str
    created_at: str
    updated_at: str
    web_url: str


# =====================================================================
# 4. HTTP CLIENT
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


def raise_for_response(res: requests.Response, what: str) -> None:
    if res.status_code >= 400:
        raise RuntimeError(f"{what} failed: HTTP {res.status_code} – {res.text[:240]}")


# =====================================================================
# 5. JIRA SERVICE
# =====================================================================

class JiraService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.session = RobustHTTPClient.create_session()
        self.auth = (config.jira_email, config.jira_api_token)
        self.headers = {"Accept": "application/json", "Content-Type": "application/json"}
        self.base = config.jira_base_url.rstrip("/")

    def test_connection(self) -> str:
        res = self.session.get(
            f"{self.base}/rest/api/3/myself", headers=self.headers, auth=self.auth, timeout=20
        )
        raise_for_response(res, "Jira connection")
        return res.json().get("displayName", "connected")

    def _sp_fields(self) -> List[str]:
        return list(dict.fromkeys([self.config.jira_story_points_field, "customfield_10026"]))

    def _fields(self) -> List[str]:
        return [
            "summary", "status", "assignee", "reporter", "priority", "issuetype", "project",
            "created", "updated", "resolutiondate", "duedate", "description", "timespent",
            "timeoriginalestimate", "timeestimate", "labels", "components", "fixVersions",
            "parent", self.config.jira_sprint_field,
        ] + self._sp_fields()

    def _search_enhanced(self, jql: str, limit: int, expand: bool) -> Optional[List[Dict[str, Any]]]:
        """New cursor-based endpoint. Returns None if the site only supports the legacy one."""
        url = f"{self.base}/rest/api/3/search/jql"
        page = 50 if expand else 100
        issues: List[Dict[str, Any]] = []
        token: Optional[str] = None
        while len(issues) < limit:
            params: Dict[str, Any] = {
                "jql": jql,
                "maxResults": min(page, limit - len(issues)),
                "fields": ",".join(self._fields()),
            }
            if expand:
                params["expand"] = "changelog"
            if token:
                params["nextPageToken"] = token
            res = self.session.get(url, headers=self.headers, auth=self.auth, params=params, timeout=60)
            if res.status_code in (404, 410) and not issues and token is None:
                return None
            raise_for_response(res, "Jira search")
            data = res.json()
            batch = data.get("issues", [])
            issues.extend(batch)
            token = data.get("nextPageToken")
            if not batch or not token or data.get("isLast"):
                break
        return issues[:limit]

    def _search_legacy(self, jql: str, limit: int, expand: bool) -> List[Dict[str, Any]]:
        url = f"{self.base}/rest/api/3/search"
        page = 50 if expand else 100
        issues: List[Dict[str, Any]] = []
        start = 0
        while len(issues) < limit:
            params: Dict[str, Any] = {
                "jql": jql,
                "startAt": start,
                "maxResults": min(page, limit - len(issues)),
                "fields": ",".join(self._fields()),
            }
            if expand:
                params["expand"] = "changelog"
            res = self.session.get(url, headers=self.headers, auth=self.auth, params=params, timeout=60)
            raise_for_response(res, "Jira search")
            data = res.json()
            batch = data.get("issues", [])
            if not batch:
                break
            issues.extend(batch)
            start += len(batch)
            if start >= data.get("total", 0):
                break
        return issues[:limit]

    def fetch_issues(
        self,
        project_keys: List[str],
        jql_filter: str = "",
        limit: int = 200,
        order: str = "Recently updated",
        include_changelog: bool = True,
    ) -> List[JiraIssueDTO]:
        keys = [re.sub(r"[^A-Z0-9_]", "", k.upper()) for k in project_keys]
        keys = [k for k in keys if k]
        if not keys:
            raise ValueError("Enter at least one Jira project key.")

        user_order = re.search(r"(?i)\border\s+by\b.*$", jql_filter or "")
        extra = re.sub(r"(?i)\border\s+by\b.*$", "", jql_filter or "").strip()
        order_by = (
            user_order.group(0)
            if user_order
            else ("ORDER BY created DESC" if order == "Recently created" else "ORDER BY updated DESC")
        )
        clause = "project in (" + ",".join(f'"{k}"' for k in keys) + ")"
        query = clause + (f" AND ({extra})" if extra else "") + f" {order_by}"

        raw = self._search_enhanced(query, limit, include_changelog)
        if raw is None:
            raw = self._search_legacy(query, limit, include_changelog)
        return [self._to_dto(r, keys[0]) for r in raw]

    def _to_dto(self, raw: Dict[str, Any], project_key: str) -> JiraIssueDTO:
        f = raw.get("fields") or {}
        key = raw.get("key", "")
        status_obj = f.get("status") or {}
        sprints = parse_sprints(f.get(self.config.jira_sprint_field))
        primary = primary_sprint(sprints)

        points = 0.0
        for fld in self._sp_fields():
            try:
                val = float(f.get(fld) or 0)
            except (TypeError, ValueError):
                val = 0.0
            if val:
                points = val
                break

        first_started, status_since, reopen = analyse_changelog(raw)
        parent = f.get("parent") or {}
        parent_fields = parent.get("fields") or {}
        desc = adf_to_text(f.get("description")).strip()
        if len(desc) > 2000:
            desc = desc[:2000] + "… [truncated]"

        return JiraIssueDTO(
            key=key,
            summary=f.get("summary", ""),
            status=status_obj.get("name", "Unknown"),
            status_category=(status_obj.get("statusCategory") or {}).get("name", "Unknown"),
            assignee=(f.get("assignee") or {}).get("displayName", "Unassigned"),
            reporter=(f.get("reporter") or {}).get("displayName", "Unknown"),
            priority=(f.get("priority") or {}).get("name", "None"),
            issue_type=(f.get("issuetype") or {}).get("name", "Unknown"),
            project=(f.get("project") or {}).get("key", project_key),
            sprint=primary["name"] if primary else "No Sprint",
            sprint_state=primary["state"] if primary else "",
            sprint_start=primary["start"] if primary else "",
            sprint_end=primary["end"] if primary else "",
            sprint_goal=primary["goal"] if primary else "",
            sprint_count=len(sprints),
            sprint_data=json.dumps(sprints),
            labels=", ".join(f.get("labels") or []),
            components=", ".join(c.get("name", "") for c in f.get("components") or []),
            fix_versions=", ".join(v.get("name", "") for v in f.get("fixVersions") or []),
            parent_key=parent.get("key", ""),
            parent_summary=parent_fields.get("summary", ""),
            created=(f.get("created") or "")[:19],
            updated=(f.get("updated") or "")[:19],
            resolved=(f.get("resolutiondate") or "")[:19],
            first_started=first_started,
            status_since=status_since,
            due_date=(f.get("duedate") or "")[:10],
            description=desc,
            time_spent_hours=round((f.get("timespent") or 0) / 3600, 2),
            original_estimate_hours=round((f.get("timeoriginalestimate") or 0) / 3600, 2),
            remaining_hours=round((f.get("timeestimate") or 0) / 3600, 2),
            story_points=points,
            reopen_count=reopen,
            url=f"{self.base}/browse/{key}",
        )


# =====================================================================
# 6. GITLAB SERVICE
# =====================================================================

class GitLabService:
    PER_PAGE = 100

    def __init__(self, config: AppConfig):
        self.config = config
        self.session = RobustHTTPClient.create_session()
        self.headers = {"PRIVATE-TOKEN": config.gitlab_private_token}
        self.base = config.gitlab_base_url.rstrip("/")

    def test_connection(self) -> str:
        res = self.session.get(f"{self.base}/api/v4/user", headers=self.headers, timeout=20)
        raise_for_response(res, "GitLab connection")
        return res.json().get("username", "connected")

    def _paginate(self, path: str, params: Dict[str, Any], limit: int) -> List[Any]:
        """Follow GitLab pagination until `limit` rows are collected (constant per_page)."""
        items: List[Any] = []
        page = 1
        while len(items) < limit:
            res = self.session.get(
                f"{self.base}{path}",
                headers=self.headers,
                params={**params, "per_page": self.PER_PAGE, "page": page},
                timeout=45,
            )
            raise_for_response(res, path)
            batch = res.json()
            if not isinstance(batch, list) or not batch:
                break
            items.extend(batch)
            nxt = res.headers.get("X-Next-Page")
            if not nxt:
                break
            page = int(nxt)
        return items[:limit]

    def fetch_projects(self, search: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "membership": "true",
            "simple": "true",
            "order_by": "last_activity_at",
            "sort": "desc",
        }
        if search.strip():
            params["search"] = search.strip()
        return self._paginate("/api/v4/projects", params, limit)

    @staticmethod
    def _mr(mr: Dict[str, Any]) -> GitLabMRDTO:
        return GitLabMRDTO(
            mr_id=mr.get("iid"),
            title=mr.get("title", ""),
            state=mr.get("state", ""),
            author=(mr.get("author") or {}).get("username", "unknown"),
            assignee=(mr.get("assignee") or {}).get("username"),
            reviewers=", ".join(r.get("username", "") for r in mr.get("reviewers") or []),
            created_at=(mr.get("created_at") or "")[:19],
            updated_at=(mr.get("updated_at") or "")[:19],
            merged_at=(mr.get("merged_at") or "")[:19] or None,
            closed_at=(mr.get("closed_at") or "")[:19] or None,
            source_branch=mr.get("source_branch", ""),
            target_branch=mr.get("target_branch", ""),
            description=(mr.get("description") or "")[:400],
            labels=", ".join(mr.get("labels") or []),
            draft=bool(mr.get("draft") or mr.get("work_in_progress")),
            has_conflicts=bool(mr.get("has_conflicts")),
            notes=int(mr.get("user_notes_count") or 0),
            web_url=mr.get("web_url", ""),
        )

    @staticmethod
    def _commit(c: Dict[str, Any]) -> GitLabCommitDTO:
        stats = c.get("stats") or {}
        return GitLabCommitDTO(
            commit_id=c.get("short_id", ""),
            title=c.get("title", ""),
            author=c.get("author_name", ""),
            created_at=(c.get("created_at") or "")[:19],
            message=(c.get("message") or "").strip(),
            additions=int(stats.get("additions") or 0),
            deletions=int(stats.get("deletions") or 0),
            web_url=c.get("web_url", ""),
        )

    @staticmethod
    def _pipeline(p: Dict[str, Any]) -> GitLabPipelineDTO:
        return GitLabPipelineDTO(
            pipeline_id=p.get("id"),
            status=p.get("status", ""),
            ref=p.get("ref", ""),
            source=p.get("source", ""),
            created_at=(p.get("created_at") or "")[:19],
            updated_at=(p.get("updated_at") or "")[:19],
            web_url=p.get("web_url", ""),
        )

    def fetch_deep_telemetry(
        self,
        project_id: int,
        max_mrs: int = 300,
        max_commits: int = 500,
        max_pipelines: int = 200,
        since_days: Optional[int] = 90,
    ) -> Dict[str, Any]:
        since = None
        if since_days:
            since = (datetime.now(timezone.utc) - timedelta(days=since_days)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        pid = int(project_id)

        def fetch_mrs():
            params: Dict[str, Any] = {"state": "all", "order_by": "updated_at", "sort": "desc"}
            if since:
                params["updated_after"] = since
            rows = self._paginate(f"/api/v4/projects/{pid}/merge_requests", params, max_mrs)
            return [self._mr(r) for r in rows]

        def fetch_commits():
            params: Dict[str, Any] = {"with_stats": "true", "all": "true"}
            if since:
                params["since"] = since
            rows = self._paginate(f"/api/v4/projects/{pid}/repository/commits", params, max_commits)
            return [self._commit(r) for r in rows]

        def fetch_pipelines():
            params: Dict[str, Any] = {"order_by": "updated_at", "sort": "desc"}
            if since:
                params["updated_after"] = since
            rows = self._paginate(f"/api/v4/projects/{pid}/pipelines", params, max_pipelines)
            return [self._pipeline(r) for r in rows]

        def safe(label: str, fn):
            try:
                return fn(), None
            except Exception as exc:  # surfaced to the UI instead of silently returning []
                logger.warning("GitLab %s fetch failed: %s", label, exc)
                return [], f"{label}: {exc}"

        tasks = {
            "Merge requests": fetch_mrs,
            "Commits": fetch_commits,
            "Pipelines": fetch_pipelines,
        }
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {n: executor.submit(safe, n, fn) for n, fn in tasks.items()}
            results = {n: f.result() for n, f in futures.items()}

        return {
            "merge_requests": [asdict(x) for x in results["Merge requests"][0]],
            "commits": [asdict(x) for x in results["Commits"][0]],
            "pipelines": [asdict(x) for x in results["Pipelines"][0]],
            "warnings": [r[1] for r in results.values() if r[1]],
        }


# =====================================================================
# 7. LLM SERVICE
# =====================================================================

class LLMAssistantService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.session = RobustHTTPClient.create_session()

    def generate_analysis(self, prompt: str, context_payload: str, audience: str = "") -> str:
        if not self.config.llm_api_key:
            raise ValueError("LLM_API_KEY is not configured.")

        url = f"{self.config.llm_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.llm_api_key}",
            "Content-Type": "application/json",
        }
        system_prompt = (
            "You are a technical program assistant for engineering leaders, scrum masters and "
            "project managers. Analyze Jira and GitLab telemetry. Focus on observable delivery "
            "signals: scope, flow, workload, aging, blockers, review queues, CI health and trends. "
            "Do not infer personal traits or judge individual employee performance. Clearly "
            "separate facts from possible explanations, state data limitations, and never invent "
            "numbers that are not in the telemetry. Return concise markdown with actionable "
            "follow-ups."
        )
        if audience:
            system_prompt += f" Write for this audience: {audience}."

        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"### TELEMETRY\n{context_payload}\n\n### TASK\n{prompt}"},
            ],
            "temperature": 0.2,
        }
        response = self.session.post(url, headers=headers, json=payload, timeout=120)
        raise_for_response(response, "LLM request")
        return response.json()["choices"][0]["message"]["content"]


# =====================================================================
# 8. THEME TOKENS (green, light + dark)
# =====================================================================

DISPLAY_FONT = "Sora"
BODY_FONT = "Manrope"

DARK_THEME: Dict[str, Any] = {
    "name": "dark", "scheme": "dark",
    "bg": "#04100B", "bg_a": "rgba(16,185,129,.20)", "bg_b": "rgba(45,212,191,.11)",
    "surface": "rgba(14,38,30,.66)", "surface_solid": "#0E261E",
    "surface2": "rgba(255,255,255,.035)", "border": "rgba(110,231,183,.17)",
    "text": "#E8F7F0", "muted": "#8FB8A6", "faint": "#5E8474",
    "accent": "#34D399", "accent_strong": "#10B981", "accent_soft": "rgba(52,211,153,.13)",
    "grid": "rgba(143,184,166,.15)", "track": "rgba(255,255,255,.09)",
    "input_bg": "rgba(6,22,16,.85)",
    "sidebar": "linear-gradient(180deg,#07180F 0%,#040F0A 100%)",
    "shadow": "0 14px 44px rgba(0,0,0,.38)",
    "gloss": "rgba(255,255,255,.11)", "gloss_soft": "rgba(255,255,255,.055)",
    "btn_top": "rgba(255,255,255,.09)", "btn_bot": "rgba(255,255,255,.03)",
    "th": "rgba(10,31,24,.96)", "row_line": "rgba(143,184,166,.10)",
    "tones": {
        "green": "#34D399", "teal": "#2DD4BF", "lime": "#A3E635",
        "amber": "#FBBF24", "red": "#FB7185", "sky": "#38BDF8",
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
    "name": "light", "scheme": "light",
    "bg": "#EDF7F1", "bg_a": "rgba(16,185,129,.22)", "bg_b": "rgba(45,212,191,.18)",
    "surface": "rgba(255,255,255,.74)", "surface_solid": "#FFFFFF",
    "surface2": "rgba(5,150,105,.045)", "border": "rgba(5,150,105,.17)",
    "text": "#092A1F", "muted": "#4A6C5D", "faint": "#7D9C8C",
    "accent": "#059669", "accent_strong": "#047857", "accent_soft": "rgba(5,150,105,.09)",
    "grid": "rgba(9,42,31,.10)", "track": "rgba(5,150,105,.13)",
    "input_bg": "rgba(255,255,255,.92)",
    "sidebar": "linear-gradient(180deg,#FFFFFF 0%,#E7F4EC 100%)",
    "shadow": "0 12px 36px rgba(6,78,59,.11)",
    "gloss": "rgba(255,255,255,.95)", "gloss_soft": "rgba(255,255,255,.65)",
    "btn_top": "rgba(255,255,255,.98)", "btn_bot": "rgba(226,243,234,.9)",
    "th": "rgba(236,247,241,.97)", "row_line": "rgba(9,42,31,.07)",
    "tones": {
        "green": "#059669", "teal": "#0D9488", "lime": "#65A30D",
        "amber": "#D97706", "red": "#E11D48", "sky": "#0284C7",
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
# 9. ANALYTICS ENGINE – JIRA
# =====================================================================

JIRA_DEFAULTS: Dict[str, Any] = {
    "status_category": "", "assignee": "Unassigned", "reporter": "Unknown", "priority": "None",
    "issue_type": "Unknown", "project": "", "sprint": "No Sprint", "sprint_state": "",
    "sprint_start": "", "sprint_end": "", "sprint_goal": "", "sprint_count": 0,
    "sprint_data": "", "labels": "", "components": "", "fix_versions": "", "parent_key": "",
    "parent_summary": "", "created": "", "updated": "", "resolved": "", "first_started": "",
    "status_since": "", "due_date": "", "description": "", "time_spent_hours": 0.0,
    "original_estimate_hours": 0.0, "remaining_hours": 0.0, "story_points": 0.0,
    "reopen_count": 0, "url": "",
}
JIRA_DATE_COLS = [
    "created", "updated", "resolved", "first_started", "status_since", "due_date",
    "sprint_start", "sprint_end",
]


def prepare_jira_dataframe(issues: List[Dict[str, Any]], stale_days: int = 5) -> pd.DataFrame:
    if not issues:
        return pd.DataFrame()

    df = pd.DataFrame(issues)
    for col, default in JIRA_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    for col in JIRA_DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    today = pd.Timestamp.now().normalize()
    status_lower = df["status"].fillna("").str.lower()
    cat_lower = df["status_category"].fillna("").str.lower()

    df["is_done"] = (cat_lower == "done") | status_lower.isin(DONE_NAMES)
    df["is_active"] = (
        (cat_lower == "in progress") | status_lower.str.contains(ACTIVE_PATTERN, regex=True, na=False)
    ) & ~df["is_done"]
    df["is_blocked"] = (
        status_lower.str.contains("block", na=False)
        | df["labels"].fillna("").str.lower().str.contains("blocked|impediment", na=False)
    ) & ~df["is_done"]

    df["done_date"] = df["resolved"].fillna(df["updated"]).where(df["is_done"])
    df["age_days"] = (today - df["created"]).dt.days.fillna(0).clip(lower=0)
    df["days_since_update"] = (today - df["updated"]).dt.days.fillna(0).clip(lower=0)
    df["days_in_status"] = (
        (today - df["status_since"].fillna(df["updated"])).dt.days.fillna(0).clip(lower=0)
    )
    df["lead_days"] = ((df["done_date"] - df["created"]).dt.total_seconds() / 86400).clip(lower=0)
    df["cycle_days"] = ((df["done_date"] - df["first_started"]).dt.total_seconds() / 86400).clip(lower=0)

    df["overdue"] = df["due_date"].notna() & (df["due_date"] < today) & ~df["is_done"]
    df["is_stale"] = (df["days_since_update"] >= stale_days) & ~df["is_done"]
    df["is_unassigned"] = (df["assignee"] == "Unassigned") & ~df["is_done"]
    df["has_estimate"] = df["story_points"] > 0
    df["has_description"] = df["description"].fillna("").str.strip() != ""
    df["is_carry_over"] = (df["sprint_count"] > 1) & ~df["is_done"]
    df["risk_flags"] = (
        df["overdue"].astype(int) + df["is_blocked"].astype(int) + df["is_stale"].astype(int)
        + df["is_carry_over"].astype(int)
    )
    return df


FILTERS = [
    ("project", "Project", "project"),
    ("assignee", "Assignee", "assignee"),
    ("issue_type", "Type", "issue_type"),
    ("sprint", "Sprint", "sprint"),
    ("status", "Status", "status"),
    ("priority", "Priority", "priority"),
]


def filtered_jira_df(df: pd.DataFrame, ignore: Tuple[str, ...] = ()) -> pd.DataFrame:
    if df.empty:
        return df
    result = df
    for name, _, field_ in FILTERS:
        if name in ignore:
            continue
        value = st.session_state.get(f"filter_{name}", "All")
        if value != "All" and field_ in result:
            result = result[result[field_].astype(str) == value]
    return result.copy()


def weekly_counts(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype="int64")
    return s.dt.to_period("W").dt.start_time.value_counts()


def multi_weekly(named: Dict[str, pd.Series], weeks: int = 16) -> pd.DataFrame:
    """Weekly event counts for several date series on a shared Monday-based axis."""
    counts = {k: weekly_counts(v) for k, v in named.items()}
    non_empty = [c for c in counts.values() if len(c)]
    if not non_empty:
        return pd.DataFrame(columns=["week", "Series", "Issues"])
    all_weeks = non_empty[0].index
    for c in non_empty[1:]:
        all_weeks = all_weeks.union(c.index)
    current_week = pd.Timestamp.now().to_period("W").start_time
    start = all_weeks.min()
    end = max(all_weeks.max(), current_week)
    full = pd.date_range(start, end, freq="7D")[-weeks:]
    rows = []
    for name, series in counts.items():
        aligned = series.reindex(full, fill_value=0) if len(series) else pd.Series(0, index=full)
        rows += [{"week": wk, "Series": name, "Issues": int(v)} for wk, v in aligned.items()]
    return pd.DataFrame(rows)


def weekly_trend(d: pd.DataFrame, weeks: int = 16) -> pd.DataFrame:
    """Created vs completed issues per week (completion uses the resolution date)."""
    return multi_weekly({"Created": d["created"], "Completed": d["done_date"]}, weeks)


def cumulative_flow(d: pd.DataFrame, weeks: int = 20) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame(columns=["week", "Series", "Issues"])
    end = pd.Timestamp.now().normalize()
    rows = []
    for w in pd.date_range(end=end, periods=weeks, freq="7D"):
        rows.append({"week": w, "Series": "Created", "Issues": int((d["created"] <= w).sum())})
        rows.append({"week": w, "Series": "Completed", "Issues": int((d["done_date"] <= w).sum())})
    return pd.DataFrame(rows)


def workload_long(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for person, g in d.groupby("assignee"):
        done = int(g["is_done"].sum())
        blocked = int(g["is_blocked"].sum())
        open_other = int(len(g) - done - blocked)
        rows += [
            {"assignee": person, "Kind": "Completed", "Issues": done},
            {"assignee": person, "Kind": "Open", "Issues": open_other},
            {"assignee": person, "Kind": "Blocked", "Issues": blocked},
        ]
    return pd.DataFrame(rows)


def percentile(series: pd.Series, q: float) -> float:
    s = series.dropna()
    return float(np.percentile(s, q)) if len(s) else float("nan")


# ---------------------------- sprint analytics -----------------------

def sprint_membership(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (issue, sprint) so carry-over issues count in every sprint they touched."""
    rows = []
    for key, raw in zip(d["key"], d["sprint_data"]):
        try:
            items = json.loads(raw) if raw else []
        except (TypeError, ValueError):
            items = []
        for s in items:
            rows.append(
                {
                    "key": key, "sprint": s["name"], "sprint_id": s["id"],
                    "sprint_state": s["state"], "sprint_start": s["start"],
                    "sprint_end": s["end"], "sprint_goal": s["goal"],
                }
            )
    if not rows:
        return pd.DataFrame()

    m = pd.DataFrame(rows)
    m["sprint_start"] = pd.to_datetime(m["sprint_start"], errors="coerce")
    m["sprint_end"] = pd.to_datetime(m["sprint_end"], errors="coerce")
    cols = [
        "key", "summary", "assignee", "status", "issue_type", "priority", "story_points",
        "is_done", "done_date", "is_blocked", "is_stale", "overdue", "is_active", "sprint_count",
    ]
    m = m.merge(d[cols], on="key", how="left")
    end = m["sprint_end"]
    m["done_in_sprint"] = m["is_done"].astype(bool) & (
        m["done_date"].isna() | end.isna() | (m["done_date"] <= end + pd.Timedelta(days=1))
    )
    m["pts_done"] = m["story_points"].where(m["done_in_sprint"], 0.0)
    return m


def sprint_stats(m: pd.DataFrame) -> pd.DataFrame:
    g = (
        m.groupby("sprint")
        .agg(
            State=("sprint_state", "first"),
            Start=("sprint_start", "first"),
            End=("sprint_end", "first"),
            Issues=("key", "nunique"),
            Done=("done_in_sprint", "sum"),
            Scope_pts=("story_points", "sum"),
            Done_pts=("pts_done", "sum"),
        )
        .reset_index()
    )
    g["Done"] = g["Done"].astype(int)
    g["Carry_over"] = g["Issues"] - g["Done"]
    g["Completion"] = np.where(g["Issues"] > 0, g["Done"] / g["Issues"] * 100, 0).round(0)
    g["Pts_completion"] = np.where(g["Scope_pts"] > 0, g["Done_pts"] / g["Scope_pts"] * 100, 0).round(0)
    return g.sort_values(["Start", "sprint"], na_position="last").reset_index(drop=True)


def burndown_frame(sel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    if len(days) == 0:
        return pd.DataFrame()
    total = float(sel["w"].sum())
    ideal = np.linspace(total, 0, len(days))
    today = pd.Timestamp.now().normalize()
    rows = []
    for i, day in enumerate(days):
        rows.append({"day": day, "Series": "Ideal", "Value": float(ideal[i])})
        if day <= today:
            done = sel.loc[
                sel["done_in_sprint"] & (sel["done_date"].dt.normalize() <= day), "w"
            ].sum()
            rows.append({"day": day, "Series": "Remaining", "Value": float(total - done)})
    return pd.DataFrame(rows)


# =====================================================================
# 10. ANALYTICS ENGINE – GITLAB
# =====================================================================

def jira_project_keys() -> set:
    return {
        str(i.get("project", "")).upper()
        for i in st.session_state.get("jira_issues", [])
        if i.get("project")
    }


def prepare_mr_df(rows: List[Dict[str, Any]], projects: Optional[set] = None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ["created_at", "updated_at", "merged_at", "closed_at"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    now = utc_now_naive()
    is_open = df["state"] == "opened"
    df["hours_to_merge"] = (df["merged_at"] - df["created_at"]).dt.total_seconds() / 3600
    df["days_to_merge"] = df["hours_to_merge"] / 24
    df["age_days"] = np.where(is_open, (now - df["created_at"]).dt.total_seconds() / 86400, np.nan)
    df["days_idle"] = np.where(is_open, (now - df["updated_at"]).dt.total_seconds() / 86400, np.nan)
    df["assignee"] = df["assignee"].fillna("")
    text = df["title"].fillna("") + " " + df["source_branch"].fillna("") + " " + df["description"].fillna("")
    df["jira_keys"] = [", ".join(extract_issue_keys(t, projects)) for t in text]
    df["has_jira_key"] = df["jira_keys"] != ""
    return df


def prepare_commit_df(rows: List[Dict[str, Any]], projects: Optional[set] = None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["day"] = df["created_at"].dt.normalize()
    df["weekday"] = df["created_at"].dt.day_name()
    df["hour"] = df["created_at"].dt.hour
    df["churn"] = df["additions"] + df["deletions"]
    df["jira_keys"] = [", ".join(extract_issue_keys(t, projects)) for t in df["message"].fillna("")]
    return df


def prepare_pipeline_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    df["duration_min"] = (df["updated_at"] - df["created_at"]).dt.total_seconds() / 60
    return df


def active_gitlab() -> Dict[str, Any]:
    pid = st.session_state.get("gitlab_active_project")
    return st.session_state.get("gitlab_cache", {}).get(pid, {}) if pid is not None else {}


def gitlab_frames(tel: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not tel:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    projects = jira_project_keys()
    return (
        prepare_mr_df(tel.get("merge_requests", []), projects),
        prepare_commit_df(tel.get("commits", []), projects),
        prepare_pipeline_df(tel.get("pipelines", [])),
    )


def pipeline_success_rate(pipes: pd.DataFrame) -> Optional[float]:
    if pipes.empty:
        return None
    finished = pipes[pipes["status"].isin(["success", "failed"])]
    if finished.empty:
        return None
    return 100 * (finished["status"] == "success").mean()


def explode_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "jira_keys" not in df:
        return pd.DataFrame(columns=["issue_key"])
    x = df.assign(issue_key=df["jira_keys"].str.split(", ")).explode("issue_key")
    return x[x["issue_key"].notna() & (x["issue_key"] != "")]


def rows_for_key(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if df.empty or "jira_keys" not in df:
        return pd.DataFrame()
    pattern = rf"(?:^|, ){re.escape(key)}(?:,|$)"
    return df[df["jira_keys"].str.contains(pattern, regex=True, na=False)]


# =====================================================================
# 11. CHART HELPERS (Altair, theme aware)
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
            labelColor=t["muted"], titleColor=t["muted"], gridColor=t["grid"],
            domainColor=t["grid"], tickColor=t["grid"],
            labelFont=BODY_FONT, titleFont=BODY_FONT, labelFontSize=11, titleFontSize=11,
        )
        .configure_legend(
            orient="bottom", labelColor=t["text"], labelFont=BODY_FONT, labelFontSize=11,
            symbolType="circle", symbolSize=90, padding=6,
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


def donut_chart(df, cat, val, center=None, height=250, colors=None):
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
                f"{cat}:N", scale=alt.Scale(domain=cats, range=rng),
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
            .mark_text(fontSize=28, fontWeight=700, color=t["text"], font=DISPLAY_FONT)
            .encode(text="t:N")
        )
    return _finish(alt.layer(*layers), height)


def hbar_chart(df, cat, val, height=None, order=None, color_map=None):
    if df is None or df.empty:
        return None
    t = T()
    df = df.copy()
    df[cat] = df[cat].astype(str)
    height = height or max(180, 34 * len(df) + 40)
    base = alt.Chart(df).encode(
        y=alt.Y(
            f"{cat}:N", sort=order if order else "-x", title=None,
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
            color=alt.Color(f"{cat}:N", scale=alt.Scale(domain=cats, range=rng), legend=None)
        )
    else:
        bars = base.mark_bar(size=18, cornerRadiusEnd=8, color=_gradient())
    labels = base.mark_text(align="left", dx=7, color=t["muted"], fontSize=11, fontWeight=700).encode(
        text=f"{val}:Q"
    )
    return _finish(alt.layer(bars, labels), height)


def vbar_chart(df, cat, val, order, height=240):
    if df is None or df.empty:
        return None
    t = T()
    df = df.copy()
    df[cat] = df[cat].astype(str)
    base = alt.Chart(df).encode(
        x=alt.X(f"{cat}:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, ticks=False, domain=False)),
        y=alt.Y(f"{val}:Q", title=None, axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{val}:Q")],
    )
    bars = base.mark_bar(size=34, cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color=_gradient(vertical=True))
    labels = base.mark_text(dy=-8, color=t["muted"], fontSize=11, fontWeight=700, baseline="bottom").encode(
        text=f"{val}:Q"
    )
    return _finish(alt.layer(bars, labels), height)


def stacked_hbar(df, cat, kind, val, colors, height=None):
    if df is None or df.empty:
        return None
    df = df.copy()
    df[cat] = df[cat].astype(str)
    order = df.groupby(cat)[val].sum().sort_values(ascending=False).index.tolist()
    height = height or max(200, 36 * len(order) + 60)
    kinds = list(colors.keys())
    chart = (
        alt.Chart(df)
        .mark_bar(size=20, cornerRadius=4)
        .encode(
            y=alt.Y(f"{cat}:N", sort=order, title=None, axis=alt.Axis(labelLimit=170, ticks=False, domain=False)),
            x=alt.X(f"sum({val}):Q", title=None, stack="zero"),
            color=alt.Color(
                f"{kind}:N",
                scale=alt.Scale(domain=kinds, range=[colors[k] for k in kinds]),
                legend=alt.Legend(title=None),
            ),
            order=alt.Order("kind_order:Q"),
            tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{kind}:N"), alt.Tooltip(f"{val}:Q")],
        )
        .transform_calculate(kind_order=f"indexof({json.dumps(kinds)}, datum.{kind})")
    )
    return _finish(chart, height)


def grouped_bar_chart(df, cat, kind, val, colors, order, height=260, rule=None, rule_label="Average"):
    if df is None or df.empty:
        return None
    t = T()
    kinds = list(colors.keys())
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X(f"{cat}:N", sort=order, title=None, axis=alt.Axis(labelAngle=-20, labelLimit=120, ticks=False)),
            xOffset=alt.XOffset(f"{kind}:N"),
            y=alt.Y(f"{val}:Q", title=None),
            color=alt.Color(
                f"{kind}:N",
                scale=alt.Scale(domain=kinds, range=[colors[k] for k in kinds]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{kind}:N"), alt.Tooltip(f"{val}:Q")],
        )
    )
    layers = [bars]
    if rule is not None and not pd.isna(rule):
        rdf = pd.DataFrame({"y": [rule], "label": [f"{rule_label} {rule:.1f}"]})
        layers.append(alt.Chart(rdf).mark_rule(strokeDash=[5, 4], color=t["muted"], strokeWidth=2).encode(y="y:Q"))
        layers.append(
            alt.Chart(rdf)
            .mark_text(align="left", dx=4, dy=-6, color=t["muted"], fontSize=11, fontWeight=700)
            .encode(y="y:Q", text="label:N", x=alt.value(0))
        )
    return _finish(alt.layer(*layers), height)


def trend_chart(df, x, y, series, height=260, colors=None):
    if df is None or df.empty:
        return None
    df = df.copy()
    kinds = df[series].unique().tolist()
    pal = palette()
    if colors:
        rng = [colors.get(k, pal[i % len(pal)]) for i, k in enumerate(kinds)]
    else:
        rng = [pal[0], pal[3]] if len(kinds) == 2 else pal[: len(kinds)]
    scale = alt.Scale(domain=kinds, range=rng)
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:T", title=None, axis=alt.Axis(format="%b %d", labelOverlap=True, grid=False)),
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


def line_chart(df, x, y, series, colors, height=260, dashed=()):
    """Multi-series line chart (used for burndown)."""
    if df is None or df.empty:
        return None
    kinds = list(colors.keys())
    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=3, point=alt.OverlayMarkDef(filled=True, size=40))
        .encode(
            x=alt.X(f"{x}:T", title=None, axis=alt.Axis(format="%b %d", labelOverlap=True, grid=False)),
            y=alt.Y(f"{y}:Q", title=None, scale=alt.Scale(zero=True)),
            color=alt.Color(
                f"{series}:N",
                scale=alt.Scale(domain=kinds, range=[colors[k] for k in kinds]),
                legend=alt.Legend(title=None),
            ),
            strokeDash=alt.StrokeDash(
                f"{series}:N",
                scale=alt.Scale(domain=kinds, range=[[6, 4] if k in dashed else [1, 0] for k in kinds]),
                legend=None,
            ),
            tooltip=[alt.Tooltip(f"{x}:T", format="%d %b"), alt.Tooltip(f"{series}:N"), alt.Tooltip(f"{y}:Q", format=".1f")],
        )
    )
    return _finish(line, height)


def bar_line_chart(df, x, bar, line, height=240):
    if df is None or df.empty:
        return None
    t = T()
    base = alt.Chart(df).encode(x=alt.X(f"{x}:T", title=None, axis=alt.Axis(format="%b %d", labelOverlap=True, grid=False)))
    bars = base.mark_bar(size=16, cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color=_gradient(vertical=True)).encode(
        y=alt.Y(f"{bar}:Q", title=None, axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip(f"{x}:T", title="Week of", format="%d %b %Y"), alt.Tooltip(f"{bar}:Q"), alt.Tooltip(f"{line}:Q", format=".1f")],
    )
    ln = base.mark_line(strokeWidth=3, interpolate="monotone", color=t["tones"]["amber"]).encode(y=alt.Y(f"{line}:Q", title=None))
    return _finish(alt.layer(bars, ln), height)


def histogram_chart(df, col, rules=None, height=240):
    if df is None or df.empty or df[col].dropna().empty:
        return None
    t = T()
    data = df[[col]].dropna()
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=_gradient(vertical=True))
        .encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=18), title="Days"),
            y=alt.Y("count():Q", title=None, axis=alt.Axis(tickMinStep=1)),
        )
    )
    layers = [bars]
    if rules:
        rdf = pd.DataFrame({"v": list(rules.values()), "label": list(rules.keys())})
        layers.append(alt.Chart(rdf).mark_rule(strokeDash=[5, 4], color=t["tones"]["amber"], strokeWidth=2).encode(x="v:Q"))
        layers.append(
            alt.Chart(rdf)
            .mark_text(align="left", dx=4, baseline="top", color=t["tones"]["amber"], fontSize=11, fontWeight=700)
            .encode(x="v:Q", y=alt.value(2), text="label:N")
        )
    return _finish(alt.layer(*layers), height)


def cycle_scatter(df, date_col, val_col, p50, p85, height=260):
    if df is None or df.empty:
        return None
    t = T()
    data = df[["key", "summary", date_col, val_col]].dropna(subset=[date_col, val_col]).copy()
    if data.empty:
        return None
    data["summary"] = data["summary"].astype(str).str.slice(0, 70)
    pts = (
        alt.Chart(data)
        .mark_circle(size=90, opacity=0.8, color=t["accent"])
        .encode(
            x=alt.X(f"{date_col}:T", title=None, axis=alt.Axis(format="%b %d", grid=False)),
            y=alt.Y(f"{val_col}:Q", title="Days"),
            tooltip=["key", "summary", alt.Tooltip(f"{date_col}:T", format="%d %b %Y"), alt.Tooltip(f"{val_col}:Q", format=".1f")],
        )
    )
    labels = [f"P50 {p50:.1f}d", f"P85 {p85:.1f}d"]
    rdf = pd.DataFrame({"v": [p50, p85], "label": labels})
    rules = (
        alt.Chart(rdf)
        .mark_rule(strokeDash=[5, 4], strokeWidth=2)
        .encode(
            y="v:Q",
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=labels, range=[t["tones"]["green"], t["tones"]["amber"]]),
                legend=alt.Legend(title=None),
            ),
        )
    )
    return _finish(alt.layer(pts, rules), height)


def boxplot_chart(df, cat, val, height=260):
    if df is None or df.empty:
        return None
    t = T()
    chart = (
        alt.Chart(df)
        .mark_boxplot(extent="min-max", size=30, color=t["accent"])
        .encode(
            x=alt.X(f"{cat}:N", title=None, axis=alt.Axis(labelAngle=0, ticks=False)),
            y=alt.Y(f"{val}:Q", title="Days in status"),
        )
    )
    return _finish(chart, height)


def heatmap_chart(df, height=230):
    if df is None or df.empty:
        return None
    t = T()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    chart = (
        alt.Chart(df)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X("hour:O", title=None, axis=alt.Axis(labelAngle=0, ticks=False)),
            y=alt.Y("weekday:O", sort=days, title=None, axis=alt.Axis(ticks=False, labelExpr="slice(datum.label,0,3)")),
            color=alt.Color("Commits:Q", scale=alt.Scale(range=[t["track"], t["accent"]]), legend=None),
            tooltip=["weekday:O", "hour:O", "Commits:Q"],
        )
    )
    return _finish(chart, height)


def daily_area_chart(df, x, y, height=240):
    if df is None or df.empty:
        return None
    t = T()
    grad = alt.Gradient(
        gradient="linear",
        stops=[alt.GradientStop(color=t["accent"], offset=0), alt.GradientStop(color="rgba(0,0,0,0)", offset=1)],
        x1=1, y1=0, x2=1, y2=1,
    )
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:T", title=None, axis=alt.Axis(format="%b %d", grid=False)),
        y=alt.Y(f"{y}:Q", title=None, axis=alt.Axis(tickMinStep=1)),
    )
    area = base.mark_area(interpolate="monotone", opacity=0.35, color=grad)
    line = base.mark_line(interpolate="monotone", strokeWidth=3, color=t["accent"])
    pts = base.mark_point(filled=True, size=50, color=t["accent"]).encode(
        tooltip=[alt.Tooltip(f"{x}:T", format="%d %b %Y"), alt.Tooltip(f"{y}:Q")]
    )
    return _finish(alt.layer(area, line, pts), height)


def scatter_chart(d, stale_days=5, height=300):
    if d is None or d.empty:
        return None
    t = T()
    data = d[["key", "summary", "assignee", "priority", "age_days", "days_since_update"]].copy()
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
                scale=alt.Scale(domain=cats, range=[pal[i % len(pal)] for i in range(len(cats))]),
                legend=alt.Legend(title=None),
            ),
            tooltip=["key", "summary", "assignee", "priority", "age_days", "days_since_update"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [stale_days]}))
        .mark_rule(strokeDash=[5, 5], color=t["tones"]["red"], opacity=0.7)
        .encode(y="y:Q")
    )
    return _finish(alt.layer(points, rule), height)


# =====================================================================
# 12. HTML / UI COMPONENTS
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
            st.markdown(f'<div class="panel-title">{esc(title)}</div>{sub}', unsafe_allow_html=True)
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


def kpi_row(items: List[tuple]):
    """items: (title, value, subtitle, tone, icon)"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            render_kpi(*item)


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


def pipeline_kind(status: str) -> str:
    s = str(status).lower()
    if s == "success":
        return "success"
    if s == "failed":
        return "danger"
    if s in ("running", "pending", "created", "manual"):
        return "warning"
    return "neutral"


PILL_RULES = {
    "status": status_kind,
    "priority": priority_kind,
    "state": mr_state_kind,
    "pipeline": pipeline_kind,
}

COLUMN_LABELS = {
    "key": "Key", "mr_id": "MR", "commit_id": "Commit", "pipeline_id": "Pipeline",
    "created_at": "Created", "updated_at": "Updated", "merged_at": "Merged",
    "issue_type": "Type", "days_since_update": "Days idle", "days_in_status": "Days in status",
    "age_days": "Age (d)", "days_idle": "Idle (d)", "hours_to_merge": "Hours to merge",
    "days_to_merge": "Days to merge", "story_points": "Points", "cycle_days": "Cycle (d)",
    "lead_days": "Lead (d)", "jira_keys": "Jira keys", "duration_min": "Duration (min)",
    "has_conflicts": "Conflicts", "notes": "Comments", "sprint_count": "Sprints",
}


def html_table(df: pd.DataFrame, max_rows: int = 300, height: int = 380, empty: str = "Nothing to show."):
    if df is None or df.empty:
        st.caption(empty)
        return

    view = df.head(max_rows)
    head = "".join(
        f"<th>{esc(COLUMN_LABELS.get(c, str(c).replace('_', ' ').title()))}</th>" for c in view.columns
    )
    body = []
    for _, row in view.iterrows():
        cells = []
        for col in view.columns:
            v = row[col]
            title = ""
            if col in PILL_RULES and isinstance(v, str) and v:
                cell = pill(v, PILL_RULES[col](v))
            elif col == "pipeline_status" and isinstance(v, str) and v:
                cell = pill(v, pipeline_kind(v))
            elif isinstance(v, (bool, np.bool_)):
                cell = pill("Yes", "warning") if v else '<span class="dash">–</span>'
            elif isinstance(v, pd.Timestamp):
                cell = esc(v.strftime("%Y-%m-%d")) if pd.notna(v) else '<span class="dash">–</span>'
            elif isinstance(v, (float, np.floating)):
                cell = '<span class="dash">–</span>' if pd.isna(v) else esc(f"{v:.1f}".rstrip("0").rstrip("."))
            elif v is None or (not isinstance(v, str) and pd.isna(v)) or v == "":
                cell = '<span class="dash">–</span>'
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
    if len(df) > max_rows:
        st.caption(f"Showing the first {max_rows} of {len(df)} rows. Use the CSV export for the full set.")


def render_hero(title: str, subtitle: str, chip: str = ""):
    chip_html = f'<span class="hero-chip"><i></i>{esc(chip)}</span>' if chip else ""
    st.markdown(
        f'<div class="hero"><div class="hero-body">{chip_html}'
        f"<h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div></div>",
        unsafe_allow_html=True,
    )


def jira_chip(default: str) -> str:
    meta = st.session_state.get("jira_meta")
    if not meta:
        return default
    return f"{default} · {meta['count']} issues · loaded {meta['fetched_at'][11:16]}"


def render_empty(title: str, body: str):
    st.markdown(
        f'<div class="empty"><div class="empty-title">{esc(title)}</div><div>{esc(body)}</div></div>',
        unsafe_allow_html=True,
    )


def render_alert(title: str, meta: str, kind: str):
    tones = T()["tones"]
    color = {"danger": tones["red"], "warning": tones["amber"], "info": tones["sky"]}.get(kind, tones["sky"])
    st.markdown(
        f'<div class="alert" style="--tone:{color}">'
        f'<div class="alert-title">{esc(title)}</div><div class="alert-meta">{esc(meta)}</div></div>',
        unsafe_allow_html=True,
    )


def render_ring(pct_value: int, label: str):
    st.markdown(
        f'<div class="ring-wrap"><div class="ring" style="--p:{int(pct_value)}">'
        f'<div class="ring-inner"><span>{int(pct_value)}%</span><small>{esc(label)}</small></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


def stat_rows(rows: List[tuple]):
    inner = "".join(
        f'<div class="stat-row"><span>{esc(k)}</span><strong>{esc(v)}</strong></div>' for k, v in rows
    )
    st.markdown(f'<div class="stat-list">{inner}</div>', unsafe_allow_html=True)


def render_progress_rows(rows: List[tuple]):
    """rows: (label, done, total, right-hand text)"""
    out = []
    for label, done, total, meta in rows:
        out.append(
            f'<div class="pbar-row"><div class="pbar-head"><span>{esc(label)}</span>'
            f"<strong>{esc(meta)}</strong></div>"
            f'<div class="pbar"><i style="width:{pct(done, total)}%"></i></div></div>'
        )
    st.markdown("".join(out), unsafe_allow_html=True)


def section(title: str, subtitle: str = ""):
    sub = f'<div class="section-sub">{esc(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="section-title">{esc(title)}</div>{sub}', unsafe_allow_html=True)


def csv_download(df: pd.DataFrame, label: str, filename: str, key: str):
    if df is None or df.empty:
        return
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", key=key)


def get_selected_issue(df: pd.DataFrame, key: str):
    if df.empty:
        return None
    rows = df[df["key"] == key]
    return rows.iloc[0] if not rows.empty else None


# =====================================================================
# 13. STYLES
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
  padding: 1.8rem 2.4rem 3.5rem; max-width: 1600px;
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
[data-testid="stSidebarNavSeparator"], [data-testid="stNavSectionHeader"] {
  color: var(--faint) !important; font-size: .68rem; letter-spacing: .08em; text-transform: uppercase;
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
.hero p { margin: 0; color: rgba(236,253,245,.86) !important; font-size: .94rem; max-width: 70ch; }
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
  box-shadow: var(--shadow), inset 0 1px 0 var(--gloss); margin-bottom: .5rem;
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
.section-title { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.15rem; color: var(--text); margin: 1.5rem 0 .3rem; }
.section-sub { color: var(--muted); font-size: .8rem; margin: 0 0 .8rem; }

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

/* ---------- progress bars ---------- */
.pbar-row { margin: .7rem 0; }
.pbar-head { display: flex; justify-content: space-between; gap: 1rem; font-size: .8rem; color: var(--muted); }
.pbar-head span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pbar-head strong { color: var(--text); white-space: nowrap; }
.pbar { height: 8px; border-radius: 999px; background: var(--track); overflow: hidden; margin-top: .35rem; }
.pbar > i { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent-strong), var(--accent)); }

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
[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 14px; background: var(--surface2); }

@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"], .main .block-container { padding: 1.1rem 1rem 2.5rem; }
  .hero { padding: 1.3rem 1.2rem; border-radius: 20px; }
  .hero h1 { font-size: 1.55rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""


def inject_custom_styles():
    t = T()
    pills = "".join(f".pill-{kind} {{ background: {bg}; color: {fg}; }}\n" for kind, (bg, fg) in t["pills"].items())
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
# 14. COMMON COMPONENTS
# =====================================================================

def reset_filters():
    for name, _, _ in FILTERS:
        st.session_state[f"filter_{name}"] = "All"
        st.session_state[f"w_filter_{name}"] = "All"


def render_global_filters(df: pd.DataFrame, hide: Tuple[str, ...] = ()):
    if df.empty:
        return
    active = [f for f in FILTERS if f[0] not in hide]
    with panel("Filters", "Applied to every Jira view and kept when you change pages"):
        cols = st.columns(len(active))
        for col, (name, label, field_) in zip(cols, active):
            options = ["All"] + sorted(df[field_].dropna().astype(str).unique().tolist())
            wkey = f"w_filter_{name}"
            current = st.session_state.get(f"filter_{name}", "All")
            if st.session_state.get(wkey) not in options:
                st.session_state[wkey] = current if current in options else "All"
            with col:
                st.session_state[f"filter_{name}"] = st.selectbox(label, options, key=wkey)

        filtered = filtered_jira_df(df, ignore=hide)
        a, b = st.columns([6, 1])
        a.caption(f"Showing {len(filtered)} of {len(df)} loaded issues")
        b.button("Reset", key="filters_reset", on_click=reset_filters)


# =====================================================================
# 15. EXECUTIVE OVERVIEW
# =====================================================================

def render_executive_overview(df: pd.DataFrame):
    render_hero(
        "Engineering overview",
        "Delivery progress, flow, workload and risk across Jira and GitLab in one view.",
        jira_chip(f"Updated {datetime.now().strftime('%d %b %Y')}"),
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
    unassigned = int(d["is_unassigned"].sum())
    carry = int(d["is_carry_over"].sum())
    progress_pct = pct(done, total)
    open_issues = d[~d["is_done"]]
    clean = pct(int((open_issues["risk_flags"] == 0).sum()), len(open_issues))

    kpi_row([
        ("Total work", total, "Filtered Jira issues", "sky", "▦"),
        ("Completed", done, f"{progress_pct}% of visible work", "green", "✓"),
        ("In progress", in_progress, "Active execution", "amber", "↻"),
        ("Blocked", blocked, "Needs attention", "red", "!"),
        ("Overdue", overdue, f"{stale} stale", "lime", "⏱"),
    ])

    mr, commits, pipes = gitlab_frames(active_gitlab())
    open_mrs = int((mr["state"] == "opened").sum()) if not mr.empty else None
    merged = mr[mr["state"] == "merged"] if not mr.empty else pd.DataFrame()
    success = pipeline_success_rate(pipes)
    kpi_row([
        ("Story points done", f"{d.loc[d['is_done'], 'story_points'].sum():g}", f"of {d['story_points'].sum():g} planned", "green", "★"),
        ("Clean flow", f"{clean}%", "Open work with no risk flag", "teal", "◎"),
        ("Median cycle time", fmt_days(d.loc[d["is_done"], "cycle_days"].median()), "Started → done", "sky", "◷"),
        ("Open merge requests", "–" if open_mrs is None else open_mrs, "GitLab review queue", "amber", "⑂"),
        ("Pipeline success", "–" if success is None else f"{success:.0f}%", "Finished pipelines", "lime", "⚙"),
    ])

    st.write("")
    a, b, c = st.columns([0.8, 1.25, 1.05])

    with a:
        with panel("Delivery progress", "Share of visible issues completed"):
            render_ring(progress_pct, "complete")
            stat_rows([
                ("Issues done", f"{done} of {total}"),
                ("Story points", f"{d.loc[d['is_done'], 'story_points'].sum():g} of {d['story_points'].sum():g}"),
                ("Logged time", f"{d['time_spent_hours'].sum():.1f} h"),
            ])

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
                render_alert(f"{stale} stale issue(s)", f"No Jira update for {st.session_state.get('stale_days', 5)}+ days.", "warning")
            if carry:
                render_alert(f"{carry} carried-over issue(s)", "Open and already planned in an earlier sprint.", "warning")
            if unassigned:
                render_alert(f"{unassigned} unassigned open issue(s)", "Work has no current assignee.", "warning")
            if not any([overdue, blocked, stale, unassigned, carry]):
                render_alert("Nothing needs attention", "Based on the Jira fields currently loaded.", "info")

    left, right = st.columns(2)
    with left:
        with panel("Created vs completed", "Weekly flow. Completion uses each issue's resolution date."):
            show_chart(trend_chart(weekly_trend(d), "week", "Issues", "Series"))
    with right:
        with panel("Team workload", "Completed, open and blocked issues per assignee"):
            tn = T()["tones"]
            show_chart(
                stacked_hbar(
                    workload_long(d), "assignee", "Kind", "Issues",
                    {"Completed": tn["green"], "Open": tn["sky"], "Blocked": tn["red"]}, height=260,
                )
            )

    section("Recent Jira activity")
    html_table(
        d.sort_values("updated", ascending=False).head(8)[["key", "summary", "status", "assignee", "priority", "updated"]],
        height=360,
    )


# =====================================================================
# 16. SPRINT ANALYSIS (Scrum Master)
# =====================================================================

def render_sprint_analysis(df: pd.DataFrame):
    render_hero(
        "Sprint analysis",
        "Burndown, scope, carry-over and velocity for the scrum master and the team.",
        jira_chip("Scrum"),
    )
    if df.empty:
        render_empty("No Jira data loaded yet", "Load a project from Data Sources first.")
        return

    d = filtered_jira_df(df, ignore=("sprint",))
    m = sprint_membership(d) if not d.empty else pd.DataFrame()
    if m.empty:
        render_empty(
            "No sprint data found",
            "Issues carry no sprint field. If your sprint field is not customfield_10020, set JIRA_SPRINT_FIELD.",
        )
        return

    stats = sprint_stats(m)
    names_desc = stats["sprint"].tolist()[::-1]
    active = stats[stats["State"] == "active"]
    default_name = active["sprint"].iloc[-1] if not active.empty else names_desc[0]
    seed("w_sprint_pick", default_name)
    if st.session_state["w_sprint_pick"] not in names_desc:
        st.session_state["w_sprint_pick"] = default_name

    c1, c2 = st.columns([2, 1])
    with c1:
        sprint_name = st.selectbox("Sprint", names_desc, key="w_sprint_pick")
    with c2:
        measure = st.radio("Measure", ["Story points", "Issue count"], horizontal=True, key="w_sprint_measure")

    sel = m[m["sprint"] == sprint_name].copy()
    info = stats[stats["sprint"] == sprint_name].iloc[0]
    has_pts = sel["story_points"].sum() > 0
    use_pts = measure == "Story points" and has_pts
    if measure == "Story points" and not has_pts:
        st.caption("No story points on this sprint's issues – showing issue counts instead.")
    sel["w"] = sel["story_points"] if use_pts else 1.0
    unit = "pts" if use_pts else "issues"

    total = float(sel["w"].sum())
    done = float(sel.loc[sel["done_in_sprint"], "w"].sum())
    remaining = total - done
    completion = pct(done, total)
    state = info["State"]
    start, end = info["Start"], info["End"]
    today = pd.Timestamp.now().normalize()

    elapsed, days_left, health, health_kind = None, None, "Not started", "neutral"
    if pd.notna(start) and pd.notna(end) and end > start:
        elapsed = float(np.clip((today - start) / (end - start), 0, 1))
        days_left = max(int((end.normalize() - today).days), 0)
    if state == "closed":
        health, health_kind = "Closed", "neutral"
    elif state == "active" and elapsed is not None:
        gap = completion / 100 - elapsed
        health, health_kind = (
            ("On track", "success") if gap >= -0.10 else ("At risk", "warning") if gap >= -0.25 else ("Behind", "danger")
        )
    elif state == "future":
        health, health_kind = "Planned", "info"

    kpi_row([
        ("Sprint scope", f"{total:g}", f"{unit} · {len(sel)} issues", "sky", "▦"),
        ("Completed", f"{done:g}", f"{completion}% of scope", "green", "✓"),
        ("Remaining", f"{remaining:g}", f"{unit} not done in sprint", "amber", "↻"),
        ("Time left", "–" if days_left is None else f"{days_left}d", f"{0 if elapsed is None else round(elapsed * 100)}% of sprint elapsed", "teal", "◷"),
        ("Blocked / stale", f"{int(sel['is_blocked'].sum())} / {int(sel['is_stale'].sum())}", "Open items in sprint", "red", "!"),
    ])
    st.markdown(f"**Pace:** {pill(health, health_kind)}", unsafe_allow_html=True)

    goal = sel["sprint_goal"].dropna().iloc[0] if sel["sprint_goal"].notna().any() else ""
    if goal:
        render_alert("Sprint goal", goal, "info")

    st.write("")
    a, b = st.columns([1.7, 1])
    tn = T()["tones"]
    with a:
        with panel("Burndown", f"Remaining {unit} per day against the ideal line (scope = current sprint membership)"):
            if pd.notna(start) and pd.notna(end):
                show_chart(
                    line_chart(
                        burndown_frame(sel, start, end), "day", "Value", "Series",
                        {"Ideal": T()["muted"], "Remaining": tn["green"]}, dashed=("Ideal",),
                    )
                )
            else:
                st.caption("This sprint has no start / end dates in Jira.")
    with b:
        with panel("Scope by status"):
            counts = sel.groupby("status")["w"].sum().reset_index().rename(columns={"status": "Status", "w": "Value"})
            show_chart(donut_chart(counts, "Status", "Value", center=f"{total:g}", height=250))

    a, b = st.columns(2)
    with a:
        with panel("Load by assignee", f"Completed vs remaining {unit}"):
            rows = []
            for person, g in sel.groupby("assignee"):
                rows += [
                    {"assignee": person, "Kind": "Completed", "Value": float(g.loc[g["done_in_sprint"], "w"].sum())},
                    {"assignee": person, "Kind": "Remaining", "Value": float(g.loc[~g["done_in_sprint"], "w"].sum())},
                ]
            show_chart(stacked_hbar(pd.DataFrame(rows), "assignee", "Kind", "Value",
                                    {"Completed": tn["green"], "Remaining": tn["amber"]}, height=260))
    with b:
        with panel("Scope by issue type"):
            ty = sel.groupby("issue_type")["w"].sum().reset_index().rename(columns={"issue_type": "Type", "w": "Value"})
            show_chart(donut_chart(ty, "Type", "Value", center=f"{total:g}", height=250))

    section("Spill-over and open work", "Items not completed inside this sprint")
    spill = sel[~sel["done_in_sprint"]].sort_values(["is_blocked", "story_points"], ascending=[False, False])
    if spill.empty:
        st.success("Everything planned in this sprint was completed.")
    else:
        html_table(spill[["key", "summary", "assignee", "status", "priority", "story_points", "sprint_count"]], height=320)
        csv_download(spill.drop(columns=["w"]), "Download spill-over (CSV)", f"{sprint_name}_spillover.csv", "dl_spill")

    # ------------------------- velocity across sprints -------------------------
    section("Velocity and predictability", "Closed sprints only. Completed means resolved by the sprint end date.")
    closed = stats[stats["State"] == "closed"].copy()
    if closed.empty:
        st.info("No closed sprints in the loaded issues yet. Fetch more history to see velocity.")
    else:
        scope_col, done_col = ("Scope_pts", "Done_pts") if closed["Scope_pts"].sum() > 0 else ("Issues", "Done")
        last3 = closed.tail(3)
        velocity = float(last3[done_col].mean())
        predictability = pct(float(closed[done_col].sum()), float(closed[scope_col].sum()))
        kpi_row([
            ("Avg velocity (last 3)", f"{velocity:.1f}", "pts" if scope_col == "Scope_pts" else "issues", "green", "➚"),
            ("Predictability", f"{predictability}%", "Completed ÷ planned, all closed", "sky", "◎"),
            ("Velocity range", f"{last3[done_col].min():g}–{last3[done_col].max():g}", "Last 3 sprints", "amber", "↕"),
            ("Avg carry-over", f"{closed['Carry_over'].mean():.1f}", "Issues per closed sprint", "red", "↷"),
        ])
        vel = closed.melt(id_vars=["sprint"], value_vars=[scope_col, done_col], var_name="Kind", value_name="Value")
        vel["Kind"] = vel["Kind"].map({scope_col: "Planned", done_col: "Completed"})
        carry = closed[["sprint", "Carry_over"]].rename(columns={"Carry_over": "Issues"})
        a, b = st.columns([1.4, 1])
        with a:
            with panel("Planned vs completed", "Dashed line is the 3-sprint average velocity"):
                show_chart(grouped_bar_chart(vel, "sprint", "Kind", "Value",
                                             {"Planned": tn["sky"], "Completed": tn["green"]},
                                             closed["sprint"].tolist(), rule=velocity, rule_label="Velocity"))
        with b:
            with panel("Carry-over per sprint", "Issues not finished in the sprint"):
                show_chart(vbar_chart(carry, "sprint", "Issues", closed["sprint"].tolist()))

    with panel("All sprints", "Scope and completion per sprint"):
        tbl = stats.copy()
        tbl = tbl[["sprint", "State", "Start", "End", "Issues", "Done", "Carry_over", "Scope_pts", "Done_pts", "Completion"]]
        tbl = tbl.rename(columns={"State": "status", "sprint": "Sprint", "Carry_over": "Carry-over",
                                  "Scope_pts": "Scope pts", "Done_pts": "Done pts", "Completion": "Completion %"})
        html_table(tbl.iloc[::-1], height=300)
        csv_download(tbl, "Download sprint table (CSV)", "sprints.csv", "dl_sprints")


# =====================================================================
# 17. FLOW METRICS (Kanban / Delivery)
# =====================================================================

def render_flow_metrics(df: pd.DataFrame):
    render_hero(
        "Flow metrics",
        "Throughput, cycle time, lead time and work-in-progress ageing.",
        jira_chip("Flow"),
    )
    if df.empty:
        render_empty("No Jira data loaded yet", "Load a project from Data Sources first.")
        return
    d = filtered_jira_df(df)
    if d.empty:
        st.warning("No issues match the current filters.")
        return

    done = d[d["is_done"]]
    cyc, lead = done["cycle_days"].dropna(), done["lead_days"].dropna()
    last28 = int((done["done_date"] >= pd.Timestamp.now() - pd.Timedelta(days=28)).sum())
    reopened = int((done["reopen_count"] > 0).sum())
    wip = int(d["is_active"].sum())

    kpi_row([
        ("Throughput", f"{last28 / 4:.1f}/wk", "Completed, last 4 weeks", "green", "➚"),
        ("Median cycle time", fmt_days(cyc.median()) if len(cyc) else "–", "Started → done", "sky", "◷"),
        ("85th pct cycle time", fmt_days(percentile(cyc, 85)), "85% finish within", "amber", "◔"),
        ("Median lead time", fmt_days(lead.median()) if len(lead) else "–", "Created → done", "teal", "⏲"),
        ("WIP", wip, "Items in progress / review", "lime", "↻"),
        ("Reopen rate", f"{pct(reopened, len(done))}%", f"{reopened} reopened", "red", "↺"),
    ])
    if cyc.empty:
        st.info("Cycle time needs status history. Enable 'Include status history' when fetching Jira issues.")

    st.write("")
    a, b = st.columns(2)
    with a:
        with panel("Throughput", "Issues completed per week with a 4-week moving average"):
            tp = multi_weekly({"Completed": done["done_date"]}, weeks=16)
            if not tp.empty:
                tp = tp.rename(columns={"Issues": "Completed"})[["week", "Completed"]]
                tp["Average"] = tp["Completed"].rolling(4, min_periods=1).mean().round(2)
                show_chart(bar_line_chart(tp, "week", "Completed", "Average"))
            else:
                st.caption("No completed issues in the loaded data.")
    with b:
        with panel("Cumulative flow", "Gap between the lines is open work (backlog + WIP)"):
            show_chart(trend_chart(cumulative_flow(d), "week", "Issues", "Series",
                                   colors={"Created": T()["tones"]["sky"], "Completed": T()["tones"]["green"]}))

    a, b = st.columns(2)
    with a:
        with panel("Cycle time scatter", "Each dot is a completed issue; lines show P50 and P85"):
            if len(cyc) >= 2:
                show_chart(cycle_scatter(done, "done_date", "cycle_days", percentile(cyc, 50), percentile(cyc, 85)))
            else:
                st.caption("Not enough completed issues with status history.")
    with b:
        with panel("Lead time distribution", "Days from creation to completion"):
            if len(lead):
                show_chart(histogram_chart(done, "lead_days", {f"P85 {percentile(lead, 85):.0f}d": percentile(lead, 85)}))
            else:
                st.caption("No completed issues in the loaded data.")

    a, b = st.columns(2)
    open_d = d[~d["is_done"]]
    with a:
        with panel("Open work by status", "Where work is waiting right now"):
            stc = open_d["status"].value_counts().rename_axis("Status").reset_index(name="Issues")
            show_chart(hbar_chart(stc, "Status", "Issues"))
    with b:
        with panel("Ageing work in progress", "Days spent in the current status (box = middle 50%)"):
            show_chart(boxplot_chart(open_d[open_d["is_active"]][["status", "days_in_status"]], "status", "days_in_status"))

    section("Oldest work in progress", "Candidates for a stand-up conversation")
    oldest = open_d[open_d["is_active"]].sort_values("days_in_status", ascending=False).head(12)
    if oldest.empty:
        st.success("No items are currently in progress.")
    else:
        html_table(oldest[["key", "summary", "assignee", "status", "days_in_status", "age_days", "priority"]], height=360)


# =====================================================================
# 18. BACKLOG & PLANNING (Project / Product Manager)
# =====================================================================

def render_planning(df: pd.DataFrame):
    render_hero(
        "Backlog & planning",
        "Backlog health, estimation coverage, epic and release progress, and a delivery forecast.",
        jira_chip("Planning"),
    )
    if df.empty:
        render_empty("No Jira data loaded yet", "Load a project from Data Sources first.")
        return
    d = filtered_jira_df(df)
    if d.empty:
        st.warning("No issues match the current filters.")
        return

    open_d = d[~d["is_done"]].copy()
    unsized = int((~open_d["has_estimate"]).sum())
    ready = open_d[open_d["has_estimate"] & open_d["has_description"]]
    not_planned = int((open_d["sprint"] == "No Sprint").sum())
    backlog_pts = float(open_d["story_points"].sum())

    kpi_row([
        ("Open backlog", len(open_d), f"{backlog_pts:g} story points", "sky", "▦"),
        ("Ready for sprint", len(ready), "Estimated and described", "green", "✓"),
        ("Unestimated", unsized, "Open without points", "amber", "?"),
        ("Not in a sprint", not_planned, "Open and unscheduled", "lime", "☐"),
        ("Avg open age", fmt_days(open_d["age_days"].mean()), "Since creation", "red", "◷"),
    ])

    st.write("")
    a, b, c = st.columns([0.8, 1.1, 1.1])
    with a:
        with panel("Refinement coverage", "Open items that are estimated and described"):
            render_ring(pct(len(ready), len(open_d)), "ready")
            stat_rows([
                ("With estimate", f"{len(open_d) - unsized} of {len(open_d)}"),
                ("With description", f"{int(open_d['has_description'].sum())} of {len(open_d)}"),
                ("Assigned", f"{int((~open_d['is_unassigned']).sum())} of {len(open_d)}"),
            ])
    with b:
        with panel("Open backlog age", "How long unfinished work has existed"):
            labels = ["0–7d", "8–14d", "15–30d", "31–60d", "60d+"]
            bins = pd.cut(open_d["age_days"], bins=[-1, 7, 14, 30, 60, float("inf")], labels=labels)
            ag = bins.value_counts().reindex(labels).fillna(0).astype(int).rename_axis("Age").reset_index(name="Issues")
            show_chart(vbar_chart(ag, "Age", "Issues", labels))
    with c:
        with panel("Open work by type"):
            ty = open_d["issue_type"].value_counts().rename_axis("Type").reset_index(name="Issues")
            show_chart(donut_chart(ty, "Type", "Issues", center=str(len(open_d)), height=250))

    # forecast from velocity
    m = sprint_membership(d)
    stats = sprint_stats(m) if not m.empty else pd.DataFrame()
    closed = stats[stats["State"] == "closed"] if not stats.empty else pd.DataFrame()
    section("Delivery forecast", "Based on the average completed points of the last three closed sprints")
    if closed.empty or closed.tail(3)["Done_pts"].mean() <= 0 or backlog_pts <= 0:
        st.info("A forecast needs closed sprints with story points and open estimated work.")
    else:
        lo, avg = closed.tail(3)["Done_pts"].min(), closed.tail(3)["Done_pts"].mean()
        hi = closed.tail(3)["Done_pts"].max()
        kpi_row([
            ("Remaining points", f"{backlog_pts:g}", "Open, estimated", "sky", "▦"),
            ("Avg velocity", f"{avg:.1f}", "pts per sprint", "green", "➚"),
            ("Sprints to clear", f"{int(np.ceil(backlog_pts / avg))}", f"Range {int(np.ceil(backlog_pts / hi))}–{int(np.ceil(backlog_pts / max(lo, 1)))}", "amber", "⏱"),
        ])

    section("Epic progress", "Progress of child issues by parent epic")
    epics = d[d["parent_key"] != ""].assign(pts_done=lambda x: x["story_points"].where(x["is_done"], 0.0))
    if epics.empty:
        st.caption("No parent / epic links found on the loaded issues.")
    else:
        g = epics.groupby("parent_key").agg(Title=("parent_summary", "first"), Total=("key", "count"), Done=("is_done", "sum")).reset_index()
        g = g.sort_values("Total", ascending=False).head(12)
        with panel("Top epics", "Completed issues per epic"):
            render_progress_rows([(f"{r.parent_key} · {str(r.Title)[:60]}", r.Done, r.Total, f"{int(r.Done)}/{int(r.Total)} · {pct(r.Done, r.Total)}%") for r in g.itertuples()])

    section("Release progress", "Grouped by Jira fix version")
    ver = d[d["fix_versions"] != ""].assign(version=lambda x: x["fix_versions"].str.split(", ")).explode("version")
    if ver.empty:
        st.caption("No fix versions found on the loaded issues.")
    else:
        g = ver.groupby("version").agg(Total=("key", "count"), Done=("is_done", "sum"), Points=("story_points", "sum")).reset_index().sort_values("Total", ascending=False).head(10)
        with panel("Versions", "Completed issues per release"):
            render_progress_rows([(r.version, r.Done, r.Total, f"{int(r.Done)}/{int(r.Total)} · {r.Points:g} pts") for r in g.itertuples()])

    a, b = st.columns(2)
    with a:
        with panel("Components", "Open work by component"):
            comp = open_d[open_d["components"] != ""].assign(c=lambda x: x["components"].str.split(", ")).explode("c")
            cc = comp["c"].value_counts().head(10).rename_axis("Component").reset_index(name="Issues")
            if cc.empty:
                st.caption("No components set.")
            else:
                show_chart(hbar_chart(cc, "Component", "Issues"))
    with b:
        with panel("Labels", "Most used labels on open work"):
            lab = open_d[open_d["labels"] != ""].assign(l=lambda x: x["labels"].str.split(", ")).explode("l")
            lc = lab["l"].value_counts().head(10).rename_axis("Label").reset_index(name="Issues")
            if lc.empty:
                st.caption("No labels set.")
            else:
                show_chart(hbar_chart(lc, "Label", "Issues"))

    section("Needs refinement", "Open items missing an estimate, a description or an owner")
    need = open_d[~open_d["has_estimate"] | ~open_d["has_description"] | open_d["is_unassigned"]].copy()
    if need.empty:
        st.success("Every open item is estimated, described and assigned.")
    else:
        need["missing"] = need.apply(
            lambda r: ", ".join([x for x, f in [("Estimate", not r["has_estimate"]), ("Description", not r["has_description"]), ("Owner", r["is_unassigned"])] if f]),
            axis=1,
        )
        html_table(need.sort_values("age_days", ascending=False)[["key", "summary", "issue_type", "priority", "missing", "age_days"]], height=360)
        csv_download(need, "Download refinement list (CSV)", "needs_refinement.csv", "dl_refine")


# =====================================================================
# 19. DELIVERY DASHBOARD
# =====================================================================

PRIORITY_ORDER = ["Highest", "Critical", "Blocker", "High", "Medium", "Low", "Lowest", "None"]


def render_delivery_dashboard(df: pd.DataFrame):
    render_hero("Delivery dashboard", "Backlog composition, ageing and delivery pressure at a glance.", jira_chip("Jira delivery"))

    if df.empty:
        render_empty("No Jira data loaded yet", "Load a project from Data Sources first.")
        return
    d = filtered_jira_df(df)
    if d.empty:
        st.warning("No issues match the current filters.")
        return

    kpi_row([
        ("Backlog", int((~d["is_done"]).sum()), "Not completed", "sky", "▦"),
        ("Completed", int(d["is_done"].sum()), "Completed issues", "green", "✓"),
        ("Average age", f'{d["age_days"].mean():.1f}d', "Across visible issues", "amber", "◷"),
        ("Logged time", f'{d["time_spent_hours"].sum():.1f}h', "Recorded work", "lime", "⏲"),
    ])

    st.write("")
    a, b = st.columns(2)
    with a:
        with panel("Priority mix", "Current work by Jira priority"):
            pr = d["priority"].value_counts().rename_axis("Priority").reset_index(name="Issues")
            known = [p for p in PRIORITY_ORDER if p in pr["Priority"].values]
            extra = [p for p in pr["Priority"] if p not in known]
            tones = T()["tones"]
            cmap = {"Highest": tones["red"], "Critical": tones["red"], "Blocker": tones["red"], "High": tones["amber"],
                    "Medium": tones["sky"], "Low": tones["teal"], "Lowest": tones["green"]}
            show_chart(hbar_chart(pr, "Priority", "Issues", order=known + extra, color_map=cmap))
    with b:
        with panel("Issue types", "Where work is concentrated"):
            ty = d["issue_type"].value_counts().rename_axis("Type").reset_index(name="Issues")
            show_chart(donut_chart(ty, "Type", "Issues", center=str(len(d)), height=250))

    a, b = st.columns(2)
    with a:
        with panel("Issue ageing", "How long visible work has existed"):
            labels = ["0–2d", "3–7d", "8–14d", "15–30d", "30d+"]
            bins = pd.cut(d["age_days"], bins=[-1, 2, 7, 14, 30, float("inf")], labels=labels)
            aging = bins.value_counts().reindex(labels).fillna(0).astype(int).rename_axis("Age").reset_index(name="Issues")
            show_chart(vbar_chart(aging, "Age", "Issues", labels))
    with b:
        with panel("Update freshness", "Time since each issue was last touched"):
            labels = ["0–1d", "2–3d", "4–5d", "6–10d", "10d+"]
            fr = pd.cut(d["days_since_update"], bins=[-1, 1, 3, 5, 10, float("inf")], labels=labels)
            fresh = fr.value_counts().reindex(labels).fillna(0).astype(int).rename_axis("Idle").reset_index(name="Issues")
            show_chart(vbar_chart(fresh, "Idle", "Issues", labels))

    stale_days = st.session_state.get("stale_days", 5)
    with panel("Age vs. idle time", f"Open issues. Points above the dashed line have been idle for {stale_days}+ days."):
        show_chart(scatter_chart(d[~d["is_done"]], stale_days=stale_days, height=320))

    section("Work needing review")
    attention = d[d["overdue"] | d["is_blocked"] | d["is_stale"]].copy()
    if attention.empty:
        st.success("No overdue, blocked or stale issues in the current filter.")
    else:
        attention["reason"] = attention.apply(
            lambda r: ", ".join([x for x, f in [("Overdue", r["overdue"]), ("Blocked", r["is_blocked"]), ("Stale", r["is_stale"])] if f]),
            axis=1,
        )
        html_table(
            attention[["key", "summary", "assignee", "status", "priority", "reason", "days_since_update"]]
            .sort_values("days_since_update", ascending=False)
        )


# =====================================================================
# 20. TEAM DASHBOARD
# =====================================================================

def render_team_dashboard(df: pd.DataFrame):
    render_hero("Team dashboard", "Workload and delivery signals for team-level coordination.", jira_chip("Team view"))

    if df.empty:
        render_empty("No Jira data loaded yet", "Load a project from Data Sources first.")
        return
    d = filtered_jira_df(df)
    if d.empty:
        st.warning("No issues match the current filters.")
        return

    d = d.assign(open_pts=d["story_points"].where(~d["is_done"], 0.0))
    team = (
        d.groupby("assignee")
        .agg(
            Issues=("key", "count"), Completed=("is_done", "sum"), In_Progress=("is_active", "sum"),
            Blocked=("is_blocked", "sum"), Overdue=("overdue", "sum"), Stale=("is_stale", "sum"),
            Open_Points=("open_pts", "sum"), Logged_Hours=("time_spent_hours", "sum"),
        )
        .reset_index()
        .sort_values(["Blocked", "Issues"], ascending=[False, False])
    )
    team["Logged_Hours"] = team["Logged_Hours"].round(1)
    tn = T()["tones"]

    left, right = st.columns([1.25, 1])
    with left:
        with panel("Team overview", "One row per assignee"):
            html_table(team, height=340)
            csv_download(team, "Download team table (CSV)", "team.csv", "dl_team")
    with right:
        with panel("Workload split", "Completed, open and blocked issues"):
            show_chart(stacked_hbar(workload_long(d), "assignee", "Kind", "Issues",
                                    {"Completed": tn["green"], "Open": tn["sky"], "Blocked": tn["red"]}, height=300))

    a, b, c = st.columns(3)
    with a:
        with panel("Open story points", "Remaining planned effort per person"):
            show_chart(hbar_chart(team[["assignee", "Open_Points"]].rename(columns={"Open_Points": "Points"}), "assignee", "Points"))
    with b:
        with panel("Logged hours", "Recorded time per person"):
            show_chart(hbar_chart(team[["assignee", "Logged_Hours"]].rename(columns={"Logged_Hours": "Hours"}), "assignee", "Hours"))
    with c:
        with panel("Attention signals", "Overdue, blocked and stale open work"):
            sig = team.melt(id_vars="assignee", value_vars=["Blocked", "Overdue", "Stale"], var_name="Kind", value_name="Issues")
            show_chart(stacked_hbar(sig, "assignee", "Kind", "Issues", {"Blocked": tn["red"], "Overdue": tn["amber"], "Stale": tn["lime"]}))

    section("Individual drill-down", "Workload view for coordination, not performance evaluation")
    selected_person = st.selectbox("Select team member", team["assignee"].tolist(), key="team_person_selector")
    render_person_detail(d, selected_person)


def render_person_detail(df: pd.DataFrame, person: str):
    p = df[df["assignee"] == person].copy()
    if p.empty:
        st.warning("No issues found for this team member.")
        return

    kpi_row([
        ("Issues", len(p), f"Assigned to {person}", "sky", "▦"),
        ("Completed", int(p["is_done"].sum()), "Completed", "green", "✓"),
        ("Active", int(p["is_active"].sum()), "In progress or review", "amber", "↻"),
        ("Attention", int(p["is_blocked"].sum() + p["overdue"].sum()),
         f"{int(p['is_blocked'].sum())} blocked, {int(p['overdue'].sum())} overdue", "red", "!"),
    ])
    st.write("")
    left, right = st.columns([1.6, 1])
    with left:
        with panel("Current work", "Open issues, most urgent first"):
            current = p[~p["is_done"]].sort_values(["is_blocked", "overdue", "updated"], ascending=[False, False, False])
            if current.empty:
                st.success("No open work in the current filter.")
            else:
                html_table(current[["key", "summary", "status", "priority", "updated", "days_since_update"]], height=340)
    with right:
        with panel("Work profile", "Issues by status"):
            prof = p["status"].value_counts().rename_axis("Status").reset_index(name="Issues")
            show_chart(donut_chart(prof, "Status", "Issues", center=str(len(p)), height=260))


# =====================================================================
# 21. RISKS & ATTENTION
# =====================================================================

def render_risks(df: pd.DataFrame):
    render_hero("Risks & attention", "A focused queue of issues that may need manager follow-up.", jira_chip("Delivery risk"))

    if df.empty:
        render_empty("No Jira data loaded yet", "Load a project from Data Sources first.")
        return
    d = filtered_jira_df(df)
    if d.empty:
        st.warning("No issues match the current filters.")
        return

    open_d = d[~d["is_done"]]
    categories = {
        "Overdue": d[d["overdue"]],
        "Blocked": d[d["is_blocked"]],
        "Stale": d[d["is_stale"]],
        "Unassigned": d[d["is_unassigned"]],
        "Unestimated": open_d[~open_d["has_estimate"]],
        "Carry-over": d[d["is_carry_over"]],
    }
    tones = {"Overdue": "red", "Blocked": "red", "Stale": "amber", "Unassigned": "lime", "Unestimated": "sky", "Carry-over": "teal"}
    icons = {"Overdue": "⏱", "Blocked": "⛔", "Stale": "◷", "Unassigned": "?", "Unestimated": "≈", "Carry-over": "↷"}
    names = list(categories)
    for chunk in (names[:3], names[3:]):
        kpi_row([(n, len(categories[n]), "Requires review", tones[n], icons[n]) for n in chunk])

    st.write("")
    tn = T()["tones"]
    colors = {"Overdue": tn["red"], "Blocked": tn["amber"], "Stale": tn["lime"], "Unassigned": tn["sky"], "Unestimated": tn["teal"], "Carry-over": tn["green"]}
    left, right = st.columns([1, 1.3])
    with left:
        with panel("Risk mix", "Share of flagged issues by signal"):
            mix = pd.DataFrame({"Signal": names, "Issues": [len(s) for s in categories.values()]})
            mix = mix[mix["Issues"] > 0]
            if mix.empty:
                st.success("No risk signals in the current filter.")
            else:
                show_chart(donut_chart(mix, "Signal", "Issues", center=str(int(mix["Issues"].sum())), colors=colors))
    with right:
        with panel("Risk by assignee", "Where flagged work sits"):
            rows = []
            for person, g in d.groupby("assignee"):
                rows += [
                    {"assignee": person, "Signal": "Overdue", "Issues": int(g["overdue"].sum())},
                    {"assignee": person, "Signal": "Blocked", "Issues": int(g["is_blocked"].sum())},
                    {"assignee": person, "Signal": "Stale", "Issues": int(g["is_stale"].sum())},
                    {"assignee": person, "Signal": "Carry-over", "Issues": int(g["is_carry_over"].sum())},
                ]
            rdf = pd.DataFrame(rows)
            if rdf.empty or rdf["Issues"].sum() == 0:
                st.success("No flagged work per assignee.")
            else:
                show_chart(stacked_hbar(rdf, "assignee", "Signal", "Issues",
                                        {k: colors[k] for k in ["Overdue", "Blocked", "Stale", "Carry-over"]}, height=260))

    section("Top risk items", "Open issues ranked by the number of risk flags")
    top = open_d[open_d["risk_flags"] > 0].copy()
    if top.empty:
        st.success("No flagged open issues.")
    else:
        top["flags"] = top.apply(
            lambda r: ", ".join([x for x, f in [("Overdue", r["overdue"]), ("Blocked", r["is_blocked"]), ("Stale", r["is_stale"]), ("Carry-over", r["is_carry_over"])] if f]),
            axis=1,
        )
        top = top.sort_values(["risk_flags", "age_days"], ascending=[False, False]).head(15)
        html_table(top[["key", "summary", "assignee", "status", "priority", "flags", "risk_flags"]], height=380)

    section("Issue queues")
    tabs = st.tabs([f"{name} ({len(subset)})" for name, subset in categories.items()])
    for tab, (name, subset) in zip(tabs, categories.items()):
        with tab:
            if subset.empty:
                st.success(f"No {name.lower()} issues in the current filter.")
            else:
                html_table(subset[["key", "summary", "assignee", "status", "priority", "updated"]])
                csv_download(subset, f"Download {name.lower()} (CSV)", f"{name.lower()}.csv", f"dl_risk_{name}")


# =====================================================================
# 22. ISSUE EXPLORER
# =====================================================================

def render_issue_explorer(df: pd.DataFrame):
    render_hero("Jira issue explorer", "Search, inspect and drill into individual Jira work items.", jira_chip("Jira workspace"))

    if df.empty:
        render_empty("No Jira data loaded yet", "Load a project from Data Sources first.")
        return
    d = filtered_jira_df(df)

    search = st.text_input("Search issues", placeholder="Search by key, summary, assignee, label or description")
    result = d.copy()
    if search.strip():
        q = search.strip().lower()
        mask = pd.Series(False, index=result.index)
        for col in ["key", "summary", "assignee", "description", "labels", "parent_key"]:
            mask |= result[col].astype(str).str.lower().str.contains(q, na=False, regex=False)
        result = result[mask]

    c1, c2 = st.columns([6, 1])
    c1.caption(f"{len(result)} issue(s)")
    with c2:
        csv_download(result.drop(columns=["description"], errors="ignore"), "Export CSV", "jira_issues.csv", "dl_issues")

    if result.empty:
        st.warning("No matching issues.")
        return

    with panel("Results", "Sorted by most recently updated"):
        html_table(
            result.sort_values("updated", ascending=False)[
                ["key", "summary", "status", "assignee", "priority", "issue_type", "sprint", "story_points", "updated", "overdue"]
            ],
            height=360,
        )

    selected_key = st.selectbox("Open issue", result["key"].tolist(), key="issue_detail_selector")
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
    kpi_row([
        ("Age", f"{int(issue['age_days'])}d", "Since creation", "sky", "◷"),
        ("Logged", f"{issue['time_spent_hours']:.1f}h", f"Estimate {issue['original_estimate_hours']:.1f}h", "lime", "⏲"),
        ("In status", f"{int(issue['days_in_status'])}d", f"Idle {int(issue['days_since_update'])}d", "amber", "↻"),
        ("Story points", f"{issue['story_points']:g}", "If configured", "green", "★"),
        ("Reopened", int(issue["reopen_count"]), f"In {int(issue['sprint_count'])} sprint(s)", "red", "↺"),
    ])

    st.write("")
    left, right = st.columns([1.5, 1])
    with left:
        with panel("Description"):
            st.write(issue["description"] or "No description available.")
    with right:
        with panel("Issue context"):
            stat_rows([
                ("Project", issue["project"]),
                ("Sprint", issue["sprint"]),
                ("Epic / parent", issue["parent_key"] or "None"),
                ("Reporter", issue["reporter"]),
                ("Due date", issue["due_date"].strftime("%Y-%m-%d") if pd.notna(issue["due_date"]) else "Not set"),
                ("Fix version", issue["fix_versions"] or "None"),
                ("Labels", issue["labels"] or "None"),
            ])

    mr, commits, _ = gitlab_frames(active_gitlab())
    linked_mr, linked_cm = rows_for_key(mr, selected_key), rows_for_key(commits, selected_key)
    if not linked_mr.empty or not linked_cm.empty:
        with panel("Linked GitLab activity", "Matched by issue key in titles, branches and commit messages"):
            if not linked_mr.empty:
                html_table(linked_mr[["mr_id", "title", "state", "author", "created_at", "merged_at"]], height=220)
            if not linked_cm.empty:
                html_table(linked_cm[["commit_id", "title", "author", "created_at"]], height=220)

    if issue["url"]:
        st.link_button("Open in Jira", issue["url"])


# =====================================================================
# 23. ENGINEERING ACTIVITY (GitLab)
# =====================================================================

GL_DEFAULTS = {"since": "90 days", "mrs": 300, "commits": 500, "pipelines": 200}
GL_LOOKBACK = {"30 days": 30, "90 days": 90, "180 days": 180, "1 year": 365, "All time": None}


def sync_gitlab_project(config: AppConfig, pid: int, name: str, settings: Dict[str, Any]) -> None:
    """Fetch one repository and store it under its own id so projects never mix."""
    cache = st.session_state.setdefault("gitlab_cache", {})
    try:
        tel = GitLabService(config).fetch_deep_telemetry(
            pid, max_mrs=settings["mrs"], max_commits=settings["commits"],
            max_pipelines=settings["pipelines"], since_days=GL_LOOKBACK[settings["since"]],
        )
    except Exception as exc:
        tel = {"merge_requests": [], "commits": [], "pipelines": [], "warnings": [str(exc)]}
    tel.update(project_id=pid, project_name=name, settings=settings, fetched_at=datetime.now().isoformat(timespec="seconds"))
    cache[pid] = tel


def render_engineering(gitlab_projects: List[Dict[str, Any]], config: AppConfig):
    render_hero("Engineering activity", "Merge requests, commits and CI health for one repository at a time.", "GitLab")

    if not gitlab_projects:
        render_empty("No repositories scanned yet", "Open Data Sources and select Scan accessible GitLab projects.")
        return

    options = {f"{p.get('name_with_namespace', p.get('name', 'Repository'))} (ID {p.get('id')})": p["id"] for p in gitlab_projects}
    labels = list(options.keys())
    active_id = st.session_state.get("gitlab_active_project")
    if st.session_state.get("w_gl_repo") not in options:
        st.session_state["w_gl_repo"] = next((l for l, i in options.items() if i == active_id), labels[0])

    gs = st.session_state.get("gl_settings", GL_DEFAULTS)
    seed("w_gl_since", gs["since"]); seed("w_gl_mrs", gs["mrs"])
    seed("w_gl_commits", gs["commits"]); seed("w_gl_pipes", gs["pipelines"]); seed("w_gl_auto", True)

    c1, c2 = st.columns([4, 1])
    with c1:
        selected = st.selectbox("Repository", labels, key="w_gl_repo")
    with c2:
        st.write("")
        st.write("")
        force = st.button("Sync now", type="primary", key="gl_sync")

    with st.expander("Fetch settings"):
        s1, s2, s3, s4, s5 = st.columns(5)
        since = s1.selectbox("Look-back", list(GL_LOOKBACK.keys()), key="w_gl_since")
        n_mrs = s2.number_input("Max merge requests", 50, 2000, step=50, key="w_gl_mrs")
        n_commits = s3.number_input("Max commits", 100, 5000, step=100, key="w_gl_commits")
        n_pipes = s4.number_input("Max pipelines", 50, 2000, step=50, key="w_gl_pipes")
        auto = s5.checkbox("Auto-sync on switch", key="w_gl_auto")
    settings = {"since": since, "mrs": int(n_mrs), "commits": int(n_commits), "pipelines": int(n_pipes)}
    st.session_state["gl_settings"] = settings

    pid = options[selected]
    st.session_state["gitlab_active_project"] = pid
    cache = st.session_state.setdefault("gitlab_cache", {})

    if force or (auto and pid not in cache):
        with st.spinner(f"Fetching merge requests, commits and pipelines for {selected}…"):
            sync_gitlab_project(config, pid, selected, settings)

    tel = cache.get(pid)
    if not tel:
        render_empty("This repository has not been synced yet", "Select Sync now to load merge requests, commits and pipelines.")
        return

    for w in tel.get("warnings", []):
        st.warning(w)
    mr, commits, pipes = gitlab_frames(tel)
    used = tel.get("settings", settings)
    st.caption(
        f"{tel['project_name']} · synced {tel['fetched_at'].replace('T', ' ')} · look-back {used['since']} · "
        f"{len(mr)} MRs · {len(commits)} commits · {len(pipes)} pipelines (all branches)"
    )
    if len(mr) >= used["mrs"] or len(commits) >= used["commits"]:
        st.caption("A limit was reached – raise the maximums in Fetch settings and sync again to load older activity.")

    merged = mr[mr["state"] == "merged"] if not mr.empty else pd.DataFrame()
    opened = mr[mr["state"] == "opened"] if not mr.empty else pd.DataFrame()
    no_review = int((merged["notes"] == 0).sum()) if not merged.empty else 0
    success = pipeline_success_rate(pipes)

    kpi_row([
        ("Merge requests", len(mr), "Loaded", "sky", "⑂"),
        ("Open", len(opened), f"{int(opened['draft'].sum()) if not opened.empty else 0} drafts", "amber", "◔"),
        ("Merged", len(merged), "Completed reviews", "green", "✓"),
        ("Median time to merge", fmt_hours(merged["hours_to_merge"].median()) if not merged.empty else "–", "Created → merged", "teal", "⏱"),
        ("Merged w/o comments", no_review, "No review discussion", "red", "!"),
    ])
    kpi_row([
        ("Commits", len(commits), "In look-back window", "lime", "●"),
        ("Contributors", commits["author"].nunique() if not commits.empty else 0, "Distinct commit authors", "sky", "☻"),
        ("Lines changed", f"{int(commits['churn'].sum()):,}" if not commits.empty else 0, "Additions + deletions", "amber", "±"),
        ("Pipeline success", "–" if success is None else f"{success:.0f}%", "Finished pipelines", "green", "⚙"),
        ("Median pipeline time", fmt_hours(pipes["duration_min"].median() / 60) if not pipes.empty else "–", "Created → updated", "teal", "◷"),
    ])

    tn = T()["tones"]
    t_over, t_mr, t_cm, t_ci = st.tabs(["Overview", "Merge requests", "Commits", "Pipelines"])

    with t_over:
        a, b = st.columns(2)
        with a:
            with panel("Merge request flow", "Opened vs merged per week"):
                flow = multi_weekly({"Opened": mr["created_at"] if not mr.empty else pd.Series(dtype="datetime64[ns]"),
                                     "Merged": mr["merged_at"] if not mr.empty else pd.Series(dtype="datetime64[ns]")})
                show_chart(trend_chart(flow, "week", "Issues", "Series", colors={"Opened": tn["sky"], "Merged": tn["green"]}))
        with b:
            with panel("Time to merge", "Days from creation to merge"):
                show_chart(histogram_chart(merged, "days_to_merge", {f"P85 {percentile(merged['days_to_merge'], 85):.1f}d": percentile(merged["days_to_merge"], 85)}) if not merged.empty else None)
        a, b = st.columns([1.3, 1])
        with a:
            with panel("Commit activity", "Commits per day"):
                if commits.empty:
                    st.caption("No commits loaded.")
                else:
                    show_chart(daily_area_chart(commits.dropna(subset=["day"]).groupby("day").size().reset_index(name="Commits"), "day", "Commits"))
        with b:
            with panel("When work happens", "Commits by weekday and hour (UTC)"):
                if commits.empty:
                    st.caption("No commits loaded.")
                else:
                    show_chart(heatmap_chart(commits.dropna(subset=["created_at"]).groupby(["weekday", "hour"]).size().reset_index(name="Commits")))

    with t_mr:
        a, b, c = st.columns([1, 1, 1.2])
        with a:
            with panel("States"):
                if mr.empty:
                    st.caption("No merge requests loaded.")
                else:
                    sc = mr["state"].value_counts().rename_axis("State").reset_index(name="MRs")
                    show_chart(donut_chart(sc, "State", "MRs", center=str(len(mr)),
                                           colors={"merged": tn["green"], "opened": tn["amber"], "closed": tn["red"]}))
        with b:
            with panel("Top authors", "Merge requests per author"):
                if not mr.empty:
                    show_chart(hbar_chart(mr["author"].value_counts().head(8).rename_axis("Author").reset_index(name="MRs"), "Author", "MRs"))
        with c:
            with panel("Review queue age", "Open merge requests by age"):
                if opened.empty:
                    st.success("No open merge requests.")
                else:
                    labels_ = ["0–2d", "3–7d", "8–14d", "15d+"]
                    bins = pd.cut(opened["age_days"], bins=[-1, 2, 7, 14, float("inf")], labels=labels_)
                    ag = bins.value_counts().reindex(labels_).fillna(0).astype(int).rename_axis("Age").reset_index(name="MRs")
                    show_chart(vbar_chart(ag, "Age", "MRs", labels_))
        section("Open review queue", "Oldest first. Conflicts and idle MRs usually need a nudge.")
        if opened.empty:
            st.caption("Nothing waiting for review.")
        else:
            html_table(opened.sort_values("age_days", ascending=False)[
                ["mr_id", "title", "author", "reviewers", "age_days", "days_idle", "draft", "has_conflicts", "notes", "jira_keys"]], height=340)
        section("Merged without review comments", "Useful for review-practice conversations, not individual blame")
        silent = merged[merged["notes"] == 0] if not merged.empty else pd.DataFrame()
        if silent.empty:
            st.caption("Every merged MR has review discussion.")
        else:
            html_table(silent.sort_values("merged_at", ascending=False)[["mr_id", "title", "author", "merged_at", "hours_to_merge"]], height=300)
        section("All merge requests")
        if not mr.empty:
            cols = ["mr_id", "title", "state", "author", "assignee", "created_at", "merged_at", "hours_to_merge", "notes", "source_branch", "target_branch", "jira_keys"]
            html_table(mr[cols], height=400)
            csv_download(mr, "Download merge requests (CSV)", "merge_requests.csv", "dl_mrs")

    with t_cm:
        a, b = st.columns(2)
        with a:
            with panel("Commits by author", "Top 10 in the window"):
                if commits.empty:
                    st.caption("No commits loaded.")
                else:
                    show_chart(hbar_chart(commits["author"].value_counts().head(10).rename_axis("Author").reset_index(name="Commits"), "Author", "Commits"))
        with b:
            with panel("Code churn per week", "Lines added vs removed"):
                if commits.empty or commits["churn"].sum() == 0:
                    st.caption("No line statistics available.")
                else:
                    cw = commits.dropna(subset=["created_at"]).copy()
                    cw["week"] = cw["created_at"].dt.to_period("W").dt.start_time
                    g = cw.groupby("week")[["additions", "deletions"]].sum().reset_index().melt(id_vars="week", var_name="Series", value_name="Issues")
                    g["Series"] = g["Series"].map({"additions": "Added", "deletions": "Removed"})
                    show_chart(trend_chart(g, "week", "Issues", "Series", colors={"Added": tn["green"], "Removed": tn["red"]}))
        section("Commits")
        if not commits.empty:
            html_table(commits[["commit_id", "title", "author", "created_at", "additions", "deletions", "jira_keys"]], height=400)
            csv_download(commits, "Download commits (CSV)", "commits.csv", "dl_commits")

    with t_ci:
        if pipes.empty:
            st.info("No pipelines found. CI may be disabled for this project or the token lacks access.")
        else:
            a, b = st.columns(2)
            with a:
                with panel("Pipeline outcomes"):
                    ps = pipes["status"].value_counts().rename_axis("Status").reset_index(name="Pipelines")
                    show_chart(donut_chart(ps, "Status", "Pipelines", center=str(len(pipes)),
                                           colors={"success": tn["green"], "failed": tn["red"], "canceled": tn["amber"], "running": tn["sky"]}))
            with b:
                with panel("Weekly success vs failed"):
                    fin = pipes[pipes["status"].isin(["success", "failed"])]
                    wk = multi_weekly({"Success": fin.loc[fin["status"] == "success", "created_at"], "Failed": fin.loc[fin["status"] == "failed", "created_at"]})
                    show_chart(trend_chart(wk, "week", "Issues", "Series", colors={"Success": tn["green"], "Failed": tn["red"]}))
            a, b = st.columns(2)
            with a:
                with panel("Median pipeline time", "Minutes per day (created → last update)"):
                    dd = pipes.dropna(subset=["created_at"]).assign(day=lambda x: x["created_at"].dt.normalize()).groupby("day")["duration_min"].median().round(1).reset_index()
                    show_chart(daily_area_chart(dd, "day", "duration_min"))
            with b:
                with panel("Failures by branch", "Where pipelines break most"):
                    fb = pipes[pipes["status"] == "failed"]["ref"].value_counts().head(8).rename_axis("Branch").reset_index(name="Failures")
                    if fb.empty:
                        st.success("No failed pipelines.")
                    else:
                        show_chart(hbar_chart(fb, "Branch", "Failures"))
            section("Recent pipelines")
            pt = pipes.rename(columns={"status": "pipeline_status"})
            html_table(pt[["pipeline_id", "pipeline_status", "ref", "source", "created_at", "duration_min"]].head(100), height=340)


# =====================================================================
# 24. TRACEABILITY (Jira <-> GitLab)
# =====================================================================

def render_traceability(df: pd.DataFrame):
    render_hero("Jira ↔ GitLab traceability", "Find work that is out of sync between your tracker and your code.", "Traceability")

    tel = active_gitlab()
    if df.empty or not tel:
        render_empty("Both sources are needed", "Load Jira issues and sync a GitLab repository, then return here.")
        return
    d = filtered_jira_df(df)
    mr, commits, _ = gitlab_frames(tel)
    if d.empty:
        st.warning("No issues match the current filters.")
        return

    mr_x, cm_x = explode_keys(mr), explode_keys(commits)
    if mr_x.empty:
        mr_agg = pd.DataFrame(columns=["MRs", "Open_MRs", "Merged_MRs"])
    else:
        mr_agg = mr_x.groupby("issue_key").agg(
            MRs=("mr_id", "count"),
            Open_MRs=("state", lambda s: int((s == "opened").sum())),
            Merged_MRs=("state", lambda s: int((s == "merged").sum())),
        )
    cm_agg = cm_x.groupby("issue_key").size().rename("Commits") if not cm_x.empty else pd.Series(dtype="int64", name="Commits")

    link = d[["key", "summary", "assignee", "status", "priority", "is_done", "is_active"]].merge(
        mr_agg, left_on="key", right_index=True, how="left").merge(cm_agg, left_on="key", right_index=True, how="left")
    for c in ["MRs", "Open_MRs", "Merged_MRs", "Commits"]:
        link[c] = link[c].fillna(0).astype(int)
    no_code = (link["MRs"] == 0) & (link["Commits"] == 0)

    done_no_code = link[link["is_done"] & no_code]
    active_no_code = link[link["is_active"] & no_code]
    done_open_mr = link[link["is_done"] & (link["Open_MRs"] > 0)]
    merged_not_done = link[~link["is_done"] & (link["Merged_MRs"] > 0) & (link["Open_MRs"] == 0)]
    mr_no_key = mr[~mr["has_jira_key"]] if not mr.empty else pd.DataFrame()
    coverage = pct(int(mr["has_jira_key"].sum()), len(mr)) if not mr.empty else 0

    st.caption(f"Repository: {tel['project_name']} · matching issue keys of Jira projects: {', '.join(sorted(jira_project_keys())) or 'none'}")
    kpi_row([
        ("MR traceability", f"{coverage}%", "MRs that reference a Jira key", "green", "⛓"),
        ("Active, no code", len(active_no_code), "In progress without MR / commit", "amber", "?"),
        ("Done, MR still open", len(done_open_mr), "Status ahead of code", "red", "!"),
        ("Merged, issue open", len(merged_not_done), "Code ahead of status", "sky", "↺"),
        ("MRs without key", len(mr_no_key), "Cannot be linked", "lime", "∅"),
    ])
    st.write("")
    a, b = st.columns([1, 1.4])
    with a:
        with panel("Merge request linkage"):
            if mr.empty:
                st.caption("No merge requests loaded.")
            else:
                mix = pd.DataFrame({"Type": ["With Jira key", "Without Jira key"], "MRs": [int(mr["has_jira_key"].sum()), int((~mr["has_jira_key"]).sum())]})
                show_chart(donut_chart(mix, "Type", "MRs", center=f"{coverage}%", colors={"With Jira key": T()["tones"]["green"], "Without Jira key": T()["tones"]["amber"]}))
    with b:
        with panel("Issues with most code activity", "MRs + commits referencing the issue"):
            top = link.assign(Activity=link["MRs"] + link["Commits"]).sort_values("Activity", ascending=False).head(8)
            top = top[top["Activity"] > 0][["key", "Activity"]]
            if top.empty:
                st.caption("No issue keys were found in merge requests or commits.")
            else:
                show_chart(hbar_chart(top, "key", "Activity"))

    st.caption("Issues finished before the GitLab look-back window, or non-code work, can legitimately show no activity.")
    tabs = st.tabs([f"In progress, no code ({len(active_no_code)})", f"Done, MR open ({len(done_open_mr)})",
                    f"Merged, issue open ({len(merged_not_done)})", f"Done, no code ({len(done_no_code)})", f"MRs without key ({len(mr_no_key)})"])
    cols = ["key", "summary", "assignee", "status", "MRs", "Open_MRs", "Merged_MRs", "Commits"]
    for tab, frame in zip(tabs[:4], [active_no_code, done_open_mr, merged_not_done, done_no_code]):
        with tab:
            html_table(frame[cols], empty="Nothing to show – in sync.")
    with tabs[4]:
        html_table(mr_no_key[["mr_id", "title", "state", "author", "created_at", "source_branch"]] if not mr_no_key.empty else mr_no_key, empty="Every MR references a Jira key.")


# =====================================================================
# 25. AI INSIGHTS
# =====================================================================

AI_TEMPLATES = {
    "Daily manager briefing": "Create a concise manager briefing. Cover delivery status, workload distribution, blockers, stale work, overdue work, code review activity and CI health.",
    "Sprint review / retrospective pack": "Summarise the most recent sprint: planned vs completed scope, carry-over, velocity trend, what went well and what to improve, as retrospective discussion prompts. Do not attribute outcomes to individuals.",
    "Sprint risk review": "Identify delivery risks visible in the telemetry. Separate direct facts from possible explanations and list the issues that deserve follow-up.",
    "Stand-up summary": "Create a short stand-up summary organised as: completed, in progress, blocked, stale/attention, and review queue.",
    "Stakeholder status report": "Write a stakeholder-ready status report: headline status (red/amber/green with reasoning), progress, upcoming milestones, risks and asks. Keep it non-technical.",
    "Release readiness": "Assess release readiness from open work, blockers, fix-version progress, pipeline health and unmerged merge requests. List go/no-go criteria that are met or missing.",
    "Backlog refinement review": "Review backlog health: unestimated or undescribed items, ageing items, unscheduled work and suggestions for the next refinement session.",
    "Jira + GitLab correlation": "Explain useful relationships between Jira delivery work and GitLab activity. Highlight where issue status and engineering activity may be out of sync.",
}
AI_AUDIENCES = ["Engineering manager", "Scrum master", "Project / program manager", "Executive stakeholders", "Team (retrospective)"]


def build_ai_context(d: pd.DataFrame, tel: Dict[str, Any], include_issues: bool, max_issues: int) -> str:
    blocks = []
    if not d.empty:
        done = d[d["is_done"]]
        metrics = {
            "issues_loaded": len(d), "done": int(d["is_done"].sum()), "in_progress": int(d["is_active"].sum()),
            "blocked": int(d["is_blocked"].sum()), "overdue": int(d["overdue"].sum()), "stale": int(d["is_stale"].sum()),
            "unassigned_open": int(d["is_unassigned"].sum()), "carry_over_open": int(d["is_carry_over"].sum()),
            "unestimated_open": int((~d["is_done"] & ~d["has_estimate"]).sum()),
            "story_points_done": float(d.loc[d["is_done"], "story_points"].sum()),
            "story_points_total": float(d["story_points"].sum()),
            "median_cycle_days": None if done["cycle_days"].dropna().empty else round(float(done["cycle_days"].median()), 1),
            "median_lead_days": None if done["lead_days"].dropna().empty else round(float(done["lead_days"].median()), 1),
            "reopened_issues": int((d["reopen_count"] > 0).sum()),
        }
        blocks.append("=== JIRA METRICS ===\n" + json.dumps(metrics))
        m = sprint_membership(d)
        if not m.empty:
            s = sprint_stats(m).tail(8).copy()
            s["Start"] = s["Start"].dt.strftime("%Y-%m-%d")
            s["End"] = s["End"].dt.strftime("%Y-%m-%d")
            blocks.append("=== SPRINTS (oldest to newest) ===\n" + s.to_json(orient="records"))
        if include_issues:
            cols = ["key", "summary", "status", "assignee", "priority", "issue_type", "sprint", "story_points", "created", "updated", "due_date", "days_since_update", "is_blocked", "overdue", "is_stale", "cycle_days"]
            c = d.sort_values("updated", ascending=False)[cols].head(max_issues).copy()
            c["summary"] = c["summary"].astype(str).str.slice(0, 90)
            for col in ["created", "updated", "due_date"]:
                c[col] = c[col].dt.strftime("%Y-%m-%d")
            blocks.append(f"=== JIRA ISSUES (latest {len(c)}) ===\n" + c.round(1).to_json(orient="records"))
    if tel:
        mr, commits, pipes = gitlab_frames(tel)
        g: Dict[str, Any] = {"repository": tel.get("project_name"), "merge_requests": len(mr), "commits": len(commits), "pipelines": len(pipes)}
        if not mr.empty:
            merged = mr[mr["state"] == "merged"]
            g.update(open_mrs=int((mr["state"] == "opened").sum()),
                     median_hours_to_merge=None if merged.empty else round(float(merged["hours_to_merge"].median()), 1),
                     mrs_without_jira_key=int((~mr["has_jira_key"]).sum()),
                     merged_without_comments=int((merged["notes"] == 0).sum()))
        sr = pipeline_success_rate(pipes)
        if sr is not None:
            g["pipeline_success_pct"] = round(sr, 1)
        blocks.append("=== GITLAB SUMMARY ===\n" + json.dumps(g))
        if not mr.empty:
            c = mr.sort_values("updated_at", ascending=False).head(60)[["mr_id", "title", "state", "author", "created_at", "merged_at", "draft", "notes"]].copy()
            c["title"] = c["title"].astype(str).str.slice(0, 80)
            for col in ["created_at", "merged_at"]:
                c[col] = c[col].dt.strftime("%Y-%m-%d")
            blocks.append("=== RECENT MERGE REQUESTS ===\n" + c.to_json(orient="records"))
    return "\n\n".join(blocks)


def render_ai_insights(df: pd.DataFrame, config: AppConfig):
    render_hero("AI engineering insights", "Turn Jira and GitLab telemetry into briefings for every audience.", "AI assistant")

    tel = active_gitlab()
    if df.empty and not tel:
        render_empty("No telemetry loaded yet", "Load Jira issues or GitLab activity from Data Sources, then come back to generate a briefing.")
        return

    with panel("Briefing setup", "Choose a template, an audience and optional focus"):
        c1, c2 = st.columns(2)
        selected = c1.selectbox("Analysis template", list(AI_TEMPLATES.keys()))
        audience = c2.selectbox("Audience", AI_AUDIENCES)
        custom = st.text_area("Additional instruction", placeholder="Optional: focus on a specific sprint, epic, release or delivery concern.", height=90)
        c3, c4, c5 = st.columns([1, 1, 1])
        include_issues = c3.checkbox("Include issue-level detail", value=True)
        max_issues = c4.slider("Max issues sent", 20, 300, 120, step=20, disabled=not include_issues)
        with c5:
            st.write("")
            generate = st.button("Generate briefing", type="primary")

    if generate:
        d = filtered_jira_df(df) if not df.empty else df
        prompt = AI_TEMPLATES[selected] + (f"\nAdditional instruction: {custom.strip()}" if custom.strip() else "")
        try:
            with st.spinner("Analysing engineering telemetry…"):
                result = LLMAssistantService(config).generate_analysis(prompt, build_ai_context(d, tel, include_issues, max_issues), audience)
            st.session_state["ai_last"] = {"title": selected, "audience": audience, "text": result, "at": datetime.now().strftime("%Y-%m-%d %H:%M")}
        except Exception as exc:
            st.error(f"AI analysis failed: {exc}")

    last = st.session_state.get("ai_last")
    if last:
        with panel(last["title"], f"For {last['audience']} · generated {last['at']}"):
            st.markdown(last["text"])
        st.download_button("Download briefing (Markdown)", f"# {last['title']}\n\n*{last['audience']} · {last['at']}*\n\n{last['text']}",
                           file_name="briefing.md", mime="text/markdown", key="dl_briefing")


# =====================================================================
# 26. DATA SOURCES
# =====================================================================

def load_jira(config: AppConfig, project_input: str, jql: str, limit: int, order: str, changelog: bool) -> int:
    keys = [k.strip() for k in project_input.split(",") if k.strip()]
    issues = JiraService(config).fetch_issues(keys, jql, limit=limit, order=order, include_changelog=changelog)
    st.session_state["jira_issues"] = [asdict(i) for i in issues]
    st.session_state["jira_meta"] = {
        "project": project_input, "jql": jql, "limit": limit, "order": order, "changelog": changelog,
        "count": len(issues), "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }
    return len(issues)


def render_data_sources(config: AppConfig):
    render_hero("Data sources", "Load the Jira and GitLab data that powers every dashboard.", "Integrations")

    def conn(label: str, ok: bool) -> str:
        return pill(f"{label} connected" if ok else f"{label} not configured", "success" if ok else "danger")

    st.markdown(
        conn("Jira", bool(config.jira_api_token and config.jira_base_url)) + " "
        + conn("GitLab", bool(config.gitlab_private_token and config.gitlab_base_url)) + " "
        + conn("LLM", bool(config.llm_api_key)),
        unsafe_allow_html=True,
    )
    st.write("")

    meta = st.session_state.get("jira_meta", {})
    seed("w_jira_project", meta.get("project", "")); seed("w_jira_jql", meta.get("jql", ""))
    seed("w_jira_limit", int(meta.get("limit", 200))); seed("w_jira_order", meta.get("order", "Recently updated"))
    seed("w_jira_changelog", meta.get("changelog", True))

    with panel("Jira", "Fetch the most recent issues from one or more projects (default 200)"):
        c1, c2 = st.columns([1.2, 2])
        project_input = c1.text_input("Project key(s)", key="w_jira_project", placeholder="CORE or CORE, PAY")
        jql = c2.text_input("Additional JQL", key="w_jira_jql", placeholder="status != Done AND priority = High")
        c3, c4, c5, c6 = st.columns([1, 1.2, 1.4, 1])
        limit = c3.number_input("Max issues", 10, 2000, step=50, key="w_jira_limit")
        order = c4.selectbox("Newest by", ["Recently updated", "Recently created"], key="w_jira_order")
        with c5:
            st.write("")
            changelog = st.checkbox("Include status history", key="w_jira_changelog", help="Enables cycle time, time-in-status and reopen counts. Slightly slower.")
        with c6:
            st.write("")
            fetch = st.button("Fetch Jira", type="primary")

        if fetch:
            if not project_input.strip():
                st.warning("Enter a Jira project key.")
            else:
                try:
                    with st.spinner("Fetching Jira issues…"):
                        n = load_jira(config, project_input, jql, int(limit), order, changelog)
                    st.success(f"Loaded {n} Jira issues.")
                except Exception as exc:
                    st.error(f"Jira fetch failed: {exc}")

    with panel("GitLab", "Scan repositories you are a member of"):
        c1, c2, c3 = st.columns([2, 1, 1])
        search = c1.text_input("Filter by name", key="w_gl_search", placeholder="optional")
        max_projects = c2.number_input("Max projects", 20, 1000, 200, step=20, key="w_gl_maxproj")
        with c3:
            st.write("")
            st.write("")
            scan = st.button("Scan accessible GitLab projects")
        if scan:
            try:
                with st.spinner("Scanning GitLab projects…"):
                    projects = GitLabService(config).fetch_projects(search, int(max_projects))
                    st.session_state["gitlab_projects"] = projects
                st.success(f"Found {len(projects)} repositories.")
            except Exception as exc:
                st.error(f"GitLab scan failed: {exc}")

        if st.session_state.get("gitlab_projects"):
            html_table(
                pd.DataFrame([
                    {"Project": p.get("name_with_namespace", p.get("name")), "ID": p.get("id"),
                     "Default branch": p.get("default_branch"), "Last activity": str(p.get("last_activity_at", ""))[:10]}
                    for p in st.session_state["gitlab_projects"]
                ]),
                height=300,
            )

    if st.button("Clear loaded telemetry"):
        for k, v in {"jira_issues": [], "gitlab_projects": [], "gitlab_cache": {}, "jira_meta": {}}.items():
            st.session_state[k] = v
        st.session_state.pop("gitlab_active_project", None)
        st.success("Loaded telemetry cleared.")
        st.rerun()


# =====================================================================
# 27. SETTINGS
# =====================================================================

def render_settings(config: AppConfig):
    render_hero("System settings", "Connection status, thresholds and runtime controls.", "Platform")

    a, b = st.columns(2)
    with a:
        with panel("Jira connection"):
            st.text_input("Base URL", value=config.jira_base_url, disabled=True, key="settings_jira_base_url")
            st.text_input("Account", value=config.jira_email, disabled=True, key="settings_jira_email")
            st.text_input("API token", value="Configured" if config.jira_api_token else "Not configured", disabled=True, key="settings_jira_api_token")
            st.text_input("Sprint field", value=config.jira_sprint_field, disabled=True, key="settings_sprint_field")
            st.text_input("Story points field", value=config.jira_story_points_field, disabled=True, key="settings_sp_field")
            if st.button("Test Jira connection", key="test_jira"):
                try:
                    st.success(f"Connected as {JiraService(config).test_connection()}")
                except Exception as exc:
                    st.error(str(exc))
        with panel("GitLab connection"):
            st.text_input("Base URL", value=config.gitlab_base_url, disabled=True, key="settings_gitlab_base_url")
            st.text_input("Private token", value="Configured" if config.gitlab_private_token else "Not configured", disabled=True, key="settings_gitlab_private_token")
            if st.button("Test GitLab connection", key="test_gitlab"):
                try:
                    st.success(f"Connected as {GitLabService(config).test_connection()}")
                except Exception as exc:
                    st.error(str(exc))

    with b:
        with panel("Analytics thresholds", "Used to flag stale work across all dashboards"):
            seed("w_stale_days", int(st.session_state.get("stale_days", 5)))
            st.session_state["stale_days"] = int(st.number_input("Stale after (days without update)", 1, 60, key="w_stale_days"))
        with panel("LLM"):
            st.text_input("Endpoint", value=config.llm_base_url, disabled=True, key="settings_llm_endpoint")
            st.text_input("Model", value=config.llm_model, disabled=True, key="settings_llm_model")
            st.text_input("API key", value="Configured" if config.llm_api_key else "Not configured", disabled=True, key="settings_llm_api_key")
        with panel("Runtime", "Reset everything loaded in this browser session"):
            if st.button("Clear session state", key="settings_clear_session"):
                st.session_state.clear()
                st.success("Runtime state cleared.")
                st.rerun()


# =====================================================================
# 28. MAIN APPLICATION
# =====================================================================

def main():
    st.set_page_config(page_title="Engineering Intelligence Hub", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

    _PANEL_N[0] = 0
    st.session_state.setdefault("dark_mode", True)
    st.session_state.setdefault("stale_days", 5)
    inject_custom_styles()

    config = AppConfig.load_from_env_or_secrets()

    st.session_state.setdefault("jira_issues", [])
    st.session_state.setdefault("jira_meta", {})
    st.session_state.setdefault("gitlab_projects", [])
    st.session_state.setdefault("gitlab_cache", {})
    for name, _, _ in FILTERS:
        st.session_state.setdefault(f"filter_{name}", "All")

    jira_df = prepare_jira_dataframe(st.session_state["jira_issues"], st.session_state["stale_days"])

    def with_filters(fn, hide: Tuple[str, ...] = ()):
        def page():
            render_global_filters(jira_df, hide)
            fn(jira_df)
        page.__name__ = fn.__name__ + "_page"
        return page

    def activity_page():
        render_engineering(st.session_state["gitlab_projects"], config)

    def trace_page():
        render_traceability(jira_df)

    def ai_page():
        render_ai_insights(jira_df, config)

    def sources_page():
        render_data_sources(config)

    def settings_page():
        render_settings(config)

    pages = {
        "Overview": [
            st.Page(with_filters(render_executive_overview), title="Executive Overview", icon="🏠", url_path="overview", default=True),
        ],
        "Agile": [
            st.Page(with_filters(render_sprint_analysis, hide=("sprint",)), title="Sprint Analysis", icon="🏃", url_path="sprints"),
            st.Page(with_filters(render_flow_metrics), title="Flow Metrics", icon="🌊", url_path="flow"),
            st.Page(with_filters(render_planning), title="Backlog & Planning", icon="🗂️", url_path="planning"),
        ],
        "Delivery": [
            st.Page(with_filters(render_delivery_dashboard), title="Delivery Dashboard", icon="📊", url_path="delivery"),
            st.Page(with_filters(render_team_dashboard), title="Team Dashboard", icon="👥", url_path="team"),
            st.Page(with_filters(render_risks), title="Risks & Attention", icon="⚠️", url_path="risks"),
        ],
        "Engineering": [
            st.Page(activity_page, title="Engineering Activity", icon="🔀", url_path="activity"),
            st.Page(trace_page, title="Traceability", icon="⛓️", url_path="traceability"),
        ],
        "Explore": [
            st.Page(with_filters(render_issue_explorer), title="Jira Issue Explorer", icon="🔎", url_path="issues"),
            st.Page(ai_page, title="AI Insights", icon="🤖", url_path="ai"),
        ],
        "Admin": [
            st.Page(sources_page, title="Data Sources", icon="🔌", url_path="sources"),
            st.Page(settings_page, title="Settings", icon="⚙️", url_path="settings"),
        ],
    }
    navigation = st.navigation(pages, position="sidebar")

    tel = active_gitlab()
    meta = st.session_state["jira_meta"]
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">⚡</div>'
            '<div><div class="brand-name">Engineering Hub</div>'
            '<div class="brand-sub">Delivery intelligence</div></div></div>',
            unsafe_allow_html=True,
        )
        st.toggle("Dark mode", key="dark_mode")
        if meta and st.button("↻ Refresh Jira", key="sidebar_refresh"):
            try:
                with st.spinner("Refreshing Jira…"):
                    load_jira(config, meta["project"], meta["jql"], meta["limit"], meta["order"], meta["changelog"])
                st.rerun()
            except Exception as exc:
                st.error(f"Jira refresh failed: {exc}")
        st.markdown("---")
        st.markdown(
            f'<div class="side-stats"><strong>Loaded data</strong><br>'
            f'Jira issues: <strong>{len(st.session_state["jira_issues"])}</strong>'
            f'{" · " + meta["fetched_at"][11:16] if meta else ""}<br>'
            f'GitLab repo: <strong>{esc(tel.get("project_name", "none")[:28]) if tel else "none"}</strong><br>'
            f'Merge requests: <strong>{len(tel.get("merge_requests", [])) if tel else 0}</strong><br>'
            f'Commits: <strong>{len(tel.get("commits", [])) if tel else 0}</strong></div>',
            unsafe_allow_html=True,
        )

    navigation.run()


if __name__ == "__main__":
    main()
