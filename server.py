# ============================================================
# Module: MCP Server Entry Point (server.py)
# 模块：MCP 服务器主入口
#
# Starts the Ombre Brain MCP service and registers memory
# operation tools for Claude to call.
# 启动 Ombre Brain MCP 服务，注册记忆操作工具供 Claude 调用。
#
# Core responsibilities:
# 核心职责：
#   - Initialize config, bucket manager, dehydrator, decay engine
#     初始化配置、记忆桶管理器、脱水器、衰减引擎
#   - Expose 6 MCP tools:
#     暴露 6 个 MCP 工具：
#       breath — Surface unresolved memories or search by keyword
#                浮现未解决记忆 或 按关键词检索
#       hold   — Store a single memory (or write a `feel` reflection)
#                存储单条记忆（或写 feel 反思）
#       grow   — Diary digest, auto-split into multiple buckets
#                日记归档，自动拆分多桶
#       trace  — Modify metadata / resolved / delete
#                修改元数据 / resolved 标记 / 删除
#       pulse  — Compact system health summary (hard-capped)
#                紧凑系统体检（硬限制输出）
#       dream  — Surface recent dynamic buckets for self-digestion
#                返回最近桶 供模型自省/写 feel
#
# Startup:
# 启动方式：
#   Local:  python server.py
#   Remote: OMBRE_TRANSPORT=streamable-http python server.py
#   Docker: docker-compose up
# ============================================================

import os
import sys
import random
import logging
import asyncio
import hashlib
import hmac
import secrets
import time
import json as _json_lib
import httpx
from datetime import datetime, timedelta, timezone


# --- Ensure same-directory modules can be imported ---
# --- 确保同目录下的模块能被正确导入 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

from bucket_manager import BucketManager
from dehydrator import Dehydrator
from decay_engine import DecayEngine
from embedding_engine import EmbeddingEngine
from import_memory import ImportEngine
from surface_audit import SurfaceAuditLog
from utils import load_config, setup_logging, strip_wikilinks, count_tokens_approx, get_ai_name

# --- Load config & init logging / 加载配置 & 初始化日志 ---
config = load_config()
setup_logging(config.get("log_level", "INFO"))
logger = logging.getLogger("ombre_brain")

# --- Runtime env vars (bind host + port + webhook) / 运行时环境变量 ---
# OMBRE_BIND_HOST: HTTP/SSE 监听地址，默认 0.0.0.0 以保持容器部署兼容。
# 裸机仅经本地反代访问时建议设为 127.0.0.1。
OMBRE_BIND_HOST = os.environ.get("OMBRE_BIND_HOST", "0.0.0.0").strip() or "0.0.0.0"

# OMBRE_PORT: HTTP/SSE 监听端口，默认 8000
try:
    OMBRE_PORT = int(os.environ.get("OMBRE_PORT", "8000") or "8000")
except ValueError:
    logger.warning("OMBRE_PORT 不是合法整数，回退到 8000")
    OMBRE_PORT = 8000

# OMBRE_HOOK_URL: 在 breath/dream 被调用后推送事件到该 URL（POST JSON）。
# OMBRE_HOOK_SKIP: 设为 true/1/yes 跳过推送。
# 详见 ENV_VARS.md。
OMBRE_HOOK_URL = os.environ.get("OMBRE_HOOK_URL", "").strip()
OMBRE_HOOK_SKIP = os.environ.get("OMBRE_HOOK_SKIP", "").strip().lower() in ("1", "true", "yes", "on")


async def _fire_webhook(event: str, payload: dict) -> None:
    """
    Fire-and-forget POST to OMBRE_HOOK_URL with the given event payload.
    Failures are logged at WARNING level only — never propagated to the caller.
    """
    if OMBRE_HOOK_SKIP or not OMBRE_HOOK_URL:
        return
    try:
        body = {
            "event": event,
            "timestamp": time.time(),
            "payload": payload,
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(OMBRE_HOOK_URL, json=body)
    except Exception as e:
        logger.warning(f"Webhook push failed ({event} → {OMBRE_HOOK_URL}): {e}")

# --- Initialize core components / 初始化核心组件 ---
embedding_engine = EmbeddingEngine(config)            # Embedding engine first (BucketManager depends on it)
bucket_mgr = BucketManager(config, embedding_engine=embedding_engine)  # Bucket manager / 记忆桶管理器
dehydrator = Dehydrator(config)                      # Dehydrator / 脱水器
decay_engine = DecayEngine(config, bucket_mgr)       # Decay engine / 衰减引擎
bucket_mgr.set_activation_callback(
    lambda _bucket_id, mode: decay_engine.relationship_clock.resume(f"bucket_{mode}")
)
import_engine = ImportEngine(config, bucket_mgr, dehydrator, embedding_engine)  # Import engine / 导入引擎
surface_audit = SurfaceAuditLog(config["buckets_dir"], max_events=50)

# --- Create MCP server instance / 创建 MCP 服务器实例 ---
# OMBRE_BIND_HOST defaults to 0.0.0.0 so Docker SSE remains externally reachable.
# stdio mode ignores host (no network)
mcp = FastMCP(
    "Ombre Brain",
    host=OMBRE_BIND_HOST,
    port=OMBRE_PORT,
)


# =============================================================
# Dashboard Auth — simple cookie-based session auth
# Dashboard 认证 —— 基于 Cookie 的会话认证
#
# Env var OMBRE_DASHBOARD_PASSWORD overrides file-stored password.
# First visit with no password set → forced setup wizard.
# Sessions stored in memory (lost on restart, 7-day expiry).
# =============================================================
_sessions: dict[str, float] = {}  # {token: expiry_timestamp}


def _get_auth_file() -> str:
    return os.path.join(config["buckets_dir"], ".dashboard_auth.json")


def _load_password_hash() -> str | None:
    try:
        auth_file = _get_auth_file()
        if os.path.exists(auth_file):
            with open(auth_file, "r", encoding="utf-8") as f:
                return _json_lib.load(f).get("password_hash")
    except Exception:
        pass
    return None


def _save_password_hash(password: str) -> None:
    salt = secrets.token_hex(16)
    h = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    auth_file = _get_auth_file()
    os.makedirs(os.path.dirname(auth_file), exist_ok=True)
    with open(auth_file, "w", encoding="utf-8") as f:
        _json_lib.dump({"password_hash": f"{salt}:{h}"}, f)


def _verify_password_hash(password: str, stored: str) -> bool:
    if ":" not in stored:
        return False
    salt, h = stored.split(":", 1)
    return hmac.compare_digest(
        h, hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    )


def _is_setup_needed() -> bool:
    """True if no password is configured (env var or file)."""
    if os.environ.get("OMBRE_DASHBOARD_PASSWORD", ""):
        return False
    return _load_password_hash() is None


def _verify_any_password(password: str) -> bool:
    """Check password against env var (first) or stored hash."""
    env_pwd = os.environ.get("OMBRE_DASHBOARD_PASSWORD", "")
    if env_pwd:
        return hmac.compare_digest(password, env_pwd)
    stored = _load_password_hash()
    if not stored:
        return False
    return _verify_password_hash(password, stored)


def _create_session() -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time() + 86400 * 7  # 7-day expiry
    return token


def _is_authenticated(request) -> bool:
    token = request.cookies.get("ombre_session")
    if not token:
        return False
    expiry = _sessions.get(token)
    if expiry is None or time.time() > expiry:
        _sessions.pop(token, None)
        return False
    return True


def _require_auth(request):
    """Return JSONResponse(401) if not authenticated, else None."""
    from starlette.responses import JSONResponse
    if not _is_authenticated(request):
        return JSONResponse(
            {"error": "Unauthorized", "setup_needed": _is_setup_needed()},
            status_code=401,
        )
    return None


# --- Auth endpoints ---
@mcp.custom_route("/auth/status", methods=["GET"])
async def auth_status(request):
    """Return auth state (authenticated, setup_needed)."""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "authenticated": _is_authenticated(request),
        "setup_needed": _is_setup_needed(),
    })


@mcp.custom_route("/auth/setup", methods=["POST"])
async def auth_setup_endpoint(request):
    """Initial password setup (only when no password is configured)."""
    from starlette.responses import JSONResponse
    if not _is_setup_needed():
        return JSONResponse({"error": "Already configured"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    password = body.get("password", "").strip()
    if len(password) < 6:
        return JSONResponse({"error": "密码不能少于6位"}, status_code=400)
    _save_password_hash(password)
    token = _create_session()
    resp = JSONResponse({"ok": True})
    resp.set_cookie("ombre_session", token, httponly=True, samesite="lax", secure=True, max_age=86400 * 7)
    return resp


@mcp.custom_route("/auth/login", methods=["POST"])
async def auth_login(request):
    """Login with password."""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    password = body.get("password", "")
    if _verify_any_password(password):
        token = _create_session()
        resp = JSONResponse({"ok": True})
        resp.set_cookie("ombre_session", token, httponly=True, samesite="lax", secure=True, max_age=86400 * 7)
        return resp
    return JSONResponse({"error": "密码错误"}, status_code=401)


@mcp.custom_route("/auth/logout", methods=["POST"])
async def auth_logout(request):
    """Invalidate session."""
    from starlette.responses import JSONResponse
    token = request.cookies.get("ombre_session")
    if token:
        _sessions.pop(token, None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("ombre_session")
    return resp


@mcp.custom_route("/auth/change-password", methods=["POST"])
async def auth_change_password(request):
    """Change dashboard password (requires current password)."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err:
        return err
    if os.environ.get("OMBRE_DASHBOARD_PASSWORD", ""):
        return JSONResponse({"error": "当前使用环境变量密码，请直接修改 OMBRE_DASHBOARD_PASSWORD"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    current = body.get("current", "")
    new_pwd = body.get("new", "").strip()
    if not _verify_any_password(current):
        return JSONResponse({"error": "当前密码错误"}, status_code=401)
    if len(new_pwd) < 6:
        return JSONResponse({"error": "新密码不能少于6位"}, status_code=400)
    _save_password_hash(new_pwd)
    _sessions.clear()
    token = _create_session()
    resp = JSONResponse({"ok": True})
    resp.set_cookie("ombre_session", token, httponly=True, samesite="lax", secure=True, max_age=86400 * 7)
    return resp


# =============================================================
# /health endpoint: lightweight keepalive
# 轻量保活接口
# For Cloudflare Tunnel or reverse proxy to ping, preventing idle timeout
# 供 Cloudflare Tunnel 或反代定期 ping，防止空闲超时断连
# =============================================================
@mcp.custom_route("/", methods=["GET"])
async def root_redirect(request):
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    from starlette.responses import JSONResponse
    try:
        # The engine is intentionally lazy-started because FastMCP owns the
        # application event loop.  /health is pinged shortly after startup and
        # every 60 seconds, so it is also our reliable post-redeploy bootstrap.
        await decay_engine.ensure_started()
        stats = await bucket_mgr.get_stats()
        return JSONResponse({
            "status": "ok",
            "buckets": stats["permanent_count"] + stats["dynamic_count"],
            "decay_engine": "running" if decay_engine.is_running else "stopped",
        })
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# =============================================================
# /breath-hook endpoint: Dedicated hook for SessionStart
# 会话启动专用挂载点
# =============================================================
DREAM_RECENT_LIMIT = 10


def _select_dream_recent(all_buckets: list[dict], limit: int = DREAM_RECENT_LIMIT) -> list[dict]:
    """Return the newest surface-level memories reserved for Dreaming.

    Both breath paths call this before weight ranking so the same memories are
    not injected twice during startup.  dream() and /dream-hook use this exact
    helper as well, keeping the reservation deterministic without shared
    per-session state.
    返回留给 Dreaming 的最新表层记忆。breath 与 dream 共用同一筛选函数，
    无需跨调用状态也能避免一次开窗重复注入。
    """
    candidates = [
        b for b in all_buckets
        if b["metadata"].get("type") not in ("permanent", "feel", "letter", "crave")
        and not b["metadata"].get("pinned", False)
        and not b["metadata"].get("protected", False)
    ]
    candidates.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
    return candidates[:limit]


def _weight_rank_snapshot(all_buckets: list[dict]) -> tuple[list[dict], dict[str, int], dict[str, float]]:
    """Snapshot Breath's pre-reservation unresolved ranking without touching buckets."""
    eligible = [
        b for b in all_buckets
        if not b["metadata"].get("resolved", False)
        and b["metadata"].get("type") not in ("permanent", "feel", "letter", "crave")
        and not b["metadata"].get("pinned", False)
        and not b["metadata"].get("protected", False)
    ]
    scores = {
        b["id"]: float(decay_engine.calculate_score(b["metadata"]))
        for b in eligible
    }
    ranked = sorted(eligible, key=lambda b: scores[b["id"]], reverse=True)
    ranks = {b["id"]: index for index, b in enumerate(ranked, start=1)}
    return ranked, ranks, scores


def _audit_bucket_entry(bucket: dict, **fields) -> dict:
    meta = bucket.get("metadata", {})
    return {
        "id": bucket.get("id", ""),
        "name": meta.get("name", bucket.get("id", "")),
        "type": meta.get("type", "dynamic"),
        "created": meta.get("created", ""),
        **fields,
    }


def _record_surface_audit(flow: str, entries: list[dict], **context) -> None:
    """Audit failures are diagnostic-only and must never break memory recall."""
    try:
        surface_audit.record(flow, entries, **context)
    except Exception as exc:
        logger.warning("Surface audit write failed (%s): %s", flow, type(exc).__name__)


@mcp.custom_route("/breath-hook", methods=["GET"])
async def breath_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        decay_engine.relationship_clock.resume("breath_hook")
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        _ranked_all, weight_ranks, score_snapshot = _weight_rank_snapshot(all_buckets)
        dream_recent = _select_dream_recent(all_buckets)
        dream_ids = {b["id"] for b in dream_recent}
        audit_entries = [
            _audit_bucket_entry(
                b,
                channel="dream_reserved",
                score=score_snapshot.get(b["id"]),
                weight_rank=weight_ranks.get(b["id"]),
                newest_position=index,
                outcome="reserved_for_dream",
            )
            for index, b in enumerate(dream_recent, start=1)
        ]
        # pinned
        pinned = [b for b in all_buckets if b["metadata"].get("pinned") or b["metadata"].get("protected")]
        # unresolved by score, excluding the newest memories reserved for dream
        unresolved = [b for b in all_buckets
                      if not b["metadata"].get("resolved", False)
                      and b["metadata"].get("type") not in ("permanent", "feel", "letter", "crave")
                      and not b["metadata"].get("pinned")
                      and not b["metadata"].get("protected")
                      and b["id"] not in dream_ids]
        scored = sorted(unresolved, key=lambda b: decay_engine.calculate_score(b["metadata"]), reverse=True)
        breath_ranks = {b["id"]: index for index, b in enumerate(scored, start=1)}

        parts = []
        token_budget = 10000
        for b in pinned:
            entry = _audit_bucket_entry(b, channel="pin", outcome="selected")
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"})
            summary_tokens = count_tokens_approx(summary)
            parts.append(f"📌 [核心准则] {summary}")
            token_budget -= summary_tokens
            entry.update(
                outcome="surfaced",
                output_position=len(parts),
                summary_tokens=summary_tokens,
            )
            audit_entries.append(entry)

        # Diversity: top-1 fixed + shuffle rest from top-20
        candidates = list(scored)
        if len(candidates) > 1:
            top1 = [candidates[0]]
            pool = candidates[1:min(20, len(candidates))]
            random.shuffle(pool)
            candidates = top1 + pool + candidates[min(20, len(candidates)):]
        # Hard cap: max 20 surfacing buckets in hook
        candidates = candidates[:20]

        candidate_entries = []
        for index, b in enumerate(candidates, start=1):
            candidate_entries.append(_audit_bucket_entry(
                b,
                channel="dynamic",
                score=score_snapshot.get(b["id"]),
                weight_rank=weight_ranks.get(b["id"]),
                breath_rank=breath_ranks.get(b["id"]),
                candidate_position=index,
                cold_start=False,
                outcome="selected",
            ))

        stopped = False
        dynamic_returned = 0
        for b, entry in zip(candidates, candidate_entries):
            if stopped:
                entry.update(outcome="not_attempted_after_break")
                continue
            if token_budget <= 0:
                entry.update(outcome="token_exhausted", budget_before=token_budget)
                stopped = True
                continue
            budget_before = token_budget
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"})
            summary_tokens = count_tokens_approx(summary)
            if summary_tokens > token_budget:
                entry.update(
                    outcome="summary_exceeds_budget",
                    reason="summary_too_large",
                    summary_tokens=summary_tokens,
                    budget_before=budget_before,
                )
                stopped = True
                continue
            await bucket_mgr.soft_touch(b["id"])
            parts.append(summary)
            token_budget -= summary_tokens
            dynamic_returned += 1
            entry.update(
                outcome="surfaced",
                output_position=dynamic_returned,
                summary_tokens=summary_tokens,
                budget_before=budget_before,
            )

        audit_entries.extend(candidate_entries)
        _record_surface_audit(
            "breath_hook",
            audit_entries,
            total_buckets=len(all_buckets),
            pinned_count=len(pinned),
            dynamic_pool_count=len(_ranked_all),
            dream_reserved_count=len(dream_recent),
            candidate_count=len(candidates),
            returned_count=len(parts),
            pinned_returned_count=len(parts) - dynamic_returned,
            dynamic_returned_count=dynamic_returned,
            max_results=20,
            max_tokens=10000,
            remaining_tokens=token_budget,
            status="complete",
        )

        if not parts:
            await _fire_webhook("breath_hook", {"surfaced": 0})
            return PlainTextResponse("")
        body_text = "[Ombre Brain - 记忆浮现]\n" + "\n---\n".join(parts)
        await _fire_webhook("breath_hook", {"surfaced": len(parts), "chars": len(body_text)})
        return PlainTextResponse(body_text)
    except Exception as e:
        logger.warning(f"Breath hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# /dream-hook endpoint: Dedicated hook for Dreaming
# Dreaming 专用挂载点
# =============================================================
@mcp.custom_route("/dream-hook", methods=["GET"])
async def dream_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        recent = _select_dream_recent(all_buckets)

        if not recent:
            _record_surface_audit(
                "dream_hook", [], total_buckets=len(all_buckets), returned_count=0, status="complete"
            )
            return PlainTextResponse("")

        try:
            ranked, weight_ranks, scores = _weight_rank_snapshot(all_buckets)
        except Exception as exc:
            logger.warning("Dream hook rank snapshot failed: %s", type(exc).__name__)
            ranked, weight_ranks, scores = [], {}, {}

        parts = []
        for b in recent:
            meta = b["metadata"]
            resolved_tag = "[已解决]" if meta.get("resolved", False) else "[未解决]"
            parts.append(
                f"{meta.get('name', b['id'])} {resolved_tag} "
                f"V{meta.get('valence', 0.5):.1f}/A{meta.get('arousal', 0.3):.1f}\n"
                f"{strip_wikilinks(b['content'][:200])}"
            )

        body_text = "[Ombre Brain - Dreaming]\n" + "\n---\n".join(parts)
        _record_surface_audit(
            "dream_hook",
            [
                _audit_bucket_entry(
                    b,
                    channel="dream",
                    score=scores.get(b["id"]),
                    weight_rank=weight_ranks.get(b["id"]),
                    newest_position=index,
                    output_position=index,
                    outcome="surfaced",
                )
                for index, b in enumerate(recent, start=1)
            ],
            total_buckets=len(all_buckets),
            dynamic_pool_count=len(ranked),
            returned_count=len(recent),
            status="complete",
        )
        await _fire_webhook("dream_hook", {"surfaced": len(parts), "chars": len(body_text)})
        return PlainTextResponse(body_text)
    except Exception as e:
        logger.warning(f"Dream hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# Internal helper: merge-or-create
# 内部辅助：检查是否可合并，可以则合并，否则新建
# Shared by hold and grow to avoid duplicate logic
# hold 和 grow 共用，避免重复逻辑
# =============================================================
async def _merge_or_create(
    content: str,
    tags: list,
    importance: int,
    domain: list,
    valence: float,
    arousal: float,
    name: str = "",
    verbatim: bool = False,
) -> tuple[str, bool]:
    """
    Check if a similar bucket exists for merging; merge if so, create if not.
    Returns (bucket_id_or_name, is_merged).
    检查是否有相似桶可合并，有则合并，无则新建。
    返回 (桶ID或名称, 是否合并)。
    """
    try:
        existing = await bucket_mgr.search(content, limit=1, domain_filter=domain or None)
    except Exception as e:
        logger.warning(f"Search for merge failed, creating new / 合并搜索失败，新建: {e}")
        existing = []

    if existing and existing[0].get("score", 0) > config.get("merge_threshold", 75):
        bucket = existing[0]
        # --- Never merge into pinned/protected buckets ---
        # --- 不合并到钉选/保护桶 ---
        if not (bucket["metadata"].get("pinned") or bucket["metadata"].get("protected")):
            try:
                merged = await dehydrator.merge(bucket["content"], content)
                old_v = bucket["metadata"].get("valence", 0.5)
                old_a = bucket["metadata"].get("arousal", 0.3)
                merged_valence = round((old_v + valence) / 2, 2)
                merged_arousal = round((old_a + arousal) / 2, 2)
                await bucket_mgr.update(
                    bucket["id"],
                    content=merged,
                    tags=list(set(bucket["metadata"].get("tags", []) + tags)),
                    importance=max(bucket["metadata"].get("importance", 5), importance),
                    domain=list(set(bucket["metadata"].get("domain", []) + domain)),
                    valence=merged_valence,
                    arousal=merged_arousal,
                )
                # --- Update embedding after merge ---
                try:
                    await embedding_engine.generate_and_store(bucket["id"], merged)
                except Exception:
                    pass
                return bucket["metadata"].get("name", bucket["id"]), True
            except Exception as e:
                logger.warning(f"Merge failed, creating new / 合并失败，新建: {e}")

    bucket_id = await bucket_mgr.create(
        content=content,
        tags=tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=name or None,
        verbatim=verbatim,
    )
    # --- Generate embedding for new bucket ---
    try:
        await embedding_engine.generate_and_store(bucket_id, content)
    except Exception:
        pass
    return bucket_id, False


# =============================================================
# Tool 1: breath — Breathe
# 工具 1：breath — 呼吸
#
# No args: surface highest-weight unresolved memories (active push)
# 无参数：浮现权重最高的未解决记忆
# With args: search by keyword + emotion coordinates
# 有参数：按关键词+情感坐标检索记忆
# =============================================================
@mcp.tool()
async def breath(
    query: str = "",
    max_tokens: int = 10000,
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    max_results: int = 20,
    importance_min: int = -1,
    bucket_id: str = "",
    limit: int = -1,
) -> str:
    """检索/浮现记忆。不传query或传空=自动浮现,有query=关键词检索。max_tokens控制返回总token上限(默认10000)。domain逗号分隔,valence/arousal 0~1(-1忽略)。max_results控制浮现模式返回数量上限(默认20,最大50)。importance_min>=1时按重要度批量拉取(不走语义搜索,按importance降序返回最多20条)。bucket_id传入桶ID时直接获取该桶内容返回(不走搜索)。limit>=1时在关键词搜索模式下限制返回条数(不填则返回所有匹配结果)。"""
    decay_engine.relationship_clock.resume("breath")
    await decay_engine.ensure_started()
    max_results = min(max_results, 50)
    max_tokens = min(max_tokens, 20000)

    # --- bucket_id mode: fetch single bucket by ID ---
    # --- 桶ID模式：直接按ID获取单个桶 ---
    if bucket_id and bucket_id.strip():
        try:
            bucket = await bucket_mgr.get(bucket_id.strip())
            if not bucket:
                return f"未找到桶 {bucket_id}"
            clean_meta = {k: v for k, v in bucket["metadata"].items() if k != "tags"}
            is_verbatim = bucket["metadata"].get("verbatim", False)
            if is_verbatim:
                clean_meta["verbatim"] = True
            summary = await dehydrator.dehydrate(strip_wikilinks(bucket["content"]), clean_meta)
            await bucket_mgr.touch(bucket["id"])
            return f"[bucket_id:{bucket['id']}] {summary}"
        except Exception as e:
            logger.error(f"Bucket ID fetch failed / 桶ID获取失败: {e}")
            return f"获取桶 {bucket_id} 失败: {e}"

    # --- importance_min mode: bulk fetch by importance threshold ---
    # --- 重要度批量拉取模式：跳过语义搜索，按 importance 降序返回 ---
    if importance_min >= 1:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            return f"记忆系统暂时无法访问: {e}"
        filtered = [
            b for b in all_buckets
            if int(b["metadata"].get("importance", 0)) >= importance_min
            and b["metadata"].get("type") not in ("feel", "letter", "crave")
        ]
        filtered.sort(key=lambda b: int(b["metadata"].get("importance", 0)), reverse=True)
        filtered = filtered[:20]
        if not filtered:
            return f"没有重要度 >= {importance_min} 的记忆。"
        results = []
        token_used = 0
        for b in filtered:
            if token_used >= max_tokens:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                t = count_tokens_approx(summary)
                if token_used + t > max_tokens:
                    break
                imp = b["metadata"].get("importance", 0)
                results.append(f"[importance:{imp}] [bucket_id:{b['id']}] {summary}")
                token_used += t
            except Exception as e:
                logger.warning(f"importance_min dehydrate failed: {e}")
        return "\n---\n".join(results) if results else "没有可以展示的记忆。"

    # --- Feel retrieval: domain="feel" is a special channel ---
    # --- Feel 检索：domain="feel" 是独立入口 ---
    if domain.strip().lower() == "feel":
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            feels.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
            if not feels:
                _record_surface_audit(
                    "feel", [], total_buckets=len(all_buckets), returned_count=0, status="complete"
                )
                return "没有留下过 feel。"
            feels = feels[:10]
            results = []
            audit_entries = []
            for newest_position, f in enumerate(feels, start=1):
                created = f["metadata"].get("created", "")
                entry = f"[{created}] [bucket_id:{f['id']}]\n{strip_wikilinks(f['content'])}"
                results.append(entry)
                audit_entries.append(_audit_bucket_entry(
                    f,
                    channel="feel",
                    newest_position=newest_position,
                    output_position=len(results),
                    outcome="surfaced",
                ))
                if count_tokens_approx("\n---\n".join(results)) > max_tokens:
                    break
            _record_surface_audit(
                "feel",
                audit_entries,
                total_buckets=len(all_buckets),
                candidate_count=len(feels),
                returned_count=len(results),
                max_results=10,
                max_tokens=max_tokens,
                status="complete",
            )
            return "=== 你留下的 feel ===\n" + "\n---\n".join(results)
        except Exception as e:
            logger.error(f"Feel retrieval failed: {e}")
            _record_surface_audit(
                "feel", [], returned_count=0, status="error", error=type(e).__name__
            )
            return "读取 feel 失败。"

    # --- Crave retrieval: domain="crave" is a special channel ---
    # --- Crave 检索：domain="crave" 是独立入口，不参与普通浮现/搜索 ---
    if domain.strip().lower() == "crave":
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            craves = [
                b for b in all_buckets
                if b["metadata"].get("type") == "crave" and not b["metadata"].get("digested", False)
            ]
            craves.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
            if not craves:
                _record_surface_audit(
                    "crave", [], total_buckets=len(all_buckets), returned_count=0, status="complete"
                )
                return "没有存过 crave。"
            craves = craves[:10]
            results = []
            audit_entries = []
            for newest_position, c in enumerate(craves, start=1):
                created = c["metadata"].get("created", "")
                title = c["metadata"].get("name", "")
                title = title if title and title != c["id"] else ""
                entry = f"[{created}] [bucket_id:{c['id']}]{(' · ' + title) if title else ''}\n{strip_wikilinks(c['content'])}"
                results.append(entry)
                audit_entries.append(_audit_bucket_entry(
                    c,
                    channel="crave",
                    newest_position=newest_position,
                    output_position=len(results),
                    outcome="surfaced",
                ))
                if count_tokens_approx("\n---\n".join(results)) > max_tokens:
                    break
            _record_surface_audit(
                "crave",
                audit_entries,
                total_buckets=len(all_buckets),
                candidate_count=len(craves),
                returned_count=len(results),
                max_results=10,
                max_tokens=max_tokens,
                status="complete",
            )
            return "=== crave ===\n" + "\n---\n".join(results)
        except Exception as e:
            logger.error(f"Crave retrieval failed: {e}")
            _record_surface_audit(
                "crave", [], returned_count=0, status="error", error=type(e).__name__
            )
            return "读取 crave 失败。"

    # --- No args or empty query: surfacing mode (weight pool active push) ---
    # --- 无参数或空query：浮现模式（权重池主动推送）---
    if not query or not query.strip():
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"Failed to list buckets for surfacing / 浮现列桶失败: {e}")
            return "记忆系统暂时无法访问。"

        # Reserve Dreaming's newest memories before breath weight ranking.
        # 在权重排序前先留出 Dreaming 的最新记忆，避免启动流程重复注入。
        ranked_all, weight_ranks, score_snapshot = _weight_rank_snapshot(all_buckets)
        dream_recent = _select_dream_recent(all_buckets)
        dream_ids = {b["id"] for b in dream_recent}
        audit_entries = [
            _audit_bucket_entry(
                b,
                channel="dream_reserved",
                score=score_snapshot.get(b["id"]),
                weight_rank=weight_ranks.get(b["id"]),
                newest_position=index,
                outcome="reserved_for_dream",
            )
            for index, b in enumerate(dream_recent, start=1)
        ]

        # --- Pinned/protected buckets: always surface as core principles ---
        # --- 钉选桶：作为核心准则，始终浮现 ---
        pinned_buckets = [
            b for b in all_buckets
            if b["metadata"].get("pinned") or b["metadata"].get("protected")
        ]
        pinned_results = []
        for b in pinned_buckets:
            audit_entry = _audit_bucket_entry(b, channel="pin", outcome="selected")
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                pinned_results.append(f"📌 [核心准则] [bucket_id:{b['id']}] {summary}")
                audit_entry.update(
                    outcome="surfaced",
                    output_position=len(pinned_results),
                    summary_tokens=count_tokens_approx(summary),
                )
            except Exception as e:
                logger.warning(f"Failed to dehydrate pinned bucket / 钉选桶脱水失败: {e}")
                audit_entry.update(outcome="dehydrate_error", reason=type(e).__name__)
            audit_entries.append(audit_entry)

        # --- Unresolved buckets: surface top N by weight ---
        # --- 未解决桶：按权重浮现前 N 条 ---
        unresolved = [
            b for b in all_buckets
            if not b["metadata"].get("resolved", False)
            and b["metadata"].get("type") not in ("permanent", "feel", "letter", "crave")
            and not b["metadata"].get("pinned", False)
            and not b["metadata"].get("protected", False)
            and b["id"] not in dream_ids
        ]

        logger.info(
            f"Breath surfacing: {len(all_buckets)} total, "
            f"{len(pinned_buckets)} pinned, {len(unresolved)} unresolved"
        )

        scored = sorted(
            unresolved,
            key=lambda b: score_snapshot[b["id"]],
            reverse=True,
        )
        breath_ranks = {b["id"]: index for index, b in enumerate(scored, start=1)}

        if scored:
            top_scores = [(b["metadata"].get("name", b["id"]), decay_engine.calculate_score(b["metadata"])) for b in scored[:5]]
            logger.info(f"Top unresolved scores: {top_scores}")

        # --- Cold-start detection: never-seen important buckets surface first ---
        # --- 冷启动检测：从未被访问过且重要度>=8的桶优先插入最前面（最多2个）---
        cold_start = [
            b for b in unresolved
            if int(b["metadata"].get("activation_count", 0)) == 0
            and int(b["metadata"].get("importance", 0)) >= 8
        ][:2]
        cold_start_ids = {b["id"] for b in cold_start}
        # Merge: cold_start first, then scored (excluding duplicates)
        scored_deduped = [b for b in scored if b["id"] not in cold_start_ids]
        scored_with_cold = cold_start + scored_deduped

        # --- Token-budgeted surfacing with diversity + hard cap ---
        # --- 按 token 预算浮现，带多样性 + 硬上限 ---
        # Top-1 always surfaces; rest sampled from top-20 for diversity
        token_budget = max_tokens
        for r in pinned_results:
            token_budget -= count_tokens_approx(r)

        candidates = list(scored_with_cold)
        if len(candidates) > 1:
            # Cold-start buckets stay at front; shuffle rest from top-20
            n_cold = len(cold_start)
            non_cold = candidates[n_cold:]
            if len(non_cold) > 1:
                top1 = [non_cold[0]]
                pool = non_cold[1:min(20, len(non_cold))]
                random.shuffle(pool)
                non_cold = top1 + pool + non_cold[min(20, len(non_cold)):]
            candidates = cold_start + non_cold
        # Hard cap: never surface more than max_results buckets
        candidates = candidates[:max_results]

        candidate_entries = [
            _audit_bucket_entry(
                b,
                channel="dynamic",
                score=score_snapshot.get(b["id"]),
                weight_rank=weight_ranks.get(b["id"]),
                breath_rank=breath_ranks.get(b["id"]),
                candidate_position=index,
                cold_start=b["id"] in cold_start_ids,
                outcome="selected",
            )
            for index, b in enumerate(candidates, start=1)
        ]

        dynamic_results = []
        stopped = False
        for b, audit_entry in zip(candidates, candidate_entries):
            if stopped:
                audit_entry.update(outcome="not_attempted_after_break")
                continue
            if token_budget <= 0:
                audit_entry.update(outcome="token_exhausted", budget_before=token_budget)
                stopped = True
                continue
            try:
                budget_before = token_budget
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                summary_tokens = count_tokens_approx(summary)
                if summary_tokens > token_budget:
                    audit_entry.update(
                        outcome="summary_exceeds_budget",
                        reason="summary_too_large",
                        summary_tokens=summary_tokens,
                        budget_before=budget_before,
                    )
                    stopped = True
                    continue
                await bucket_mgr.soft_touch(b["id"])
                score = decay_engine.calculate_score(b["metadata"])
                dynamic_results.append(f"[权重:{score:.2f}] [bucket_id:{b['id']}] {summary}")
                token_budget -= summary_tokens
                audit_entry.update(
                    outcome="surfaced",
                    output_position=len(dynamic_results),
                    summary_tokens=summary_tokens,
                    budget_before=budget_before,
                )
            except Exception as e:
                logger.warning(f"Failed to dehydrate surfaced bucket / 浮现脱水失败: {e}")
                audit_entry.update(outcome="dehydrate_error", reason=type(e).__name__)

        audit_entries.extend(candidate_entries)
        _record_surface_audit(
            "breath",
            audit_entries,
            total_buckets=len(all_buckets),
            pinned_count=len(pinned_buckets),
            dynamic_pool_count=len(ranked_all),
            dream_reserved_count=len(dream_recent),
            candidate_count=len(candidates),
            returned_count=len(pinned_results) + len(dynamic_results),
            pinned_returned_count=len(pinned_results),
            dynamic_returned_count=len(dynamic_results),
            max_results=max_results,
            max_tokens=max_tokens,
            remaining_tokens=token_budget,
            status="complete",
        )

        if not pinned_results and not dynamic_results:
            return "权重池平静，没有需要处理的记忆。"

        parts = []
        if pinned_results:
            parts.append("=== 核心准则 ===\n" + "\n---\n".join(pinned_results))
        if dynamic_results:
            parts.append("=== 浮现记忆 ===\n" + "\n---\n".join(dynamic_results))
        return "\n\n".join(parts)

    # --- With args: search mode (keyword + vector dual channel) ---
    # --- 有参数：检索模式（关键词 + 向量双通道）---
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None

    try:
        matches = await bucket_mgr.search(
            query,
            limit=max(max_results, 20),
            domain_filter=domain_filter,
            query_valence=q_valence,
            query_arousal=q_arousal,
        )
    except Exception as e:
        logger.error(f"Search failed / 检索失败: {e}")
        return "检索过程出错，请稍后重试。"

    # --- Pinned/protected buckets are now included in search results ---
    # --- 钉选桶现在也参与搜索结果 ---

    # --- Vector similarity channel: find semantically related buckets ---
    # --- 向量相似度通道：找到语义相关的桶 ---
    matched_ids = {b["id"] for b in matches}
    try:
        vector_results = await embedding_engine.search_similar(query, top_k=max(max_results, 20))
        for vid, sim_score in vector_results:
            if vid not in matched_ids and sim_score > 0.5:
                bucket = await bucket_mgr.get(vid)
                if bucket and bucket["metadata"].get("type") not in ("feel", "letter", "crave"):
                    bucket["score"] = round(sim_score * 100, 2)
                    bucket["vector_match"] = True
                    matches.append(bucket)
                    matched_ids.add(vid)
    except Exception as e:
        logger.warning(f"Vector search failed, using keyword only / 向量搜索失败: {e}")

    # --- Apply limit to search results if specified ---
    # --- 如果指定了 limit，截取搜索结果前 N 条 ---
    if limit >= 1:
        matches = matches[:limit]

    results = []
    token_used = 0
    for bucket in matches:
        if token_used >= max_tokens:
            break
        try:
            clean_meta = {k: v for k, v in bucket["metadata"].items() if k != "tags"}
            # --- Memory reconstruction: shift displayed valence by current mood ---
            # --- 记忆重构：根据当前情绪微调展示层 valence（±0.1）---
            if q_valence is not None and "valence" in clean_meta:
                original_v = float(clean_meta.get("valence", 0.5))
                shift = (q_valence - 0.5) * 0.2  # ±0.1 max shift
                clean_meta["valence"] = max(0.0, min(1.0, original_v + shift))
            summary = await dehydrator.dehydrate(strip_wikilinks(bucket["content"]), clean_meta)
            summary_tokens = count_tokens_approx(summary)
            if token_used + summary_tokens > max_tokens:
                break
            await bucket_mgr.touch(bucket["id"])
            if bucket.get("vector_match"):
                summary = f"[语义关联] [bucket_id:{bucket['id']}] {summary}"
            else:
                summary = f"[bucket_id:{bucket['id']}] {summary}"
            results.append(summary)
            token_used += summary_tokens
        except Exception as e:
            logger.warning(f"Failed to dehydrate search result / 检索结果脱水失败: {e}")
            continue

    # --- Random surfacing: when search returns < 3, 40% chance to float old memories ---
    # --- 随机浮现：检索结果不足 3 条时，40% 概率从低权重旧桶里漂上来 ---
    if len(matches) < 3 and random.random() < 0.4:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            matched_ids = {b["id"] for b in matches}
            low_weight = [
                b for b in all_buckets
                if b["id"] not in matched_ids
                and decay_engine.calculate_score(b["metadata"]) < 2.0
            ]
            if low_weight:
                drifted = random.sample(low_weight, min(random.randint(1, 3), len(low_weight)))
                drift_results = []
                for b in drifted:
                    clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                    summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                    drift_results.append(f"[surface_type: random]\n{summary}")
                results.append("--- 忽然想起来 ---\n" + "\n---\n".join(drift_results))
        except Exception as e:
            logger.warning(f"Random surfacing failed / 随机浮现失败: {e}")

    if not results:
        await _fire_webhook("breath", {"mode": "empty", "matches": 0})
        return "未找到相关记忆。"

    final_text = "\n---\n".join(results)
    await _fire_webhook("breath", {"mode": "ok", "matches": len(matches), "chars": len(final_text)})
    return final_text


def _tagging_truncation_note(content: str) -> str:
    """
    Non-blocking warning appended to hold/grow's return string when content
    exceeds the tagging call's input limit — content is stored in full either
    way, this only means domain/tags may not reflect anything past that cut.
    Storage never blocks on this: re-entering content to "fix" tags isn't
    worth the friction, the caller just needs to know it happened.
    非阻断式提醒——超过打标输入上限时附加在 hold/grow 返回值里。内容本身
    完整存储，只是标签/分类可能没覆盖到截断点之后的部分。不会因此拒绝存储
    (重新录入去"修"标签不值得这个摩擦成本)，只是让调用方知道发生过这事。
    """
    limit = Dehydrator.ANALYZE_INPUT_LIMIT
    if len(content) > limit:
        return f"\n⚠️内容{len(content)}字，超过打标上限{limit}字，标签/分类只覆盖前{limit}字（内容已完整存储，未截断）"
    return ""


# =============================================================
# Tool 2: hold — Hold on to this
# 工具 2：hold — 握住，留下来
# =============================================================
@mcp.tool()
async def hold(
    content: str,
    tags: str = "",
    importance: int = 5,
    pinned: bool = False,
    feel: bool = False,
    crave: bool = False,
    title: str = "",
    source_bucket: str = "",    valence: float = -1,
    arousal: float = -1,
    verbatim: bool = False,
) -> str:
    """存储单条记忆,自动打标+合并。tags逗号分隔,importance 1-10。pinned=True创建永久钉选桶。feel=True存储你的第一人称感受(不参与普通浮现，必须由你自己提供title作为日记标题)。crave=True存储色色内容,独立池子不参与普通浮现/打标/脱水,只能通过breath(domain="crave")读取,不占用其他记忆的名额。title=feel 必填、crave 可选。source_bucket=被消化的记忆桶ID(feel模式下,标记源记忆为已消化)。verbatim=True原样保留内容不脱水(适用于操作手册、代码等需要精确保留的内容)。"""
    await decay_engine.ensure_started()

    # --- Input validation / 输入校验 ---
    if not content or not content.strip():
        return "内容为空，无法存储。"
    if feel and not crave and (not title or not title.strip()):
        return "feel 需要你自己写一个 title（日记标题）；本次未保存，请补上标题后重试。"

    importance = max(1, min(10, importance))
    extra_tags = [t.strip() for t in tags.split(",") if t.strip()]

    # --- Crave mode: store as crave type, isolated pool, no tagging/dehydration ---
    # --- Crave 模式：存为 crave 类型，独立池子，不打标不脱水 ---
    if crave:
        crave_valence = valence if 0 <= valence <= 1 else 0.5
        crave_arousal = arousal if 0 <= arousal <= 1 else 0.5
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=["__crave__"] + extra_tags,
            importance=importance,
            domain=["crave"],
            valence=crave_valence,
            arousal=crave_arousal,
            name=(title.strip()[:60] or None) if title else None,
            bucket_type="crave",
            verbatim=True,
        )
        return f"🔥crave→{bucket_id}"

    # --- Feel mode: store as feel type, minimal metadata ---
    # --- Feel 模式：存为 feel 类型，最少元数据 ---
    if feel:
        # Feel valence/arousal = model's own perspective
        feel_valence = valence if 0 <= valence <= 1 else 0.5
        feel_arousal = arousal if 0 <= arousal <= 1 else 0.3
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=[],
            importance=5,
            domain=[],
            valence=feel_valence,
            arousal=feel_arousal,
            name=(title.strip()[:60] or None) if title else None,
            bucket_type="feel",
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception:
            pass
        # --- Mark source memory as digested + store model's valence perspective ---
        # --- 标记源记忆为已消化 + 存储模型视角的 valence ---
        if source_bucket and source_bucket.strip():
            try:
                update_kwargs = {"digested": True}
                if 0 <= valence <= 1:
                    update_kwargs["model_valence"] = feel_valence
                await bucket_mgr.update(source_bucket.strip(), **update_kwargs)
            except Exception as e:
                logger.warning(f"Failed to mark source as digested / 标记已消化失败: {e}")
        return f"🫧feel→{bucket_id}"

    # --- Step 1: auto-tagging / 自动打标 ---
    try:
        analysis = await dehydrator.analyze(content)
    except Exception as e:
        logger.warning(f"Auto-tagging failed, using defaults / 自动打标失败: {e}")
        analysis = {
            "domain": ["未分类"], "valence": 0.5, "arousal": 0.3,
            "tags": [], "suggested_name": "",
        }

    domain = analysis["domain"]
    auto_valence = analysis["valence"]
    auto_arousal = analysis["arousal"]
    auto_tags = analysis["tags"]
    suggested_name = analysis.get("suggested_name", "")

    # --- User-supplied valence/arousal takes priority over analyze() result ---
    # --- 用户显式传入的 valence/arousal 优先，analyze() 结果作为 fallback ---
    final_valence = valence if 0 <= valence <= 1 else auto_valence
    final_arousal = arousal if 0 <= arousal <= 1 else auto_arousal

    all_tags = list(dict.fromkeys(auto_tags + extra_tags))

    # --- Pinned buckets bypass merge and are created directly in permanent dir ---
    # --- 钉选桶跳过合并，直接新建到 permanent 目录 ---
    if pinned:
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=all_tags,
            importance=10,
            domain=domain,
            valence=final_valence,
            arousal=final_arousal,
            name=suggested_name or None,
            bucket_type="permanent",
            pinned=True,
            verbatim=verbatim,
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception:
            pass
        return f"📌钉选→{bucket_id} {','.join(domain)}{_tagging_truncation_note(content)}"

    # --- Step 2: merge or create / 合并或新建 ---
    result_name, is_merged = await _merge_or_create(
        content=content,
        tags=all_tags,
        importance=importance,
        domain=domain,
        valence=final_valence,
        arousal=final_arousal,
        name=suggested_name,
        verbatim=verbatim,
    )

    action = "合并→" if is_merged else "新建→"
    return f"{action}{result_name} {','.join(domain)}{_tagging_truncation_note(content)}"


# =============================================================
# Tool 3: grow — Grow, fragments become memories
# 工具 3：grow — 生长，一天的碎片长成记忆
# =============================================================
@mcp.tool()
async def grow(content: str) -> str:
    """日记归档,自动拆分为多桶。短内容(<30字)走快速路径。"""
    await decay_engine.ensure_started()

    if not content or not content.strip():
        return "内容为空，无法整理。"

    # --- Short content fast path: skip digest, use hold logic directly ---
    # --- 短内容快速路径：跳过 digest 拆分，直接走 hold 逻辑省一次 API ---
    # For very short inputs (like "1"), calling digest is wasteful:
    # it sends the full DIGEST_PROMPT (~800 tokens) to DeepSeek for nothing.
    # Instead, run analyze + create directly.
    if len(content.strip()) < 30:
        logger.info(f"grow short-content fast path: {len(content.strip())} chars")
        try:
            analysis = await dehydrator.analyze(content)
        except Exception as e:
            logger.warning(f"Fast-path analyze failed / 快速路径打标失败: {e}")
            analysis = {
                "domain": ["未分类"], "valence": 0.5, "arousal": 0.3,
                "tags": [], "suggested_name": "",
            }
        result_name, is_merged = await _merge_or_create(
            content=content.strip(),
            tags=analysis.get("tags", []),
            importance=analysis.get("importance", 5) if isinstance(analysis.get("importance"), int) else 5,
            domain=analysis.get("domain", ["未分类"]),
            valence=analysis.get("valence", 0.5),
            arousal=analysis.get("arousal", 0.3),
            name=analysis.get("suggested_name", ""),
        )
        action = "合并" if is_merged else "新建"
        return f"{action} → {result_name} | {','.join(analysis.get('domain', []))} V{analysis.get('valence', 0.5):.1f}/A{analysis.get('arousal', 0.3):.1f}"

    # --- Step 1: let API split and organize / 让 API 拆分整理 ---
    try:
        items = await dehydrator.digest(content)
    except Exception as e:
        logger.error(f"Diary digest failed / 日记整理失败: {e}")
        return f"日记整理失败: {e}"

    if not items:
        return "内容为空或整理失败。"

    results = []
    created = 0
    merged = 0

    # --- Step 2: merge or create each item (with per-item error handling) ---
    # --- 逐条合并或新建（单条失败不影响其他）---
    for item in items:
        try:
            result_name, is_merged = await _merge_or_create(
                content=item["content"],
                tags=item.get("tags", []),
                importance=item.get("importance", 5),
                domain=item.get("domain", ["未分类"]),
                valence=item.get("valence", 0.5),
                arousal=item.get("arousal", 0.3),
                name=item.get("name", ""),
            )

            trunc_mark = "⚠️打标截断" if item.get("tagging_truncated") else ""
            if is_merged:
                results.append(f"📎{result_name}{trunc_mark}")
                merged += 1
            else:
                results.append(f"📝{item.get('name', result_name)}{trunc_mark}")
                created += 1
        except Exception as e:
            logger.warning(
                f"Failed to process diary item / 日记条目处理失败: "
                f"{item.get('name', '?')}: {e}"
            )
            results.append(f"⚠️{item.get('name', '?')}")

    return f"{len(items)}条|新{created}合{merged}\n" + "\n".join(results)


# =============================================================
# Tool 4: trace — Trace, redraw the outline of a memory
# 工具 4：trace — 描摹，重新勾勒记忆的轮廓
# Also handles deletion (delete=True)
# 同时承接删除功能
# =============================================================
@mcp.tool()
async def trace(
    bucket_id: str,
    name: str = "",
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    importance: int = -1,
    tags: str = "",
    resolved: int = -1,
    pinned: int = -1,
    digested: int = -1,
    content: str = "",
    delete: bool = False,
) -> str:
    """修改记忆元数据或内容。resolved=1沉底/0激活,pinned=1钉选/0取消,digested=1隐藏(保留但不浮现)/0取消隐藏,content=替换桶正文,delete=True删除。只传需改的,-1或空=不改。"""

    if not bucket_id or not bucket_id.strip():
        return "请提供有效的 bucket_id。"

    # --- Delete mode / 删除模式 ---
    if delete:
        success = await bucket_mgr.delete(bucket_id)
        if success:
            embedding_engine.delete_embedding(bucket_id)
        return f"已遗忘记忆桶: {bucket_id}" if success else f"未找到记忆桶: {bucket_id}"

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return f"未找到记忆桶: {bucket_id}"

    # --- Collect only fields actually passed / 只收集用户实际传入的字段 ---
    updates = {}
    if name:
        updates["name"] = name
    if domain:
        updates["domain"] = [d.strip() for d in domain.split(",") if d.strip()]
    if 0 <= valence <= 1:
        updates["valence"] = valence
    if 0 <= arousal <= 1:
        updates["arousal"] = arousal
    if 1 <= importance <= 10:
        updates["importance"] = importance
    if tags:
        updates["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if resolved in (0, 1):
        updates["resolved"] = bool(resolved)
    if pinned in (0, 1):
        updates["pinned"] = bool(pinned)
        if pinned == 1:
            updates["importance"] = 10  # pinned → lock importance
    if digested in (0, 1):
        updates["digested"] = bool(digested)
    if content:
        updates["content"] = content

    if not updates:
        return "没有任何字段需要修改。"

    success = await bucket_mgr.update(bucket_id, **updates)
    if not success:
        return f"修改失败: {bucket_id}"

    # Re-generate embedding if content changed
    if "content" in updates:
        try:
            await embedding_engine.generate_and_store(bucket_id, updates["content"])
        except Exception:
            pass

    changed = ", ".join(f"{k}={v}" for k, v in updates.items() if k != "content")
    if "content" in updates:
        changed += (", content=已替换" if changed else "content=已替换")
    # Explicit hint about resolved state change semantics
    # 特别提示 resolved 状态变化的语义
    if "resolved" in updates:
        if updates["resolved"]:
            changed += " → 已沉底，只在关键词触发时重新浮现"
        else:
            changed += " → 已重新激活，将参与浮现排序"
    if "digested" in updates:
        if updates["digested"]:
            changed += " → 已隐藏，保留但不再浮现"
        else:
            changed += " → 已取消隐藏，重新参与浮现"
    return f"已修改记忆桶 {bucket_id}: {changed}"


# =============================================================
# Tool 5: pulse — Heartbeat, compact system health summary
# 工具 5：pulse — 脉搏，紧凑系统体检
# =============================================================
PULSE_MAX_TOKENS = 500
PULSE_MAX_ANOMALIES = 3


def _cap_pulse_output(text: str, max_tokens: int = PULSE_MAX_TOKENS) -> str:
    """Hard-stop pulse output before it can flood the model context."""
    if count_tokens_approx(text) <= max_tokens:
        return text

    suffix = "\n[输出已截断]"
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = text[:mid].rstrip() + suffix
        if count_tokens_approx(candidate) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return text[:low].rstrip() + suffix


async def _collect_diagnostics(include_archive: bool = True) -> dict:
    """Collect read-only health data shared by pulse and the admin CLI API."""
    buckets = await bucket_mgr.list_all(include_archive=include_archive)
    today = datetime.now().date().isoformat()
    active = [
        b for b in buckets
        if b.get("metadata", {}).get("type") != "archived"
    ]

    type_counts = {
        "dynamic": 0,
        "permanent": 0,
        "feel": 0,
        "letter": 0,
        "archived": 0,
    }
    today_added = 0
    undigested = 0
    unclassified_ids = []
    invalid_created_ids = []

    for bucket in buckets:
        meta = bucket.get("metadata", {})
        btype = meta.get("type", "dynamic")
        type_counts[btype] = type_counts.get(btype, 0) + 1
        created = str(meta.get("created", ""))
        if created[:10] == today:
            today_added += 1
        elif created:
            try:
                datetime.fromisoformat(created)
            except (TypeError, ValueError):
                invalid_created_ids.append(bucket["id"])

        if (
            btype == "dynamic"
            and not meta.get("digested", False)
            and not meta.get("pinned", False)
            and not meta.get("protected", False)
        ):
            undigested += 1
        if (
            btype not in ("feel", "letter", "crave", "archived")
            and (not meta.get("domain") or meta.get("domain") == ["未分类"])
        ):
            unclassified_ids.append(bucket["id"])

    tagging_failure_count = len(dehydrator.recent_tagging_failures)
    disk_ids = {
        b["id"] for b in buckets
        if str(b.get("content") or "").strip()
    }
    index_ids = set()
    missing_embedding_ids = []
    orphan_embedding_ids = []
    if embedding_engine and embedding_engine.enabled:
        try:
            index_ids = embedding_engine.list_all_ids()
            missing_embedding_ids = sorted(disk_ids - index_ids)
            orphan_embedding_ids = sorted(index_ids - disk_ids)
        except Exception as e:
            logger.warning(f"Diagnostics embedding check failed: {e}")

    anomalies = []
    if tagging_failure_count:
        anomalies.append({
            "code": "tagging_failures",
            "count": tagging_failure_count,
            "message": f"本次部署记录到 {tagging_failure_count} 次打标失败",
            "sample_ids": [],
        })
    if unclassified_ids:
        anomalies.append({
            "code": "unclassified",
            "count": len(unclassified_ids),
            "message": f"{len(unclassified_ids)} 个桶仍未分类",
            "sample_ids": unclassified_ids[:10],
        })
    if missing_embedding_ids:
        anomalies.append({
            "code": "missing_embeddings",
            "count": len(missing_embedding_ids),
            "message": f"{len(missing_embedding_ids)} 个桶缺少向量索引",
            "sample_ids": missing_embedding_ids[:10],
        })
    if orphan_embedding_ids:
        anomalies.append({
            "code": "orphan_embeddings",
            "count": len(orphan_embedding_ids),
            "message": f"{len(orphan_embedding_ids)} 条向量索引没有对应桶",
            "sample_ids": orphan_embedding_ids[:10],
        })
    if invalid_created_ids:
        anomalies.append({
            "code": "invalid_created",
            "count": len(invalid_created_ids),
            "message": f"{len(invalid_created_ids)} 个桶的创建时间无效",
            "sample_ids": invalid_created_ids[:10],
        })
    if not decay_engine.is_running:
        anomalies.insert(0, {
            "code": "decay_stopped",
            "count": 1,
            "message": "衰减引擎未运行",
            "sample_ids": [],
        })

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total_buckets": len(buckets),
            "active_buckets": len(active),
            "today_added": today_added,
            "undigested": undigested,
            "feel_count": type_counts.get("feel", 0),
            "tagging_failure_count": tagging_failure_count,
            "decay_engine": "running" if decay_engine.is_running else "stopped",
            "embedding_enabled": bool(embedding_engine and embedding_engine.enabled),
            "types": type_counts,
        },
        "anomalies": anomalies,
        "buckets": buckets,
        "index_ids": index_ids,
    }


@mcp.tool()
async def pulse(include_archive: bool = False) -> str:
    """500 token以内的系统体检摘要；不返回桶清单或正文。include_archive仅为旧客户端兼容保留。"""
    # pulse is often the first tool called after a deployment.  Start the lazy
    # background task before reporting status so "stopped" is not a false alarm.
    await decay_engine.ensure_started()
    try:
        diagnostics = await _collect_diagnostics(include_archive=True)
    except Exception as e:
        return _cap_pulse_output(f"Ombre Brain 体检失败: {type(e).__name__}")

    summary = diagnostics["summary"]
    anomalies = diagnostics["anomalies"][:PULSE_MAX_ANOMALIES]
    anomaly_lines = [f"- {item['message']}" for item in anomalies] or ["- 无"]
    result = (
        "=== Ombre Brain 简短体检 ===\n"
        f"总桶数: {summary['total_buckets']}\n"
        f"今日新增: {summary['today_added']}\n"
        f"未消化/feel数: {summary['undigested']}/{summary['feel_count']}\n"
        f"打标失败计数: {summary['tagging_failure_count']}\n"
        "异常摘要（最多3条）:\n"
        + "\n".join(anomaly_lines)
    )
    return _cap_pulse_output(result)


# =============================================================
# Tool 6: dream — Dreaming, digest recent memories
# 工具 6：dream — 做梦，消化最近的记忆
#
# Reads recent surface-level buckets (≤10), returns them for
# Claude to introspect under prompt guidance.
# 读取最近新增的表层桶（≤10个），返回给 Claude 在提示词引导下自主思考。
# Claude then decides: resolve some, write feels, or do nothing.
# =============================================================
@mcp.tool()
async def dream() -> str:
    """做梦——读取最近新增的记忆桶,供你自省。读完后可以trace(resolved=1)放下,或hold(feel=True)写感受。"""
    await decay_engine.ensure_started()

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.error(f"Dream failed to list buckets: {e}")
        return "记忆系统暂时无法访问。"

    # --- Same deterministic reservation used by both breath paths ---
    # --- 与两条 breath 路径共用同一组确定性的最新记忆 ---
    recent = _select_dream_recent(all_buckets)

    if not recent:
        _record_surface_audit(
            "dream", [], total_buckets=len(all_buckets), returned_count=0, status="complete"
        )
        return "没有需要消化的新记忆。"

    try:
        ranked, weight_ranks, score_snapshot = _weight_rank_snapshot(all_buckets)
    except Exception as exc:
        logger.warning("Dream rank snapshot failed: %s", type(exc).__name__)
        ranked, weight_ranks, score_snapshot = [], {}, {}

    parts = []
    for b in recent:
        meta = b["metadata"]
        resolved_tag = " [已解决]" if meta.get("resolved", False) else " [未解决]"
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        created = meta.get("created", "")
        parts.append(
            f"[{meta.get('name', b['id'])}]{resolved_tag} "
            f"主题:{domains} V{val:.1f}/A{aro:.1f} "
            f"创建:{created}\n"
            f"ID: {b['id']}\n"
            f"{strip_wikilinks(b['content'][:500])}"
        )

    header = (
        "=== Dreaming ===\n"
        "以下是你最近的记忆。用第一人称想：\n"
        "- 这些东西里有什么在你这里留下了重量？\n"
        "- 有什么还没想清楚？\n"
        "- 有什么可以放下了？\n"
        "想完之后：值得放下的用 trace(bucket_id, resolved=1)；\n"
        "有沉淀的用 hold(content=\"...\", feel=True, source_bucket=\"bucket_id\", valence=你的感受) 写下来。\n"
        "valence 是你对这段记忆的感受，不是事件本身的情绪。\n"
        "没有沉淀就不写，不强迫产出。\n"
    )

    # --- Connection hint: find most similar pair via embeddings ---
    connection_hint = ""
    if embedding_engine and embedding_engine.enabled and len(recent) >= 2:
        try:
            best_pair = None
            best_sim = 0.0
            ids = [b["id"] for b in recent]
            names = {b["id"]: b["metadata"].get("name", b["id"]) for b in recent}
            embeddings = {}
            for bid in ids:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb
            for i, id_a in enumerate(ids):
                for id_b in ids[i+1:]:
                    if id_a in embeddings and id_b in embeddings:
                        sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                        if sim > best_sim:
                            best_sim = sim
                            best_pair = (id_a, id_b)
            if best_pair and best_sim > 0.5:
                connection_hint = (
                    f"\n💭 [{names[best_pair[0]]}] 和 [{names[best_pair[1]]}] "
                    f"似乎有关联 (相似度:{best_sim:.2f})——不替你下结论，你自己想。\n"
                )
        except Exception as e:
            logger.warning(f"Dream connection hint failed: {e}")

    # --- Feel crystallization hint: detect repeated feel themes ---
    crystal_hint = ""
    if embedding_engine and embedding_engine.enabled:
        try:
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            if len(feels) >= 3:
                feel_embeddings = {}
                for f in feels:
                    emb = await embedding_engine.get_embedding(f["id"])
                    if emb is not None:
                        feel_embeddings[f["id"]] = emb
                # Find clusters: feels with similarity > 0.7 to at least 2 others
                for fid, femb in feel_embeddings.items():
                    similar_feels = []
                    for oid, oemb in feel_embeddings.items():
                        if oid != fid:
                            sim = embedding_engine._cosine_similarity(femb, oemb)
                            if sim > 0.7:
                                similar_feels.append(oid)
                    if len(similar_feels) >= 2:
                        feel_bucket = next((f for f in feels if f["id"] == fid), None)
                        if feel_bucket and not feel_bucket["metadata"].get("pinned"):
                            content_preview = strip_wikilinks(feel_bucket["content"][:80])
                            crystal_hint = (
                                f"\n🔮 你已经写过 {len(similar_feels)+1} 条相似的 feel "
                                f"（围绕「{content_preview}…」）。"
                                f"如果这已经是确信而不只是感受了，"
                                f"你可以用 hold(content=\"...\", pinned=True) 升级它。"
                                f"不急，你自己决定。\n"
                            )
                            break
        except Exception as e:
            logger.warning(f"Dream crystallization hint failed: {e}")

    final_text = header + "\n---\n".join(parts) + connection_hint + crystal_hint
    _record_surface_audit(
        "dream",
        [
            _audit_bucket_entry(
                b,
                channel="dream",
                score=score_snapshot.get(b["id"]),
                weight_rank=weight_ranks.get(b["id"]),
                newest_position=index,
                output_position=index,
                outcome="surfaced",
            )
            for index, b in enumerate(recent, start=1)
        ],
        total_buckets=len(all_buckets),
        dynamic_pool_count=len(ranked),
        candidate_count=len(recent),
        returned_count=len(recent),
        max_results=DREAM_RECENT_LIMIT,
        status="complete",
    )
    await _fire_webhook("dream", {"recent": len(recent), "chars": len(final_text)})
    return final_text


# =============================================================
# Tool: letter_write — 写信，用户与你之间的永久信件
# =============================================================
@mcp.tool()
async def letter_write(
    author: str,
    content: str,
    user_name: str = "",
    title: str = "",
    date: str = "",
) -> str:
    """写一封信,永久保存,不参与衰减/合并/普通浮现,只能通过 letter_read 读取。author='user'表示这封信是用户执笔;'ai'或你自己的名字表示你执笔(统一存为 AI_NAME 环境变量配置的署名)。user_name可选,标注用户具体署名。title可选标题。date可选(YYYY-MM-DD,不填用当前时间)。"""
    await decay_engine.ensure_started()
    if user_name is None: user_name = ""
    if title is None: title = ""
    if date is None: date = ""
    if not author or not author.strip():
        return "author 不能为空。"
    if not content or not content.strip():
        return "信件内容不能为空。"

    ai_name = get_ai_name()
    raw = author.strip()
    low = raw.lower()
    if low == "user":
        a = "user"
    elif low in ("ai", "claude") or raw == ai_name:
        a = ai_name
    else:
        a = raw

    extra_meta = {"author": a}
    if user_name.strip():
        extra_meta["user_name"] = user_name.strip()
    if title.strip():
        extra_meta["title"] = title.strip()[:120]
    if date.strip():
        extra_meta["letter_date"] = date.strip()

    bucket_id = await bucket_mgr.create(
        content=content.strip(),
        tags=["__letter__"],
        importance=10,
        domain=["letter"],
        valence=0.5,
        arousal=0.3,
        name=(title.strip()[:60] or f"{a}_{date.strip() or 'letter'}"),
        bucket_type="letter",
        verbatim=True,
    )
    try:
        await bucket_mgr.update(bucket_id, **extra_meta)
    except Exception as e:
        logger.warning(f"letter_write update meta failed / 写信元数据更新失败: {e}")
    try:
        await embedding_engine.generate_and_store(bucket_id, content)
    except Exception:
        pass
    return f"💌letter→{bucket_id} [{a}]"


# =============================================================
# Tool: letter_read — 读信
# =============================================================
@mcp.tool()
async def letter_read(
    query: str = "",
    limit: int = 10,
    author: str = "",
    date_from: str = "",
    date_to: str = "",
) -> str:
    """读信。不传query=按时间倒序返回最近的信;传query=语义/关键词检索。author筛选:'user'/'ai'/具体署名。date_from/date_to按YYYY-MM-DD筛选letter_date或创建时间(闭区间)。limit最多50,默认10。"""
    if query is None: query = ""
    if limit is None: limit = 10
    if author is None: author = ""
    if date_from is None: date_from = ""
    if date_to is None: date_to = ""
    limit = max(1, min(50, limit))
    try:
        all_b = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        return f"读取信件失败: {e}"
    letters = [b for b in all_b if b["metadata"].get("type") == "letter"]

    af = author.strip()
    if af:
        ai_name = get_ai_name()
        af_low = af.lower()
        if af_low == "user":
            letters = [b for b in letters if b["metadata"].get("author") == "user"]
        elif af_low in ("ai", "claude") or af == ai_name:
            ai_aliases = {ai_name, "claude"}
            letters = [b for b in letters if b["metadata"].get("author") in ai_aliases]
        else:
            letters = [b for b in letters if b["metadata"].get("author") == af]

    def _within(b):
        d = b["metadata"].get("letter_date") or b["metadata"].get("created", "")
        if date_from and d and d < date_from:
            return False
        if date_to and d and d > date_to:
            return False
        return True

    letters = [b for b in letters if _within(b)]

    query_text = query.strip()

    def _matches_query(b):
        if not query_text:
            return True
        meta = b.get("metadata", {})
        parts = [
            b.get("content", ""),
            str(meta.get("name") or ""),
            str(meta.get("title") or ""),
            str(meta.get("author") or ""),
        ]
        parts.extend(str(t) for t in (meta.get("tags") or []))
        return query_text.lower() in "\n".join(parts).lower()

    if query_text and embedding_engine and getattr(embedding_engine, "enabled", False):
        try:
            sims = await embedding_engine.search_similar(query_text, top_k=limit * 3)
            id_score = {bid: sc for bid, sc in sims}
            vector_matches = [b for b in letters if b["id"] in id_score]
            if vector_matches:
                letters = vector_matches
                letters.sort(key=lambda b: id_score.get(b["id"], 0.0), reverse=True)
            else:
                letters = [b for b in letters if _matches_query(b)]
                letters.sort(key=lambda b: b["metadata"].get("letter_date") or b["metadata"].get("created", ""), reverse=True)
        except Exception as e:
            logger.warning(f"letter_read vector search failed / 信件向量检索失败: {e}")
            letters = [b for b in letters if _matches_query(b)]
            letters.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
    else:
        if query_text:
            letters = [b for b in letters if _matches_query(b)]
        letters.sort(key=lambda b: b["metadata"].get("letter_date") or b["metadata"].get("created", ""), reverse=True)

    letters = letters[:limit]
    if not letters:
        return "没有找到匹配的信件。"
    parts = []
    for b in letters:
        m = b["metadata"]
        a = m.get("author", "?")
        d = (m.get("letter_date") or m.get("created", ""))[:10]
        title = m.get("title") or m.get("name", "")
        parts.append(
            f"[{b['id']}] {a} · {d}{(' · ' + title) if title else ''}\n"
            + strip_wikilinks(b["content"])
        )
    return "=== 信件 ===\n" + "\n\n---\n\n".join(parts)


# =============================================================
# Dashboard API endpoints (for lightweight Web UI)
# 仪表板 API（轻量 Web UI 用）
# =============================================================
@mcp.custom_route("/api/decay-freeze", methods=["GET"])
async def api_decay_freeze_status(request):
    """Return the persistent relationship-clock freeze state."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    return JSONResponse(decay_engine.relationship_clock.status())


@mcp.custom_route("/api/decay-freeze", methods=["POST"])
async def api_decay_freeze_update(request):
    """Freeze decay manually or cancel it from the authenticated Dashboard."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not isinstance(body, dict) or not isinstance(body.get("frozen"), bool):
        return JSONResponse({"error": "frozen must be boolean"}, status_code=400)
    try:
        if body["frozen"]:
            status = decay_engine.relationship_clock.freeze()
        else:
            status = decay_engine.relationship_clock.resume("dashboard_manual")
        return JSONResponse(status)
    except OSError:
        logger.exception("Failed to persist decay freeze state")
        return JSONResponse({"error": "freeze state write failed"}, status_code=500)


@mcp.custom_route("/api/buckets", methods=["GET"])
async def api_buckets(request):
    """List all buckets with metadata (no content for efficiency)."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True)
        result = []
        for b in all_buckets:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "tags": meta.get("tags", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "model_valence": meta.get("model_valence"),
                "importance": meta.get("importance", 5),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "digested": meta.get("digested", False),
                "created": meta.get("created", ""),
                "last_active": meta.get("last_active", ""),
                "activation_count": meta.get("activation_count", 1),
                "score": decay_engine.calculate_score(meta),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
            })
        result.sort(key=lambda x: x["score"], reverse=True)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/bucket/{bucket_id}", methods=["GET"])
async def api_bucket_detail(request):
    """Get full bucket content by ID."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    meta = bucket.get("metadata", {})
    content = strip_wikilinks(bucket.get("content", ""))
    edited_hash = meta.get("dehydration_edited_hash")
    dehydration_stale = bool(edited_hash) and edited_hash != hashlib.sha256(content.encode()).hexdigest()
    return JSONResponse({
        "id": bucket["id"],
        "metadata": meta,
        "content": content,
        "score": decay_engine.calculate_score(meta),
        "dehydration_stale": dehydration_stale,
    })


@mcp.custom_route("/api/bucket/{bucket_id}", methods=["POST"])
async def api_bucket_update(request):
    """Update bucket metadata and/or content from dashboard."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    bucket_id = request.path_params["bucket_id"]
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    # --- Delete mode ---
    if body.get("_delete"):
        success = await bucket_mgr.delete(bucket_id)
        if success:
            embedding_engine.delete_embedding(bucket_id)
            return JSONResponse({"ok": True, "deleted": bucket_id})
        return JSONResponse({"error": "not found"}, status_code=404)

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)

    updates = {}
    if "name" in body and body["name"]:
        updates["name"] = body["name"]
    if "domain" in body:
        if isinstance(body["domain"], list):
            updates["domain"] = body["domain"]
        elif isinstance(body["domain"], str):
            updates["domain"] = [d.strip() for d in body["domain"].split(",") if d.strip()]
    if "tags" in body:
        if isinstance(body["tags"], list):
            updates["tags"] = body["tags"]
        elif isinstance(body["tags"], str):
            updates["tags"] = [t.strip() for t in body["tags"].split(",") if t.strip()]
    if "valence" in body:
        v = float(body["valence"])
        if 0 <= v <= 1:
            updates["valence"] = v
    if "arousal" in body:
        a = float(body["arousal"])
        if 0 <= a <= 1:
            updates["arousal"] = a
    if "importance" in body:
        imp = int(body["importance"])
        if 1 <= imp <= 10:
            updates["importance"] = imp
    if "resolved" in body:
        updates["resolved"] = bool(body["resolved"])
    if "pinned" in body:
        updates["pinned"] = bool(body["pinned"])
        if body["pinned"]:
            updates["importance"] = 10
    if "digested" in body:
        updates["digested"] = bool(body["digested"])
    if "dehydration_mode" in body and body["dehydration_mode"] in ("auto", "facts", "summary"):
        updates["dehydration_mode"] = body["dehydration_mode"]
    if "verbatim" in body:
        updates["verbatim"] = bool(body["verbatim"])
    if "dehydration_edited_hash" in body:
        updates["dehydration_edited_hash"] = str(body["dehydration_edited_hash"] or "")
    if "content" in body and body["content"]:
        updates["content"] = body["content"]

    if not updates:
        return JSONResponse({"error": "no fields to update"}, status_code=400)

    success = await bucket_mgr.update(bucket_id, touch=False, **updates)
    if not success:
        return JSONResponse({"error": "update failed"}, status_code=500)

    if "content" in updates:
        try:
            await embedding_engine.generate_and_store(bucket_id, updates["content"])
        except Exception:
            pass

    return JSONResponse({
        "ok": True,
        "updated": list(updates.keys()),
        "activation_preserved": True,
    })


async def _dashboard_readonly_search(query: str, limit: int = 30) -> list[dict]:
    """Mirror Breath's keyword + vector search without any activation.

    Dashboard editing also needs guaranteed direct lookup by title and bucket
    ID.  Those direct hits are merged ahead of the normal Breath-style fuzzy
    ranking and explicit semantic-vector channel, never filtered out by the
    embedding candidate set.
    """
    query = (query or "").strip()
    if not query:
        return []
    limit = max(1, min(50, int(limit)))

    async def _vector_results():
        if not embedding_engine or not getattr(embedding_engine, "enabled", False):
            return []
        try:
            return await embedding_engine.search_similar(query, top_k=max(limit, 20))
        except Exception as exc:
            logger.warning(f"Dashboard vector search failed, using text only: {exc}")
            return []

    all_buckets, keyword_matches, vector_results = await asyncio.gather(
        bucket_mgr.list_all(include_archive=False),
        bucket_mgr.search(
            query,
            limit=max(limit, 20),
            use_embedding_prefilter=False,
        ),
        _vector_results(),
    )
    eligible = {
        bucket["id"]: bucket
        for bucket in all_buckets
        if bucket.get("metadata", {}).get("type") not in ("feel", "letter", "crave")
    }
    records = {}

    def add(bucket_id: str, reason: str, rank: tuple, score: float, vector_similarity=None):
        bucket = eligible.get(bucket_id)
        if not bucket:
            return
        current = records.get(bucket_id)
        if current is None:
            current = {
                "bucket": bucket,
                "reasons": [],
                "rank": rank,
                "score": score,
                "vector_similarity": None,
            }
            records[bucket_id] = current
        else:
            current["rank"] = min(current["rank"], rank)
            current["score"] = max(current["score"], score)
        if reason not in current["reasons"]:
            current["reasons"].append(reason)
        if vector_similarity is not None:
            current["vector_similarity"] = vector_similarity

    lowered = query.casefold()
    direct_hits = []
    for bucket_id, bucket in eligible.items():
        name = str(bucket.get("metadata", {}).get("name") or bucket_id)
        bid_lower = bucket_id.casefold()
        name_lower = name.casefold()
        if bid_lower == lowered:
            direct_hits.append((0, bucket_id, "桶ID", 100.0))
        elif bid_lower.startswith(lowered):
            direct_hits.append((1, bucket_id, "桶ID", 99.0))
        elif lowered in bid_lower:
            direct_hits.append((2, bucket_id, "桶ID", 98.0))
        if name_lower == lowered:
            direct_hits.append((3, bucket_id, "标题", 100.0))
        elif lowered in name_lower:
            direct_hits.append((4, bucket_id, "标题", 97.0))
    direct_hits.sort(key=lambda item: (item[0], item[1]))
    for order, (_, bucket_id, reason, score) in enumerate(direct_hits):
        add(bucket_id, reason, (0, order), score)

    for order, match in enumerate(keyword_matches):
        add(match["id"], "关键词", (1, order), float(match.get("score", 0) or 0))

    for order, (bucket_id, similarity) in enumerate(vector_results):
        if similarity > 0.5:
            add(
                bucket_id,
                "语义向量",
                (2, order),
                round(float(similarity) * 100, 2),
                vector_similarity=round(float(similarity), 4),
            )

    ordered = sorted(records.values(), key=lambda item: item["rank"])
    results = []
    for record in ordered[:limit]:
        bucket = record["bucket"]
        copied = dict(bucket)
        copied["score"] = round(record["score"], 2)
        copied["match_reasons"] = record["reasons"]
        copied["vector_similarity"] = record["vector_similarity"]
        results.append(copied)
    return results


@mcp.custom_route("/api/search", methods=["GET"])
async def api_search(request):
    """Read-only Dashboard search; never touch or soft-touch a bucket."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    query = request.query_params.get("q", "")
    if not query:
        return JSONResponse({"error": "missing q parameter"}, status_code=400)
    try:
        try:
            limit = int(request.query_params.get("limit", "30"))
        except ValueError:
            return JSONResponse({"error": "invalid limit"}, status_code=400)
        matches = await _dashboard_readonly_search(query, limit=limit)
        result = []
        for b in matches:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "tags": meta.get("tags", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "model_valence": meta.get("model_valence"),
                "importance": meta.get("importance", 5),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "digested": meta.get("digested", False),
                "created": meta.get("created", ""),
                "last_active": meta.get("last_active", ""),
                "activation_count": meta.get("activation_count", 1),
                "score": b.get("score", 0) or decay_engine.calculate_score(meta),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
                "match_reasons": b.get("match_reasons", []),
                "vector_similarity": b.get("vector_similarity"),
            })
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/network", methods=["GET"])
async def api_network(request):
    """Get embedding similarity network for visualization."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        nodes = []
        edges = []
        embeddings = {}

        for b in all_buckets:
            meta = b.get("metadata", {})
            bid = b["id"]
            nodes.append({
                "id": bid,
                "name": meta.get("name", bid),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "score": decay_engine.calculate_score(meta),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "digested": meta.get("digested", False),
            })
            if embedding_engine and embedding_engine.enabled:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb

        # Build edges from embeddings (similarity > 0.5)
        ids = list(embeddings.keys())
        for i, id_a in enumerate(ids):
            for id_b in ids[i+1:]:
                sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                if sim > 0.5:
                    edges.append({"source": id_a, "target": id_b, "similarity": round(sim, 3)})

        return JSONResponse({"nodes": nodes, "edges": edges})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================
# /api/admin/backfill-embeddings — one-off catch-up for buckets
# that existed before embedding was correctly configured.
# Writes only to the embeddings SQLite DB via generate_and_store;
# deliberately does NOT go through bucket_mgr.update(), so it never
# touches last_active / activation_count on any bucket.
# =============================================================
_backfill_state = {
    "running": False,
    "total": 0,
    "done": 0,
    "success": 0,
    "failed": 0,
    "skipped": 0,
    "finished": False,
}


async def _run_backfill():
    global _backfill_state
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True)
        missing = []
        for b in all_buckets:
            emb = await embedding_engine.get_embedding(b["id"])
            if emb is None:
                missing.append(b)

        _backfill_state.update({
            "running": True, "total": len(missing), "done": 0,
            "success": 0, "failed": 0, "skipped": 0, "finished": False,
        })

        batch_size = 20
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            for b in batch:
                content = b.get("content", "")
                if not content or not content.strip():
                    _backfill_state["skipped"] += 1
                else:
                    try:
                        ok = await embedding_engine.generate_and_store(b["id"], content)
                        _backfill_state["success" if ok else "failed"] += 1
                    except Exception as e:
                        logger.warning(f"Backfill embedding failed for {b['id']}: {e}")
                        _backfill_state["failed"] += 1
                _backfill_state["done"] += 1
            if i + batch_size < len(missing):
                await asyncio.sleep(2)
    except Exception as e:
        logger.error(f"Backfill run failed: {e}")
    finally:
        _backfill_state["running"] = False
        _backfill_state["finished"] = True


@mcp.custom_route("/api/admin/backfill-embeddings", methods=["POST"])
async def api_admin_backfill_embeddings(request):
    """Start a one-off backfill of missing embeddings for existing buckets."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    if not embedding_engine.enabled:
        return JSONResponse({"error": "Embedding not enabled"}, status_code=400)
    if _backfill_state["running"]:
        return JSONResponse({"error": "Backfill already running"}, status_code=409)
    asyncio.create_task(_run_backfill())
    return JSONResponse({"status": "started"})


@mcp.custom_route("/api/admin/backfill-embeddings", methods=["GET"])
async def api_admin_backfill_status(request):
    """Get progress of the running/last backfill."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    return JSONResponse(_backfill_state)


# =============================================================
# /api/admin/backfill-tags — one-off re-tagging for buckets that got
# stuck with domain=["未分类"] because deepseek-v4-flash intermittently
# omits the domain key (see dehydrator._api_analyze retry logic above).
# Writes only domain/tags/valence/arousal/name via bucket_mgr.update(),
# always with touch=False so the decay clock is not reset.
# =============================================================
_tag_backfill_state = {
    "running": False,
    "total": 0,
    "done": 0,
    "fixed": 0,
    "still_unclassified": 0,
    "skipped": 0,
    "finished": False,
}


async def _run_tag_backfill():
    global _tag_backfill_state
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True)
        targets = [
            b for b in all_buckets
            if b.get("metadata", {}).get("domain") == ["未分类"]
            and b.get("metadata", {}).get("type") not in ("feel", "letter", "crave")
        ]

        _tag_backfill_state.update({
            "running": True, "total": len(targets), "done": 0,
            "fixed": 0, "still_unclassified": 0, "skipped": 0, "finished": False,
        })

        for b in targets:
            content = b.get("content", "")
            if not content or not content.strip():
                _tag_backfill_state["skipped"] += 1
                _tag_backfill_state["done"] += 1
                continue
            try:
                analysis = await dehydrator.analyze(content)
            except Exception as e:
                logger.warning(f"Tag backfill analyze failed for {b['id']}: {e}")
                _tag_backfill_state["still_unclassified"] += 1
                _tag_backfill_state["done"] += 1
                continue

            if analysis.get("domain") == ["未分类"]:
                _tag_backfill_state["still_unclassified"] += 1
            else:
                update_kwargs = {
                    "domain": analysis["domain"],
                    "tags": analysis.get("tags", []),
                    "valence": analysis.get("valence", 0.5),
                    "arousal": analysis.get("arousal", 0.3),
                }
                meta_name = b.get("metadata", {}).get("name", "")
                if (not meta_name or meta_name == b["id"]) and analysis.get("suggested_name"):
                    update_kwargs["name"] = analysis["suggested_name"]
                await bucket_mgr.update(b["id"], touch=False, **update_kwargs)
                _tag_backfill_state["fixed"] += 1
            _tag_backfill_state["done"] += 1
            await asyncio.sleep(0.5)
    except Exception as e:
        logger.error(f"Tag backfill run failed: {e}")
    finally:
        _tag_backfill_state["running"] = False
        _tag_backfill_state["finished"] = True


@mcp.custom_route("/api/admin/backfill-tags", methods=["POST"])
async def api_admin_backfill_tags(request):
    """Start a one-off re-tagging pass for buckets stuck at domain=未分类."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    if _tag_backfill_state["running"]:
        return JSONResponse({"error": "Tag backfill already running"}, status_code=409)
    asyncio.create_task(_run_tag_backfill())
    return JSONResponse({"status": "started"})


@mcp.custom_route("/api/admin/backfill-tags", methods=["GET"])
async def api_admin_backfill_tags_status(request):
    """Get progress of the running/last tag backfill."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    return JSONResponse(_tag_backfill_state)


@mcp.custom_route("/api/admin/tagging-diagnostics", methods=["GET"])
async def api_admin_tagging_diagnostics(request):
    """Recent auto-tagging failures (raw LLM response + finish_reason per
    attempt), visible from the Dashboard without needing Zeabur log access."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    return JSONResponse(list(dehydrator.recent_tagging_failures))


@mcp.custom_route("/api/admin/surface-audit", methods=["GET"])
async def api_admin_surface_audit(request):
    """Authenticated metadata-only history of Breath/Dream/Feel selections."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        limit = int(request.query_params.get("limit", 20))
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(surface_audit.max_events, limit))
    try:
        return JSONResponse({
            "retention": surface_audit.max_events,
            "events": surface_audit.recent(limit),
        })
    except Exception as exc:
        logger.exception("Surface audit read failed")
        return JSONResponse({"error": type(exc).__name__}, status_code=500)


@mcp.custom_route("/api/admin/diagnostics", methods=["GET"])
async def api_admin_diagnostics(request):
    """Read-only, authenticated diagnostics for the maintainer CLI.

    This route is deliberately not an MCP tool and has no Dashboard control.
    It returns metadata only, never bucket content.
    """
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err

    def bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(request.query_params.get(name, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    include_archive = str(
        request.query_params.get("include_archive", "true")
    ).lower() not in ("0", "false", "no")
    offset = bounded_int("offset", 0, 0, 1_000_000)
    limit = bounded_int("limit", 25, 1, 100)

    try:
        diagnostics = await _collect_diagnostics(include_archive=include_archive)
        buckets = diagnostics.pop("buckets")
        index_ids = diagnostics.pop("index_ids")
        buckets.sort(
            key=lambda b: str(b.get("metadata", {}).get("created", "")),
            reverse=True,
        )
        page = []
        for bucket in buckets[offset:offset + limit]:
            meta = bucket.get("metadata", {})
            page.append({
                "id": bucket["id"],
                "name": meta.get("name", bucket["id"]),
                "type": meta.get("type", "dynamic"),
                "created": meta.get("created", ""),
                "last_active": meta.get("last_active", ""),
                "domain": meta.get("domain", []),
                "tag_count": len(meta.get("tags", []) or []),
                "importance": meta.get("importance", 5),
                "resolved": bool(meta.get("resolved", False)),
                "digested": bool(meta.get("digested", False)),
                "score": decay_engine.calculate_score(meta),
                "has_embedding": bucket["id"] in index_ids if embedding_engine.enabled else None,
            })
        diagnostics["pagination"] = {
            "offset": offset,
            "limit": limit,
            "returned": len(page),
            "total": len(buckets),
            "include_archive": include_archive,
        }
        diagnostics["buckets"] = page
        return JSONResponse(diagnostics)
    except Exception as e:
        logger.exception("Admin diagnostics failed")
        return JSONResponse({"error": type(e).__name__}, status_code=500)


@mcp.custom_route("/api/breath-debug", methods=["GET"])
async def api_breath_debug(request):
    """Debug endpoint: simulate breath scoring and return per-bucket breakdown."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    query = request.query_params.get("q", "")
    q_valence = request.query_params.get("valence")
    q_arousal = request.query_params.get("arousal")
    q_valence = float(q_valence) if q_valence else None
    q_arousal = float(q_arousal) if q_arousal else None

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        results = []
        w = {
            "topic": bucket_mgr.w_topic,
            "emotion": bucket_mgr.w_emotion,
            "time": bucket_mgr.w_time,
            "importance": bucket_mgr.w_importance,
        }
        w_sum = sum(w.values())

        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            bid = bucket["id"]
            try:
                topic = bucket_mgr._calc_topic_score(query, bucket) if query else 0.0
                emotion = bucket_mgr._calc_emotion_score(q_valence, q_arousal, meta)
                time_s = bucket_mgr._calc_time_score(meta)
                imp = max(1, min(10, int(meta.get("importance", 5)))) / 10.0

                raw_total = (
                    topic * w["topic"]
                    + emotion * w["emotion"]
                    + time_s * w["time"]
                    + imp * w["importance"]
                )
                normalized = (raw_total / w_sum) * 100 if w_sum > 0 else 0
                resolved = meta.get("resolved", False)
                if resolved:
                    normalized *= 0.3

                results.append({
                    "id": bid,
                    "name": meta.get("name", bid),
                    "domain": meta.get("domain", []),
                    "type": meta.get("type", "dynamic"),
                    "resolved": resolved,
                    "pinned": meta.get("pinned", False),
                    "scores": {
                        "topic": round(topic, 4),
                        "emotion": round(emotion, 4),
                        "time": round(time_s, 4),
                        "importance": round(imp, 4),
                    },
                    "weights": w,
                    "raw_total": round(raw_total, 4),
                    "normalized": round(normalized, 2),
                    "passed_threshold": normalized >= bucket_mgr.fuzzy_threshold,
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["normalized"], reverse=True)
        passed = [r for r in results if r["passed_threshold"]]
        return JSONResponse({
            "query": query,
            "valence": q_valence,
            "arousal": q_arousal,
            "weights": w,
            "threshold": bucket_mgr.fuzzy_threshold,
            "total_candidates": len(results),
            "passed_count": len(passed),
            "results": results[:50],  # top 50 for debug
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================
# /api/simulate/* — non-activating wake-flow simulation for the
# dashboard. Mirrors breath()/dream()'s real selection logic so the
# user can see exactly what would surface right now, without ever
# calling touch()/soft_touch() (no activation side effects).
# 唤醒流程模拟：照抄 breath()/dream() 的真实选桶逻辑，但绝不调用
# touch/soft_touch，纯粹用于核对权重、不影响记忆的激活状态。
# =============================================================

def _sim_row(bucket: dict, score: float, **extra) -> dict:
    meta = bucket.get("metadata", {})
    return {
        "id": bucket["id"],
        "name": meta.get("name", bucket["id"]),
        "type": meta.get("type", "dynamic"),
        "domain": meta.get("domain", []),
        "score": round(score, 4) if score is not None else None,
        "content_preview": strip_wikilinks(bucket.get("content", ""))[:200],
        **extra,
    }


async def _dehydrate_concurrent(buckets: list[dict], concurrency: int = 5) -> dict:
    """Dehydrate multiple buckets in parallel (bounded), one dehydrator.dehydrate()
    call per bucket. Cache hits return instantly; cache misses (real API calls)
    overlap instead of queuing one after another.
    Returns {bucket_id: (summary_or_None, error_or_None)} — never raises, so a
    single bad bucket can't take the rest down with it.
    并发脱水多个桶（限流），缓存命中的秒回，未命中的（真调API）互相重叠而不是排队。
    返回 {bucket_id: (摘要或None, 异常或None)}，不抛异常，一个桶坏了不连累其他桶。
    """
    sem = asyncio.Semaphore(concurrency)

    async def _one(b: dict):
        async with sem:
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                return b["id"], (summary, None)
            except Exception as e:
                return b["id"], (None, e)

    pairs = await asyncio.gather(*(_one(b) for b in buckets))
    return dict(pairs)


@mcp.custom_route("/api/simulate/breath", methods=["GET"])
async def api_simulate_breath(request):
    """Dry-run of breath()'s no-query surfacing mode (line ~838 in breath())."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        max_results = min(int(request.query_params.get("max_results", 20)), 50)
        max_tokens = min(int(request.query_params.get("max_tokens", 10000)), 20000)
    except (TypeError, ValueError):
        max_results, max_tokens = 20, 10000

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    ranked_all, weight_ranks, score_snapshot = _weight_rank_snapshot(all_buckets)
    dream_recent = _select_dream_recent(all_buckets)
    dream_ids = {b["id"] for b in dream_recent}

    def score_of(b):
        return score_snapshot.get(b["id"], decay_engine.calculate_score(b["metadata"]))

    pinned_buckets = [
        b for b in all_buckets
        if b["metadata"].get("pinned") or b["metadata"].get("protected")
    ]
    pinned_ids = {b["id"] for b in pinned_buckets}

    unresolved = [
        b for b in all_buckets
        if not b["metadata"].get("resolved", False)
        and b["metadata"].get("type") not in ("permanent", "feel", "letter", "crave")
        and not b["metadata"].get("pinned", False)
        and not b["metadata"].get("protected", False)
        and b["id"] not in dream_ids
    ]
    unresolved_ids = {b["id"] for b in unresolved}

    surfaced = []
    not_surfaced = []

    # --- everything else that isn't in `unresolved`: classify why it's excluded ---
    for b in all_buckets:
        bid = b["id"]
        if bid in pinned_ids or bid in unresolved_ids:
            continue
        meta = b["metadata"]
        if bid in dream_ids:
            not_surfaced.append(_sim_row(b, score_of(b), reason="reserved_for_dream"))
        elif meta.get("type") in ("permanent", "feel", "letter", "crave"):
            not_surfaced.append(_sim_row(b, score_of(b), reason="excluded_type"))
        elif meta.get("resolved", False):
            not_surfaced.append(_sim_row(b, score_of(b), reason="resolved"))

    # --- cold-start + weight ranking + diversity shuffle, exactly as breath() ---
    scored = sorted(unresolved, key=lambda b: score_snapshot[b["id"]], reverse=True)
    cold_start = [
        b for b in unresolved
        if int(b["metadata"].get("activation_count", 0)) == 0
        and int(b["metadata"].get("importance", 0)) >= 8
    ][:2]
    cold_start_ids = {b["id"] for b in cold_start}
    scored_deduped = [b for b in scored if b["id"] not in cold_start_ids]
    candidates = cold_start + scored_deduped

    if len(candidates) > 1:
        n_cold = len(cold_start)
        non_cold = candidates[n_cold:]
        if len(non_cold) > 1:
            top1 = [non_cold[0]]
            pool = non_cold[1:min(20, len(non_cold))]
            random.shuffle(pool)
            non_cold = top1 + pool + non_cold[min(20, len(non_cold)):]
        candidates = cold_start + non_cold

    chosen = candidates[:max_results]
    beyond_cap = candidates[max_results:]
    for b in beyond_cap:
        not_surfaced.append(_sim_row(b, score_of(b), reason="beyond_max_results"))

    # --- dehydrate pinned + chosen concurrently (bounded), THEN apply the exact
    # same sequential token-budget bookkeeping breath() uses. Concurrency only
    # speeds up the I/O; the selection outcome is identical to doing it one at a time. ---
    # --- 钉选桶 + 候选桶并发脱水（限流），之后再按 breath() 原本的顺序逐条扣 token 预算。
    # 并发只是让"等结果"这一步重叠，选桶/扣预算的结果跟排队一个个做完全一样。---
    dehydrated = await _dehydrate_concurrent(pinned_buckets + chosen, concurrency=5)

    token_budget = max_tokens
    for b in pinned_buckets:
        summary, err = dehydrated.get(b["id"], (None, RuntimeError("missing")))
        if err:
            not_surfaced.append(_sim_row(b, score_of(b), reason="dehydrate_error", detail=str(err)))
            continue
        token_budget -= count_tokens_approx(summary)
        surfaced.append(_sim_row(b, score_of(b), channel="pin", summary=summary))

    stopped = False
    for b in chosen:
        if stopped:
            not_surfaced.append(_sim_row(b, score_of(b), reason="token_exhausted"))
            continue
        if token_budget <= 0:
            not_surfaced.append(_sim_row(b, score_of(b), reason="token_exhausted"))
            stopped = True
            continue
        summary, err = dehydrated.get(b["id"], (None, RuntimeError("missing")))
        if err:
            not_surfaced.append(_sim_row(b, score_of(b), reason="dehydrate_error", detail=str(err)))
            continue
        summary_tokens = count_tokens_approx(summary)
        if summary_tokens > token_budget:
            not_surfaced.append(_sim_row(b, score_of(b), reason="summary_exceeds_budget"))
            stopped = True
            continue
        token_budget -= summary_tokens
        channel = "cold_start" if b["id"] in cold_start_ids else "dynamic"
        surfaced.append(_sim_row(b, score_of(b), channel=channel, summary=summary))

    not_surfaced.sort(key=lambda r: r["score"] if r["score"] is not None else -1, reverse=True)
    return JSONResponse({
        "mode": "breath",
        "total_buckets": len(all_buckets),
        "surfaced": surfaced,
        "not_surfaced": not_surfaced,
    })


@mcp.custom_route("/api/simulate/dream", methods=["GET"])
async def api_simulate_dream(request):
    """Dry-run of dream() (line ~1644) — it never touches buckets in production either."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    _ranked_all, weight_ranks, score_snapshot = _weight_rank_snapshot(all_buckets)
    recent = _select_dream_recent(all_buckets)
    recent_ids = {b["id"] for b in recent}

    def score_of(b):
        return score_snapshot.get(b["id"], decay_engine.calculate_score(b["metadata"]))

    surfaced = [
        _sim_row(b, score_of(b), channel="dream", content_preview=strip_wikilinks(b.get("content", ""))[:500])
        for b in recent
    ]
    not_surfaced = []
    for b in all_buckets:
        if b["id"] in recent_ids:
            continue
        meta = b["metadata"]
        if meta.get("pinned") or meta.get("protected"):
            not_surfaced.append(_sim_row(b, score_of(b), reason="pinned_excluded"))
        elif meta.get("type") in ("permanent", "feel", "letter", "crave"):
            not_surfaced.append(_sim_row(b, score_of(b), reason="excluded_type"))
        else:
            not_surfaced.append(_sim_row(b, score_of(b), reason="not_recent_enough"))

    not_surfaced.sort(key=lambda r: r["score"] if r["score"] is not None else -1, reverse=True)
    return JSONResponse({
        "mode": "dream",
        "total_buckets": len(all_buckets),
        "surfaced": surfaced,
        "not_surfaced": not_surfaced,
    })


@mcp.custom_route("/api/simulate/feel", methods=["GET"])
async def api_simulate_feel(request):
    """Dry-run of breath(domain='feel') (line ~741) — never dehydrates or touches."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
    feels.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
    top10, rest = feels[:10], feels[10:]

    surfaced = [
        _sim_row(b, None, channel="feel", created=b["metadata"].get("created", ""),
                 content_preview=strip_wikilinks(b.get("content", ""))[:500])
        for b in top10
    ]
    not_surfaced = [
        _sim_row(b, None, reason="beyond_top10", created=b["metadata"].get("created", ""))
        for b in rest
    ]
    return JSONResponse({
        "mode": "feel",
        "total_buckets": len(all_buckets),
        "surfaced": surfaced,
        "not_surfaced": not_surfaced,
    })


@mcp.custom_route("/api/simulate/crave", methods=["GET"])
async def api_simulate_crave(request):
    """Dry-run of breath(domain='crave') (line ~787) — never dehydrates or touches."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    craves = [b for b in all_buckets if b["metadata"].get("type") == "crave"]
    active = [b for b in craves if not b["metadata"].get("digested", False)]
    digested = [b for b in craves if b["metadata"].get("digested", False)]
    active.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
    top10, rest = active[:10], active[10:]

    surfaced = [
        _sim_row(b, None, channel="crave", created=b["metadata"].get("created", ""),
                 content_preview=strip_wikilinks(b.get("content", ""))[:500])
        for b in top10
    ]
    not_surfaced = [
        _sim_row(b, None, reason="beyond_top10", created=b["metadata"].get("created", ""))
        for b in rest
    ] + [
        _sim_row(b, None, reason="digested", created=b["metadata"].get("created", ""))
        for b in digested
    ]
    return JSONResponse({
        "mode": "crave",
        "total_buckets": len(all_buckets),
        "surfaced": surfaced,
        "not_surfaced": not_surfaced,
    })


@mcp.custom_route("/api/dehydrate-preview/{bucket_id}", methods=["GET"])
async def api_dehydrate_preview(request):
    """On-demand dehydration preview for a single bucket. Read-only: dehydrate()
    only touches its own SQLite cache, never bucket metadata / activation state.

    ?raw=1 returns the LLM's structured JSON (core_facts/summary — older
    cache entries from before todos/keywords/emotion_state were dropped may
    still carry those extra fields, harmlessly ignored) plus both renderable
    variants ("summary" and "facts", see dehydrator._render_dehydrated) and
    which one auto mode would currently pick — so the dashboard can show a
    side-by-side comparison before the user decides whether to override
    dehydration_mode.
    """
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    content = strip_wikilinks(bucket["content"])
    clean_meta = {k: v for k, v in bucket["metadata"].items() if k != "tags"}
    try:
        if request.query_params.get("raw") == "1":
            if clean_meta.get("verbatim") or count_tokens_approx(content) < 100:
                return JSONResponse({"id": bucket_id, "raw": None, "note": "verbatim 或内容过短，未经过 JSON 脱水"})
            cached = dehydrator._get_cached_summary(content)
            parsed = None
            if cached:
                try:
                    parsed = dehydrator._parse_dehydration(cached, source=content[:3000])
                except ValueError:
                    cached = None
            raw_text = cached if cached else await dehydrator._api_dehydrate(content)
            if parsed is None:
                parsed = dehydrator._parse_dehydration(raw_text, source=content[:3000])
            if not cached:
                dehydrator._set_cached_summary(content, raw_text)
            variants = None
            auto_picks = None
            if isinstance(parsed, dict) and "summary" in parsed:
                variants = {
                    "summary": dehydrator._render_dehydrated(parsed, "summary"),
                    "facts": dehydrator._render_dehydrated(parsed, "facts"),
                }
                auto_picks = dehydrator._pick_dehydration_mode(parsed, "auto")
            return JSONResponse({
                "id": bucket_id,
                "raw": parsed,
                "raw_text": raw_text,
                "from_cache": bool(cached),
                "variants": variants,
                "auto_picks": auto_picks,
                "current_override": clean_meta.get("dehydration_mode", "auto"),
            })
        summary = await dehydrator.dehydrate(content, clean_meta)
        return JSONResponse({"id": bucket_id, "summary": summary})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/dehydrate-preview/{bucket_id}", methods=["POST"])
async def api_dehydrate_preview_save(request):
    """Save a hand-edited dehydration draft (core_facts + summary), overwriting
    whatever the LLM produced in the cache. Pins the edit to the bucket's current
    content hash — if the raw content is edited afterward the hash won't match
    and GET /api/bucket/{id} will report dehydration_stale=true so the dashboard
    can warn instead of silently discarding the manual edit."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    core_facts = body.get("core_facts")
    summary = body.get("summary")
    if not isinstance(core_facts, list) or not isinstance(summary, str) or not summary.strip():
        return JSONResponse({"error": "core_facts (list) and summary (non-empty string) required"}, status_code=400)
    mode = body.get("dehydration_mode")
    if mode is not None and mode not in ("auto", "facts", "summary"):
        return JSONResponse({"error": "invalid dehydration_mode"}, status_code=400)
    content = strip_wikilinks(bucket["content"])
    parsed = {"core_facts": [str(f) for f in core_facts], "summary": summary.strip()}
    content_hash = dehydrator.set_manual_summary(content, parsed)
    updates = {
        "dehydration_edited_hash": content_hash,
        "verbatim": False,
    }
    if mode is not None:
        updates["dehydration_mode"] = mode
    updated = await bucket_mgr.update(bucket_id, touch=False, **updates)
    if not updated:
        return JSONResponse({"error": "update failed"}, status_code=500)
    return JSONResponse({"ok": True})


def _dehydration_queue_rows(
    all_buckets: list[dict],
    scope: str,
    today=None,
    timezone_offset_minutes: int = 0,
) -> list[dict]:
    """Build the manual-review queue without reading or changing any cache."""
    today = today or datetime.now().date()
    oldest = today - timedelta(days=6)
    rows = []
    for bucket in all_buckets:
        meta = bucket.get("metadata", {})
        if meta.get("type", "dynamic") not in ("dynamic", "permanent"):
            continue
        content = strip_wikilinks(bucket.get("content", ""))
        if meta.get("verbatim") or count_tokens_approx(content) < 100:
            continue
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        edited_hash = str(meta.get("dehydration_edited_hash") or "")
        if edited_hash == content_hash:
            continue
        created = str(meta.get("created") or "")
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            # Browser getTimezoneOffset is UTC - local (China = -480).
            # Bucket timestamps are currently server-local naive timestamps;
            # production runs in UTC, so translate them to the reviewer's day.
            if created_dt.tzinfo is not None:
                created_dt = created_dt.astimezone(timezone.utc).replace(tzinfo=None)
            created_date = (created_dt - timedelta(minutes=timezone_offset_minutes)).date()
        except (ValueError, TypeError):
            created_date = None
        if scope == "today" and created_date != today:
            continue
        if scope == "week" and (created_date is None or created_date < oldest or created_date > today):
            continue
        rows.append({
            "id": bucket["id"],
            "name": meta.get("name") or bucket["id"],
            "type": meta.get("type", "dynamic"),
            "domain": meta.get("domain", []),
            "created": created,
            "content_chars": len(content),
            "estimated_tokens": count_tokens_approx(content),
            "content_preview": content[:160],
            "stale_manual": bool(edited_hash),
        })
    rows.sort(key=lambda row: row["created"], reverse=True)
    return rows


@mcp.custom_route("/api/dehydration-queue", methods=["GET"])
async def api_dehydration_queue(request):
    """List normal buckets that still need a human-approved dehydration."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    scope = request.query_params.get("scope", "today")
    if scope not in ("today", "week", "all"):
        return JSONResponse({"error": "scope must be today, week or all"}, status_code=400)
    try:
        reviewer_today = datetime.fromisoformat(
            request.query_params.get("today", datetime.now().date().isoformat())
        ).date()
        timezone_offset = int(request.query_params.get("tz_offset", "0"))
        if not -840 <= timezone_offset <= 840:
            raise ValueError
    except (ValueError, TypeError):
        return JSONResponse({"error": "invalid reviewer date or timezone"}, status_code=400)
    try:
        buckets = await bucket_mgr.list_all(include_archive=False)
        rows = _dehydration_queue_rows(
            buckets,
            scope,
            today=reviewer_today,
            timezone_offset_minutes=timezone_offset,
        )
        return JSONResponse({"scope": scope, "count": len(rows), "items": rows})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@mcp.custom_route("/api/dehydration-review/{bucket_id}", methods=["GET"])
async def api_dehydration_review(request):
    """Open a review without triggering the LLM or mutating the cache."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    meta = bucket.get("metadata", {})
    content = strip_wikilinks(bucket.get("content", ""))
    cached = dehydrator._get_cached_summary(content)
    parsed = None
    cache_issue = ""
    if cached:
        try:
            parsed = dehydrator._parse_dehydration(cached, source=content[:3000])
        except ValueError as exc:
            cache_issue = str(exc)
    return JSONResponse({
        "id": bucket_id,
        "name": meta.get("name") or bucket_id,
        "created": meta.get("created", ""),
        "domain": meta.get("domain", []),
        "content": content,
        "content_chars": len(content),
        "estimated_tokens": count_tokens_approx(content),
        "dehydration_mode": meta.get("dehydration_mode", "auto"),
        "cached": parsed,
        "cache_issue": cache_issue,
        "manual_current": bool(meta.get("dehydration_edited_hash")) and (
            meta.get("dehydration_edited_hash") == hashlib.sha256(content.encode()).hexdigest()
        ),
    })


@mcp.custom_route("/api/dehydration-review/{bucket_id}/draft", methods=["POST"])
async def api_dehydration_review_draft(request):
    """Generate an uncached AI draft, split against the complete source."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    content = strip_wikilinks(bucket.get("content", ""))
    try:
        draft = await dehydrator.generate_review_draft(content)
        return JSONResponse({"id": bucket_id, **draft})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@mcp.custom_route("/api/dehydration-review/{bucket_id}/verbatim", methods=["POST"])
async def api_dehydration_review_verbatim(request):
    """Approve keeping the complete original, without activating the bucket."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    bucket_id = request.path_params["bucket_id"]
    if not await bucket_mgr.get(bucket_id):
        return JSONResponse({"error": "not found"}, status_code=404)
    updated = await bucket_mgr.update(bucket_id, touch=False, verbatim=True)
    if not updated:
        return JSONResponse({"error": "update failed"}, status_code=500)
    return JSONResponse({"ok": True})


@mcp.custom_route("/dashboard", methods=["GET"])
async def dashboard(request):
    """Serve the dashboard HTML page."""
    from starlette.responses import HTMLResponse
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)


@mcp.custom_route("/api/config", methods=["GET"])
async def api_config_get(request):
    """Get current runtime config (safe fields only, API key masked)."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    dehy = config.get("dehydration", {})
    emb = config.get("embedding", {})
    api_key = dehy.get("api_key", "")
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else ("***" if api_key else "")
    return JSONResponse({
        "dehydration": {
            "model": dehy.get("model", ""),
            "base_url": dehy.get("base_url", ""),
            "api_key_masked": masked_key,
            "max_tokens": dehy.get("max_tokens", 1024),
            "temperature": dehy.get("temperature", 0.1),
        },
        "embedding": {
            "enabled": emb.get("enabled", False),
            "model": emb.get("model", ""),
        },
        "merge_threshold": config.get("merge_threshold", 75),
        "transport": config.get("transport", "stdio"),
        "buckets_dir": config.get("buckets_dir", ""),
    })


@mcp.custom_route("/api/config", methods=["POST"])
async def api_config_update(request):
    """Hot-update runtime config. Optionally persist to config.yaml."""
    from starlette.responses import JSONResponse
    import yaml
    err = _require_auth(request)
    if err: return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    updated = []

    # --- Dehydration config ---
    if "dehydration" in body:
        d = body["dehydration"]
        dehy = config.setdefault("dehydration", {})
        for key in ("model", "base_url", "max_tokens", "temperature"):
            if key in d:
                dehy[key] = d[key]
                updated.append(f"dehydration.{key}")
        if "api_key" in d and d["api_key"]:
            dehy["api_key"] = d["api_key"]
            updated.append("dehydration.api_key")
        # Hot-reload dehydrator
        dehydrator.model = dehy.get("model", "deepseek-v4-flash")
        dehydrator.base_url = dehy.get("base_url", "")
        dehydrator.api_key = dehy.get("api_key", "")
        if hasattr(dehydrator, "client") and dehydrator.api_key:
            from openai import AsyncOpenAI
            dehydrator.client = AsyncOpenAI(
                api_key=dehydrator.api_key,
                base_url=dehydrator.base_url,
            )

    # --- Embedding config ---
    if "embedding" in body:
        e = body["embedding"]
        emb = config.setdefault("embedding", {})
        if "enabled" in e:
            emb["enabled"] = bool(e["enabled"])
            embedding_engine.enabled = emb["enabled"]
            updated.append("embedding.enabled")
        if "model" in e:
            emb["model"] = e["model"]
            embedding_engine.model = emb["model"]
            updated.append("embedding.model")

    # --- Merge threshold ---
    if "merge_threshold" in body:
        config["merge_threshold"] = int(body["merge_threshold"])
        updated.append("merge_threshold")

    # --- Persist to config.yaml if requested ---
    if body.get("persist", False):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        try:
            save_config = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    save_config = yaml.safe_load(f) or {}

            if "dehydration" in body:
                sc_dehy = save_config.setdefault("dehydration", {})
                for key in ("model", "base_url", "max_tokens", "temperature"):
                    if key in body["dehydration"]:
                        sc_dehy[key] = body["dehydration"][key]
                # Never persist api_key to yaml (use env var)

            if "embedding" in body:
                sc_emb = save_config.setdefault("embedding", {})
                for key in ("enabled", "model"):
                    if key in body["embedding"]:
                        sc_emb[key] = body["embedding"][key]

            if "merge_threshold" in body:
                save_config["merge_threshold"] = int(body["merge_threshold"])

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(save_config, f, default_flow_style=False, allow_unicode=True)
            updated.append("persisted_to_yaml")
        except Exception as e:
            return JSONResponse({"error": f"persist failed: {e}", "updated": updated}, status_code=500)

    return JSONResponse({"updated": updated, "ok": True})


# =============================================================
# /api/host-vault — read/write the host-side OMBRE_HOST_VAULT_DIR
# 用于在 Dashboard 设置 docker-compose 挂载的宿主机记忆桶目录。
# 写入项目根目录的 .env 文件，需 docker compose down/up 才能生效。
# =============================================================

def _project_env_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _read_env_var(name: str) -> str:
    """Return current value of `name` from process env first, then .env file (best-effort)."""
    val = os.environ.get(name, "").strip()
    if val:
        return val
    env_path = _project_env_path()
    if not os.path.exists(env_path):
        return ""
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == name:
                    return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _write_env_var(name: str, value: str) -> None:
    """
    Idempotent upsert of `NAME=value` in project .env. Creates the file if missing.
    Preserves other entries verbatim. Quotes values containing spaces.
    """
    env_path = _project_env_path()
    quoted = f'"{value}"' if value and (" " in value or "#" in value) else value
    new_line = f"{name}={quoted}\n"

    lines: list[str] = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    replaced = False
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, _, _v = stripped.partition("=")
        if k.strip() == name:
            lines[i] = new_line
            replaced = True
            break
    if not replaced:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(new_line)

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


@mcp.custom_route("/api/host-vault", methods=["GET"])
async def api_host_vault_get(request):
    """Read the current OMBRE_HOST_VAULT_DIR (process env > project .env)."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    value = _read_env_var("OMBRE_HOST_VAULT_DIR")
    return JSONResponse({
        "value": value,
        "source": "env" if os.environ.get("OMBRE_HOST_VAULT_DIR", "").strip() else ("file" if value else ""),
        "env_file": _project_env_path(),
    })


@mcp.custom_route("/api/host-vault", methods=["POST"])
async def api_host_vault_set(request):
    """
    Persist OMBRE_HOST_VAULT_DIR to the project .env file.
    Body: {"value": "/path/to/vault"}  (empty string clears the entry)
    Note: container restart is required for docker-compose to pick up the new mount.
    """
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    raw = body.get("value", "")
    if not isinstance(raw, str):
        return JSONResponse({"error": "value must be a string"}, status_code=400)
    value = raw.strip()

    # Reject characters that would break .env / shell parsing
    if "\n" in value or "\r" in value or '"' in value or "'" in value:
        return JSONResponse({"error": "value must not contain quotes or newlines"}, status_code=400)

    try:
        _write_env_var("OMBRE_HOST_VAULT_DIR", value)
    except Exception as e:
        return JSONResponse({"error": f"failed to write .env: {e}"}, status_code=500)

    return JSONResponse({
        "ok": True,
        "value": value,
        "env_file": _project_env_path(),
        "note": "已写入 .env；需在宿主机执行 `docker compose down && docker compose up -d` 让新挂载生效。",
    })


# =============================================================
# Import API — conversation history import
# 导入 API — 对话历史导入
# =============================================================

@mcp.custom_route("/api/import/upload", methods=["POST"])
async def api_import_upload(request):
    """Upload a conversation file and start import."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err

    if import_engine.is_running:
        return JSONResponse({"error": "Import already running"}, status_code=409)

    content_type = request.headers.get("content-type", "")
    filename = ""

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            file_field = form.get("file")
            if not file_field:
                return JSONResponse({"error": "No file field"}, status_code=400)
            raw_bytes = await file_field.read()
            filename = getattr(file_field, "filename", "upload")
            raw_content = raw_bytes.decode("utf-8", errors="replace")
        else:
            body = await request.body()
            raw_content = body.decode("utf-8", errors="replace")
            # Try to get filename from query params
            filename = request.query_params.get("filename", "upload")

        if not raw_content.strip():
            return JSONResponse({"error": "Empty file"}, status_code=400)

        preserve_raw = request.query_params.get("preserve_raw", "").lower() in ("1", "true")
        resume = request.query_params.get("resume", "").lower() in ("1", "true")

    except Exception as e:
        return JSONResponse({"error": f"Failed to read upload: {e}"}, status_code=400)

    # Start import in background
    async def _run_import():
        try:
            await import_engine.start(raw_content, filename, preserve_raw, resume)
        except Exception as e:
            logger.error(f"Import failed: {e}")

    asyncio.create_task(_run_import())

    return JSONResponse({
        "status": "started",
        "filename": filename,
        "size_bytes": len(raw_content.encode()),
    })


@mcp.custom_route("/api/import/status", methods=["GET"])
async def api_import_status(request):
    """Get current import progress."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    return JSONResponse(import_engine.get_status())


@mcp.custom_route("/api/import/pause", methods=["POST"])
async def api_import_pause(request):
    """Pause the running import."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    if not import_engine.is_running:
        return JSONResponse({"error": "No import running"}, status_code=400)
    import_engine.pause()
    return JSONResponse({"status": "pause_requested"})


@mcp.custom_route("/api/import/patterns", methods=["GET"])
async def api_import_patterns(request):
    """Detect high-frequency patterns after import."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        patterns = await import_engine.detect_patterns()
        return JSONResponse({"patterns": patterns})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/results", methods=["GET"])
async def api_import_results(request):
    """List recently imported/created buckets for review."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        limit = int(request.query_params.get("limit", "50"))
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # Sort by created time, newest first
        all_buckets.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
        results = []
        for b in all_buckets[:limit]:
            results.append({
                "id": b["id"],
                "name": b["metadata"].get("name", ""),
                "content": b["content"][:300],
                "type": b["metadata"].get("type", ""),
                "domain": b["metadata"].get("domain", []),
                "tags": b["metadata"].get("tags", []),
                "importance": b["metadata"].get("importance", 5),
                "created": b["metadata"].get("created", ""),
            })
        return JSONResponse({"buckets": results, "total": len(all_buckets)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/review", methods=["POST"])
async def api_import_review(request):
    """Apply review decisions: mark buckets as important/noise/pinned."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    decisions = body.get("decisions", [])
    if not decisions:
        return JSONResponse({"error": "No decisions provided"}, status_code=400)

    applied = 0
    errors = 0
    for d in decisions:
        bid = d.get("bucket_id", "")
        action = d.get("action", "")
        if not bid or not action:
            continue
        try:
            if action == "important":
                await bucket_mgr.update(bid, touch=False, importance=9)
            elif action == "pin":
                await bucket_mgr.update(bid, touch=False, pinned=True)
            elif action == "noise":
                await bucket_mgr.update(bid, touch=False, resolved=True, importance=1)
            elif action == "delete":
                file_path = bucket_mgr._find_bucket_file(bid)
                if file_path:
                    os.remove(file_path)
            applied += 1
        except Exception as e:
            logger.warning(f"Review action failed for {bid}: {e}")
            errors += 1

    return JSONResponse({"applied": applied, "errors": errors})


# =============================================================
# /api/status — system status for Dashboard settings tab
# /api/status — Dashboard 设置页用系统状态
# =============================================================
@mcp.custom_route("/api/status", methods=["GET"])
async def api_system_status(request):
    """Return detailed system status for the settings panel."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        stats = await bucket_mgr.get_stats()
        return JSONResponse({
            "decay_engine": "running" if decay_engine.is_running else "stopped",
            "embedding_enabled": embedding_engine.enabled,
            "buckets": {
                "permanent": stats.get("permanent_count", 0),
                "dynamic": stats.get("dynamic_count", 0),
                "archive": stats.get("archive_count", 0),
                "total": stats.get("permanent_count", 0) + stats.get("dynamic_count", 0),
            },
            "using_env_password": bool(os.environ.get("OMBRE_DASHBOARD_PASSWORD", "")),
            "version": "1.3.0",
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# --- Entry point / 启动入口 ---
if __name__ == "__main__":
    transport = config.get("transport", "stdio")
    logger.info(f"Ombre Brain starting | transport: {transport}")

    if transport in ("sse", "streamable-http"):
        import threading
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        # --- Application-level keepalive: ping /health every 60s ---
        # --- 应用层保活：每 60 秒 ping 一次 /health，防止 Cloudflare Tunnel 空闲断连 ---
        async def _keepalive_loop():
            await asyncio.sleep(10)  # Wait for server to fully start
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        await client.get(f"http://localhost:{OMBRE_PORT}/health", timeout=5)
                        logger.debug("Keepalive ping OK / 保活 ping 成功")
                    except Exception as e:
                        logger.warning(f"Keepalive ping failed / 保活 ping 失败: {e}")
                    await asyncio.sleep(60)

        def _start_keepalive():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_keepalive_loop())

        t = threading.Thread(target=_start_keepalive, daemon=True)
        t.start()

        # --- Add CORS middleware so remote clients (Cloudflare Tunnel / ngrok) can connect ---
        # --- 添加 CORS 中间件，让远程客户端（Cloudflare Tunnel / ngrok）能正常连接 ---
        if transport == "streamable-http":
            _app = mcp.streamable_http_app()
        else:
            _app = mcp.sse_app()
        _app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
        logger.info("CORS middleware enabled for remote transport / 已启用 CORS 中间件")
        uvicorn.run(_app, host=OMBRE_BIND_HOST, port=OMBRE_PORT)
    else:
        mcp.run(transport=transport)
