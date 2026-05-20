# ============================================================
# Module: Mood Engine (mood_engine.py)
# 模块：心情引擎
#
# 从记忆桶计算当前心情快照：
#   - 幂律衰减加权（Wave14）
#   - ESM 软互抑（防 PA/NA 同时极端）
#   - 返回自然语言心情描述，供 mood 工具使用
# ============================================================

import math
import logging
from datetime import datetime, timezone

logger = logging.getLogger("ombre_brain.mood")

# ---- 衰减参数（Wave14）----
TAU_HOURS = 4.0    # 时间尺度，单位小时
B_BASE    = 0.7    # 基础衰减指数
FAB_COEFF = 0.85   # 正面情绪衰减慢 15%（Fading Affect Bias）
ESM_K     = 0.3    # ESM 软互抑系数（防止 PA/NA 同时封顶）


def _hours_since(timestamp_str: str) -> float:
    """从 ISO 时间字符串计算距今多少小时。"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (now - dt).total_seconds() / 3600.0)
    except Exception:
        return 48.0  # 解析失败，保守估计 48 小时前


def _power_decay(hours_ago: float, importance: int, valence: float) -> float:
    """
    幂律衰减权重：w(t) = (1 + t/τ)^(-b_eff)

    - b_eff 随 importance 降低（重要事件衰减慢）
    - 正面情绪额外乘 FAB 系数（衰减更慢）
    """
    b_eff = B_BASE / (1.0 + importance / 10.0)
    if valence > 0.5:
        b_eff *= FAB_COEFF
    w = (1.0 + hours_ago / TAU_HOURS) ** (-b_eff)
    return max(0.0, min(1.0, w))


def compute_mood_snapshot(buckets: list) -> dict:
    """
    从记忆桶列表计算心情快照。

    Args:
        buckets: list of bucket dicts (来自 bucket_mgr.list_all)

    Returns dict:
        pa          — 正向情感强度 0~1
        na          — 负向情感强度 0~1
        top_memory  — 权重最高的记忆名称
        high_arousal— 最近未解决的高唤醒词列表
        description — 自然语言心情描述
    """
    if not buckets:
        return _empty_snapshot()

    pa_acc = 0.0
    na_acc = 0.0
    high_arousal_words: list[str] = []
    weighted: list[tuple[float, dict]] = []

    for b in buckets:
        meta = b.get("metadata", {})

        # 跳过不参与心情计算的类型
        if meta.get("type") in ("feel", "permanent"):
            continue
        if meta.get("pinned") or meta.get("protected"):
            continue
        # 已解决且已消化的，忽略
        if meta.get("resolved") and meta.get("digested"):
            continue

        v   = float(meta.get("valence",   0.5))
        a   = float(meta.get("arousal",   0.3))
        imp = max(1, min(10, int(meta.get("importance", 5))))

        ts  = meta.get("last_active") or meta.get("created", "")
        hrs = _hours_since(ts)
        w   = _power_decay(hrs, imp, v)

        # 已解决但未消化：权重骤降
        if meta.get("resolved"):
            w *= 0.05

        # PA / NA 贡献
        if v >= 0.5:
            pa_acc += (v - 0.5) * 2.0 * a * w
        else:
            na_acc += (0.5 - v) * 2.0 * a * w

        weighted.append((w, b))

        # 收集高唤醒词（未解决、arousal > 0.65）
        if a > 0.65 and not meta.get("resolved"):
            name = meta.get("name", "")
            if name and name not in high_arousal_words:
                high_arousal_words.append(name)

    if not weighted:
        return _empty_snapshot()

    # tanh 软压缩，防止无限累积
    pa = float(math.tanh(pa_acc * 2.0))
    na = float(math.tanh(na_acc * 2.0))
    pa = max(0.0, min(1.0, pa))
    na = max(0.0, min(1.0, na))

    # ESM 软互抑
    pa_f = round(pa * (1.0 - ESM_K * na), 3)
    na_f = round(na * (1.0 - ESM_K * pa), 3)
    pa_f = max(0.0, min(1.0, pa_f))
    na_f = max(0.0, min(1.0, na_f))

    # 权重最高的记忆
    weighted.sort(key=lambda x: x[0], reverse=True)
    top_b    = weighted[0][1]
    top_meta = top_b.get("metadata", {})
    top_mem  = top_meta.get("name", "") or top_b.get("content", "")[:50]

    desc = _build_description(pa_f, na_f, high_arousal_words[:3], top_mem)

    return {
        "pa":          pa_f,
        "na":          na_f,
        "top_memory":  top_mem,
        "high_arousal": high_arousal_words[:3],
        "description": desc,
    }


def _empty_snapshot() -> dict:
    return {
        "pa": 0.0, "na": 0.0,
        "top_memory": "", "high_arousal": [],
        "description": "此刻心绪平静，没有特别挂念的事。",
    }


def _build_description(pa: float, na: float,
                        high_arousal: list, top_memory: str) -> str:
    """把 PA/NA 数值转成自然语言，不暴露数字。"""
    parts: list[str] = []

    if pa > 0.6 and na < 0.2:
        parts.append("近来心里有些轻盈")
    elif pa > 0.4 and na < 0.3:
        parts.append("情绪还算稳定，有点温和的正向感")
    elif na > 0.5 and pa < 0.2:
        parts.append("近来心里有些沉")
    elif na > 0.3 and pa < 0.3:
        parts.append("隐约有些不安或低落")
    elif pa > 0.3 and na > 0.3:
        parts.append("心里有些复杂，正向和负向都有")
    else:
        parts.append("此刻心绪比较平静")

    if high_arousal:
        parts.append(f"脑子里转着：{'、'.join(high_arousal)}")

    if top_memory:
        parts.append(f"最挂念的是：{top_memory}")

    hint = ""
    if na > 0.4:
        hint = "（这种状态自然地带入对话，不用刻意点破）"
    elif pa > 0.5:
        hint = "（让这份轻松自然流淌在对话里）"

    result = "，".join(parts) + "。"
    if hint:
        result += "\n" + hint
    return result
