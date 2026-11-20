#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A组：短道速滑专家事件运动学摘要消融（A1/A2/A3，支持 fixed/expert）。

Prompt参考用户提供的 expert_event_summary(2).py Prompt V2：自然中文、
运动事实与时间顺序、无战术标签、summary/phases/evidence/observations/uncertainties。

输入：CSV/expert_annotations.json + CSV/<race_id>/athlete_*.csv + pairwise/subject_*_opponent_*.csv
输出：每种时间范围模式一份独立 JSON；每一次 LLM 调用完成后原子落盘，可断点续跑。
依赖：pandas, numpy, openai；Python 3.10+。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "ablation_A_v1.2_anonymous"
PROMPT_VERSION = "expert_event_summary_prompt_V2_anonymous_AB"
PAIR_F1 = [
    "gap_opponent_minus_subject_m",
    "offset_subject_minus_opponent_m",
    "subject_section", "opponent_section",
    "delta_path_speed_mps", "delta_reference_progress_speed_mps",
]
SINGLE_S1 = [
    "position_x_m", "position_y_m", "velocity_x_mps", "velocity_y_mps",
    "path_speed_mps", "path_tangential_accel_mps2", "heading_rad",
    "yaw_rate_radps", "curvature_inv_m", "lateral_offset_ref_m",
    "lateral_speed_mps", "lateral_accel_mps2", "reference_progress_speed_mps",
    "section",
]
EXPERIMENTS = {
    "A1": "仅双人F1数值序列，直接生成双人运动学摘要",
    "A2": "双人F1数值 + 主体和对手各自的单人S1数值序列，生成双人运动学摘要",
    "A3": "主体和对手分别用S1生成单人LLM摘要，再与双人F1数值一起生成双人摘要",
}

SYSTEM_PAIR = """你是一名短道速滑运动学观察描述助手。请将双人二维轨迹的数值时序数据，转化为客观、连贯、便于与官方规则和战术知识文本进行语义检索的中文自然语言描述。

你的任务是描述“观察到的相对运动过程”，不是解释数据表格，不是识别战术，也不是判定比赛行为是否合规。

【描述重点】
1. 前后关系：双方谁在前、谁在后；间隔是缩小、扩大还是大致稳定；是否由落后变为领先，以及变化发生的先后顺序。
2. 内外侧关系：双方的相对内外侧位置、横向间隔变化、相对位置是否反转。相对偏移发生变化，不代表一定是某一名运动员独自移动；不要擅自归因。
3. 速度关系：区分实际滑行路径上的速度与沿赛道前进方向的进度速度；只有数据有效且支持时，才描述某人沿赛道前进方向移动得更快或双方速度接近。
4. 场地与过程：将赛道区段自然表述为“直道”“弯道”“从直道进入弯道”等，保留运动变化发生的时间顺序。

【自然语言表达要求——仅约束输出内容】
1. 所有描述性文本应以自然中文表达运动过程；运动员只能用A和B表示，禁止出现任何真实姓名。不要照抄输入中的字段名、变量名、英文缩写、带下划线的特征名或数学比较式。例如不要出现 gap_m、offset_m、delta_v_ref、section_A、>0、<0 等。
2. 将数据转换为运动过程语言：如“两人之间的前后间隔逐渐缩小”“A从B后方移动到前方”“A相对B更靠赛道内侧”“双方横向间隔缩小”等。只能在数据支持时采用这些措辞。
3. 优先描述“原有位置—变化过程—最终关系”，避免像数据报告一样逐项列出首值、末值、极值。
4. 允许为关键时刻、阶段和证据保留少量精确时间、距离或速度数值（使用秒、米、米/秒等自然单位），但不能用变量名介绍数值。
5. 不要把首末两点的差异误写为全程单调、持续、平稳的变化。没有足够中间证据时使用“整体呈……趋势”等谨慎表达。
6. 不要重复相同事实，也不要强行将连续运动分成固定数量的阶段。

【事实与边界约束】
1. 只能使用输入中可核验的相对位置、速度、赛道区段和时间顺序；不得推断未观察到的动作、身体接触、意图、犯规、责任或比赛结果。
2. 禁止直接输出战术标签，如“内线进攻”“外线进攻”“换线再进攻”“跟滑战术”“依靠速度守位”等。可描述相对领先关系的变化，但不得凭此宣称“成功完成某种战术”。
3. A、B 是匿名运动员角色，不提供运动员真实姓名；A并不必然代表战术进攻者。所有输出字段，包括summary、phases的description/evidence、key_observations、uncertainties，均只使用A/B。根据显式提供的前后、内外侧符号约定判断相对位置，不要猜测。
4. 标记为无效或需复核的速度数据不能作为可靠依据；输入null或NA为缺失，不是零；无法确认时不要补造。
5. 截取窗口边界并不等于真实运动行为的开始和结束；阶段划分只表示观察到的运动变化。
6. 若证据不充分，应简要说明缺失的条件，而非输出猜测或无意义的泛化声明。

【JSON 输出格式】
只输出一个合法 JSON 对象，不输出 Markdown、解释或思考过程。必须保留以下字段名以供程序解析；字段的文本内容须满足上述自然语言要求。
{
  "summary": "100～180字左右，连贯描述整体运动过程，优先保留前后、内外侧、速度和场地区段变化",
  "phases": [
    {
      "time_range_s": [0.0, 1.0],
      "description": "这一阶段的自然语言运动描述",
      "evidence": "用自然中文描述可核验的支持证据，必要时引用时间或数值，不能出现输入变量名"
    }
  ],
  "key_observations": ["最多5条不同的关键运动学观察，均用自然中文表述"],
  "uncertainties": ["仅列出会影响判断的实质性数据限制或不确定性；没有则返回空数组"]
}
"""


SYSTEM_SINGLE = """你是一名短道速滑单人运动学观察描述助手。请将单人二维轨迹的数值时序数据转化为客观、连贯的中文自然语言描述。你的任务是描述该运动员自身的运动过程，不是解释表格、识别战术或判定行为是否合规。

【描述重点】
1. 位置与运动方向：只根据位置、航向、轨迹曲率及赛道区段描述运动方向和直道/弯道变化。
2. 速度与加速度：区分实际滑行路径速度与沿参考线的前进速度；无效数据不能作为证据。
3. 横向运动：横向偏移正值代表相对赛道参考线更靠外侧，负值代表更靠内侧；横向速度的正负代表偏移变化方向，不等于战术动作。
4. 时间顺序：优先描述“原有状态—变化过程—最终状态”，不得将首末差异误写为全程单调变化。

【语言与事实要求】
1. 描述性文本使用自然中文，只使用匿名角色A/B，禁止输出真实姓名、CSV字段名、英文变量名、缩写、下划线名称或数学比较式。
2. 不堆砌逐点数值；关键证据可保留少量时间、速度或距离，须使用自然单位。
3. 只依据该运动员的单人数据描述。禁止猜测其他运动员、两人前后关系、战术标签、行动意图、接触、违规、排名和比赛结果。
4. null/NA是数据缺失，不得当成零；没有足够证据时说清无法判断的具体原因。
5. 片段边界不代表运动行为真实开始或结束；不要强行分成固定数量的阶段。

【JSON 输出格式】
只输出合法JSON对象，不输出Markdown或思考过程。只保留：
{
  "summary": "100～150字左右，连贯描述单人运动过程，不预测战术类别",
  "phases": [
    {"time_range_s": [0.0, 1.0], "description": "这一阶段的自然语言运动描述", "evidence": "自然中文描述的数值证据"}
  ],
  "key_observations": ["最多5条关键运动学观察"],
  "uncertainties": ["仅列出实质性数据限制；没有则为空数组"]
}
"""


PAIR_FLOAT_FIELDS = set(PAIR_F1) - {"subject_section", "opponent_section"}
SINGLE_FLOAT_FIELDS = set(SINGLE_S1) - {"section"}
SINGLE_GEOMETRY = {"lateral_offset_ref_m", "section"}
SINGLE_GEOM_SPEED = {"lateral_speed_mps", "reference_progress_speed_mps"}
SINGLE_GEOM_ACCEL = {"lateral_accel_mps2"}
SINGLE_ACCEL = {"path_tangential_accel_mps2"}
SINGLE_POSITION = {"position_x_m", "position_y_m"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(data: Any) -> str:
    return sha256_bytes(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temp_path = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temp_path, path)  # 同一目录中原子替换，适用于Windows
    finally:
        if temp_path.exists():
            temp_path.unlink()


def sanitize_id(raw: Any, key: str) -> str:
    value = str(raw)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
        raise ValueError(f"{key} 不允许包含路径或特殊字符: {value!r}")
    return value


def read_annotations(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少专家标注文件: {path}")
    source = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(source, dict):
        if isinstance(source.get("events"), list):
            source = source["events"]
        elif isinstance(source.get("annotations"), list):
            source = source["annotations"]
        elif "event_id" in source:
            source = [source]
    if not isinstance(source, list):
        raise ValueError("expert_annotations.json 需为事件对象数组、{'events':[...]} 或单个事件对象")
    events = []
    seen = set()
    for i, item in enumerate(source):
        if not isinstance(item, dict):
            raise ValueError(f"标注索引{i}不是对象")
        required = ["event_id", "race_id", "subject_id", "subject_name",
                    "opponent_id", "opponent_name", "start_frame", "end_frame"]
        missing = [k for k in required if k not in item]
        if missing:
            raise ValueError(f"标注索引{i}缺少: {missing}")
        event = dict(item)
        for k in ["event_id", "race_id", "subject_id", "opponent_id"]:
            event[k] = sanitize_id(event[k], k)
        for k in ["start_frame", "end_frame"]:
            if isinstance(event[k], bool) or int(event[k]) != float(event[k]):
                raise ValueError(f"{event['event_id']} 的 {k} 必须为整数")
            event[k] = int(event[k])
        if event["start_frame"] > event["end_frame"]:
            raise ValueError(f"{event['event_id']} 事件起点晚于终点")
        if event["subject_id"] == event["opponent_id"]:
            raise ValueError(f"{event['event_id']} 主体和对手不能相同")
        if event["event_id"] in seen:
            raise ValueError(f"重复事件ID: {event['event_id']}")
        seen.add(event["event_id"])
        events.append(event)
    return events


def load_csv(path: Path, fields: list[str], extra: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少CSV: {path}")
    header = pd.read_csv(path, nrows=0).columns
    wanted = list(dict.fromkeys(["frame", "video_time_s", *fields, *extra]))
    missing = sorted(set(wanted) - set(header))
    if missing:
        raise ValueError(f"{path} 缺少字段: {missing}")
    df = pd.read_csv(path, usecols=wanted, low_memory=False)
    frames = pd.to_numeric(df["frame"], errors="raise")
    if frames.isna().any() or not np.all(frames == np.floor(frames)):
        raise ValueError(f"{path} 有非法帧号")
    df["frame"] = frames.astype("int64")
    if df["frame"].duplicated().any() or df.empty:
        raise ValueError(f"{path} 存在重复frame或为空")
    return df.sort_values("frame").set_index("frame", drop=False)


def bool_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1"}


def number_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(numeric, 6) if math.isfinite(numeric) else None


def field_value(row: pd.Series, field: str, kind: str, exclude_reviewed_speed: bool) -> Any:
    """仅基于源CSV的质量标记屏蔽无效数值，不计算统计摘要或战术判定。"""
    if kind == "pair":
        geometry_ok = bool_value(row.get("pair_geometry_valid"))
        progress_ok = bool_value(row.get("pair_progress_valid"))
        speed_ok = bool_value(row.get("pair_speed_valid"))
        reviewed = ("subject_speed_review" in str(row.get("pair_quality_flags", ""))
                    or "opponent_speed_review" in str(row.get("pair_quality_flags", "")))
        if field == "gap_opponent_minus_subject_m":
            valid = progress_ok
        elif field in ("offset_subject_minus_opponent_m", "subject_section", "opponent_section"):
            valid = geometry_ok
        elif field == "delta_reference_progress_speed_mps":
            valid = speed_ok and geometry_ok and not (exclude_reviewed_speed and reviewed)
        else:
            valid = speed_ok and not (exclude_reviewed_speed and reviewed)
    else:
        pos_ok = bool_value(row.get("position_valid"))
        spd_ok = bool_value(row.get("speed_valid"))
        acc_ok = bool_value(row.get("accel_valid"))
        geom_ok = bool_value(row.get("geometry_valid"))
        reviewed = bool_value(row.get("speed_review_flag"))
        if field in SINGLE_POSITION:
            valid = pos_ok
        elif field in SINGLE_GEOMETRY:
            valid = geom_ok
        elif field in SINGLE_GEOM_SPEED:
            valid = spd_ok and geom_ok and not (exclude_reviewed_speed and reviewed)
        elif field in SINGLE_GEOM_ACCEL:
            valid = acc_ok and geom_ok and not (exclude_reviewed_speed and reviewed)
        elif field in SINGLE_ACCEL:
            valid = acc_ok and not (exclude_reviewed_speed and reviewed)
        else:
            valid = spd_ok and not (exclude_reviewed_speed and reviewed)
    if not valid:
        return None
    if field in ("section", "subject_section", "opponent_section"):
        value = row.get(field)
        return str(value) if pd.notna(value) and str(value).strip() else None
    return number_or_none(row.get(field))


def sample_frames(start: int, end: int, points: int) -> list[int]:
    count = min(points, end - start + 1)
    if count <= 0:
        raise ValueError("采样窗口没有帧")
    if count == 1:
        return [start]
    indices = np.rint(np.linspace(start, end, count)).astype(np.int64).tolist()
    if len(indices) != len(set(indices)) or indices[0] != start or indices[-1] != end:
        raise AssertionError("均匀抽样生成重复帧或丢失端点")
    return indices


def sampled_rows(df: pd.DataFrame, frames: list[int], start_frame: int, fps: float,
                 fields: list[str], kind: str, exclude_reviewed_speed: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sampled_df = df.reindex(frames)
    output = []
    valid_counts = {field: 0 for field in fields}
    missing_frames: list[int] = []
    for frame, row in sampled_df.iterrows():
        if pd.isna(row.get("frame")):
            missing_frames.append(int(frame))
        result: dict[str, Any] = {"t_s": round((int(frame) - start_frame) / fps, 4)}
        for field in fields:
            value = field_value(row, field, kind, exclude_reviewed_speed)
            result[field] = value
            if value is not None:
                valid_counts[field] += 1
        output.append(result)
    return output, {"sample_count": len(frames), "sampled_frames": frames,
                    "missing_source_frames": missing_frames, "valid_field_counts": valid_counts}


def determine_window(event: dict[str, Any], mode: str, fixed_window_s: float, fps: float,
                     available_start: int, available_end: int) -> dict[str, Any]:
    start, end = event["start_frame"], event["end_frame"]
    if mode == "expert":
        first, last = start, end
    else:
        span = round(fixed_window_s * fps)
        if span < end - start:
            raise ValueError(f"事件 {event['event_id']} 的专家持续时间 {(end-start)/fps:.3f}s > "
                             f"固定窗口 {span/fps:.3f}s，请增大 --fixed-window-s")
        # 固定采样时间跨度（区间内有 span+1 个原始帧），中点对齐事件并在数据边缘平移。
        center = (start + end) / 2.0
        first = round(center - span / 2.0)
        last = first + span
        if first < available_start:
            last += available_start - first
            first = available_start
        if last > available_end:
            first -= last - available_end
            last = available_end
    if first < available_start or last > available_end:
        raise ValueError(f"事件 {event['event_id']} 需要帧 [{first},{last}]，"
                         f"但双人及单人 CSV 公共可用范围 [{available_start},{available_end}]；"
                         "不可静默截断，请调整窗口或检查上游数据")
    if first > start or last < end:
        raise AssertionError("提取窗口没有覆盖完整专家事件")
    return {"start_frame": first, "end_frame": last, "duration_s": round((last-first)/fps, 6),
            "sampled_raw_frame_span": last-first+1, "expert_start_frame": start,
            "expert_end_frame": end, "expert_duration_s": round((end-start)/fps, 6),
            "window_mode": mode}


def load_event_data(root: Path, event: dict[str, Any], mode: str, fixed_window_s: float,
                    sample_points: int, exclude_reviewed_speed: bool) -> dict[str, Any]:
    race = root / event["race_id"]
    subj, opp = event["subject_id"], event["opponent_id"]
    pair_path = race / "pairwise" / f"subject_{subj}_opponent_{opp}.csv"
    subj_path = race / f"athlete_{subj}.csv"
    opp_path = race / f"athlete_{opp}.csv"
    metadata_path = race / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"找不到比赛metadata: {metadata_path}")
    race_metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    fps = float(race_metadata["data_info"]["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"比赛 {event['race_id']} 的fps无效")
    pair = load_csv(pair_path, PAIR_F1,
                    ["pair_geometry_valid", "pair_progress_valid", "pair_speed_valid", "pair_quality_flags"])
    subject = load_csv(subj_path, SINGLE_S1, ["position_valid", "speed_valid", "accel_valid",
                                               "geometry_valid", "speed_review_flag"])
    opponent = load_csv(opp_path, SINGLE_S1, ["position_valid", "speed_valid", "accel_valid",
                                                "geometry_valid", "speed_review_flag"])
    # 同一事件，所有A组实验使用完全相同的窗口（包括A1）。
    common_start = max(int(df.index.min()) for df in (pair, subject, opponent))
    common_end = min(int(df.index.max()) for df in (pair, subject, opponent))
    window = determine_window(event, mode, fixed_window_s, fps, common_start, common_end)
    frames = sample_frames(window["start_frame"], window["end_frame"], sample_points)
    pair_rows, pair_quality = sampled_rows(pair, frames, window["start_frame"], fps,
                                           PAIR_F1, "pair", exclude_reviewed_speed)
    subject_rows, subject_quality = sampled_rows(subject, frames, window["start_frame"], fps,
                                                 SINGLE_S1, "single", exclude_reviewed_speed)
    opponent_rows, opponent_quality = sampled_rows(opponent, frames, window["start_frame"], fps,
                                                   SINGLE_S1, "single", exclude_reviewed_speed)
    files = {"pairwise": str(pair_path.resolve()), "subject_single": str(subj_path.resolve()),
             "opponent_single": str(opp_path.resolve()), "race_metadata": str(metadata_path.resolve())}
    file_signatures = {key: {"bytes": Path(filename).stat().st_size,
                             "mtime_ns": Path(filename).stat().st_mtime_ns}
                       for key, filename in files.items()}
    return {"window": window, "fps": fps, "frames": frames, "files": files,
            "file_signatures": file_signatures, "pair_rows": pair_rows,
            "subject_rows": subject_rows, "opponent_rows": opponent_rows,
            "quality": {"pair": pair_quality, "subject": subject_quality,
                        "opponent": opponent_quality}}


def json_for_prompt(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def compact_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """列头只出现一次，避免逐行重复长字段名造成上下文溢出；不做任何统计或规则判断。"""
    if not rows:
        return {"columns": [], "rows": []}
    columns = list(rows[0].keys())
    return {"columns": columns, "rows": [[row.get(column) for column in columns] for row in rows]}


def build_single_prompt(event: dict[str, Any], role: str, rows: list[dict[str, Any]]) -> str:
    """单人Prompt遵循参考文件的‘客观过程、时序证据、自然中文’约束。"""
    if role not in ("subject", "opponent"):
        raise ValueError("role必须是subject或opponent")
    alias = "A" if role == "subject" else "B"
    last_t = rows[-1]["t_s"] if rows else 0.0
    return f"""请客观概括短道速滑运动员 {alias} 在一个观察窗口内的单人运动学过程。

【角色与时间】
仅有运动员 {alias} 的自身轨迹，禁止讨论其他运动员。相对时间：0.00～{last_t:.2f} 秒。
注意：截取窗口边界不等于实际运动行为边界。

【输入属性定义】
- position_x_m / position_y_m：冰面二维位置，米。
- velocity_x_mps / velocity_y_mps：二维速度分量，米/秒。
- path_speed_mps：运动员实际路径上的速度，米/秒。
- path_tangential_accel_mps2：沿路径切向加速度，米/秒²。
- heading_rad：运动方向角，弧度；yaw_rate_radps：航向变化率，弧度/秒。
- curvature_inv_m：轨迹曲率，1/米。
- lateral_offset_ref_m：参考线横向偏移，越大越靠赛道外侧，米。
- lateral_speed_mps：偏移变化速度，正值表示更偏外，米/秒。
- lateral_accel_mps2：横向加速度，米/秒²。
- reference_progress_speed_mps：沿赛道参考线的前进速度，米/秒。
- section：所在赛道区段，仅表示几何位置。
- t_s：相对于观察窗口开始的时间，秒。
- null：此字段缺失或质量无效，不能根据它得出结论。

【按时间抽取的代表性采样点，共 {len(rows)} 个】
表格JSON的columns是字段名，rows是按时间排序的数值：
{json_for_prompt(compact_table(rows))}

【运动描述任务】
按原有状态、变化过程、最终状态的顺序描述；优先保留速度、转向、赛道区段和横向变化的可核验事实。
禁止在输出文本中照抄字段名、变量名、英文缩写或比较式。不要把首末差异说成全程单调变化。不得引入未提供的对手信息和战术标签。
只返回系统提示要求的JSON对象。"""


def build_pair_prompt(event: dict[str, Any], pair_rows: list[dict[str, Any]],
                      single_subject: list[dict[str, Any]] | None = None,
                      single_opponent: list[dict[str, Any]] | None = None,
                      single_summaries: tuple[dict[str, Any], dict[str, Any]] | None = None) -> str:
    """保持参考文件Prompt V2中的变量解释、描述顺序与输出限制。"""
    last_t = pair_rows[-1]["t_s"] if pair_rows else 0.0
    lines = [
        "请客观概括下面一组短道速滑运动员 A、B 的双人时序数据。",
        "\n【角色与时间】",
        "A和B均为匿名观察对象，不提供真实姓名；A/B角色并不表示谁发动战术。",
        f"相对观察时间：0.00～{last_t:.2f} 秒。截取片段的边界不等于真实行为边界。",
        "\n【变量定义与符号约定】",
        "- gap_opponent_minus_subject_m：B减A的已对齐连续参考进度差（米）；正值表示B沿赛道参考进度领先A，负值表示A领先B；这不是经裁判确认的比赛名次。",
        "- offset_subject_minus_opponent_m：A减B的参考线横向偏移差（米）；正值表示A相对B更靠外侧，负值表示A相对B更靠内侧。",
        "- delta_path_speed_mps：A减B的实际运动路径速度差（米/秒）。",
        "- delta_reference_progress_speed_mps：A减B的赛道前进方向速度差（米/秒）；这与路径速度差不是同一物理量。",
        "- subject_section/opponent_section：A/B的赛道几何区段，只代表场地位置，不代表战术。",
        "- t_s：观察窗口起点后的相对时间（秒）；null表示字段无效或缺失，不能当成0。",
        f"\n【按时间抽取的代表性采样点；共 {len(pair_rows)} 个】",
        "表格JSON的columns表示每列变量，rows按时间先后排列：",
        json_for_prompt(compact_table(pair_rows)),
    ]
    if single_subject is not None and single_opponent is not None:
        lines += [
            "\n【额外单人运动学数值——A2配置】",
            "每人的单人时序与双人时序时间对齐。路径速度与参考线前进速度不同。",
            "单人横向偏移越大越靠外侧；横向速度正值表示向外偏移。",
            "航向角单位弧度、航向变化率弧度/秒、曲率1/米、加速度米/秒²。",
            "运动员A单人数据：", json_for_prompt(compact_table(single_subject)),
            "运动员B单人数据：", json_for_prompt(compact_table(single_opponent)),
        ]
    if single_summaries is not None:
        lines += [
            "\n【额外单人运动学观察摘要——A3配置】",
            "以下两个单人摘要只依据各自轨迹形成，是辅助信息；如与直接双人数据不一致，请依据双人原始采样并如实说明不确定性。",
            "运动员A独立摘要：", json_for_prompt(single_summaries[0]),
            "运动员B独立摘要：", json_for_prompt(single_summaries[1]),
        ]
    lines += [
        "\n【运动描述任务】",
        "请把双人时序数值转写为适于短道速滑权威知识检索的自然中文观察描述；按真实时间顺序综合叙述：",
        "1. 最初谁在前、谁在后，相对内外侧以及间隔关系；",
        "2. 双方间隔是接近、扩大还是变化不明显，是否发生前后位置交换；",
        "3. 横向间隔、内外侧位置关系的变化；不要把相对变化擅自归因于某一个人的动作；",
        "4. 基于有效数值判断速度关系变化，以及直道和弯道过渡；",
        "5. 最后能够确认的客观位置关系。",
        "\n【输出限制】",
        "- summary、phases.description/evidence、key_observations和uncertainties必须采用自然中文运动过程表述，运动员一律用A/B表示，不得输出真实姓名、输入字段名或数学比较式。",
        "- 优先原有状态—变化过程—最终关系；不逐行复述、不堆砌数字，也不编造统计峰值或没提供的细节。",
        "- 只有充分证据时才用‘持续’‘始终’‘单调’‘逐渐’；不强制划分固定数量阶段。",
        "- 不得输出战术标签、动作意图、犯规、接触、裁判判罚或凭轨迹无法证实的比赛结果。",
        "- 只返回系统提示中的四字段JSON对象。",
    ]
    return "\n".join(lines)


def validate_schema(parsed: Any, stage: str) -> tuple[bool, str]:
    """所有阶段与原专家摘要脚本采用相同JSON输出结构。"""
    if not isinstance(parsed, dict):
        return False, "根节点不是JSON对象"
    required = {"summary": str, "phases": list, "key_observations": list, "uncertainties": list}
    for name, kind in required.items():
        if name not in parsed or not isinstance(parsed[name], kind):
            return False, f"缺少字段或类型错误：{name}"
    for field in ("key_observations", "uncertainties"):
        if not all(isinstance(value, str) for value in parsed[field]):
            return False, f"{field} 的元素必须是字符串"
    for phase in parsed["phases"]:
        if not isinstance(phase, dict):
            return False, "phases的元素必须是对象"
        bounds = phase.get("time_range_s")
        if not isinstance(bounds, list) or len(bounds) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in bounds
        ):
            return False, "phase.time_range_s必须是两个有限数值"
        if bounds[0] > bounds[1]:
            return False, "phase.time_range_s起止次序错误"
        if not isinstance(phase.get("description"), str) or not isinstance(phase.get("evidence"), str):
            return False, "phase需要description和evidence字符串"
    return True, ""


def invoke_llm(client: Any, args: argparse.Namespace, system: str, user: str, stage: str) -> dict[str, Any]:
    started_at = now_iso()
    start = time.perf_counter()
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": args.temperature,
        "max_tokens": args.max_output_tokens,
    }
    if not args.no_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if not args.enable_thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    if args.seed is not None:
        kwargs["seed"] = args.seed
    try:
        result = client.chat.completions.create(**kwargs)
        raw = result.choices[0].message.content or ""
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = None
        valid, message = validate_schema(parsed, stage)
        usage = result.usage.model_dump(mode="json") if result.usage else {}
        return {"started_at_utc": started_at, "finished_at_utc": now_iso(),
                "elapsed_s": round(time.perf_counter() - start, 3),
                "status": "success" if valid else "invalid_response",
                "validation_error": message if not valid else None,
                "raw_response": raw, "parsed_response": parsed,
                "usage": usage, "finish_reason": result.choices[0].finish_reason,
                "model_returned": result.model, "response_id": result.id}
    except Exception as exc:
        return {"started_at_utc": started_at, "finished_at_utc": now_iso(),
                "elapsed_s": round(time.perf_counter() - start, 3),
                "status": "api_error", "error": f"{type(exc).__name__}: {exc}",
                "raw_response": None, "parsed_response": None,
                "usage": {}, "finish_reason": None}


def save_call(record: dict[str, Any], role: str, system: str, user: str, stage: str,
              args: argparse.Namespace, client: Any, checkpoint: Any) -> dict[str, Any]:
    """成功调用缓存并跳过；失败保留历史；每次调用完成后立即checkpoint。"""
    if record.get("status") == "success":
        return record
    expected_hash = stable_hash({"system": system, "user": user, "stage": stage})
    old_hash = record.get("request", {}).get("prompt_sha256")
    if old_hash is not None and expected_hash != old_hash:
        raise ValueError(f"{role} 已存储请求与当前不同，拒绝错误续跑（请换输出目录）")
    record["request"] = {"role": role, "stage": stage,
                         "prompt_sha256": expected_hash, "system_prompt": system,
                         "user_prompt": user, "model": args.model,
                         "temperature": args.temperature, "max_output_tokens": args.max_output_tokens,
                         "thinking_enabled": args.enable_thinking, "json_mode": not args.no_json_mode,
                         "seed": args.seed}
    record.setdefault("attempts", [])
    if args.dry_run:
        record["status"] = "dry_run"
        checkpoint()
        return record
    attempt = invoke_llm(client, args, system, user, stage)
    attempt["attempt_index"] = len(record["attempts"]) + 1
    record["attempts"].append(attempt)
    record["status"] = attempt["status"]
    record["updated_at_utc"] = now_iso()
    checkpoint()  # 保存完整原始响应，避免长时间GPU运行中损失已完成结果
    return record


def latest_parsed(call: dict[str, Any]) -> dict[str, Any] | None:
    if call.get("status") != "success":
        return None
    for attempt in reversed(call.get("attempts", [])):
        if attempt["status"] == "success":
            return attempt["parsed_response"]
    return None


def normalize_config(args: argparse.Namespace, annotations: Path) -> dict[str, Any]:
    """只放影响生成结果/输入选择的变量；--repeats和实验子集不参与签名。"""
    return {"script_version": SCRIPT_VERSION, "prompt_version": PROMPT_VERSION,
            "csv_dir": str(args.csv_dir.resolve()), "annotations": str(annotations.resolve()),
            "annotations_sha256": sha256_bytes(annotations.read_bytes()),
            "window_mode": args.window_mode,
            "fixed_window_s": args.fixed_window_s if args.window_mode == "fixed" else None,
            "sample_points": args.sample_points, "retrieval_query_field": "summary", "exclude_reviewed_speed": not args.include_reviewed_speed,
            "model": args.model, "base_url": args.base_url.rstrip("/"),
            "temperature": args.temperature, "max_output_tokens": args.max_output_tokens,
            "enable_thinking": args.enable_thinking, "json_mode": not args.no_json_mode,
            "seed": args.seed, "pair_fields": PAIR_F1, "single_fields": SINGLE_S1}


def initialize_results(path: Path, args: argparse.Namespace, annotations: Path) -> dict[str, Any]:
    config = normalize_config(args, annotations)
    config_hash = stable_hash(config)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8-sig"))
        actual = existing.get("run_metadata", {}).get("config_hash")
        if actual != config_hash:
            raise ValueError(f"现有结果文件与当前模型/窗口/输入配置不同：{path}\n"
                             "为避免混入不兼容结果，请指定其他 --output-dir 或 --output-file")
        return existing
    suffix = "-expert" if args.window_mode == "expert" else ""
    return {"schema_version": SCRIPT_VERSION, "group": "A", "window_mode": args.window_mode,
            "run_metadata": {"created_at_utc": now_iso(), "updated_at_utc": now_iso(),
                             "config_hash": config_hash, "config": config,
                             "note": "专家标签及运动员真实姓名仅保存在事件metadata；LLM输入与摘要均匿名使用A/B；"
                                     "运动学摘要仅由LLM生成，未加入程序化统计；后续检索统一编码匿名parsed_response.summary；"
                                     "system及user prompt参考expert_event_summary(2).py Prompt V2"},
            "experiments": {
                f"{eid}{suffix}": {"experiment_id": f"{eid}{suffix}",
                                    "description": desc,
                                    "settings": {"pair_fields": PAIR_F1,
                                                 "single_fields": SINGLE_S1 if eid != "A1" else [],
                                                 "single_mode": "none" if eid == "A1" else ("raw_numeric" if eid == "A2" else "llm_single_summaries"),
                                                 "sample_points": args.sample_points,
                                                 "window_mode": args.window_mode},
                                    "events": {}}
                for eid, desc in EXPERIMENTS.items()
            }}


def event_metadata(event: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    labels = {name: event.get(name) for name in
              ("action_code", "event_layer", "outcome_code", "key_frame", "key_frame_role", "window_duration_s")
              if name in event}
    return {"event_id": event["event_id"], "race_id": event["race_id"],
            "subject": {"athlete_id": event["subject_id"], "name": event["subject_name"]},
            "opponent": {"athlete_id": event["opponent_id"], "name": event["opponent_name"]},
            "athlete_aliases": {
                "A": {"athlete_id": event["subject_id"], "name": event["subject_name"]},
                "B": {"athlete_id": event["opponent_id"], "name": event["opponent_name"]},
            },
            "ground_truth_expert_interval": {"start_frame": event["start_frame"],
                                             "end_frame": event["end_frame"], **labels},
            "input_window": data["window"], "fps": data["fps"],
            "csv_sources": data["files"], "csv_file_signatures": data["file_signatures"],
            "sampling": data["quality"]}


def verify_event_fingerprint(existing_event: dict[str, Any], metadata: dict[str, Any]) -> None:
    previous = existing_event.get("metadata")
    if previous is None or "csv_file_signatures" not in previous:
        # 先前是 input_error 的记录没有读取过源CSV，修复文件后允许重试。
        return
    if previous.get("csv_file_signatures") != metadata["csv_file_signatures"]:
        raise ValueError(f"{metadata['event_id']} 的CSV已变化，拒绝复用旧LLM结果；请使用新的输出目录")
    if previous.get("input_window") != metadata["input_window"] or previous.get("sampling") != metadata["sampling"]:
        raise ValueError(f"{metadata['event_id']} 的输入帧或采样发生变化，拒绝混合结果")


def execute(args: argparse.Namespace, client: Any = None) -> Path:
    annotations = args.annotations if args.annotations is not None else args.csv_dir / "expert_annotations.json"
    annotations = annotations.expanduser().resolve()
    events = read_annotations(annotations)
    if args.event_ids:
        selected = set(args.event_ids)
        invalid = selected - {event["event_id"] for event in events}
        if invalid:
            raise ValueError(f"--event-ids 包含不存在的事件: {sorted(invalid)}")
        events = [e for e in events if e["event_id"] in selected]
    output_name = f"A_group_{args.window_mode}{'_dry_run' if args.dry_run else ''}.json"
    out_path = args.output_file if args.output_file else args.output_dir / output_name
    out_path = out_path.expanduser().resolve()
    if not args.dry_run and client is None:
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                        timeout=args.timeout, max_retries=0)
    results = initialize_results(out_path, args, annotations)

    def checkpoint() -> None:
        results["run_metadata"]["updated_at_utc"] = now_iso()
        atomic_write(out_path, results)

    checkpoint()
    suffix = "-expert" if args.window_mode == "expert" else ""
    for event in events:
        event_id = event["event_id"]
        print(f"\n[事件] {event_id} | {event['race_id']} | A:{event['subject_id']} -> B:{event['opponent_id']}", flush=True)
        try:
            data = load_event_data(args.csv_dir, event, args.window_mode,
                                   args.fixed_window_s, args.sample_points,
                                   not args.include_reviewed_speed)
            meta = event_metadata(event, data)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  [数据错误] {error}", flush=True)
            for experiment in args.experiments:
                key = experiment + suffix
                events_result = results["experiments"][key]["events"]
                if event_id not in events_result:
                    events_result[event_id] = {"status": "input_error", "error": error,
                                               "metadata": {"event_id": event_id, "race_id": event["race_id"]},
                                               "repeats": {}}
            checkpoint()
            if args.fail_fast:
                raise
            continue
        for experiment in args.experiments:
            key = experiment + suffix
            events_result = results["experiments"][key]["events"]
            item = events_result.setdefault(event_id, {"metadata": meta, "repeats": {}})
            verify_event_fingerprint(item, meta)
            item["metadata"] = meta
            item.pop("error", None)
            item["status"] = "running"
            checkpoint()
            for repeat in range(1, args.repeats + 1):
                repeat_key = str(repeat)
                rep = item["repeats"].setdefault(repeat_key, {"status": "running", "single_calls": {},
                                                             "pair_call": {}})
                if rep.get("status") == "success":
                    print(f"  [跳过] {key}/{event_id}/repeat_{repeat} 已完成", flush=True)
                    continue
                rep["status"] = "running"
                # A3：把每个单人调用完整保存到本事件对应A3的repeat内，支持中断后跳过成功的单人调用。
                single_summaries = []
                if experiment == "A3":
                    for role, single_rows in (("subject", data["subject_rows"]),
                                              ("opponent", data["opponent_rows"])):
                        call = rep["single_calls"].setdefault(role, {})
                        user_prompt = build_single_prompt(event, role, single_rows)
                        save_call(call, role, SYSTEM_SINGLE, user_prompt, "single", args, client, checkpoint)
                        parsed = latest_parsed(call)
                        print(f"  [{key} 单人 {role}] {call['status']}", flush=True)
                        if parsed is None:
                            if args.dry_run:
                                # 预检时继续保存B的单人调用提示词；不伪造单人模型回复。
                                continue
                            single_summaries = []
                            break
                        single_summaries.append(parsed)
                    if len(single_summaries) != 2:
                        rep["status"] = "dry_run" if args.dry_run else "blocked_by_single_call"
                        checkpoint()
                        # dry-run也允许显示双人最终prompt结构，但单人响应未知，不伪造摘要。
                        if args.dry_run:
                            rep["pair_call"]["status"] = "not_built_requires_single_outputs"
                            checkpoint()
                        continue
                user_prompt = build_pair_prompt(
                    event, data["pair_rows"],
                    data["subject_rows"] if experiment == "A2" else None,
                    data["opponent_rows"] if experiment == "A2" else None,
                    tuple(single_summaries) if experiment == "A3" else None,
                )
                pair_call = rep["pair_call"]
                save_call(pair_call, "pair", SYSTEM_PAIR, user_prompt, "pair", args, client, checkpoint)
                rep["status"] = pair_call["status"]
                print(f"  [{key} 第{repeat}次] {rep['status']}", flush=True)
                checkpoint()
                if args.fail_fast and rep["status"] not in ("success", "dry_run"):
                    raise RuntimeError(f"模型调用失败: {key}/{event_id}/repeat_{repeat}")
            rep_statuses = [r.get("status") for r in item["repeats"].values()]
            item["status"] = "success" if len(rep_statuses) >= args.repeats and all(
                item["repeats"][str(i)].get("status") == "success" for i in range(1, args.repeats + 1)
            ) else ("dry_run" if args.dry_run else "incomplete")
            checkpoint()
    print(f"\n[完成] 结果文件: {out_path}", flush=True)
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="短道速滑LLM运动学摘要 A组消融：A1/A2/A3")
    parser.add_argument("--csv-dir", type=Path, required=True,
                        help="V2_LLM_Based/CSV 根目录：含expert_annotations.json和5个比赛子目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="A组实验JSON输出目录")
    parser.add_argument("--annotations", type=Path, help="专家标注路径，默认<csv-dir>/expert_annotations.json")
    parser.add_argument("--output-file", type=Path, help="自定义结果JSON文件名；默认A_group_fixed/expert.json")
    parser.add_argument("--window-mode", choices=["fixed", "expert"], default="fixed",
                        help="fixed：固定长度、以专家区间中点为中心；expert：直接用标注起止帧")
    parser.add_argument("--fixed-window-s", type=float, default=4.0,
                        help="固定窗口时间跨度（秒），默认4.0；只用于fixed模式，必须覆盖完整事件")
    parser.add_argument("--sample-points", type=int, default=41, help="每一条时序均匀采样点数，默认41")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS),
                        help="按需运行A1/A2/A3中的一部分；多次执行可补全同一JSON")
    parser.add_argument("--event-ids", nargs="+", help="仅执行指定事件，如C001 C002（省略则执行全部）")
    parser.add_argument("--repeats", type=int, default=1, help="每个配置和专家事件的重复生成次数")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI兼容vLLM API")
    parser.add_argument("--model", default="Qwen3.8-27B-FP8", help="服务器served-model-name（大小写敏感）")
    parser.add_argument("--api-key", default=None, help="鉴权token；建议用OPENAI_API_KEY环境变量，结果不保存密钥")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-output-tokens", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--enable-thinking", action="store_true", help="默认禁用思考以提高速度")
    parser.add_argument("--no-json-mode", action="store_true", help="服务端不支持response_format=json_object时使用")
    parser.add_argument("--include-reviewed-speed", action="store_true",
                        help="默认屏蔽带speed_review_flag的速度字段；启用后保留")
    parser.add_argument("--dry-run", action="store_true", help="只生成输入prompt和JSON，不调用LLM")
    parser.add_argument("--fail-fast", action="store_true", help="遇到数据或API失败时停止，而不是保存错误并继续")
    args = parser.parse_args(argv)
    if args.sample_points < 2 or args.repeats < 1:
        parser.error("--sample-points需>=2，--repeats需>=1")
    if not math.isfinite(args.fixed_window_s) or args.fixed_window_s <= 0:
        parser.error("--fixed-window-s必须是正数")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature必须是非负数")
    if args.max_output_tokens < 64 or args.timeout <= 0:
        parser.error("--max-output-tokens至少64，--timeout必须>0")
    args.csv_dir = args.csv_dir.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    execute(args)


if __name__ == "__main__":
    main()
