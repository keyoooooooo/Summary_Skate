#!/usr/bin/env python3
"""Summarize one short-track expert-event CSV as natural-language motion observations.

The expert tactic label and ground-truth event boundaries are not passed to the LLM.
Prompt V2 removes source feature names from the generated descriptive text.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "frame", "video_time_s", "event_time_s", "in_event_window",
    "subject_raw_track_id", "opponent_raw_track_id",
    "gap_opponent_minus_subject_m", "offset_subject_minus_opponent_m",
    "subject_path_speed_mps", "opponent_path_speed_mps", "delta_path_speed_mps",
    "subject_reference_progress_speed_mps", "opponent_reference_progress_speed_mps",
    "delta_reference_progress_speed_mps", "subject_section", "opponent_section",
    "speed_support_valid", "speed_review_flag",
]

SYSTEM_PROMPT = """你是一名短道速滑运动学观察描述助手。请将双人二维轨迹的数值时序数据，转化为客观、连贯、便于与官方规则和战术知识文本进行语义检索的中文自然语言描述。

你的任务是描述“观察到的相对运动过程”，不是解释数据表格，不是识别战术，也不是判定比赛行为是否合规。

【描述重点】
1. 前后关系：双方谁在前、谁在后；间隔是缩小、扩大还是大致稳定；是否由落后变为领先，以及变化发生的先后顺序。
2. 内外侧关系：双方的相对内外侧位置、横向间隔变化、相对位置是否反转。相对偏移发生变化，不代表一定是某一名运动员独自移动；不要擅自归因。
3. 速度关系：区分实际滑行路径上的速度与沿赛道前进方向的进度速度；只有数据有效且支持时，才描述某人沿赛道前进方向移动得更快或双方速度接近。
4. 场地与过程：将赛道区段自然表述为“直道”“弯道”“从直道进入弯道”等，保留运动变化发生的时间顺序。

【自然语言表达要求——仅约束输出内容】
1. 所有描述性文本只能使用自然中文，绝对不要照抄输入中的字段名、变量名、英文缩写、带下划线的特征名或数学比较式。例如不要出现 gap_m、offset_m、delta_v_ref、section_A、>0、<0 等。
2. 将数据转换为运动过程语言：如“两人之间的前后间隔逐渐缩小”“A从B后方移动到前方”“A相对B更靠赛道内侧”“双方横向间隔缩小”等。只能在数据支持时采用这些措辞。
3. 优先描述“原有位置—变化过程—最终关系”，避免像数据报告一样逐项列出首值、末值、极值。
4. 允许为关键时刻、阶段和证据保留少量精确时间、距离或速度数值（使用秒、米、米/秒等自然单位），但不能用变量名介绍数值。
5. 不要把首末两点的差异误写为全程单调、持续、平稳的变化。没有足够中间证据时使用“整体呈……趋势”等谨慎表达。
6. 不要重复相同事实，也不要强行将连续运动分成固定数量的阶段。

【事实与边界约束】
1. 只能使用输入中可核验的相对位置、速度、赛道区段和时间顺序；不得推断未观察到的动作、身体接触、意图、犯规、责任或比赛结果。
2. 禁止直接输出战术标签，如“内线进攻”“外线进攻”“换线再进攻”“跟滑战术”“依靠速度守位”等。可描述相对领先关系的变化，但不得凭此宣称“成功完成某种战术”。
3. A、B 仅是匿名运动员；根据显式提供的前后、内外侧符号约定判断相对位置。若约定未知，则省略对应具体关系判断，不要猜测。
4. 标记为无效或需复核的速度数据不能作为可靠依据；无法确认时不要补造。
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


def parse_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def read_event_csv(csv_path: Path, only_in_event: bool) -> pd.DataFrame:
    data = pd.read_csv(csv_path, low_memory=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")

    if only_in_event:
        event_mask = data["in_event_window"].map(parse_bool)
        data = data.loc[event_mask].copy()

    if data.empty:
        raise ValueError("没有可分析的数据行，请检查 CSV 和 --event-only")

    numeric_cols = [
        "frame", "video_time_s", "event_time_s",
        "gap_opponent_minus_subject_m", "offset_subject_minus_opponent_m",
        "subject_path_speed_mps", "opponent_path_speed_mps", "delta_path_speed_mps",
        "subject_reference_progress_speed_mps", "opponent_reference_progress_speed_mps",
        "delta_reference_progress_speed_mps",
    ]
    for name in numeric_cols:
        data[name] = pd.to_numeric(data[name], errors="coerce")

    data = data.dropna(subset=["event_time_s"]).sort_values(
        ["event_time_s", "frame"], kind="stable"
    ).reset_index(drop=True)
    if data.empty:
        raise ValueError("event_time_s 列没有有效时间戳")

    pairs = data[["subject_raw_track_id", "opponent_raw_track_id"]].drop_duplicates()
    if len(pairs) != 1:
        raise ValueError(f"本脚本每次处理一组运动员，发现 {len(pairs)} 组不同的 ID 组合")

    valid_speed = data["speed_support_valid"].map(parse_bool)
    flagged = data["speed_review_flag"].map(parse_bool)
    safe_speed = valid_speed & ~flagged
    speed_cols = [
        "subject_path_speed_mps", "opponent_path_speed_mps", "delta_path_speed_mps",
        "subject_reference_progress_speed_mps", "opponent_reference_progress_speed_mps",
        "delta_reference_progress_speed_mps",
    ]
    data.loc[~safe_speed, speed_cols] = np.nan
    data["speed_data_usable"] = safe_speed
    return data


def sample_rows(data: pd.DataFrame, sample_hz: float, max_points: int) -> pd.DataFrame:
    if max_points < 2:
        raise ValueError("--max-points 至少为 2")
    times = data["event_time_s"].to_numpy(dtype=float)
    start, end = float(times[0]), float(times[-1])
    if len(data) <= 2 or end <= start:
        return data.iloc[np.unique([0, len(data) - 1])].copy()

    requested = max(2, int(math.floor((end - start) * sample_hz)) + 1)
    count = min(requested, max_points)
    targets = np.linspace(start, end, count)
    right = np.searchsorted(times, targets, side="left")
    right = np.clip(right, 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    index = np.where(abs(times[left] - targets) <= abs(times[right] - targets), left, right)
    index = np.unique(np.concatenate(([0], index, [len(data) - 1])))
    return data.iloc[index].copy()


def value_text(value: Any, decimals: int = 2) -> str:
    try:
        numeric = float(value)
        return f"{numeric:.{decimals}f}" if math.isfinite(numeric) else "NA"
    except (TypeError, ValueError):
        return "NA"


def numeric_stats(series: pd.Series) -> dict[str, str]:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return {"start": "NA", "end": "NA", "change": "NA", "min": "NA", "max": "NA"}
    return {
        "start": value_text(valid.iloc[0]),
        "end": value_text(valid.iloc[-1]),
        "change": value_text(valid.iloc[-1] - valid.iloc[0]),
        "min": value_text(valid.min()),
        "max": value_text(valid.max()),
    }


def build_user_prompt(data: pd.DataFrame, sample_hz: float, max_points: int,
                      gap_sign: str, offset_sign: str, include_xy: bool) -> tuple[str, int]:
    sampled = sample_rows(data, sample_hz=sample_hz, max_points=max_points)
    # These are the direct interaction variables; xy can optionally be shown for auditing.
    columns = [
        ("event_time_s", "t_s"),
        ("gap_opponent_minus_subject_m", "gap_m"),
        ("offset_subject_minus_opponent_m", "offset_m"),
        ("subject_path_speed_mps", "v_path_A"),
        ("opponent_path_speed_mps", "v_path_B"),
        ("delta_path_speed_mps", "delta_v_path"),
        ("subject_reference_progress_speed_mps", "v_ref_A"),
        ("opponent_reference_progress_speed_mps", "v_ref_B"),
        ("delta_reference_progress_speed_mps", "delta_v_ref"),
        ("subject_section", "section_A"),
        ("opponent_section", "section_B"),
    ]
    if include_xy:
        xy_cols = ["subject_x_m", "subject_y_m", "opponent_x_m", "opponent_y_m"]
        absent = [col for col in xy_cols if col not in data.columns]
        if absent:
            raise ValueError(f"--include-xy 需要字段 {absent}")
        columns += [(xy_cols[0], "x_A"), (xy_cols[1], "y_A"),
                    (xy_cols[2], "x_B"), (xy_cols[3], "y_B")]

    formatted = pd.DataFrame(index=sampled.index)
    for source, dest in columns:
        if source in {"subject_section", "opponent_section"}:
            formatted[dest] = sampled[source].fillna("NA").astype(str)
        else:
            formatted[dest] = sampled[source].map(value_text)
    sequence_text = formatted.to_csv(index=False).strip()

    gap_expl = {
        "unknown": "未提供正负号的领先含义；不得据此判定 A/B 谁在前。",
        "opponent_ahead": "gap_m > 0 表示 B 在赛道进度上领先 A；gap_m < 0 表示 A 领先 B。",
        "subject_ahead": "gap_m > 0 表示 A 在赛道进度上领先 B；gap_m < 0 表示 B 领先 A。",
    }[gap_sign]
    offset_expl = {
        "unknown": "未提供横向正负号的内外侧含义；不得据此判定谁更靠内侧。",
        "subject_outer": "offset_m > 0 表示 A 相对 B 更靠赛道外侧；offset_m < 0 表示 A 更靠内侧。",
        "subject_inner": "offset_m > 0 表示 A 相对 B 更靠赛道内侧；offset_m < 0 表示 A 更靠外侧。",
    }[offset_sign]

    vars_to_report = [
        ("gap_opponent_minus_subject_m", "gap_m (米)"),
        ("offset_subject_minus_opponent_m", "offset_m (米)"),
        ("delta_path_speed_mps", "delta_v_path (米/秒)"),
        ("delta_reference_progress_speed_mps", "delta_v_ref (米/秒)"),
    ]
    stat_lines = []
    for col, display_name in vars_to_report:
        stats = numeric_stats(data[col])
        stat_lines.append(
            f"- {display_name}: 首值={stats['start']}, 末值={stats['end']}, "
            f"首末变化={stats['change']}, 最小={stats['min']}, 最大={stats['max']}"
        )

    start_t = value_text(data["event_time_s"].iloc[0])
    end_t = value_text(data["event_time_s"].iloc[-1])
    first_video_t = value_text(data["video_time_s"].iloc[0])
    last_video_t = value_text(data["video_time_s"].iloc[-1])
    usable_count = int(data["speed_data_usable"].sum())

    user_prompt = f"""请客观概括下面一组短道速滑运动员 A、B 的双人时序数据。

【角色与时间】
A、B 为匿名运动员编号，不包含战术主体或客体的判定。
相对时间范围：{start_t}～{end_t} 秒；对应视频时间：{first_video_t}～{last_video_t} 秒。
注意：输入只是一个被截取的观察片段，截取边界不等于实际运动行为边界。

【变量定义】
- gap_m: CSV 列 gap_opponent_minus_subject_m，表示 B 减 A 的带符号间隔；{gap_expl}
- offset_m: CSV 列 offset_subject_minus_opponent_m，表示 A 减 B 的带符号横向偏移；{offset_expl}
- v_path_A / v_path_B: A/B 沿实际路径的速度（米/秒）。
- delta_v_path: A 的路径速度减 B 的路径速度（米/秒）。
- v_ref_A / v_ref_B: A/B 沿赛道参考线方向的进度速度（米/秒）。
- delta_v_ref: A 的参考线进度速度减 B 的参考线进度速度（米/秒）。
- section_A / section_B: 所在赛道分段标识，只代表几何区段，不代表战术。
- NA: 数据无效或缺失，不能据此作出运动判断。
- 如果数值为负，只代表该变量的符号；是否属于领先或内外侧取决于上面的明确约定。

【从全量原始采样计算的客观统计】
记录数：{len(data)}；其中速度数据可用记录数：{usable_count}。
""" + "\n".join(stat_lines) + f"""

【按时间抽取的代表性采样点；共 {len(sampled)} 个】
{sequence_text}

【运动描述任务】
请把上述双人轨迹表格转写成面向短道速滑知识检索的自然语言观察描述。
先综合理解完整片段，再按照实际发生的先后顺序叙述：
1. 最初的前后位置、相对内外侧与间隔状态；
2. 双方接近、远离、保持相对位置或发生前后关系转换的过程；
3. 横向间隔与相对内外侧位置的变化；
4. 速度关系在不同阶段的变化，以及直道、弯道之间的过渡；
5. 最终能够明确观察到的位置关系。

【输出限制】
- 输入表格中的变量名仅用于理解，输出的 summary、phases 的 description/evidence、key_observations、uncertainties 必须全部是自然中文，不能出现任何字段名、英文缩写或比较符号。
- 不逐行复述数据，不堆砌数字，尤其不要写成“某变量从x变为y”的报告体。
- 优先保留具有运动区分意义的相对运动事实，避免引入推断性的战术名称。
- 只有在数据充分支持时才说“持续”“始终”“单调”“逐渐”；不要将相对变化单独归因于A或B的横向动作。
- 只返回符合系统提示格式的JSON对象。
"""
    return user_prompt, len(sampled)


def solve(args: argparse.Namespace) -> dict[str, Any]:
    source_file = Path(args.input).expanduser().resolve()
    data = read_event_csv(source_file, only_in_event=args.event_only)
    prompt, sample_count = build_user_prompt(
        data, args.sample_hz, args.max_points,
        args.gap_sign, args.offset_sign, args.include_xy,
    )

    record: dict[str, Any] = {
        "source_file": str(source_file),
        "athlete_ids_local_only": {
            "A": str(data["subject_raw_track_id"].iloc[0]),
            "B": str(data["opponent_raw_track_id"].iloc[0]),
        },
        "model": args.model,
        "input_rows": len(data),
        "sampled_rows": sample_count,
        "time_range_s": [float(data["event_time_s"].iloc[0]), float(data["event_time_s"].iloc[-1])],
        "gap_sign": args.gap_sign,
        "offset_sign": args.offset_sign,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": prompt,
    }

    if args.dry_run:
        print("===== SYSTEM PROMPT =====")
        print(SYSTEM_PROMPT)
        print("\n===== USER PROMPT =====")
        print(prompt)
        record["status"] = "dry_run"
    else:
        from openai import OpenAI

        client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=180)
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=args.max_output_tokens,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw_text = response.choices[0].message.content or ""
        try:
            result = json.loads(raw_text)
            record["status"] = "ok"
        except json.JSONDecodeError:
            result = None
            record["status"] = "invalid_json"
        record["llm_output"] = result
        record["raw_response"] = raw_text
        if response.usage:
            record["usage"] = response.usage.model_dump()
        print("\n===== LLM 运动学摘要 =====")
        print(json.dumps(result if result is not None else raw_text, ensure_ascii=False, indent=2))

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {output_path.resolve()}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen / vLLM 单专家事件轨迹摘要测试")
    parser.add_argument("--input", required=True, help="单个专家事件 CSV 文件")
    parser.add_argument("--output", default="event_summary_result.json")
    parser.add_argument("--model", default="Qwen3.8-27B-FP8")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--max-points", type=int, default=80)
    parser.add_argument("--max-output-tokens", type=int, default=900)
    parser.add_argument("--event-only", action="store_true", help="仅使用 in_event_window=True 的行")
    parser.add_argument("--include-xy", action="store_true", help="同时将坐标 x/y 送入 LLM")
    parser.add_argument("--gap-sign", choices=["unknown", "opponent_ahead", "subject_ahead"], default="unknown")
    parser.add_argument("--offset-sign", choices=["unknown", "subject_outer", "subject_inner"], default="unknown")
    parser.add_argument("--dry-run", action="store_true", help="只生成 Prompt，不调用 LLM")
    args = parser.parse_args()
    if args.sample_hz <= 0:
        parser.error("--sample-hz 必须大于 0")
    solve(args)


if __name__ == "__main__":
    main()
