#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通过 OpenAI 兼容接口测试本地 vLLM 的 Qwen 推理服务。

示例：
    python test_qwen_inference.py \
        --base-url http://127.0.0.1:8000/v1 \
        --model Qwen3.8-27B-FP8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from openai import OpenAI, OpenAIError


SYSTEM_PROMPT = "你是一名短道速滑运动学分析助手，只根据提供的事实回答。"
DEFAULT_USER_PROMPT = (
    "运动员A原本位于运动员B后方，随后沿内侧接近B并移动至B前方。"
    "请用一句客观的中文描述这一运动过程，不要推断动机或犯规。"
)


def solve(args: argparse.Namespace) -> dict[str, Any]:
    client = OpenAI(
        base_url=args.base_url.rstrip("/"),
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=0,
    )

    print(f"服务地址: {args.base_url}")
    print(f"模型名称: {args.model}")
    print("正在请求模型生成...", flush=True)

    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": args.prompt},
        ],
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    elapsed_seconds = time.perf_counter() - start_time

    content = response.choices[0].message.content or ""
    result = {
        "model": response.model,
        "base_url": args.base_url,
        "prompt": args.prompt,
        "response": content,
        "finish_reason": response.choices[0].finish_reason,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "usage": response.usage.model_dump() if response.usage else None,
    }

    print("\n===== Qwen 生成结果 =====")
    print(content if content else "(模型未返回可见文本，请检查 max_output_tokens 和 thinking 配置)")
    print(f"\n耗时: {elapsed_seconds:.2f} 秒")
    print(f"结束原因: {result['finish_reason']}")
    if response.usage:
        print("Token 用量:", json.dumps(result["usage"], ensure_ascii=False))

    if args.output:
        from pathlib import Path

        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"已保存结果: {output_path.resolve()}")

    if not content.strip():
        raise RuntimeError("请求成功，但模型返回了空文本")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="测试本地 vLLM / Qwen 推理服务")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3.8-27B-FP8")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--prompt", default=DEFAULT_USER_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", default="", help="可选的 JSON 结果文件路径")
    args = parser.parse_args()

    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens 必须大于 0")

    try:
        solve(args)
    except (OpenAIError, RuntimeError, OSError) as error:
        print(f"\n推理测试失败: {error}", file=sys.stderr)
        print(
            "请检查 vLLM 是否启动、端口是否正确，以及 --model 是否与 "
            "--served-model-name 一致。",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
