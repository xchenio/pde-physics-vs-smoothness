"""只读提取 Jupyter notebook 的代码单元，供第三方代码审计使用。

不执行 notebook、不修改原仓库，只把 code cell 的源码按关键词过滤后打印/导出，
便于在审计记录里引用"原始代码第几个 cell 写了什么"。

用法：
    D:/miniconda3/envs/pytorch/python.exe tools/notebook_peek.py external/pi_deeponet/Burger/PI_DeepONet_Burger.ipynb
    D:/miniconda3/envs/pytorch/python.exe tools/notebook_peek.py <path> --grep "u0,branch,reshape" --max-chars 600
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def code_cells(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        text = "".join(src) if isinstance(src, list) else str(src)
        yield i, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--grep", default="", help="逗号分隔的关键词；命中任一即打印该 cell")
    ap.add_argument("--max-chars", type=int, default=1200, help="每个 cell 最多打印的字符数")
    ap.add_argument("--all", action="store_true", help="打印全部 code cell")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    keys = [k.strip() for k in args.grep.split(",") if k.strip()]

    n = 0
    for idx, text in code_cells(path):
        hit = args.all or not keys or any(k in text for k in keys)
        if not hit:
            continue
        n += 1
        print(f"\n===== cell {idx} ({len(text)} chars) =====")
        print(text[: args.max_chars])
        if len(text) > args.max_chars:
            print(f"... [{len(text) - args.max_chars} chars truncated]")
    print(f"\n[{path.name}] matched {n} code cells", file=sys.stderr)


if __name__ == "__main__":
    main()
