"""登记新下载的公开数据文件（先算摘要，由人核对来源后再写进 pin 表）。

为什么不做成"自动写入"：SHA-256 pin 的意义是**人核对过来源**。
脚本只负责算数并打印，写入 `src/public_benchmark.py` 的 `EXPECTED_SHA256`
这一步留给人，避免把"下错了的文件"默默 pin 成权威来源。

用法：
    D:/miniconda3/envs/pytorch/python.exe scripts/register_public_file.py external_data/pdebench/1D_Burgers_Sols_Nu0.01.hdf5
    # 顺便看一眼 HDF5 里的 Nu，确认粘度确实是想要的那个
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.public_benchmark import EXPECTED_SHA256, file_hash  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true", help="只输出 JSON，便于脚本消费")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise SystemExit(f"not found: {p}")
    digest = file_hash(p)

    nu = None
    shape = None
    try:
        import h5py
        with h5py.File(p, "r") as f:
            nu = float(f.attrs["Nu"])
            shape = tuple(int(x) for x in f["tensor"].shape)
    except Exception as exc:                                   # noqa: BLE001
        print(f"[warn] could not read HDF5 attrs: {exc}", file=sys.stderr)

    info = {
        "file": p.name,
        "bytes": p.stat().st_size,
        "sha256": digest,
        "Nu": nu,
        "tensor_shape": shape,
        "already_pinned": EXPECTED_SHA256.get(p.name) == digest,
    }
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return
    print(f"file           : {info['file']}")
    print(f"size           : {info['bytes'] / 1e9:.2f} GB")
    print(f"Nu (HDF5 attr) : {nu}")
    print(f"tensor shape   : {shape}")
    print(f"sha256         : {digest}")
    print(f"already pinned : {info['already_pinned']}")
    print()
    if not info["already_pinned"]:
        print("核对来源后，把它加进 src/public_benchmark.py 的 EXPECTED_SHA256：")
        print(f"    '{p.name}':")
        print(f"        '{digest}',")


if __name__ == "__main__":
    main()
