"""Versioned JSON-scalar repair; preserves v1 sealed source/partial results.

Delegates unchanged numerical diagnosis to v1 with explicitly versioned output
roots and a scalar-only serialization adapter. No model or numerical-rule change.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.diagnose_physics_stress as engine
from scripts.field_derivative_diagnostics import band_errors as numerical_band_errors
from scripts.run_public_ridge_residual import locked_json
from src.public_benchmark import file_hash
OUT=ROOT/'results/physics_diagnosis_v2';DATA=ROOT/'external_data/physics_diagnosis_v2'


def python_scalars(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:python_scalars(v) for k,v in value.items()}
    if isinstance(value,list):return [python_scalars(v) for v in value]
    return value


def safe_band_errors(*args,**kwargs):return python_scalars(numerical_band_errors(*args,**kwargs))


def plan():
    parent=ROOT/'results/physics_diagnosis_v1/plan.json';p=json.loads(parent.read_text())
    for name,h in p['source'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('sealed v1 numerical source changed')
    p['supersedes_plan_sha256']=file_hash(parent)
    p['serialization_repair']='Cast NumPy scalar values to native Python scalars before JSON, unchanged numerical rules. V1 partial artifacts preserved.'
    p['source'][Path(__file__).relative_to(ROOT).as_posix()]=file_hash(Path(__file__))
    locked_json(OUT/'plan.json',p);return p


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','run']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan()
    # Explicit compatibility adapter, recorded in v2 plan/source fingerprints.
    engine.OUT=OUT;engine.DATA=DATA;engine.band_errors=safe_band_errors
    if args.stage=='run':engine.run(p)
