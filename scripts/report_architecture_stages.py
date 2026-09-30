"""Generate a data-only Markdown table without selecting or opening confirmations."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_architecture_stages import OUT


def report():
    lines=['# Architecture development results','',
           'Mean per-sample relative L2; lower is better. New runs: three seeds, fixed final checkpoints. Development selection only; no confirmatory claims.','',
           '| Task | Stage / setting | Train | Development mean ± seed SD | Real parameters | Optimization seconds |',
           '|---|---|---:|---:|---:|---:|']
    for task in ('burgers_original','burgers_independent','allen_cahn'):
        for stage in ('fit','spatial','fno'):
            p=OUT/task/(stage+'_summary.json')
            if not p.exists():continue
            s=json.loads(p.read_text())
            if stage=='fit':
                lines.append(f"| {task} | frozen linear | {s['linear']['train']['mean_relative_l2']:.6f} | {s['linear']['val']['mean_relative_l2']:.6f} | — | — |")
            for label,g in s['groups'].items():
                v=g['metrics']['val']['mean_relative_l2'];t=g['metrics']['train']['mean_relative_l2']['mean']
                lines.append(f"| {task} | {stage}: {label} | {t:.6f} | {v['mean']:.6f} ± {v['std']:.6f} | {g['parameters_real']} | {g['mean_optimization_seconds']:.1f} |")
    lines+=['','## Historical strong controls','',
            'Previous five-seed development means, retained as context (not paired three-seed comparisons; different checkpoint selection).','',
            '| Task | All modes | Old continuous shrinkage |','|---|---:|---:|']
    for task in ('burgers_original','burgers_independent','allen_cahn'):
        h=json.loads((ROOT/'results/public_uncertainty_modal_v1'/task/'development.json').read_text())
        lines.append(f"| {task} | {h['means']['all']:.6f} | {h['means']['old_shrink']:.6f} |")
    lines+=['','Source: public_architecture_stages_v1/*/*_summary.json; historical controls from public_uncertainty_modal_v1/*/development.json.',
            'Budgets also change cosine schedules. CNN uses the same POD; FNO predicts the full field and has more parameters. Transferred MLP-selected settings are not architecture-specific tuning. Optimization times exclude evaluation/checkpoint overhead and are hardware/load dependent.','']
    return '\n'.join(lines)


if __name__=='__main__':print(report())
