"""Generate the baseline report figures from the archived training and evaluation files."""
import argparse
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train.references import build_reference
from train.evaluate import build_scoring_target
from score import config, scoring
from score.evaluate_core import evaluate


def training(exp, out):
    segments = [pd.read_csv(p) for p in sorted((exp / "training").glob("**/logs/reward_components.csv"))]
    history = pd.concat(segments).sort_values("timesteps")
    assert history.timesteps.is_unique and history.timesteps.iloc[-1] == 10002432
    names = [k for k in history if (k.endswith("_score") or k in ("reward","failure_penalty")) and history[k].notna().any()]
    fig, axes = plt.subplots(4, 3, figsize=(13,10), sharex=True, constrained_layout=True)
    for ax, key in zip(axes.flat, names):
        for idx, segment in enumerate(segments):
            x = segment.timesteps / 1e6
            ax.plot(x, segment[key], color="#9bc6d5", lw=.55)
            ax.plot(x, segment[key].rolling(20, min_periods=1).mean(), color="#105875", lw=1.3)
            if idx: ax.axvline(x.iloc[0], color="#a8592e", ls=":", lw=1)
        ax.set_title(key, fontsize=10)
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(names):]: ax.axis("off")
    fig.supxlabel("Training transitions (millions)")
    fig.suptitle("F2b baseline | logged reward and score components")
    fig.savefig(out / "training_rewards.png", dpi=145)
    plt.close(fig)


def evaluation(exp, split, out):
    folder = exp / "evaluation" / split
    entries = json.loads((folder/"infer_result.json").read_text())
    summary = json.loads((folder/"score_summary.json").read_text())
    target = build_scoring_target(build_reference(ROOT/"configs/xpt_reference_targets.json")[0])
    weights = config.metric_weights()
    n = config.get_total_steps()
    curves = []
    for row in summary["rollouts"]:
        v = entries[row["rollout_id"]]
        result = evaluate({"F2b": v}, {"F2b":target})["per_task_results"][0]
        raw = scoring.compute_per_metric_step_scores(result.per_step_epsilons, weights, config.EPSILON_MAX,
                                                      n, result.K_eff, result.topo_mask)
        factor = result.gamma * result.eta * result.mu * result.rho / n
        frame = pd.DataFrame({k: x * factor * int(not result.timeout) for k,x in raw.items()})
        for k in weights: np.testing.assert_allclose(frame[k].sum(),row["metric_scores"][k],atol=1e-9,rtol=0)
        frame["Total"] = frame.sum(axis=1)
        np.testing.assert_allclose(frame.Total.sum(), row["task_score"],atol=1e-9,rtol=0)
        curves.append((row,frame))
    for cumulative in (False, True):
        fig,axes = plt.subplots(2,4,figsize=(14,6),constrained_layout=True)
        for ax,k in zip(axes.flat,[*weights,"Total"]):
            for row,frame in curves:
                values = frame[k].cumsum() if cumulative else frame[k]
                # After early termination the remaining contributions are exactly zero.
                end = min(n, max(r["n_steps"] for r,_ in curves)+2)
                ax.plot(np.arange(1,end+1),values.iloc[:end],label=f"Episode {row['episode']}",lw=1.2,alpha=.8)
            ax.set_title(k)
            ax.grid(alpha=.22)
        axes.flat[0].legend(fontsize=8)
        fig.supxlabel("Episode step")
        mode = "cumulative" if cumulative else "per_step"
        fig.suptitle(f"{split.title()} | {curves[0][0]['shot_id']} | {mode.replace('_',' ')} contribution (points)")
        fig.savefig(out/f"{split}_{mode}.png",dpi=145)
        plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path(__file__).parent)
    p.add_argument("--output",type=Path,default=Path(__file__).parent/"picture")
    args=p.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    training(args.data,args.output)
    for split in ("validation","test"): evaluation(args.data,split,args.output)
