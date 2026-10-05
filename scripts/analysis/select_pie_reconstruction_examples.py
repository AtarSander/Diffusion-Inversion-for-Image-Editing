#!/usr/bin/env python3
"""Save linked PIE-Bench panels ranked by a reconstruction-only composite."""
from __future__ import annotations
import argparse, csv, json, math, os, shutil
from pathlib import Path

METRICS = {
    "structure_distance": False,
    "lpips_unedit_part": False,
    "mse_unedit_part": False,
    "psnr_unedit_part": True,
    "ssim_unedit_part": True,
}
METHODS = ("pix2pix-zero", "masactrl", "pnp")

def number(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError): return None

def rank_fraction(rows, key, higher):
    valid=[row for row in rows if row["values"].get(key) is not None]
    valid.sort(key=lambda row: row["values"][key], reverse=higher)
    n=len(valid)
    for index,row in enumerate(valid): row["ranks"][key]=index / max(n-1, 1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--count", type=int, default=25)
    ap.add_argument("--require-all-metrics", action="store_true")
    ap.add_argument("--score-mode", choices=("percentile", "mean_degradation"), default="percentile")
    args=ap.parse_args()
    if args.count < 1: raise ValueError("--count must be positive")
    mapping=json.loads((args.data / "mapping_file.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    all_summary={"methodology": "score_mode=mean_degradation uses the equally weighted mean of signed per-metric z-score deviations: high structure distance/LPIPS/MSE and low PSNR/SSIM are bad. It therefore identifies samples that pull the separately reported metric means downward. CLIP is excluded.", "score_mode": args.score_mode, "require_all_metrics": args.require_all_metrics}
    for editor in METHODS:
        method=f"lora+directinversion+{editor}"
        metric_path=args.root / "metrics" / f"{method}.csv"
        panel_root=args.root / method / "annotation_images"
        with metric_path.open(newline="", encoding="utf-8") as handle:
            raw=list(csv.DictReader(handle))
        rows=[]
        for record in raw:
            file_id=record["file_id"]
            item=mapping.get(file_id)
            if item is None: continue
            values={metric: number(record.get(f"{method}|{metric}")) for metric in METRICS}
            rows.append({"file_id":file_id,"image_path":item["image_path"],"values":values,"ranks":{}})
        for metric,higher in METRICS.items(): rank_fraction(rows, metric, higher)
        for row in rows:
            ranks=list(row["ranks"].values())
            row["composite_reconstruction_rank"]=sum(ranks)/len(ranks) if ranks else None
            row["metrics_used"]=len(ranks)
        if args.score_mode == "mean_degradation":
            # Equal-weight standardized signed deviations identify samples that
            # pull the separately reported metric means in the bad direction.
            for metric, higher in METRICS.items():
                values=[row["values"][metric] for row in rows if row["values"].get(metric) is not None]
                mean=sum(values)/len(values)
                variance=sum((value-mean)**2 for value in values)/len(values)
                std=variance**0.5
                for row in rows:
                    value=row["values"].get(metric)
                    if value is not None:
                        row.setdefault("degradations", {})[metric]=((mean-value) if higher else (value-mean)) / std if std else 0.0
            for row in rows:
                contributions=list(row.get("degradations", {}).values())
                row["composite_reconstruction_rank"]=sum(contributions)/len(contributions) if contributions else None
        eligible=[row for row in rows if row["composite_reconstruction_rank"] is not None and (not args.require_all_metrics or row["metrics_used"] == len(METRICS))]
        eligible.sort(key=lambda row: row["composite_reconstruction_rank"])
        selections={"best":eligible[:args.count], "worst":list(reversed(eligible[-args.count:]))}
        method_out=args.output / editor
        for label, selected in selections.items():
            target=method_out / label
            target.mkdir(parents=True, exist_ok=True)
            manifest=[]
            for rank,row in enumerate(selected, start=1):
                source=panel_root / row["image_path"]
                suffix=source.suffix or ".png"
                destination=target / f"{rank:02d}_{row['file_id']}{suffix}"
                if not source.is_file(): raise FileNotFoundError(source)
                if destination.exists() or destination.is_symlink(): destination.unlink()
                try: os.link(source, destination)
                except OSError: shutil.copy2(source, destination)
                manifest.append({"rank":rank, "panel":destination.name, **row})
            (method_out / f"{label}.json").write_text(json.dumps(manifest, indent=2)+"\n")
            with (method_out / f"{label}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer=csv.DictWriter(handle, fieldnames=["rank","panel","file_id","image_path","composite_reconstruction_rank","metrics_used",*METRICS])
                writer.writeheader()
                for entry in manifest:
                    writer.writerow({"rank":entry["rank"],"panel":entry["panel"],"file_id":entry["file_id"],"image_path":entry["image_path"],"composite_reconstruction_rank":entry["composite_reconstruction_rank"],"metrics_used":entry["metrics_used"],**entry["values"]})
        all_summary[editor]={"method":method,"images_ranked":len(eligible),"best_ids":[r["file_id"] for r in selections["best"]],"worst_ids":[r["file_id"] for r in selections["worst"]]}
    (args.output / "README.json").write_text(json.dumps(all_summary, indent=2)+"\n")

if __name__ == "__main__": main()
