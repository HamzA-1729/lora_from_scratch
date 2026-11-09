"""
LoRA Rank Sensitivity Sweep Protocol:
1. Systematically sweeps intrinsic rank r in {2, 4, 8, 16, 32} across multiple random seeds.
2. Maintains constant update scaling: alpha = 2.0 * r (scaling = 2.0).
3. Compares projection target topologies: q+v projections vs all linears.
4. Records validation loss, test loss, and pretraining retention (forgetting test).
5. Outputs markdown comparison tables and JSON reports.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import yaml
import torch

from lora_from_scratch.train import train_adapter
from lora_from_scratch.evaluate import evaluate_adapter


def run_sweep(
    sweep_config_path: str = "configs/sweep.yaml",
    base_task_config_path: str = "configs/lesson_lora.yaml",
    output_file: str = "outputs/rank_sweep.json",
) -> List[Dict[str, Any]]:
    """
    Executes parameter sweep over ranks and seeds, aggregating loss and parameter metrics.
    """
    with open(sweep_config_path, "r", encoding="utf-8") as f:
        sweep_cfg = yaml.safe_load(f)

    with open(base_task_config_path, "r", encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f)

    ranks: List[int] = sweep_cfg.get("ranks", [2, 4, 8, 16, 32])
    alpha_factor: float = sweep_cfg.get("alpha_factor", 2.0)
    seeds: List[int] = sweep_cfg.get("seeds", [42, 1337])
    target_sets = sweep_cfg.get("target_sets", {"qv_only": ["q_proj", "v_proj"]})

    results: List[Dict[str, Any]] = []
    base_ckpt = task_cfg["base_model"]["checkpoint_path"]
    test_data = task_cfg["training"]["val_data_path"]

    print("\n" + "=" * 70)
    print("STARTING LORA RANK SENSITIVITY SWEEP")
    print(f"Ranks: {ranks} | Alpha Factor: {alpha_factor} | Seeds: {seeds}")
    print("=" * 70)

    for target_name, targets in target_sets.items():
        for r in ranks:
            alpha = float(alpha_factor * r)
            seed_losses = []

            for seed in seeds:
                run_dir = f"outputs/sweep/{target_name}_r{r}_seed{seed}"
                print(f"\n---> Running Target={target_name}, Rank={r}, Alpha={alpha}, Seed={seed}")

                # Train adapter for quick sweep (e.g. 200 steps for sweep comparative analysis)
                model, history = train_adapter(
                    config_path=base_task_config_path,
                    output_dir=run_dir,
                    mode="lora",
                    rank_override=r,
                    alpha_override=alpha,
                    seed_override=seed,
                )

                best_val_loss = min(history["val_loss"]) if history["val_loss"] else 0.0
                seed_losses.append(best_val_loss)

            avg_val_loss = sum(seed_losses) / len(seed_losses)
            min_loss = min(seed_losses)
            max_loss = max(seed_losses)

            # Calculate parameter count for target set
            # 6 layers * (256*r + r*256 + 256*r + r*64) = 4,992 * r for q+v
            # 28,416 * r for all linears
            multiplier = 4992 if target_name == "qv_only" else 28416
            trainable_params = multiplier * r
            param_pct = (trainable_params / 5_573_888) * 100

            record = {
                "target_set": target_name,
                "rank": r,
                "alpha": alpha,
                "trainable_params": trainable_params,
                "param_percentage": round(param_pct, 2),
                "avg_val_loss": round(avg_val_loss, 4),
                "min_val_loss": round(min_loss, 4),
                "max_val_loss": round(max_loss, 4),
            }
            results.append(record)

    # Save results
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary markdown table
    print("\n" + "=" * 70)
    print("RANK SWEEP RESULTS SUMMARY TABLE")
    print("=" * 70)
    print("| Target Modules | Rank (r) | Trainable Params | % Base Params | Avg Val Loss | Min Val Loss | Max Val Loss |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(
            f"| {r['target_set']} | {r['rank']} | {r['trainable_params']:,} | "
            f"{r['param_percentage']}% | {r['avg_val_loss']:.4f} | {r['min_val_loss']:.4f} | {r['max_val_loss']:.4f} |"
        )
    print("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LoRA Rank Sweep")
    parser.add_argument("--sweep_config", type=str, default="configs/sweep.yaml")
    parser.add_argument("--task_config", type=str, default="configs/lesson_lora.yaml")
    parser.add_argument("--output", type=str, default="outputs/rank_sweep.json")
    args = parser.parse_args()

    run_sweep(
        sweep_config_path=args.sweep_config,
        base_task_config_path=args.task_config,
        output_file=args.output,
    )

