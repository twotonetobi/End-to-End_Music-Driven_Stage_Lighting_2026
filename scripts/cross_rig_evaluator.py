"""
Cross-Rig Evaluator for Table 2: Intention Preservation Under Cross-Rig Application
"""

import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from offline_processor import OfflineProcessor
from rig_renderer import (
    RigRenderer,
    RIG_PROFILES,
    compute_mean_intensity_correlation,
    compute_peak_intensity_correlation,
    compute_dynamic_range_correlation,
    compute_color_correlation,
)


class CrossRigEvaluator:
    def __init__(self, configs_dir: Path):
        self.configs_dir = configs_dir
        self.processor = OfflineProcessor()

        with open(configs_dir / "paths.yaml") as f:
            self.paths = yaml.safe_load(f)

    def evaluate_segment(
        self,
        geo_path: str,
        pas_path: str,
        bpm: float,
        rig_name: str,
    ) -> Dict[str, float]:
        results = self.processor.process_segment(geo_path, pas_path, bpm)
        renderer = RigRenderer(rig_name)

        all_mean = []
        all_peak = []
        all_range = []
        all_hue = []
        all_sat = []

        for lx_num in ["lx1", "lx2", "lx3"]:
            original_rgb = results[lx_num]["rgb"]
            rendered_rgb = renderer.render(original_rgb)

            mean_corr = compute_mean_intensity_correlation(original_rgb, rendered_rgb)
            peak_corr = compute_peak_intensity_correlation(original_rgb, rendered_rgb)
            range_corr = compute_dynamic_range_correlation(original_rgb, rendered_rgb)
            hue_corr, sat_corr = compute_color_correlation(original_rgb, rendered_rgb)

            all_mean.append(mean_corr)
            all_peak.append(peak_corr)
            all_range.append(range_corr)
            all_hue.append(hue_corr)
            all_sat.append(sat_corr)

        return {
            "mean_intensity_corr": float(np.mean(all_mean)),
            "peak_intensity_corr": float(np.mean(all_peak)),
            "dynamic_range_corr": float(np.mean(all_range)),
            "hue_corr": float(np.mean(all_hue)),
            "sat_corr": float(np.mean(all_sat)),
            "unique_positions": renderer.get_effective_resolution(),
        }

    def evaluate_all_segments(
        self, rig_names: Optional[List[str]] = None, limit: Optional[int] = None
    ) -> pd.DataFrame:
        if rig_names is None:
            rig_names = ["direct_club", "club", "concert", "led_bars", "reference"]

        geo_dir = Path(self.paths["inference_data"]["oscillator"])
        pas_dir = Path(self.paths["inference_data"]["diffusion"])
        timings_dir = Path(self.paths["song_timings"])

        results = []
        geo_files = sorted(geo_dir.glob("*.pkl"))

        if limit:
            geo_files = geo_files[:limit]

        for geo_file in geo_files:
            pas_file = pas_dir / geo_file.name

            if not pas_file.exists():
                continue

            parts = geo_file.stem.split("_part_")
            if len(parts) != 2:
                continue

            song_name = parts[0]
            segment_info = parts[1]

            timing_file = timings_dir / f"{song_name}.json"

            if timing_file.exists():
                with open(timing_file) as f:
                    metadata = json.load(f)
                bpm = metadata["bpm"]
            else:
                bpm = 120.0

            for rig_name in rig_names:
                try:
                    metrics = self.evaluate_segment(
                        str(geo_file), str(pas_file), bpm, rig_name
                    )

                    results.append(
                        {
                            "song_name": song_name,
                            "segment": segment_info,
                            "rig": rig_name,
                            **metrics,
                        }
                    )

                except Exception as e:
                    print(f"Error processing {geo_file.stem} with {rig_name}: {e}")

        return pd.DataFrame(results)

    def compute_summary_statistics(self, df: pd.DataFrame) -> Dict:
        summary = {}

        for rig in df["rig"].unique():
            rig_data = df[df["rig"] == rig]
            profile = RIG_PROFILES.get(rig)

            summary[rig] = {
                "name": profile.name if profile else rig,
                "unique_positions": profile.unique_positions_per_group
                if profile
                else "?",
                "mirroring": profile.mirroring if profile else False,
                "mean_intensity_corr": {
                    "mean": float(rig_data["mean_intensity_corr"].mean()),
                    "std": float(rig_data["mean_intensity_corr"].std()),
                },
                "peak_intensity_corr": {
                    "mean": float(rig_data["peak_intensity_corr"].mean()),
                    "std": float(rig_data["peak_intensity_corr"].std()),
                },
                "dynamic_range_corr": {
                    "mean": float(rig_data["dynamic_range_corr"].mean()),
                    "std": float(rig_data["dynamic_range_corr"].std()),
                },
                "hue_corr": {
                    "mean": float(rig_data["hue_corr"].mean()),
                    "std": float(rig_data["hue_corr"].std()),
                },
                "sat_corr": {
                    "mean": float(rig_data["sat_corr"].mean()),
                    "std": float(rig_data["sat_corr"].std()),
                },
            }

        return summary

    def print_summary_table(self, summary: Dict):
        print("\n" + "=" * 90)
        print("TABLE 2: Intention Preservation Under Cross-Rig Application")
        print("=" * 90)

        rig_order = ["direct_club", "club", "concert", "led_bars", "reference"]

        header = f"{'Rig':<22} {'Unique':<8} {'Mean I':<10} {'Peak I':<10} {'Range':<10} {'Hue':<10} {'Sat':<10}"
        print(header)
        print("-" * 90)

        for rig in rig_order:
            if rig not in summary:
                continue

            data = summary[rig]
            unique = data["unique_positions"]
            mirror_str = " (M)" if data["mirroring"] else ""

            mean_i = f"{data['mean_intensity_corr']['mean']:.3f}"
            peak_i = f"{data['peak_intensity_corr']['mean']:.3f}"
            range_c = f"{data['dynamic_range_corr']['mean']:.3f}"
            hue_c = f"{data['hue_corr']['mean']:.3f}"
            sat_c = f"{data['sat_corr']['mean']:.3f}"

            print(
                f"{data['name']:<22} {unique}{mirror_str:<8} {mean_i:<10} {peak_i:<10} {range_c:<10} {hue_c:<10} {sat_c:<10}"
            )

        print("-" * 90)
        print(
            "(M) = Center-mirrored rig. All values are Pearson correlations (higher = better)."
        )
        print()

    def export_latex_table(self, summary: Dict, output_path: Path) -> str:
        latex = r"""\begin{table}[htbp]
\centering
\caption{Intention Preservation Under Cross-Rig Application}
\label{tab:cross-rig}
\begin{tabular}{lcccccc}
\toprule
Rig Configuration & Unique & $\rho_{\bar{I}}$ & $\rho_{I_{peak}}$ & $\rho_{\Delta I}$ & $\rho_{H}$ & $\rho_{S}$ \\
\midrule
"""

        rig_order = ["direct_club", "club", "concert", "led_bars", "reference"]

        for rig in rig_order:
            if rig not in summary:
                continue

            data = summary[rig]
            name = data["name"]
            unique = data["unique_positions"]
            mirror = "$^*$" if data["mirroring"] else ""
            baseline = "$^\\dagger$" if rig.startswith("direct_") else ""

            mean_i = f"{data['mean_intensity_corr']['mean']:.2f}"
            peak_i = f"{data['peak_intensity_corr']['mean']:.2f}"
            range_c = f"{data['dynamic_range_corr']['mean']:.2f}"
            hue_c = f"{data['hue_corr']['mean']:.2f}"
            sat_c = f"{data['sat_corr']['mean']:.2f}"

            latex += f"{name}{baseline} & {unique}{mirror} & {mean_i} & {peak_i} & {range_c} & {hue_c} & {sat_c} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\vspace{0.5em}
\begin{flushleft}
\footnotesize{$^\dagger$Naive baseline (no intention preservation). $^*$Center-mirrored. All values are Pearson correlations.}
\end{flushleft}
\end{table}"""

        with open(output_path, "w") as f:
            f.write(latex)

        return latex


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs")
    parser.add_argument("--output", type=str, default="../outputs")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    configs_dir = Path(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    evaluator = CrossRigEvaluator(configs_dir)

    print("Starting Cross-Rig Evaluation...")
    df = evaluator.evaluate_all_segments(limit=args.limit)

    csv_path = output_dir / "table2_cross_rig.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    summary = evaluator.compute_summary_statistics(df)
    evaluator.print_summary_table(summary)

    latex_path = output_dir / "table2_latex.tex"
    evaluator.export_latex_table(summary, latex_path)
    print(f"LaTeX table saved to {latex_path}")


if __name__ == "__main__":
    main()
