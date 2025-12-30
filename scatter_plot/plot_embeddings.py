"""plot_embeddings.py

Visualize driver style embeddings produced by module_c_driver_style.py.

Usage
-----
python plot_embeddings.py driver_style_embedding.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt


def load_embedding_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_driver_style(embedding_json: Dict[str, Any], save_path: str | None = None) -> None:
    drivers = embedding_json["drivers"]
    laps = embedding_json["laps"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- visual encoding ---
    # color  : driver (provided by JSON)
    # size   : race phase (opening < middle < ending)
    phase_size = {
        "opening": 40,
        "middle": 80,
        "ending": 140,
    }

    # group laps by (driver, phase)
    grouped: Dict[tuple, list] = {}
    for lap in laps:
        key = (lap["driver"], lap["phase"])
        grouped.setdefault(key, []).append(lap)

    driver_label = {d["id"]: d.get("label", f"Driver {d['id']}") for d in drivers}
    driver_color = {d["id"]: d.get("color", None) for d in drivers}

    for (driver_id, phase), lap_list in grouped.items():
        xs = [lap["embed_x"] for lap in lap_list]
        ys = [lap["embed_y"] for lap in lap_list]

        label = f"{driver_label.get(driver_id, driver_id)} - {phase}"
        size = phase_size.get(phase, 60)
        color = driver_color.get(driver_id)

        ax.scatter(
            xs,
            ys,
            label=label,
            alpha=0.8,
            s=size,
            marker="o",
            edgecolors="none",
            c=color,
        )

    # 如果之後要在互動版上強調代表性圈，可以用 laps 裡的 is_focus 自行處理。
    # 這個靜態版本先不加 Lap 編號文字，避免干擾判讀。

    ax.set_xlabel("PC1 – Aggressiveness (← conservative | aggressive →)")
    ax.set_ylabel("PC2 – Race context (↑ clean air | traffic & battles ↓)")
    ax.set_title(f"Driver Style Map (session {embedding_json.get('session_key')})")

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    if save_path:
        out_path = Path(save_path)
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_embeddings.py driver_style_embedding.json")
        sys.exit(1)

    json_path = sys.argv[1]
    data = load_embedding_json(json_path)

    out_png = Path(json_path).with_suffix(".png")
    plot_driver_style(data, save_path=str(out_png))