"""Screen M3FD temporal boundaries and equal-dHash groups without changing data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--rows-per-page", type=int, default=20)
    parser.add_argument("--overview-step", type=int, default=10)
    parser.add_argument("--overview-columns", type=int, default=5)
    parser.add_argument("--overview-rows", type=int, default=6)
    return parser.parse_args()


def thumbnail(path: Path, size=(48, 36)):
    with Image.open(path) as image:
        return np.asarray(ImageOps.fit(image.convert("L"), size, Image.Resampling.BILINEAR), dtype=np.float32)


def dhash(path: Path):
    with Image.open(path) as image:
        values = np.asarray(ImageOps.fit(image.convert("L"), (9, 8), Image.Resampling.BILINEAR))
    bits = values[:, 1:] > values[:, :-1]
    return int("".join("1" if bit else "0" for bit in bits.flat), 2)


def panel(path: Path, size=(160, 120)):
    with Image.open(path) as image:
        return ImageOps.fit(image.convert("RGB"), size, Image.Resampling.BILINEAR)


def write_contact_pages(rows, ir, vis, folder, rows_per_page):
    folder.mkdir(parents=True, exist_ok=True)
    for page_start in range(0, len(rows), rows_per_page):
        selected = rows[page_start:page_start + rows_per_page]
        canvas = Image.new("RGB", (640, len(selected) * 145), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, row in enumerate(selected):
            y = row_index * 145
            before, after = row["before"], row["after"]
            for column, path in enumerate((ir[before], ir[after], vis[before], vis[after])):
                canvas.paste(panel(path), (column * 160, y + 20))
            draw.text((4, y + 3), f"{before}->{after} score={row['score']:.4f}", fill="black")
        page = page_start // rows_per_page + 1
        canvas.save(folder / f"boundary_candidates_{page:02d}.jpg", quality=92)


def write_overview_pages(stems, ir, vis, folder, step, columns, rows):
    """Render a coarse whole-dataset overview for complete-sequence review."""
    selected = stems[::step]
    per_page = columns * rows
    cell_width, cell_height = 240, 115
    for page_start in range(0, len(selected), per_page):
        page_stems = selected[page_start:page_start + per_page]
        canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        draw = ImageDraw.Draw(canvas)
        for index, stem in enumerate(page_stems):
            row, column = divmod(index, columns)
            x, y = column * cell_width, row * cell_height
            canvas.paste(panel(ir[stem], (120, 90)), (x, y + 20))
            canvas.paste(panel(vis[stem], (120, 90)), (x + 120, y + 20))
            draw.text((x + 4, y + 3), stem, fill="black")
        page = page_start // per_page + 1
        canvas.save(folder / f"sequence_overview_{page:02d}.jpg", quality=90)


def index(folder: Path):
    return {path.stem: path for path in folder.glob("*.png")}


def main():
    args = parse_args()
    ir, vis = index(args.root / "Ir"), index(args.root / "Vis")
    stems = sorted(set(ir) & set(vis))
    if not stems:
        raise RuntimeError("No paired PNG images found")

    features = {}
    hash_groups = defaultdict(list)
    for number, stem in enumerate(stems, 1):
        features[stem] = (thumbnail(ir[stem]), thumbnail(vis[stem]))
        hash_groups[(dhash(ir[stem]), dhash(vis[stem]))].append(stem)
        if number % 500 == 0:
            print(f"decoded {number}/{len(stems)}", flush=True)

    transitions = []
    for before, after in zip(stems, stems[1:]):
        ir_a, vis_a = features[before]
        ir_b, vis_b = features[after]
        ir_mae = float(np.mean(np.abs(ir_a - ir_b)) / 255.0)
        vis_mae = float(np.mean(np.abs(vis_a - vis_b)) / 255.0)
        transitions.append(
            {"before": before, "after": after, "ir_mae": ir_mae, "vis_mae": vis_mae,
             "score": (ir_mae + vis_mae) / 2.0}
        )
    chronological_transitions = transitions
    ranked_transitions = sorted(transitions, key=lambda row: row["score"], reverse=True)
    top = ranked_transitions[: args.top]
    scores = np.asarray([row["score"] for row in chronological_transitions])
    duplicates = [group for group in hash_groups.values() if len(group) > 1]
    duplicates.sort(key=lambda group: (-len(group), group[0]))
    result = {
        "root": str(args.root.resolve()),
        "paired_images": len(stems),
        "method": "mean IR/VIS grayscale thumbnail MAE; candidates only, not automatic scene labels",
        "score_quantiles": {
            str(q): float(np.quantile(scores, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.98, 0.99)
        },
        "top_transition_candidates": top,
        "chronological_transitions": chronological_transitions,
        "equal_combined_dhash_groups": duplicates,
        "equal_combined_dhash_group_count": len(duplicates),
        "limitations": [
            "Large camera motion can look like a scene boundary and gradual scene changes can be missed.",
            "Equal dHash is a screening constraint, not proof that two frames are identical.",
            "Contact sheets or original metadata must verify scene membership before manifest freezing.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_contact_pages(top, ir, vis, args.contact_dir, args.rows_per_page)
    write_overview_pages(
        stems, ir, vis, args.contact_dir, args.overview_step, args.overview_columns, args.overview_rows
    )
    print(json.dumps({key: result[key] for key in ("paired_images", "equal_combined_dhash_group_count")}, indent=2))


if __name__ == "__main__":
    main()
