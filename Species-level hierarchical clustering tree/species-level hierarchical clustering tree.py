from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist


SPECIAL_LABEL_FILES = {
    "Limosilactobacillus_reuteri.xlsx",
    "Bifidobacterium_longum.xlsx",
}


def build_cluster_colors(k: int):
    if k <= 10:
        cmap = plt.get_cmap("tab10", k)
        colors = [cmap(i) for i in range(k)]
    elif k <= 20:
        cmap = plt.get_cmap("tab20", k)
        colors = [cmap(i) for i in range(k)]
    else:
        cmap = plt.get_cmap("gist_ncar", k)
        colors = [cmap(i) for i in range(k)]
    return [to_hex(c) for c in colors]


def load_subsp_mapping(results_dir: Path) -> dict[str, str]:
    ref_candidates = [p for p in (results_dir / "..").glob("2_*/*.xlsx") if not p.name.startswith("~$")]
    if not ref_candidates:
        return {}
    ref_file = ref_candidates[0]
    ref = pd.read_excel(ref_file, sheet_name=3)

    mapping = {}
    for _, row in ref.iterrows():
        true_name = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
        subsp = str(row.iloc[2]) if pd.notna(row.iloc[2]) else ""
        if true_name:
            mapping[true_name] = subsp
        first_name = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
        if first_name and first_name not in mapping:
            mapping[first_name] = subsp
    return mapping


def clean_strain_label(full_name: str, species_prefix: str) -> str:
    label = str(full_name)
    prefix = species_prefix + "_"
    if label.startswith(prefix):
        label = label[len(prefix):]
    if label.endswith("_ctg"):
        label = label[:-4]
    return label.strip("_ ")


def abbreviate_subsp_label(subsp: str, file_name: str) -> str:
    text = str(subsp).replace("_", " ").strip()
    lower = text.lower()
    subtype = ""
    if "subsp." in lower:
        subtype = text.split("subsp.", 1)[1].strip()
    elif "subsp" in lower:
        subtype = text.split("subsp", 1)[1].strip()
    else:
        parts = text.split()
        subtype = parts[-1] if parts else text

    if file_name == "Bifidobacterium_longum.xlsx":
        return f"Bl. {subtype}"
    if file_name == "Limosilactobacillus_reuteri.xlsx":
        return f"Lr. {subtype}"
    return text


def build_display_labels(sample_names: pd.Series, file_name: str, file_stem: str, subsp_map: dict[str, str]) -> list[str]:
    labels = []
    for name in sample_names.astype(str):
        strain = clean_strain_label(name, file_stem)
        if file_name in SPECIAL_LABEL_FILES:
            subsp = subsp_map.get(name, "")
            if subsp and subsp.lower() != "nan":
                short_subsp = abbreviate_subsp_label(subsp, file_name)
                labels.append(f"{short_subsp} {strain}".strip())
            else:
                labels.append(strain)
        else:
            labels.append(strain)
    return labels


def draw_cluster_tree(file_path: Path, out_dir: Path, subsp_map: dict[str, str], threshold: float = 0.1) -> None:
    df = pd.read_excel(file_path, sheet_name=0)
    sample_names = df.iloc[:, 0].astype(str)
    X = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    X_bin = (X > 0).astype(float)

    n = X_bin.shape[0]
    if n == 1:
        return

    condensed = pdist(X_bin, metric="hamming")
    Z = linkage(condensed, method="average")
    cluster_labels = fcluster(Z, threshold, criterion="distance")
    uniq = sorted(np.unique(cluster_labels))
    palette = build_cluster_colors(max(len(uniq), 1))
    cluster_color_map = {cid: palette[i] for i, cid in enumerate(uniq)}
    default_color = "#6f6f6f"

    # subtree -> cluster id if pure cluster else 0
    subtree_clusters = {}
    for leaf_id in range(n):
        subtree_clusters[leaf_id] = int(cluster_labels[leaf_id])
    for i in range(Z.shape[0]):
        node_id = n + i
        a = int(Z[i, 0])
        b = int(Z[i, 1])
        ca, cb = subtree_clusters[a], subtree_clusters[b]
        subtree_clusters[node_id] = ca if ca == cb else 0

    def link_color_func(node_id: int) -> str:
        cid = subtree_clusters[node_id]
        return cluster_color_map[cid] if cid in cluster_color_map else default_color

    labels = build_display_labels(sample_names, file_path.name, file_path.stem, subsp_map)

    fig_w = 16
    fig_h = max(6, n * 0.18)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    dendrogram(
        Z,
        orientation="right",
        labels=labels,
        leaf_font_size=7,
        color_threshold=0,
        above_threshold_color=default_color,
        link_color_func=link_color_func,
        ax=ax,
    )

    ax.set_xlabel("Hamming distance (binary matrix)")
    ax.set_ylabel("")
    title_name = file_path.stem.replace("_", " ")
    ax.set_title(f"{title_name}: hierarchical clustering")
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    out_png = out_dir / f"{file_path.stem}_hamming_binary_clusters.png"
    out_svg = out_dir / f"{file_path.stem}_hamming_binary_clusters.svg"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base = Path("1_diamond_k0_split_by_species")
    if not base.exists():
        raise RuntimeError(f"Missing folder: {base}")

    out_dir = Path("1_diamond_k0_split_by_species_hamming_binary_trees")
    out_dir.mkdir(exist_ok=True)

    subsp_map = load_subsp_mapping(Path("."))
    files = sorted([p for p in base.glob("*.xlsx") if not p.name.startswith("~$")])
    if not files:
        raise RuntimeError("No per-species xlsx files found.")

    processed = []
    for fp in files:
        draw_cluster_tree(fp, out_dir, subsp_map, threshold=0.1)
        processed.append(fp.name)
        print("done", fp.name)

    print(f"Processed files: {len(processed)}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
