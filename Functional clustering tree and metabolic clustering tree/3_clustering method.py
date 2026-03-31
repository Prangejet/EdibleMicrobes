from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


FILE_FUNCTION_ESC = "1_function.xlsx"
FILE_METAB_ESC = "2_metabolism.xlsx"
SHEET_METAB = "mean_output"


def clean_str(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def decode_esc(s: str) -> str:
    return s.encode("utf-8").decode("unicode_escape")


def find_file_by_escaped_name(root: Path, escaped_name: str) -> Path:
    for p in root.rglob("*.xlsx"):
        if p.name.encode("unicode_escape").decode() == escaped_name:
            return p
    raise FileNotFoundError(f"Cannot find file: {escaped_name}")


def canonical_species_name(species_text: str) -> str:
    s = clean_str(species_text).replace("_", " ").lower()
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t]
    if len(toks) >= 2:
        return f"{toks[0]}_{toks[1]}"
    return toks[0] if toks else ""


def canonical_abbreviation(abbr_text: str) -> str:
    s = clean_str(abbr_text)
    s = re.sub(r"\s+", " ", s)
    return s


def build_species_colors(species_series: pd.Series) -> Dict[str, str]:
    # Keep exactly the palette strategy used in Chapter-2 script.
    species_unique = list(pd.unique(species_series))
    k = len(species_unique)
    if k <= 20:
        cmap = plt.get_cmap("tab20", k)
        colors = [cmap(i) for i in range(k)]
    elif k <= 40:
        cmap1 = plt.get_cmap("tab20", 20)
        cmap2 = plt.get_cmap("tab20b", k - 20)
        colors = [cmap1(i) for i in range(20)] + [cmap2(i) for i in range(k - 20)]
    else:
        cmap = plt.get_cmap("gist_ncar", k)
        colors = [cmap(i) for i in range(k)]
    return dict(zip(species_unique, [to_hex(c) for c in colors]))


def preprocess(name: str, x: np.ndarray) -> np.ndarray:
    if name == "binary":
        return (x != 0).astype(float)
    if name == "minmax_col":
        mn = x.min(axis=0, keepdims=True)
        mx = x.max(axis=0, keepdims=True)
        den = mx - mn
        den[den == 0] = 1.0
        return (x - mn) / den
    if name == "zscore_col":
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        return (x - mu) / sd
    raise ValueError(name)


def dist(name: str, x: np.ndarray) -> np.ndarray:
    if name == "jensenshannon":
        x2 = x.copy()
        x2[x2 < 0] = 0.0
        s = x2.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        p = x2 / s
        return pdist(p, metric="jensenshannon")
    return pdist(x, metric=name)


def build_node_color_map(
    z: np.ndarray,
    strains: List[str],
    strain_to_species: Dict[str, str],
    species_to_color: Dict[str, str],
    mixed_color: str = "#8A8A8A",
) -> Dict[int, str]:
    n = len(strains)
    children = {n + i: (int(z[i, 0]), int(z[i, 1])) for i in range(z.shape[0])}
    leaf_species = {i: strain_to_species.get(strains[i], "") for i in range(n)}
    cache: Dict[int, Set[str]] = {}

    def collect(node_id: int) -> Set[str]:
        if node_id in cache:
            return cache[node_id]
        if node_id < n:
            sp = leaf_species[node_id]
            cache[node_id] = {sp} if sp else set()
            return cache[node_id]
        a, b = children[node_id]
        cache[node_id] = collect(a) | collect(b)
        return cache[node_id]

    out = {}
    for node_id in range(n, 2 * n - 1):
        sp_set = collect(node_id)
        out[node_id] = species_to_color[next(iter(sp_set))] if len(sp_set) == 1 else mixed_color
    return out


def tick_pos(ax: plt.Axes) -> Dict[str, float]:
    return dict(zip([t.get_text() for t in ax.get_yticklabels()], ax.get_yticks()))


def color_ticks(ax: plt.Axes, strain_to_species: Dict[str, str], species_to_color: Dict[str, str]) -> None:
    for t in ax.get_yticklabels():
        sp = strain_to_species.get(t.get_text(), "")
        t.set_color(species_to_color.get(sp, "#404040"))


def draw(
    strains: List[str],
    strain_to_abbr: Dict[str, str],
    abbr_to_color: Dict[str, str],
    z_left: np.ndarray,
    z_right: np.ndarray,
    title: str,
    left_title: str,
    right_title: str,
    out_png: Path,
    out_pdf: Path,
    out_svg: Path,
) -> None:
    mixed = "#9A9A9A"
    left_c = build_node_color_map(z_left, strains, strain_to_abbr, abbr_to_color, mixed)
    right_c = build_node_color_map(z_right, strains, strain_to_abbr, abbr_to_color, mixed)

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.55, 1.15], wspace=0.02)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[0, 2])

    dendrogram(
        z_left,
        labels=strains,
        orientation="left",
        leaf_font_size=8,
        ax=ax_l,
        above_threshold_color=mixed,
        link_color_func=lambda nid: left_c.get(nid, mixed),
    )
    dendrogram(
        z_right,
        labels=strains,
        orientation="right",
        leaf_font_size=8,
        ax=ax_r,
        above_threshold_color=mixed,
        link_color_func=lambda nid: right_c.get(nid, mixed),
    )
    color_ticks(ax_l, strain_to_abbr, abbr_to_color)
    color_ticks(ax_r, strain_to_abbr, abbr_to_color)
    ax_l.set_title(left_title)
    ax_r.set_title(right_title)
    ax_l.set_xlabel("Distance")
    ax_r.set_xlabel("Distance")

    lp = tick_pos(ax_l)
    rp = tick_pos(ax_r)
    yl = ax_l.get_ylim()
    yr = ax_r.get_ylim()
    y0 = min(yl[0], yl[1], yr[0], yr[1])
    y1 = max(yl[0], yl[1], yr[0], yr[1])

    ax_m.set_xlim(0, 1)
    ax_m.set_ylim(y0, y1)
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    for s in ax_m.spines.values():
        s.set_visible(False)
    for st in strains:
        if st in lp and st in rp:
            c = abbr_to_color.get(strain_to_abbr.get(st, ""), "#B0B0B0")
            ax_m.plot([0, 1], [lp[st], rp[st]], color=c, alpha=0.45, lw=0.9)

    abbr_order = list(pd.unique(pd.Series([strain_to_abbr[s] for s in strains if strain_to_abbr.get(s, "")])))
    handles = [Line2D([0], [0], color=abbr_to_color[s], lw=2.2) for s in abbr_order]
    fig.legend(handles, abbr_order, title="Abbreviation", loc="center left", bbox_to_anchor=(0.985, 0.5), frameon=False, fontsize=8, title_fontsize=9)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.97, 0.95])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Keep text editable in vector outputs and force Arial.
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    root = Path.cwd()
    out_dir = root / decode_esc("\\u7b2c\\u4e09\\u7ae0") / decode_esc("1_kbuffer\\u548ccdm\\u7684\\u4ee3\\u8c22\\u4ea7\\u91cf")
    out_dir.mkdir(parents=True, exist_ok=True)

    f_func = find_file_by_escaped_name(root, FILE_FUNCTION_ESC)
    f_metab = find_file_by_escaped_name(root, FILE_METAB_ESC)
    df1 = pd.read_excel(f_func, sheet_name=decode_esc(SHEET_FUNCTION_ESC))
    df2 = pd.read_excel(f_metab, sheet_name=SHEET_METAB)

    df1.iloc[:, 0] = df1.iloc[:, 0].map(clean_str)
    df2.iloc[:, 0] = df2.iloc[:, 0].map(clean_str)
    df1 = df1[df1.iloc[:, 0] != ""].drop_duplicates(subset=df1.columns[0], keep="first").set_index(df1.columns[0], drop=False)
    df2 = df2[df2.iloc[:, 0] != ""].drop_duplicates(subset=df2.columns[0], keep="first").set_index(df2.columns[0], drop=False)

    strains = [s for s in df1.index.tolist() if s in set(df2.index.tolist())]
    if len(strains) < 3:
        raise RuntimeError("Too few common strains.")

    # Use Abbreviation column (4th column) from both tables.
    # Prefer Abbreviation from metabolite table, fallback to function table.
    ab1 = df1.loc[strains].iloc[:, 3].map(clean_str).map(canonical_abbreviation)
    ab2 = df2.loc[strains].iloc[:, 3].map(clean_str).map(canonical_abbreviation)
    strain_to_abbr = {s: (ab2.loc[s] or ab1.loc[s]) for s in strains}
    abbr_series = pd.Series([strain_to_abbr[s] for s in strains if strain_to_abbr[s]])
    abbr_to_color = build_species_colors(abbr_series)

    pd.DataFrame({"abbreviation": list(abbr_to_color.keys()), "color_hex": list(abbr_to_color.values())}).to_csv(
        out_dir / "3_abbreviation_palette_mapping_current_tables.csv", index=False, encoding="utf-8-sig"
    )

    x1 = df1.loc[strains].iloc[:, 4:].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
    x2 = df2.loc[strains].iloc[:, 4:].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)

    # Balanced pair (recommended for interpretation).
    x1b = preprocess("binary", x1)
    d1b = dist("sokalsneath", x1b)
    z1b = linkage(d1b, method="weighted", optimal_ordering=True)

    x2b = preprocess("minmax_col", x2)
    d2b = dist("jensenshannon", x2b)
    z2b = linkage(d2b, method="weighted", optimal_ordering=True)

    draw(
        strains=strains,
        strain_to_abbr=strain_to_abbr,
        abbr_to_color=abbr_to_color,
        z_left=z1b,
        z_right=z2b,
        title="Paired Dendrograms (Palette style from Chapter 2, Abbreviation mapped from current tables)",
        left_title="Tree1: sokalsneath + weighted",
        right_title="Tree2: jensenshannon + weighted",
        out_png=out_dir / "3_paired_dendrogram_balanced_palette_by_current_species.png",
        out_pdf=out_dir / "3_paired_dendrogram_balanced_palette_by_current_species.pdf",
        out_svg=out_dir / "3_paired_dendrogram_balanced_palette_by_current_species.svg",
    )

    # Best-by-between pair.
    x1c = preprocess("binary", x1)
    d1c = dist("yule", x1c)
    z1c = linkage(d1c, method="single", optimal_ordering=True)

    x2c = preprocess("zscore_col", x2)
    d2c = dist("sqeuclidean", x2c)
    z2c = linkage(d2c, method="complete", optimal_ordering=True)

    draw(
        strains=strains,
        strain_to_abbr=strain_to_abbr,
        abbr_to_color=abbr_to_color,
        z_left=z1c,
        z_right=z2c,
        title="Paired Dendrograms Best Correlation (Palette style from Chapter 2, Abbreviation mapped from current tables)",
        left_title="Tree1: yule + single",
        right_title="Tree2: sqeuclidean + complete",
        out_png=out_dir / "3_paired_dendrogram_best_palette_by_current_species.png",
        out_pdf=out_dir / "3_paired_dendrogram_best_palette_by_current_species.pdf",
        out_svg=out_dir / "3_paired_dendrogram_best_palette_by_current_species.svg",
    )

    print("Finished redraw with palette-style mapping by Abbreviation.")
    print(f"Common strains: {len(strains)}")
    print(out_dir / "3_abbreviation_palette_mapping_current_tables.csv")
    print(out_dir / "3_paired_dendrogram_balanced_palette_by_current_species.png")
    print(out_dir / "3_paired_dendrogram_best_palette_by_current_species.png")


if __name__ == "__main__":
    main()
