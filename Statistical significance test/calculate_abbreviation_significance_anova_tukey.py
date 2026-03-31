from __future__ import annotations

from pathlib import Path

import math
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


VALUE_COLUMNS = [
    "ILA_CDM",
    "ILA_KBUFFER",
    "IALD_CDM",
    "IALD_KBUFFER",
    "IE_CDM",
    "IE_KBUFFER",
    "IAA_CDM",
    "IAA_KBUFFER",
]

# Direct input path for the current workbook.
INPUT_PATH = Path(
    "Source_Data.xlsx"
)
OUTPUT_PATH = INPUT_PATH.with_name("output.xlsx")


def excel_letter(index: int) -> str:
    result = ""
    current = index
    while True:
        current, rem = divmod(current, 26)
        result = chr(97 + rem) + result
        if current == 0:
            return result
        current -= 1


def compact_letter_display(
    groups: list[str], nonsignificant: dict[str, dict[str, bool]]
) -> dict[str, str]:
    letter_groups: list[list[str]] = []
    assigned: dict[str, list[int]] = {group: [] for group in groups}

    for group in groups:
        for idx, members in enumerate(letter_groups):
            if all(nonsignificant[group][member] for member in members):
                members.append(group)
                assigned[group].append(idx)
        if not assigned[group]:
            letter_groups.append([group])
            assigned[group].append(len(letter_groups) - 1)

    keep = [True] * len(letter_groups)
    member_sets = [set(members) for members in letter_groups]
    for i, set_i in enumerate(member_sets):
        for j, set_j in enumerate(member_sets):
            if i != j and set_i and set_i.issubset(set_j) and len(set_i) < len(set_j):
                keep[i] = False

    remap: dict[int, int] = {}
    next_index = 0
    for old_index, keep_flag in enumerate(keep):
        if keep_flag:
            remap[old_index] = next_index
            next_index += 1

    final = {}
    for group in groups:
        letters = [excel_letter(remap[idx]) for idx in assigned[group] if idx in remap]
        final[group] = "".join(letters) if letters else "a"
    return final


def compute_letters(
    df: pd.DataFrame, group_col: str, value_col: str, group_order: list[str]
) -> tuple[dict[str, str], float, list[str]]:
    subset = df[[group_col, value_col]].dropna().copy()
    grouped_values = {}
    for group in group_order:
        values = subset.loc[subset[group_col] == group, value_col].to_numpy(dtype=float)
        if len(values) > 0:
            grouped_values[group] = values

    groups = [group for group in group_order if group in grouped_values]
    groups = sorted(
        groups,
        key=lambda group: (
            -float(np.median(grouped_values[group])),
            -float(np.mean(grouped_values[group])),
            group,
        ),
    )
    if len(groups) <= 1:
        return {group: "a" for group in groups}, math.nan, groups

    overall_p = f_oneway(*(grouped_values[group] for group in groups)).pvalue
    if not np.isfinite(overall_p) or overall_p >= 0.05:
        return {group: "a" for group in groups}, overall_p, groups

    values = np.concatenate([grouped_values[group] for group in groups])
    labels = np.concatenate(
        [np.array([group] * len(grouped_values[group]), dtype=object) for group in groups]
    )
    tukey = pairwise_tukeyhsd(endog=values, groups=labels, alpha=0.05)

    nonsignificant = {group: {other: True for other in groups} for group in groups}
    for row in tukey.summary().data[1:]:
        group_a, group_b, _, _, _, _, reject = row
        is_nonsignificant = not bool(reject)
        nonsignificant[group_a][group_b] = is_nonsignificant
        nonsignificant[group_b][group_a] = is_nonsignificant

    return compact_letter_display(groups, nonsignificant), overall_p, groups


def resolve_writable_output_path(output_path: Path) -> Path:
    if not output_path.exists():
        return output_path
    try:
        with output_path.open("ab"):
            return output_path
    except PermissionError:
        pass

    for index in range(2, 100):
        candidate = output_path.with_name(f"{output_path.stem}_v{index}{output_path.suffix}")
        if not candidate.exists():
            return candidate
        try:
            with candidate.open("ab"):
                return candidate
        except PermissionError:
            continue
    raise PermissionError(f"No writable output path available near: {output_path}")


def build_output(input_path: Path, output_path: Path) -> None:
    df = pd.read_excel(input_path)
    group_col = "Abbreviation"
    group_order = list(pd.unique(df[group_col].dropna()))

    summary = pd.DataFrame({group_col: group_order})
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for value_col in VALUE_COLUMNS:
            letters, overall_p, sorted_groups = compute_letters(df, group_col, value_col, group_order)
            rank_map = {group: rank for rank, group in enumerate(sorted_groups, start=1)}
            rows = []
            for group in group_order:
                values = df.loc[df[group_col] == group, value_col].dropna()
                if len(values) == 0:
                    continue
                rows.append(
                    {
                        "rank_desc_median": rank_map.get(group),
                        "Abbreviation": group,
                        "n": int(len(values)),
                        "mean": round(float(values.mean()), 2),
                        "median": round(float(values.median()), 2),
                        "letter": letters.get(group, ""),
                        "overall_p": overall_p,
                        "posthoc_method": "Tukey HSD",
                    }
                )

            detail_df = pd.DataFrame(rows)
            detail_df.to_excel(writer, sheet_name=value_col[:31], index=False)
            summary[value_col] = summary[group_col].map(
                {row["Abbreviation"]: row["letter"] for row in rows}
            )

        summary.to_excel(writer, sheet_name="summary_letters", index=False)


def main() -> None:
    output_path = resolve_writable_output_path(OUTPUT_PATH)
    build_output(input_path=INPUT_PATH, output_path=output_path)
    print(f"INPUT={INPUT_PATH}")
    print(f"OUTPUT={output_path}")


if __name__ == "__main__":
    main()
