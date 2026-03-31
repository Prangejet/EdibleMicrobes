from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


def locate_train_file() -> Path:
    matches = list(Path(".").glob("1_annotation.xlsx"))
    if matches:
        return matches[0]
    files = [f for f in Path(".").glob("*.xlsx") if not f.name.startswith("~$")]
    if not files:
        raise FileNotFoundError("No training xlsx file found.")
    return files[0]


def locate_pred_file() -> Path:
    matches = list(Path(".").glob("2_allstrains.xlsx"))
    if matches:
        return matches[0]
    matches = list(Path(".").glob("4_*.xlsx"))
    if matches:
        return matches[0]
    raise FileNotFoundError("No prediction xlsx file found.")


def main() -> None:
    train_file = locate_train_file()
    pred_file = locate_pred_file()
    print(f"[INFO] Train file: {train_file}")
    print(f"[INFO] Predict file: {pred_file}")

    train_df = pd.read_excel(train_file)
    pred_df = pd.read_excel(pred_file)

    if "ILA_CDM" not in train_df.columns or "IAA_KBUFFER" not in train_df.columns:
        raise ValueError("Cannot find metabolite boundary columns in training file.")

    metabolite_cols = list(train_df.loc[:, "ILA_CDM":"IAA_KBUFFER"].columns)
    train_feature_cols = list(train_df.columns[train_df.columns.get_loc("IAA_KBUFFER") + 1 :])

    X_train = train_df[train_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_train = (X_train > 0).astype(np.int8)
    Y_train = train_df[metabolite_cols].apply(pd.to_numeric, errors="coerce")

    # Multi-output model: predict all metabolites in one model.
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train.to_numpy(dtype=np.float32), Y_train.to_numpy(dtype=np.float64))

    common_features = [c for c in train_feature_cols if c in pred_df.columns]
    missing_in_pred = [c for c in train_feature_cols if c not in pred_df.columns]
    extra_in_pred = [c for c in pred_df.columns if c not in train_feature_cols]

    # Align prediction matrix strictly to training feature order.
    X_pred_aligned = pd.DataFrame(0, index=pred_df.index, columns=train_feature_cols, dtype=np.int8)
    if common_features:
        tmp = pred_df[common_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        X_pred_aligned.loc[:, common_features] = (tmp > 0).astype(np.int8)

    y_pred = model.predict(X_pred_aligned.to_numpy(dtype=np.float32))

    id_cols = [c for c in ["name", "genus", "sp", "species"] if c in pred_df.columns]
    out_df = pred_df[id_cols].copy() if id_cols else pd.DataFrame(index=pred_df.index)
    for i, met in enumerate(metabolite_cols):
        out_df[f"pred_{met}"] = y_pred[:, i]

    align_info = pd.DataFrame(
        {
            "metric": [
                "n_train_samples",
                "n_predict_samples",
                "n_targets",
                "n_train_features",
                "n_common_features",
                "n_missing_features_in_predict",
                "n_extra_columns_in_predict_dropped",
            ],
            "value": [
                len(train_df),
                len(pred_df),
                len(metabolite_cols),
                len(train_feature_cols),
                len(common_features),
                len(missing_in_pred),
                len(extra_in_pred),
            ],
        }
    )

    missing_df = pd.DataFrame({"missing_feature_in_predict": missing_in_pred})
    extra_df = pd.DataFrame({"extra_column_in_predict_dropped": extra_in_pred})

    output_xlsx = Path("output.xlsx")
    output_csv = Path("output.csv")
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="predictions", index=False)
        align_info.to_excel(writer, sheet_name="alignment_info", index=False)
        missing_df.to_excel(writer, sheet_name="missing_features", index=False)
        extra_df.to_excel(writer, sheet_name="dropped_extra_cols", index=False)

    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("[DONE] Prediction completed.")
    print(f"[DONE] Output xlsx: {output_xlsx.resolve()}")
    print(f"[DONE] Output csv : {output_csv.resolve()}")
    print(
        f"[INFO] Feature alignment: common={len(common_features)}, "
        f"missing={len(missing_in_pred)}, extra_dropped={len(extra_in_pred)}"
    )


if __name__ == "__main__":
    main()
