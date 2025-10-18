# -*- coding: utf-8 -*-
# ==============================================================================
# M5 — 10门店批量：最小特征 + XGBoost（稳健版）
# - 不做 EDA（批量跑更快）
# - 保持与单店版本相同的最小特征、训练参数、预测一致性
# ==============================================================================

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import gc
import traceback

# ----------------- 目录与门店列表（按你本地路径） -----------------
OUT_DIR = "./ouput/feature_by_store"
STORE_FILES = [
    "./m5-forecasting-accuracy/m5_outputs/long_parts/long_store_CA_1.parquet",
    "./m5-forecasting-accuracy/m5_outputs/long_parts/long_store_CA_2.parquet",
    "./m5-forecasting-accuracy/m5_outputs/long_parts/long_store_CA_3.parquet",
    "./m5-forecasting-accuracy/m5_outputs/long_parts/long_store_CA_4.parquet",
    "./m5-forecasting-accuracy/m5_outputs/long_parts/long_store_TX_1.parquet",
    "./m5-forecasting-accuracy/m5_outputs/long_parts/long_store_TX_2.parquet",
    "./m5-forecasting-accuracy/m5_outputs/long_parts/long_store_TX_3.parquet",
    "./m5-forecasting-accuracy/m5_outputs/long_parts/long_store_WI_1.parquet",
    "./m5-forecasting-accuracy/m5_outputs/long_parts/long_store_WI_2.parquet",
    "./m5-forecasting-accuracy/m5_outputs/long_parts/long_store_WI_3.parquet",
]
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------- 训练/验证窗口 -----------------
TRAIN_END = 1913
VALID_START, VALID_END = 1914, 1941

# ----------------- 小工具 -----------------
def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated(keep="first")]

def _vec1d(x, dtype=None):
    if isinstance(x, pd.DataFrame):
        x = x.squeeze("columns") if x.shape[1] == 1 else x.iloc[:, 0]
    arr = np.asarray(x)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr

# ----------------- 最小特征工程 -----------------
def build_min_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'd' not in df.columns:
        raise ValueError("输入数据缺少列 'd'。")
    if pd.api.types.is_string_dtype(df['d']) or df['d'].dtype == object:
        df['d'] = df['d'].astype(str).str.extract(r'(\d+)').astype(np.int16)
    else:
        df['d'] = df['d'].astype(np.int16)

    for c, t in [('wday','int8'), ('month','int8'), ('year','int16')]:
        if c in df.columns:
            df[c] = df[c].astype(t)

    for need in ['id', 'sales']:
        if need not in df.columns:
            raise ValueError(f"输入数据缺少列 '{need}'。")

    df = df.sort_values(['id', 'd']).reset_index(drop=True)

    if 'item_id' not in df.columns:
        raise ValueError("输入数据缺少列 'item_id'，无法构建 item_id_freq。")
    item_id_freq_map = df['item_id'].value_counts().to_dict()
    df['item_id_freq'] = df['item_id'].map(item_id_freq_map).astype(np.int32)

    # 滞后
    for lag in (7, 14, 28):
        col = f"sales_lag_{lag}"
        if col not in df.columns:
            df[col] = (
                df.groupby('id', observed=True)['sales']
                  .shift(lag).astype(np.float32)
            )

    # rolling mean（shift 28 后）
    for window in (7, 14, 28):
        col = f"rolling_mean_{window}"
        if col not in df.columns:
            df[col] = (
                df.groupby('id', observed=True)['sales']
                  .transform(lambda x: x.shift(28).rolling(window).mean())
                  .astype(np.float32)
            )

    return _dedup_columns(df)

# ----------------- 训练并验证（单店） -----------------
def train_one_store(parq_path: str):
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error

    base = os.path.splitext(os.path.basename(parq_path))[0]
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> Store: {base}")

    if not os.path.exists(parq_path):
        print(f"  [SKIP] Not found: {parq_path}")
        return None

    # 读 + 特征
    df = pd.read_parquet(parq_path, engine="pyarrow")
    df = _dedup_columns(df)
    df = build_min_features(df)

    # 保存带特征数据（可审）
    feats_out = os.path.join(OUT_DIR, f"{base}_with_min_feats.parquet")
    df.to_parquet(feats_out, index=False, engine="pyarrow")

    # 特征列表（最小集）
    maybe_feats = [
        'item_id_freq',
        'dept_id','cat_id','store_id','state_id',
        'wday','month','year',
        'event_name_1','event_type_1','event_name_2','event_type_2',
        'snap_CA','snap_TX','snap_WI',
        'sell_price',
        'sales_lag_7','sales_lag_14','sales_lag_28',
        'rolling_mean_7','rolling_mean_14','rolling_mean_28'
    ]
    features = [c for c in maybe_feats if c in df.columns]
    features = pd.Index(features).drop_duplicates().tolist()

    # 切分并确保 lag 可用
    need_lags = ['sales_lag_7','sales_lag_14','sales_lag_28',
                 'rolling_mean_7','rolling_mean_14','rolling_mean_28']
    sub_ok = [c for c in need_lags if c in df.columns]
    train_df = df[df['d'] <= TRAIN_END].dropna(subset=sub_ok).copy()
    valid_df = df[(df['d'] >= VALID_START) & (df['d'] <= VALID_END)].dropna(subset=sub_ok).copy()
    print(f"  Data: train={len(train_df):,} | valid={len(valid_df):,} | feats={len(features)}")

    # 类别对齐
    cat_cols = ['dept_id','cat_id','store_id','state_id',
                'event_name_1','event_type_1','event_name_2','event_type_2']
    for c in cat_cols:
        if c in train_df.columns:
            train_df[c] = train_df[c].astype('category')
            valid_df[c] = valid_df[c].astype('category')
            valid_df[c] = valid_df[c].cat.set_categories(train_df[c].cat.categories)

    # 清理 sell_price 的 inf
    if 'sell_price' in train_df.columns:
        train_df['sell_price'] = train_df['sell_price'].replace([np.inf, -np.inf], np.nan)
        valid_df['sell_price'] = valid_df['sell_price'].replace([np.inf, -np.inf], np.nan)

    # DMatrix
    X_tr = _dedup_columns(train_df[features].copy())
    X_va = _dedup_columns(valid_df[features].copy())
    y_tr = _vec1d(train_df['sales'], dtype=np.float32)
    y_va = _vec1d(valid_df['sales'], dtype=np.float32)

    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)

    # XGB 参数（与单店版本一致）
    params = {
        'objective': 'reg:tweedie',
        'tweedie_variance_power': 1.1,
        'eval_metric': 'rmse',
        # 'tree_method': 'hist',
        'tree_method': 'hist',
        'device': 'cuda',  # 指定使用CUDA
        'eta': 0.02,
        'max_depth': 7,
        'min_child_weight': 150,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'lambda': 0.1,
        'alpha': 0.1,
        'max_bin': 63,
        'seed': 42
    }
    NUM_BOOST_ROUND = 4000
    EARLY_STOP_ROUNDS = 300

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        verbose_eval=200
    )
    best_iter = int(model.best_iteration)
    print(f"  Best iteration: {best_iter}")

    # 保存模型
    model_out = os.path.join(OUT_DIR, f"model_{base}_xgb.json")
    model.save_model(model_out)

    # 验证集预测（严格按 best_iteration）
    pred_va = model.predict(dvalid, iteration_range=(0, best_iter + 1))
    pred_va = np.maximum(0.0, pred_va)
    rmse_model = float(np.sqrt(((y_va - pred_va) ** 2).mean()))

    if 'sales_lag_28' in valid_df.columns:
        m = valid_df['sales_lag_28'].notna()
        if m.any():
            y_nv = _vec1d(valid_df.loc[m, 'sales'], dtype=np.float64)
            n_nv = _vec1d(valid_df.loc[m, 'sales_lag_28'], dtype=np.float64)
            rmse_naive = float(np.sqrt(((y_nv - n_nv) ** 2).mean()))
        else:
            rmse_naive = float('nan')
    else:
        rmse_naive = float('nan')

    # 保存验证集预测
    val_pred_csv = os.path.join(OUT_DIR, f"valid_preds_{base}.csv")
    out_df = valid_df[['id','store_id','d','sales']].copy()
    out_df['pred'] = pred_va.astype('float32')
    out_df.to_csv(val_pred_csv, index=False)

    # 返回指标
    metrics = {
        "store": base,
        "model_path": model_out,
        "features_used": features,
        "features_used_count": len(features),
        "best_iteration": best_iter,
        "rmse_valid_model": rmse_model,
        "rmse_seasonal_naive": rmse_naive,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
    }

    # 单店 metrics.json
    with open(os.path.join(OUT_DIR, f"metrics_{base}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Done: RMSE(model)={rmse_model:.5f} | RMSE(naive28)={rmse_naive:.5f}")
    return metrics

# ----------------- 主流程 -----------------
def main():
    print("========== M5 10 Stores | Minimal Features + XGB (batch) ==========")
    print("OUT_DIR:", OUT_DIR)
    print("========================================================\n")

    all_metrics = []
    for path in STORE_FILES:
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            m = train_one_store(path)
            if m is not None:
                all_metrics.append(m)
        except Exception as e:
            print(f"  [ERROR] {base} failed: {e}")
            traceback.print_exc()
        finally:
            gc.collect()

    # 汇总指标
    if all_metrics:
        dfm = pd.DataFrame(all_metrics)
        dfm = dfm[['store','best_iteration','features_used_count','rmse_valid_model','rmse_seasonal_naive','train_rows','valid_rows']]
        out_csv = os.path.join(OUT_DIR, "all_stores_metrics.csv")
        dfm.to_csv(out_csv, index=False)
        print("\nSaved metrics:", out_csv)

    print("\nAll done.")

if __name__ == "__main__":
    main()
