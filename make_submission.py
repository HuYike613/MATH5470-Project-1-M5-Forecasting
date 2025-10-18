# -*- coding: utf-8 -*-
"""
Build Kaggle submission (validation + evaluation) from per-store trained models.
Assumes models were trained with the **minimal feature set** only.

Outputs (in OUT_DIR):
  - submission.csv
  - validation_actual_pred_1914_1941.csv
"""

import os
import numpy as np
import pandas as pd

# ========= 路径 =========
OUT_DIR = "/Users/mengqi/Desktop/hkust/MATH 5470/first_assignment/m5-forecasting-accuracy/m5_outputs/feature_by_store"
STORE_FILES = [
    "/Users/mengqi/Desktop/hkust/MATH 5470/first_assignment/m5-forecasting-accuracy/m5_outputs/long_parts/long_store_CA_1.parquet",
    "/Users/mengqi/Desktop/hkust/MATH 5470/first_assignment/m5-forecasting-accuracy/m5_outputs/long_parts/long_store_CA_2.parquet",
    "/Users/mengqi/Desktop/hkust/MATH 5470/first_assignment/m5-forecasting-accuracy/m5_outputs/long_parts/long_store_CA_3.parquet",
    "/Users/mengqi/Desktop/hkust/MATH 5470/first_assignment/m5-forecasting-accuracy/m5_outputs/long_parts/long_store_CA_4.parquet",
    "/Users/mengqi/Desktop/hkust/MATH 5470/first_assignment/m5-forecasting-accuracy/m5_outputs/long_parts/long_store_TX_1.parquet",
    "/Users/mengqi/Desktop/hkust/MATH 5470/first_assignment/m5-forecasting-accuracy/m5_outputs/long_parts/long_store_TX_2.parquet",
    "/Users/mengqi/Desktop/hkust/MATH 5470/first_assignment/m5-forecasting-accuracy/m5_outputs/long_parts/long_store_TX_3.parquet",
    "/Users/mengqi/Desktop/hkust/MATH 5470/first_assignment/m5-forecasting-accuracy/m5_outputs/long_parts/long_store_WI_1.parquet",
    "/Users/mengqi/Desktop/hkust/MATH 5470/first_assignment/m5-forecasting-accuracy/m5_outputs/long_parts/long_store_WI_2.parquet",
    "/Users/mengqi/Desktop/hkust/MATH 5470/first_assignment/m5-forecasting-accuracy/m5_outputs/long_parts/long_store_WI_3.parquet",
]
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 窗口（M5 口径） =========
TRAIN_END = 1913
VALID_START, VALID_END = 1914, 1941
EVAL_START,  EVAL_END  = 1942, 1969
F_COLS = [f"F{i}" for i in range(1, 29)]

# ========= 与训练保持一致的“最小特征集” =========
MINIMAL_FEATS = [
    'item_id_freq',
    'dept_id','cat_id','store_id','state_id',
    'wday','month','year',
    'event_name_1','event_type_1','event_name_2','event_type_2',
    'snap_CA','snap_TX','snap_WI',
    'sell_price',
    'sales_lag_7','sales_lag_14','sales_lag_28',
    'rolling_mean_7','rolling_mean_14','rolling_mean_28'
]

CAT_COLS = ['dept_id','cat_id','store_id','state_id',
            'event_name_1','event_type_1','event_name_2','event_type_2']

def _safe_div(a, b):
    return np.where((b==0) | (~np.isfinite(b)), np.nan, a / b)

def build_full_frame(df_store: pd.DataFrame) -> pd.DataFrame:
    # d -> int
    if pd.api.types.is_string_dtype(df_store['d']) or df_store['d'].dtype == object:
        df_store['d'] = df_store['d'].astype(str).str.extract(r'(\d+)').astype('int32')
    else:
        df_store['d'] = df_store['d'].astype('int32')
    df_store = df_store.sort_values(['id','d']).reset_index(drop=True)

    max_d = df_store['d'].max()
    if max_d >= EVAL_END:
        return df_store.copy()

    # 补全到 d=1..1969 的“日历面板”（复用已存在的日历信息）
    cal_cols = ['date','wday','weekday','month','year',
                'event_name_1','event_type_1','event_name_2','event_type_2',
                'snap_CA','snap_TX','snap_WI','wm_yr_wk']
    cal_cols = [c for c in cal_cols if c in df_store.columns]

    cal_map = (df_store[['d'] + cal_cols]
               .drop_duplicates('d')
               .set_index('d')
               .sort_index())

    full = pd.DataFrame({'d': np.arange(1, EVAL_END+1, dtype='int32')})
    full = full.join(cal_map, on='d')

    # 兜底填充
    if 'wday' not in full.columns or full['wday'].isna().any():
        if 'wday' not in full.columns:
            full['wday'] = ((full['d'] - 1) % 7 + 1).astype('int8')
        else:
            full['wday'] = full['wday'].astype('float32').fillna(((full['d'] - 1) % 7 + 1))
    if 'month' not in full.columns:
        full['month'] = 1
    if 'year' not in full.columns:
        full['year'] = 2011
    for c in ['event_name_1','event_type_1','event_name_2','event_type_2']:
        if c not in full.columns:
            full[c] = pd.Series(pd.Categorical([np.nan]*len(full)))
    for c in ['snap_CA','snap_TX','snap_WI']:
        if c not in full.columns:
            full[c] = 0

    # 扩展到 id × d
    ids = df_store[['id','item_id','dept_id','cat_id','store_id','state_id']].drop_duplicates('id')
    try:
        base = ids.merge(full, how='cross')  # 需要 pandas >= 1.2
    except TypeError:
        # 兼容旧 pandas：人造 key 交叉连接
        ids['_k'] = 1; full['_k'] = 1
        base = ids.merge(full, on='_k').drop(columns=['_k'])
        ids.drop(columns=['_k'], inplace=True); full.drop(columns=['_k'], inplace=True)

    df_store_base = df_store[['id','d','sales','sell_price']]
    out = base.merge(df_store_base, on=['id','d'], how='left')
    out = out.sort_values(['id','d']).reset_index(drop=True)
    out['sell_price'] = (out.groupby(['store_id','item_id'])['sell_price'].ffill())
    return out

def add_minimal_feats(df: pd.DataFrame) -> pd.DataFrame:
    """只构建 minimal 所需特征：item_id_freq、lags 7/14/28、shift(28) rolling_mean 7/14/28。"""
    df = df.sort_values(['id','d']).reset_index(drop=True)

    # item_id_freq
    df['item_id_freq'] = df['item_id'].map(df['item_id'].value_counts()).astype('int32')

    # 滞后
    for lag in (7, 14, 28):
        col = f'sales_lag_{lag}'
        if col not in df.columns:
            df[col] = df.groupby('id', observed=True)['sales'].shift(lag).astype('float32')

    # rolling mean（shift 28 后）
    for win in (7, 14, 28):
        col = f'rolling_mean_{win}'
        if col not in df.columns:
            df[col] = (df.groupby('id', observed=True)['sales']
                         .transform(lambda s: s.shift(28).rolling(win).mean())
                         .astype('float32'))
    return df

def predict_store(df_store: pd.DataFrame, model_path: str):
    import xgboost as xgb

    full = build_full_frame(df_store)
    full = add_minimal_feats(full)

    # 加载已训练模型（最小特征）
    booster = xgb.Booster()
    booster.load_model(model_path)

    feats = [c for c in MINIMAL_FEATS if c in full.columns]  # 顺序对齐
    if len(feats) != len(MINIMAL_FEATS):
        missing = [c for c in MINIMAL_FEATS if c not in feats]
        print(f"   [WARN] missing minimal features on this store: {missing}")

    # 验证窗一次性预测
    val_rows = (full['d'] >= VALID_START) & (full['d'] <= VALID_END)
    X_val = full.loc[val_rows, feats].copy()
    for c in CAT_COLS:
        if c in X_val.columns:
            X_val[c] = X_val[c].astype('category')
    dval = xgb.DMatrix(X_val, enable_categorical=True)
    full.loc[val_rows, 'pred'] = np.clip(booster.predict(dval), 0, None)

    # 评测窗逐日滚动预测（为了正确更新滞后 & rolling）
    for day in range(EVAL_START, EVAL_END+1):
        mask_today = (full['d'] == day)

        # 用到的 minimal 特征都重算（销售用最新的预测回填）
        #   - lags: 7/14/28
        for lag in (7, 14, 28):
            col = f'sales_lag_{lag}'
            full.loc[mask_today, col] = (full.groupby('id', observed=True)['sales']
                                           .shift(lag)).loc[mask_today].astype('float32')

        #   - rolling_mean_(7/14/28) with shift(28)
        for win in (7, 14, 28):
            col = f'rolling_mean_{win}'
            full.loc[mask_today, col] = (full.groupby('id', observed=True)['sales']
                                           .transform(lambda s: s.shift(28).rolling(win).mean())
                                           ).loc[mask_today].astype('float32')

        # 预测 day
        X_day = full.loc[mask_today, feats].copy()
        for c in CAT_COLS:
            if c in X_day.columns:
                X_day[c] = X_day[c].astype('category')
        dday = xgb.DMatrix(X_day, enable_categorical=True)
        yhat_day = np.clip(booster.predict(dday), 0, None)
        full.loc[mask_today, 'pred'] = yhat_day

        # 把预测写回 sales，供后面天的滞后/rolling 使用
        full.loc[mask_today, 'sales'] = full.loc[mask_today, 'pred']

    # 宽表
    def to_wide(df_part, d_start, d_end):
        tmp = df_part.loc[(df_part['d']>=d_start)&(df_part['d']<=d_end),
                          ['id','d','pred']].copy()
        tmp['F'] = tmp['d'] - d_start + 1
        wide = tmp.pivot(index='id', columns='F', values='pred').reindex(columns=range(1,29))
        wide.columns = [f'F{i}' for i in range(1,29)]
        wide = wide.reset_index()
        return wide

    val_wide = to_wide(full, VALID_START, VALID_END)
    eval_wide = to_wide(full, EVAL_START, EVAL_END)

    # id 后缀修正：validation 保持，evaluation 替换
    val_wide['id']  = val_wide['id'].astype(str).str.replace('_validation','_validation', regex=False)
    eval_wide['id'] = eval_wide['id'].astype(str).str.replace('_validation','_evaluation', regex=False)

    return val_wide, eval_wide, full

def main():
    print(">>> Building submission from trained models (MINIMAL features)")
    all_val, all_eval, val_detail_rows = [], [], []

    for path in STORE_FILES:
        base = os.path.splitext(os.path.basename(path))[0]
        model_path = os.path.join(OUT_DIR, f"model_{base}_xgb.json")
        if not os.path.exists(path):
            print(f"[WARN] parquet missing, skip: {path}")
            continue
        if not os.path.exists(model_path):
            print(f"[WARN] model missing, skip: {model_path}")
            continue

        print(f">>> Filling store: {base}")
        df_s = pd.read_parquet(path, engine="pyarrow")
        need_cols = ['id','item_id','dept_id','cat_id','store_id','state_id',
                     'd','sales','date','wm_yr_wk','sell_price','weekday','wday','month','year',
                     'event_name_1','event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI']
        miss = [c for c in need_cols if c not in df_s.columns]
        if miss:
            print(f"   [ERROR] missing columns {miss}, skip store.")
            continue

        try:
            val_wide, eval_wide, full = predict_store(df_s, model_path)
            all_val.append(val_wide)
            all_eval.append(eval_wide)

            val_detail = full.loc[(full['d']>=VALID_START)&(full['d']<=VALID_END),
                                  ['id','store_id','d','sales','pred']].copy()
            val_detail['store_file'] = base
            val_detail_rows.append(val_detail)
        except Exception as e:
            print(f"   [ERROR] {base} -> {e}")
            continue

    if not all_val or not all_eval:
        print("No panel collected. Abort.")
        return

    sub_val  = pd.concat(all_val, axis=0, ignore_index=True)
    sub_eval = pd.concat(all_eval, axis=0, ignore_index=True)

    # 提交流程：validation 在上，evaluation 在下
    submission = pd.concat([sub_val, sub_eval], axis=0, ignore_index=True)
    # 缺失填 0（极少数冷启动行）
    submission[F_COLS] = submission[F_COLS].fillna(0)

    sub_path = os.path.join(OUT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print("Saved submission:", sub_path, f"({submission.shape[0]} rows)")

    if val_detail_rows:
        detail = pd.concat(val_detail_rows, axis=0, ignore_index=True)
        detail.to_csv(os.path.join(OUT_DIR, "validation_actual_pred_1914_1941.csv"), index=False)
        print("Saved:", os.path.join(OUT_DIR, "validation_actual_pred_1914_1941.csv"))

if __name__ == "__main__":
    main()
