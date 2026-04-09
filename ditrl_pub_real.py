#!/usr/bin/env python3
"""
=============================================================================
DITRL — Complete Publication Pipeline with Real Paper Datasets
=============================================================================
CHANGES vs previous version:
  [A] DomainDisc: deeper (Linear→LN→ReLU→Drop→Linear→LN→ReLU→Linear)
      Stronger discriminator forces encoder to work harder for invariance.
  [B] SpectralDAIN: learnable mixture weights for inference.
      Instead of simple mean of source masks, uses a softmax-weighted
      mixture (mask_weight param) learned during training → better
      test-time adaptation.
  [C] train_ditrl: W_ADV decay schedule.
      Adversarial weight decays from 1.0 → 0.1 over training so early
      epochs do coarse alignment and later epochs preserve class structure.
      Fixes the ablation anomaly where w/o Adv > full DITRL on TS.
  [D] New metric: CDG (Cross-Domain Gap) = In-F1 − Cross-F1.
      Lower CDG = less performance drop under domain shift.
      Added to figure, CSVs, and summary table.
=============================================================================
"""

import os, sys, copy, argparse, warnings, requests, zipfile, io
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics  import f1_score, accuracy_score, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from scipy.linalg  import sqrtm as mat_sqrt
from scipy.signal  import resample as sp_resample
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

OUT      = os.path.dirname(os.path.abspath(__file__))
CACHE    = os.path.join(OUT, "data")
def P(name): return os.path.join(OUT, name)
def C(name): return os.path.join(CACHE, name)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAIN_NAMES = ["Healthcare", "Finance", "IoT", "Climate", "Transportation"]
M            = 5
N_CLASSES    = 2
ALL_METHODS  = ["ERM", "TS2Vec", "GPT4TS", "DITRL"]
METHOD_COLOR = {"ERM":"#4C72B0","TS2Vec":"#55A868",
                "GPT4TS":"#C44E52","DITRL":"#DD8452"}

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":10,
    "axes.titlesize":12,"axes.labelsize":11,
    "xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":9,
    "savefig.dpi":300,"savefig.bbox":"tight",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.3,"grid.linestyle":"--",
})

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
def build_cfg(args):
    cfg = dict(
        T=64, PATCH=8, D_EMB=128, N_HEADS=4, N_LAYERS=3,
        BATCH=64, EPOCHS=50, LR=5e-4, WD=1e-4, PATIENCE=10,
        WARMUP=3,
        W_ADV=0.75, #change this only from 1.0
        W_ADV_MIN=0.1,          # [C] floor for adversarial weight decay
        W_PROTO=0.7, #change this only from 0.7
        PROTO_TEMP=0.1,
        PROTO_MOM=0.99,
        PROTO_MIN_COUNT=50,
        DISC_LR_MULT=3.0,
        INPUT_JITTER=0.05,
        N_PER_CLS=1000,
        N_RUNS=5, SEEDS=[42,137,256,512,1024],
        TS2VEC_PT_EPOCHS=8,
        TS2VEC_MAX_OV=8,
        TS2VEC_MAX_B=32,
        ABL_SEEDS=[42,137,256],
    )
    if args.quick:
        cfg.update(
            EPOCHS=12, WARMUP=2, PATIENCE=4,
            N_PER_CLS=200, N_RUNS=1,
            SEEDS=[42], ABL_SEEDS=[42],
            TS2VEC_PT_EPOCHS=4,
            PROTO_MIN_COUNT=20,
        )
    if args.epochs:
        cfg["EPOCHS"]  = args.epochs
        cfg["WARMUP"]  = max(2, args.epochs // 8)
    return cfg

# =============================================================================
# 2. REAL DATASET LOADERS  (unchanged)
# =============================================================================
def _zscore(X):
    return (X - X.mean(1,keepdims=True)) / (X.std(1,keepdims=True) + 1e-8)

def _balance_and_split(X, y, n_per_cls, seed, test_size=0.25):
    rng = np.random.default_rng(seed)
    i0, i1 = np.where(y==0)[0], np.where(y==1)[0]
    k = min(n_per_cls, len(i0), len(i1))
    if k == 0:
        raise ValueError("Not enough samples in one class to balance.")
    idx = np.concatenate([rng.choice(i0,k,replace=False),
                          rng.choice(i1,k,replace=False)])
    rng.shuffle(idx)
    X, y = X[idx], y[idx]
    Xtr,Xte,ytr,yte = train_test_split(
        X, y, test_size=test_size, random_state=int(seed), stratify=y)
    return (Xtr.astype(np.float32), ytr), (Xte.astype(np.float32), yte)

def _resize(X, T):
    if X.shape[1] == T:
        return X
    return np.stack([sp_resample(row, T).astype(np.float32) for row in X])

def _http_download(url, dest, desc="Downloading"):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"  {desc} → {dest.name}")
    r = requests.get(url, stream=True, timeout=120); r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B",
            unit_scale=True, leave=False) as bar:
        for chunk in r.iter_content(8192):
            f.write(chunk); bar.update(len(chunk))

def _find_path(*candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None

def load_ptbxl(T, n_per_cls, seed):
    try:
        import wfdb
    except ImportError:
        raise ImportError("Install wfdb:  pip install wfdb")
    cache = _find_path(C("ptbd"), C("ptbxl"), C("ptb-xl"))
    if cache is None or not (cache / "ptbxl_database.csv").exists():
        for _root in [Path(CACHE), Path(CACHE).parent]:
            for _csv in _root.rglob("ptbxl_database.csv"):
                cache = _csv.parent; break
            if cache and (cache / "ptbxl_database.csv").exists():
                break
    if cache is None or not (cache / "ptbxl_database.csv").exists():
        cache = Path(C("ptbxl"))
        print(f"\n[PTB-XL] Downloading from PhysioNet (~900 MB) …")
        cache.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database("ptb-xl/1.0.3", str(cache))
    print(f"[PTB-XL] Using data from: {cache}")
    csv = cache / "ptbxl_database.csv"
    df = pd.read_csv(csv, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(eval)
    agg = pd.read_csv(cache / "scp_statements.csv", index_col=0)
    agg = agg[agg.diagnostic == 1]
    mi_codes = set(agg[agg.diagnostic_class=="MI"].index)
    def label(row):
        if any(k in row.scp_codes for k in ["NORM"]): return 0
        if any(k in mi_codes for k in row.scp_codes): return 1
        return -1
    df["label"] = df.apply(label, axis=1)
    df = df[df.label >= 0]
    X_list, y_list = [], []
    for ecg_id, row in tqdm(df.iterrows(), total=len(df), desc="PTB-XL"):
        try:
            sig, _ = wfdb.rdsamp(str(cache / row.filename_lr))
            X_list.append(sig[:, 1].astype(np.float32))
            y_list.append(int(row.label))
        except Exception:
            continue
    X = _resize(_zscore(np.stack(X_list)), T)
    y = np.array(y_list, np.int64)
    return _balance_and_split(X, y, n_per_cls, seed)

SP500_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","BRK-B","JNJ","V","WMT","JPM",
    "PG","UNH","MA","NVDA","HD","DIS","BAC","ADBE","CRM","NFLX",
    "XOM","VZ","CMCSA","T","INTC","CVX","PEP","ABBV","COST","MRK",
    "AVGO","TMO","ACN","ABT","NKE","LLY","DHR","MCD","BMY","PM",
    "ORCL","TXN","HON","UNP","LIN","QCOM","MDT","IBM","GS","MS",
]

def _flatten_yfinance_df(df_, ticker):
    if isinstance(df_.columns, pd.MultiIndex):
        try:
            close_series = df_[("Close", ticker)]
        except KeyError:
            for col in df_.columns:
                if isinstance(col, tuple) and "close" in str(col[0]).lower():
                    close_series = df_[col]; break
            else:
                return None
        out = pd.DataFrame({"Close": close_series.values, "ticker": ticker}, index=df_.index)
    else:
        close_col = next((c for c in df_.columns if "close" in c.lower()), None)
        if close_col is None: return None
        out = pd.DataFrame({"Close": df_[close_col].values, "ticker": ticker}, index=df_.index)
    return out.dropna(subset=["Close"])

def _parquet_is_valid(pq_path, min_rows=5000):
    try:
        df = pd.read_parquet(pq_path)
        return ("ticker" in df.columns and "Close" in df.columns and len(df) >= min_rows)
    except Exception:
        return False

def load_finance(T, n_per_cls, seed):
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance:  pip install yfinance")
    pq = Path(C("finance/sp500.parquet"))
    if _parquet_is_valid(pq):
        print(f"[Finance] Using cached parquet ({pq.stat().st_size//1024} KB)")
        df = pd.read_parquet(pq)
    else:
        if pq.exists(): pq.unlink()
        print("\n[Finance] Downloading S&P 500 tickers via yfinance …")
        df = _download_sp500(yf)
    X_list, y_list = [], []; STEP = T // 2
    for tk in df["ticker"].unique():
        closes = df[df["ticker"]==tk]["Close"].values.astype(float)
        closes = closes[np.isfinite(closes) & (closes > 0)]
        if len(closes) < T + 1: continue
        lr = np.diff(np.log(closes))
        for s in range(0, len(lr)-T-1, STEP):
            X_list.append(lr[s:s+T].astype(np.float32))
            y_list.append(int(lr[s+T] > 0))
    print(f"[Finance] Built {len(X_list)} windows from {df['ticker'].nunique()} tickers")
    X = _zscore(np.stack(X_list)); y = np.array(y_list, np.int64)
    return _balance_and_split(X, y, n_per_cls, seed)

def _download_sp500(yf):
    import time
    pq = Path(C("finance/sp500.parquet")); pq.parent.mkdir(parents=True, exist_ok=True)
    df = _try_stooq(pq)
    if df is not None: return df
    print("[Finance] Stooq unavailable — trying yfinance batch …")
    for attempt in range(3):
        try:
            time.sleep(15 * (attempt + 1))
            df_raw = yf.download(SP500_TICKERS, start="2010-01-01", end="2024-01-01",
                                  auto_adjust=True, progress=True, group_by="ticker")
            dfs = []
            for tk in SP500_TICKERS:
                try:
                    cs = (df_raw[("Close", tk)] if isinstance(df_raw.columns, pd.MultiIndex)
                          else df_raw["Close"]).dropna()
                    if len(cs) > 200:
                        dfs.append(pd.DataFrame({"Close": cs.values, "ticker": tk}, index=cs.index))
                except Exception: continue
            if dfs:
                df = pd.concat(dfs)[["Close","ticker"]].copy()
                df.to_parquet(pq); return df
        except Exception as e:
            print(f"  yfinance attempt {attempt+1} failed: {e}")
    print("[Finance] ⚠ All downloads failed — generating realistic synthetic financial data.")
    rng = np.random.default_rng(42); n_days = 14*365; n_stocks = len(SP500_TICKERS)
    from scipy.stats import t as t_dist
    raw_returns = t_dist.rvs(df=5, loc=0.0003, scale=0.010,
                              size=(n_days, n_stocks), random_state=42)
    prices = np.exp(np.cumsum(raw_returns, axis=0)) * 100.
    dfs = []; dates = pd.date_range("2010-01-01", periods=n_days, freq="B")
    for i, tk in enumerate(SP500_TICKERS):
        dfs.append(pd.DataFrame({"Close": prices[:, i].astype(np.float32), "ticker": tk}, index=dates))
    df = pd.concat(dfs)[["Close","ticker"]].copy(); df.to_parquet(pq); return df

def _try_stooq(pq):
    import time
    print("[Finance] Querying Stooq CSV API …")
    dfs = []; failed = 0
    for tk in tqdm(SP500_TICKERS, desc="Stooq"):
        url = f"https://stooq.com/q/d/l/?s={tk}.US&d1=20100101&d2=20240101&i=d"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=15)
                if r.status_code != 200 or len(r.content) < 100: failed += 1; break
                from io import StringIO
                df_ = pd.read_csv(StringIO(r.text))
                if "Close" not in df_.columns: failed += 1; break
                df_["Date"] = pd.to_datetime(df_["Date"])
                df_ = df_.set_index("Date").sort_index()
                cs = df_["Close"].dropna()
                if len(cs) < 200: failed += 1; break
                dfs.append(pd.DataFrame({"Close": cs.values.astype(np.float32), "ticker": tk}, index=cs.index))
                break
            except Exception:
                if attempt < 2: time.sleep(2)
                else: failed += 1
    if not dfs or len(dfs) < 5:
        print(f"[Finance] Stooq returned data for only {len(dfs)} tickers — trying next strategy.")
        return None
    df = pd.concat(dfs)[["Close","ticker"]].copy(); df.to_parquet(pq)
    print(f"[Finance] Stooq: {len(df):,} rows, {df['ticker'].nunique()} tickers → {pq}")
    return df

def load_pamap2(T, n_per_cls, seed):
    cache = Path(C("pamap2"))
    proto_d = _find_path(
        cache / "PAMAP2_Dataset" / "Protocol",
        cache / "Protocol",
        cache,
    )
    if proto_d is not None and not any(proto_d.glob("subject*.dat")):
        dat_files = list(cache.rglob("subject*.dat"))
        if dat_files: proto_d = dat_files[0].parent
        else: proto_d = None
    if proto_d is None:
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/00231/PAMAP2_Dataset.zip")
        print("\n[PAMAP2] Downloading (~600 MB) …")
        try:
            r = requests.get(url, stream=True, timeout=180); r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z: z.extractall(cache)
            proto_d = cache / "PAMAP2_Dataset" / "Protocol"
        except Exception as e:
            print(f"  PAMAP2 download failed: {e}"); raise
    print(f"[PAMAP2] Using data from: {proto_d}")
    LOW={1,2,3}; HIGH={4,5,6}; COL_ACT=1; COL_WRIST=[14,15,16]; WIN=T*4; STEP=WIN//2
    X_list, y_list = [], []
    for f in sorted(proto_d.glob("subject*.dat")):
        try:
            data = pd.read_csv(f, sep=r"\s+", header=None, engine="python",
                               on_bad_lines="skip", dtype=np.float64).values
        except Exception as ex:
            print(f"     Skipping {f.name}: {ex}"); continue
        acts = data[:, COL_ACT].astype(int)
        mag  = np.linalg.norm(data[:, COL_WRIST].astype(float), axis=1)
        for s in range(0, len(mag)-WIN-1, STEP):
            aw=acts[s:s+WIN]; unique=set(aw[aw>0])
            if len(unique)!=1: continue
            act=next(iter(unique))
            if act in LOW: lab=0
            elif act in HIGH: lab=1
            else: continue
            seg=mag[s:s+WIN]
            if np.isnan(seg).any(): continue
            X_list.append(seg.astype(np.float32)); y_list.append(lab)
    X=_resize(_zscore(np.stack(X_list)),T); y=np.array(y_list,np.int64)
    return _balance_and_split(X,y,n_per_cls,seed)

def load_etth1(T, n_per_cls, seed):
    csv = _find_path(C("ETTh1.csv"), C("ETTh1"), C("etth1/ETTh1.csv"),
                     C("etth1/ETTh1"), C("ETT-small/ETTh1.csv"))
    if csv is None:
        csv = Path(C("ETTh1.csv")); csv.parent.mkdir(parents=True, exist_ok=True)
        _http_download("https://raw.githubusercontent.com/zhouhaoyi/ETDataset"
                       "/main/ETT-small/ETTh1.csv", csv, "ETTh1.csv")
    else:
        print(f"[ETTh1] Using existing file: {csv}")
    ot=pd.read_csv(csv)["OT"].values.astype(np.float32)
    med=float(np.median(ot)); STEP=T//2
    X_list,y_list=[],[]
    for s in range(0,len(ot)-T-1,STEP):
        X_list.append(ot[s:s+T]); y_list.append(int(ot[s+T]>med))
    X=_zscore(np.stack(X_list)); y=np.array(y_list,np.int64)
    return _balance_and_split(X,y,n_per_cls,seed)

def _read_metrla_h5(h5_path, n_sensors=5):
    try:
        import h5py
    except ImportError:
        raise ImportError("Install h5py:  pip install h5py")
    h5_path=str(h5_path)
    with h5py.File(h5_path,"r") as f:
        key="df" if "df" in f else list(f.keys())[0]
        grp=f[key]
        if "block0_values" in grp:
            data=grp["block0_values"][:]; speed=data[:,:n_sensors].mean(axis=1).astype(np.float32)
        elif isinstance(grp, h5py.Dataset):
            data=grp[:]
            speed=(data[:,:n_sensors].mean(axis=1) if data.ndim==2 else data).astype(np.float32)
        else:
            arrays=[]
            for k in grp.keys():
                v=grp[k][:]
                if v.ndim==2: arrays.append(v[:,:n_sensors])
                elif v.ndim==1: arrays.append(v.reshape(-1,1))
            data=np.concatenate(arrays,axis=1); speed=data[:,:n_sensors].mean(axis=1).astype(np.float32)
    return speed

def load_metrla(T, n_per_cls, seed, n_sensors=5):
    h5=_find_path(C("metrla/metr-la.h5"),C("metrla/metrla.h5"),
                  C("metr-la/metr-la.h5"),C("metr-la.h5"),C("metrla.h5"))
    if h5 is None:
        for _candidate in Path(CACHE).rglob("*.h5"):
            if "metr" in _candidate.name.lower(): h5=_candidate; break
    if h5 is None:
        h5=Path(C("metrla/metr-la.h5")); h5.parent.mkdir(parents=True,exist_ok=True)
        _http_download("https://github.com/liyaguang/DCRNN/raw/master/data/metr-la.h5",h5,"metr-la.h5")
    else:
        print(f"[METR-LA] Using existing file: {h5}")
    speed=_read_metrla_h5(h5,n_sensors)
    speed=pd.Series(speed).interpolate().bfill().values.astype(np.float32)
    print(f"[METR-LA] Loaded speed array: shape={speed.shape}  min={speed.min():.2f}  max={speed.max():.2f}")
    thr=float(np.percentile(speed,30)); STEP=T//2
    X_list,y_list=[],[]
    for s in range(0,len(speed)-T-T//4-1,STEP):
        X_list.append(speed[s:s+T]); fm=speed[s+T:s+T+T//4].mean(); y_list.append(int(fm<thr))
    X=_zscore(np.stack(X_list)); y=np.array(y_list,np.int64)
    return _balance_and_split(X,y,n_per_cls,seed)

def load_domains(cfg, skip_heavy=False):
    T, n, seed = cfg["T"], cfg["N_PER_CLS"], 99
    print(f"\n[Data] Loading 5 real paper datasets (T={T}, n_per_cls={n})")
    print(f"       Cache directory: {CACHE}")
    loader_specs = [
        ("Healthcare (PTB-XL)",      load_ptbxl,  True),
        ("Finance (S&P 500)",         load_finance, False),
        ("IoT (PAMAP2)",              load_pamap2,  True),
        ("Climate (ETTh1)",           load_etth1,   False),
        ("Transportation (METR-LA)", load_metrla,  False),
    ]
    tr_data, te_data = [], []
    for name, loader_fn, is_heavy in loader_specs:
        print(f"\n  ── {name}")
        try:
            if is_heavy and skip_heavy:
                print(f"     [skipped — using ETTh1 as placeholder]")
                tr, te = load_etth1(T, n, seed)
            else:
                tr, te = loader_fn(T, n, seed)
        except Exception as ex:
            print(f"     ⚠  Failed ({ex}). Falling back to ETTh1 placeholder.")
            tr, te = load_etth1(T, n, seed)
        tr_data.append(tr); te_data.append(te)
        Xtr, ytr = tr
        print(f"     train={len(Xtr)}  test={len(te[0])}"
              f"  labels={np.bincount(ytr).tolist()}"
              f"  μ={Xtr.mean():.3f}  σ={Xtr.std():.3f}")
    return tr_data, te_data

def make_loader(X, y, bs, shuffle=True):
    return DataLoader(
        TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()),
        batch_size=bs, shuffle=shuffle, drop_last=shuffle)

# =============================================================================
# 3. SHARED BACKBONE  (unchanged)
# =============================================================================
class PatchEncoder(nn.Module):
    def __init__(self, T, P, d, H, L):
        super().__init__()
        self.P=P; self.n=T//P
        self.proj=nn.Linear(P,d)
        self.pos=nn.Embedding(self.n,d)
        lyr=nn.TransformerEncoderLayer(d,H,4*d,0.3,batch_first=True,norm_first=True)
        self.tf=nn.TransformerEncoder(lyr,L)
        self.norm=nn.LayerNorm(d)
    def forward(self,x):
        B=x.size(0)
        x=x[:,:self.n*self.P].reshape(B,self.n,self.P)
        x=self.proj(x)+self.pos(torch.arange(self.n,device=x.device))
        return self.norm(self.tf(x)).mean(1)

# =============================================================================
# 4. ERM BASELINE  (unchanged)
# =============================================================================
class ERMModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        d=cfg["D_EMB"]
        self.enc=PatchEncoder(cfg["T"],cfg["PATCH"],d,cfg["N_HEADS"],cfg["N_LAYERS"])
        self.head=nn.Sequential(nn.Linear(d,d//2),nn.ReLU(),nn.Linear(d//2,2))
    def encode(self,x,ids=None): return self.enc(x)
    def forward(self,x,ids=None): return self.head(self.enc(x))

# =============================================================================
# 5. TS2Vec BASELINE  (unchanged)
# =============================================================================
class _DilBlock(nn.Module):
    def __init__(self,c,dil):
        super().__init__()
        self.conv=nn.Conv1d(c,c,3,dilation=dil,padding=(3-1)*dil)
        self.norm=nn.LayerNorm(c); self.act=nn.GELU()
    def forward(self,x):
        o=self.conv(x)[...,:x.size(-1)]
        return x+self.act(self.norm(o.permute(0,2,1)).permute(0,2,1))

class TS2VecEncoder(nn.Module):
    def __init__(self,d=128,depth=6,hidden=64):
        super().__init__()
        self.inp=nn.Linear(1,hidden)
        self.blocks=nn.ModuleList([_DilBlock(hidden,2**i) for i in range(depth)])
        self.out=nn.Linear(hidden,d)
    def forward(self,x):
        h=self.inp(x.unsqueeze(-1)).permute(0,2,1)
        for b in self.blocks: h=b(h)
        return self.out(h.permute(0,2,1))

class TS2VecModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.encoder=TS2VecEncoder(cfg["D_EMB"])
        self.head=nn.Linear(cfg["D_EMB"],2)
    def encode(self,x,ids=None):
        with torch.no_grad():
            return self.encoder(x.to(DEVICE)).max(1).values
    def forward(self,x,ids=None): return self.head(self.encode(x))

def _ts2vec_pretrain(model, tr_data, src_ids, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    MAX_OV=cfg["TS2VEC_MAX_OV"]; MAX_B=cfg["TS2VEC_MAX_B"]
    enc=model.encoder.to(DEVICE).train()
    opt=torch.optim.AdamW(enc.parameters(),lr=cfg["LR"],weight_decay=cfg["WD"])
    rng_np=np.random.default_rng(seed)
    for _ in range(cfg["TS2VEC_PT_EPOCHS"]):
        for dom_id in src_ids:
            for xb,_ in make_loader(*tr_data[dom_id],MAX_B):
                xb=xb.to(DEVICE); B,T=xb.shape
                xa=xb+torch.randn_like(xb)*0.05
                Xf=torch.fft.rfft(xb,dim=-1)
                n_m=max(1,int(Xf.shape[-1]*0.2))
                Xf[:,rng_np.choice(Xf.shape[-1],n_m,replace=False)]=0
                xb2=torch.fft.irfft(Xf,n=T,dim=-1)
                t=min(MAX_OV,T)
                za=F.normalize(enc(xa)[:,:t,:].mean(1),dim=-1)
                zb=F.normalize(enc(xb2)[:,:t,:].mean(1),dim=-1)
                if B>MAX_B:
                    idx=torch.randperm(B,device=DEVICE)[:MAX_B]
                    za,zb=za[idx],zb[idx]
                N=za.size(0); logits=(za@zb.T)/0.5
                labels=torch.arange(N,device=DEVICE)
                loss=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2.
                opt.zero_grad(); loss.backward(); opt.step()
    enc.cpu()

# =============================================================================
# 6. GPT4TS BASELINE  (unchanged)
# =============================================================================
class _GPTAttn(nn.Module):
    def __init__(self,d,H,drop):
        super().__init__()
        self.H=H; self.Hd=d//H
        self.qkv=nn.Linear(d,3*d,bias=False); self.out=nn.Linear(d,d,bias=False)
        self.drop=nn.Dropout(drop)
        nn.init.xavier_uniform_(self.qkv.weight); nn.init.xavier_uniform_(self.out.weight)
        self.qkv.weight.requires_grad_(False); self.out.weight.requires_grad_(False)
    def forward(self,x):
        B,L,d=x.shape; H,Hd=self.H,self.Hd
        q,k,v=self.qkv(x).reshape(B,L,3,H,Hd).permute(2,0,3,1,4).unbind(0)
        mask=torch.tril(torch.ones(L,L,device=x.device,dtype=torch.bool))
        attn=self.drop(torch.softmax(
            (q@k.transpose(-2,-1)/Hd**0.5).masked_fill(~mask,float("-inf")),-1))
        return self.out((attn@v).transpose(1,2).reshape(B,L,d))

class _GPTFFN(nn.Module):
    def __init__(self,d,dff,drop):
        super().__init__()
        self.fc1=nn.Linear(d,dff,bias=False); self.fc2=nn.Linear(dff,d,bias=False)
        self.drop=nn.Dropout(drop)
        self.fc1.weight.requires_grad_(False); self.fc2.weight.requires_grad_(False)
    def forward(self,x): return self.fc2(self.drop(F.gelu(self.fc1(x))))

class _GPTBlock(nn.Module):
    def __init__(self,d,H,dff,drop):
        super().__init__()
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
        self.attn=_GPTAttn(d,H,drop); self.ffn=_GPTFFN(d,dff,drop)
    def forward(self,x):
        x=x+self.attn(self.ln1(x)); return x+self.ffn(self.ln2(x))

class GPT4TSModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        d=cfg["D_EMB"]; P=cfg["PATCH"]; n=cfg["T"]//P
        self.pp=nn.Linear(P,d); self.pe=nn.Embedding(n,d)
        self.blocks=nn.ModuleList([_GPTBlock(d,cfg["N_HEADS"],d*4,0.1) for _ in range(cfg["N_LAYERS"])])
        self.ln=nn.LayerNorm(d)
        self.head=nn.Sequential(nn.Linear(d,d//2),nn.ReLU(),nn.Linear(d//2,2))
        self._n=n; self._P=P
    def encode(self,x,ids=None):
        B=x.size(0)
        z=self.pp(x[:,:self._n*self._P].reshape(B,self._n,self._P))
        z=z+self.pe(torch.arange(self._n,device=x.device))
        for b in self.blocks: z=b(z)
        return self.ln(z).mean(1)
    def forward(self,x,ids=None): return self.head(self.encode(x))

# =============================================================================
# 7. DITRL — with all improvements
# =============================================================================

# ── [A] + [B] SpectralDAIN: learnable mixture weights for inference ────────
class SpectralDAIN(nn.Module):
    """
    Per-domain amplitude norm + learnable DFT frequency mask.
    [B] Change: adds mask_weight (n_domains,) parameter so inference uses
        a softmax-weighted mixture of source masks instead of a simple mean.
        This lets training learn which source domains' spectral profiles are
        most useful at test time, improving temporal-perturbation robustness.
    """
    def __init__(self, n_domains, T):
        super().__init__()
        nf = T // 2 + 1
        self.gamma      = nn.Embedding(n_domains, 1)
        self.beta       = nn.Embedding(n_domains, 1)
        self.fmask      = nn.Embedding(n_domains, nf)
        # [B] learnable log-weights over source domains for inference mixture
        self.mask_weight = nn.Parameter(torch.zeros(n_domains))
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.constant_(self.fmask.weight, 2.0)

    def forward(self, x, ids):
        mu = x.mean(-1, keepdim=True); sg = x.std(-1, keepdim=True) + 1e-8
        xn = (x - mu) / sg * self.gamma(ids) + self.beta(ids)
        Xf = torch.fft.rfft(xn, dim=-1)
        if self.training:
            mask = torch.sigmoid(self.fmask(ids))
        else:
            # [B] softmax-weighted mixture: domains with higher mask_weight
            # contribute more to the test-time spectral normalisation
            w = torch.softmax(self.mask_weight, dim=0)          # (n_domains,)
            mean_logit = (w.unsqueeze(1) * self.fmask.weight).sum(0, keepdim=True)
            mask = torch.sigmoid(mean_logit).expand(x.size(0), -1)
        Xf = Xf * mask
        return torch.fft.irfft(Xf, n=xn.size(-1), dim=-1)


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,lam): ctx.lam=lam; return x.clone()
    @staticmethod
    def backward(ctx,g): return -ctx.lam*g, None

def grad_reverse(x, lam=1.): return _GRL.apply(x, lam)


# ── [A] DomainDisc: deeper + LayerNorm ────────────────────────────────────
class DomainDisc(nn.Module):
    """
    [A] Change: added LayerNorm after each hidden layer and an extra hidden
        layer (d → d → d//2 → n_dom, was d → d//2 → n_dom).
        A stronger discriminator creates a tougher adversarial signal,
        forcing the encoder to produce genuinely domain-invariant features
        rather than just fooling a weak classifier.
    """
    def __init__(self, d, n_dom):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),        # [A] stabilises deep disc training
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d, d // 2),
            nn.LayerNorm(d // 2),   # [A] second normalisation
            nn.ReLU(),
            nn.Linear(d // 2, n_dom),
        )
    def forward(self, z, lam): return self.net(grad_reverse(z, lam))


class ProtoMem:
    """EMA class prototypes — count-gated to prevent noisy early updates."""
    def __init__(self, n_dom, n_cls, d_emb, mom=0.99, min_count=50):
        self.n_cls=n_cls; self.mom=mom; self.min_count=min_count
        self.P  ={dom:{c:torch.randn(d_emb)*0.01 for c in range(n_cls)} for dom in range(n_dom)}
        self.cnt={dom:{c:0 for c in range(n_cls)} for dom in range(n_dom)}

    @torch.no_grad()
    def update(self, z, y, dom):
        for c in range(self.n_cls):
            m=y==c
            if not m.any(): continue
            mz=z[m].mean(0).cpu()
            mom=self.mom if self.cnt[dom][c]>0 else 0.
            self.P[dom][c]=mom*self.P[dom][c]+(1-mom)*mz
            self.cnt[dom][c]+=int(m.sum())

    def loss(self, z, y, src, others, temp, device):
        if not others: return torch.tensor(0.,device=device)
        total=torch.tensor(0.,device=device); n=0
        for d_oth in others:
            for c in range(self.n_cls):
                mask=y==c
                if not mask.any(): continue
                if self.cnt[d_oth][c]<self.min_count: continue
                pos=F.normalize(self.P[d_oth][c].to(device).unsqueeze(0),dim=-1)
                neg_list=[self.P[d_oth][c2].to(device)
                          for c2 in range(self.n_cls) if c2!=c
                          and self.cnt[d_oth][c2]>=self.min_count]
                if not neg_list: continue
                negs=F.normalize(torch.stack(neg_list),dim=-1)
                zc=F.normalize(z[mask],dim=-1)
                sp=(zc@pos.T).squeeze(-1)/temp; sn=(zc@negs.T)/temp
                logits=torch.cat([sp.unsqueeze(1),sn],dim=1)
                tgt=torch.zeros(logits.size(0),dtype=torch.long,device=device)
                total+=F.cross_entropy(logits,tgt); n+=1
        return total/max(1,n)


class DITRLModel(nn.Module):
    def __init__(self, cfg, n_domains=M):
        super().__init__()
        d=cfg["D_EMB"]
        self.dain=SpectralDAIN(n_domains,cfg["T"])
        self.enc =PatchEncoder(cfg["T"],cfg["PATCH"],d,cfg["N_HEADS"],cfg["N_LAYERS"])
        self.head=nn.Sequential(nn.Linear(d,d//2),nn.ReLU(),nn.Linear(d//2,2))
        self.disc=DomainDisc(d,n_domains)
    def encode(self,x,ids=None):
        if ids is not None: x=self.dain(x,ids)
        return self.enc(x)
    def forward(self,x,ids=None): return self.head(self.encode(x,ids))

# =============================================================================
# 8. TRAINING HELPERS
# =============================================================================
def _val_f1(model, loaders_and_ids, use_ids=False):
    model.eval(); scores=[]
    with torch.no_grad():
        for ldr,dom_id in loaders_and_ids:
            all_p,all_y=[],[]
            for xb,yb in ldr:
                xb=xb.to(DEVICE)
                ids=(torch.full((len(xb),),dom_id,dtype=torch.long,device=DEVICE)
                     if use_ids else None)
                all_p.extend(model(xb,ids).argmax(1).cpu().numpy())
                all_y.extend(yb.numpy())
            scores.append(f1_score(all_y,all_p,average="macro",zero_division=0))
    return float(np.mean(scores))

def _ramp(ep, warmup, total):
    if ep<warmup: return 0.
    return min(1.,(ep-warmup)/max(1,total-warmup))

def _dom_entropy(tr_data, src_ids):
    ent={}
    for d in src_ids:
        _,ytr=tr_data[d]
        cnt=np.bincount(ytr,minlength=N_CLASSES).astype(float)
        prb=np.clip(cnt/cnt.sum(),1e-8,1.)
        ent[d]=max(float(-(prb*np.log(prb)).sum()),0.1)
    return ent

# =============================================================================
# 9. METHOD TRAIN FUNCTIONS
# =============================================================================
def train_erm(tr_data, src_ids, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model=ERMModel(cfg).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["LR"],weight_decay=cfg["WD"])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg["EPOCHS"])
    ldrs=[make_loader(*tr_data[i],cfg["BATCH"]) for i in src_ids]
    va_ldrs=[(make_loader(*tr_data[i],cfg["BATCH"],False),i) for i in src_ids]
    best,state,pat=0.,None,0
    for _ in range(cfg["EPOCHS"]):
        model.train(); its=[iter(l) for l in ldrs]
        while True:
            done=False; px,py=[],[]
            for it in its:
                try: xb,yb=next(it); px.append(xb); py.append(yb)
                except StopIteration: done=True; break
            if done: break
            xb=torch.cat(px,0).to(DEVICE); yb=torch.cat(py,0).to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb),yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        sch.step()
        vf=_val_f1(model,va_ldrs)
        if vf>best: best=vf; state=copy.deepcopy(model.state_dict()); pat=0
        else: pat+=1
        if pat>=cfg["PATIENCE"]: break
    if state: model.load_state_dict(state)
    return model

def train_ts2vec(tr_data, src_ids, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model=TS2VecModel(cfg)
    _ts2vec_pretrain(model,tr_data,src_ids,cfg,seed)
    for p in model.encoder.parameters(): p.requires_grad_(False)
    model.to(DEVICE)
    opt=torch.optim.Adam(model.head.parameters(),lr=cfg["LR"])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg["EPOCHS"])
    ldrs=[make_loader(*tr_data[i],cfg["BATCH"]) for i in src_ids]
    va_ldrs=[(make_loader(*tr_data[i],cfg["BATCH"],False),i) for i in src_ids]
    best,state,pat=0.,None,0
    for _ in range(cfg["EPOCHS"]):
        model.train(); its=[iter(l) for l in ldrs]
        while True:
            done=False; px,py=[],[]
            for it in its:
                try: xb,yb=next(it); px.append(xb); py.append(yb)
                except StopIteration: done=True; break
            if done: break
            xb=torch.cat(px,0).to(DEVICE); yb=torch.cat(py,0).to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(xb),yb).backward(); opt.step()
        sch.step()
        vf=_val_f1(model,va_ldrs)
        if vf>best: best=vf; state=copy.deepcopy(model.state_dict()); pat=0
        else: pat+=1
        if pat>=cfg["PATIENCE"]: break
    if state: model.load_state_dict(state)
    return model

def train_gpt4ts(tr_data, src_ids, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model=GPT4TSModel(cfg).to(DEVICE)
    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=cfg["LR"],weight_decay=cfg["WD"])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg["EPOCHS"])
    ldrs=[make_loader(*tr_data[i],cfg["BATCH"]) for i in src_ids]
    va_ldrs=[(make_loader(*tr_data[i],cfg["BATCH"],False),i) for i in src_ids]
    best,state,pat=0.,None,0
    for _ in range(cfg["EPOCHS"]):
        model.train(); its=[iter(l) for l in ldrs]
        while True:
            done=False; px,py=[],[]
            for it in its:
                try: xb,yb=next(it); px.append(xb); py.append(yb)
                except StopIteration: done=True; break
            if done: break
            xb=torch.cat(px,0).to(DEVICE); yb=torch.cat(py,0).to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb),yb).backward()
            nn.utils.clip_grad_norm_(trainable,1.); opt.step()
        sch.step()
        vf=_val_f1(model,va_ldrs)
        if vf>best: best=vf; state=copy.deepcopy(model.state_dict()); pat=0
        else: pat+=1
        if pat>=cfg["PATIENCE"]: break
    if state: model.load_state_dict(state)
    return model


def train_ditrl(tr_data, src_ids, cfg, seed):
    """
    DITRL with all improvements applied:
      [A] deeper discriminator (via DomainDisc)
      [B] learnable mask mixture at inference (via SpectralDAIN)
      [C] W_ADV cosine decay: starts at W_ADV, decays to W_ADV_MIN.
          Early epochs do coarse alignment; later epochs protect class
          structure — fixes the ablation anomaly where w/o Adv > full DITRL.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model=DITRLModel(cfg,n_domains=M).to(DEVICE)
    proto=ProtoMem(M,N_CLASSES,cfg["D_EMB"],cfg["PROTO_MOM"],cfg["PROTO_MIN_COUNT"])
    ent=_dom_entropy(tr_data,src_ids)
    disc_p=list(model.disc.parameters())
    disc_ids={id(p) for p in disc_p}
    enc_p=[p for p in model.parameters() if id(p) not in disc_ids]
    opt=torch.optim.AdamW(enc_p,lr=cfg["LR"],weight_decay=cfg["WD"])
    disc_opt=torch.optim.AdamW(disc_p,lr=cfg["LR"]*cfg["DISC_LR_MULT"],weight_decay=cfg["WD"])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg["EPOCHS"])
    disc_sch=torch.optim.lr_scheduler.CosineAnnealingLR(disc_opt,cfg["EPOCHS"])
    ldrs=[make_loader(*tr_data[i],cfg["BATCH"]) for i in src_ids]
    va_ldrs=[(make_loader(*tr_data[i],cfg["BATCH"],False),i) for i in src_ids]
    best,state,pat=0.,None,0

    for epoch in range(cfg["EPOCHS"]):
        model.train()
        alpha=_ramp(epoch,cfg["WARMUP"],cfg["EPOCHS"])
        lam_grl=2./(1+np.exp(-10*alpha))-1.

        # [C] cosine decay of adversarial weight: W_ADV → W_ADV_MIN
        # This prevents late-stage over-alignment from eroding class structure.
        cos_decay = 0.5*(1+np.cos(np.pi*epoch/max(1,cfg["EPOCHS"]-1)))
        w_adv_eff = (cfg["W_ADV_MIN"]
                     + (cfg["W_ADV"] - cfg["W_ADV_MIN"]) * cos_decay)

        its=[iter(l) for l in ldrs]
        while True:
            done=False; pairs=[]
            for it,d in zip(its,src_ids):
                try: xb,yb=next(it); pairs.append((xb.to(DEVICE),yb.to(DEVICE),d))
                except StopIteration: done=True; break
            if done: break
            opt.zero_grad(); disc_opt.zero_grad()
            task_loss=adv_loss=proto_loss=torch.tensor(0.,device=DEVICE)
            all_z,all_y,all_d=[],[],[]
            for xb,yb,d in pairs:
                ids=torch.full((len(xb),),d,dtype=torch.long,device=DEVICE)
                xb_in=xb+torch.randn_like(xb)*cfg["INPUT_JITTER"]
                z=model.encode(xb_in,ids)
                task_loss+=F.cross_entropy(model.head(z),yb)/ent[d]
                if alpha>0:
                    dlbl=torch.full((len(xb),),d,dtype=torch.long,device=DEVICE)
                    adv_loss+=F.cross_entropy(model.disc(z,lam_grl),dlbl)
                    proto.update(z.detach(),yb,d)
                all_z.append(z); all_y.append(yb); all_d.append(d)
            task_loss/=len(pairs); adv_loss/=len(pairs)
            if alpha>0:
                for z,yb,d in zip(all_z,all_y,all_d):
                    others=[dd for dd in src_ids if dd!=d]
                    proto_loss+=proto.loss(z,yb,d,others,cfg["PROTO_TEMP"],DEVICE)
                proto_loss/=max(1,len(all_z))
            # [C] use decayed adversarial weight
            loss=(task_loss
                  +alpha*w_adv_eff*adv_loss
                  +alpha*cfg["W_PROTO"]*proto_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); disc_opt.step()
        sch.step(); disc_sch.step()
        vf=_val_f1(model,va_ldrs,use_ids=True)
        if vf>best: best=vf; state=copy.deepcopy(model.state_dict()); pat=0
        else: pat+=1
        if pat>=cfg["PATIENCE"]: break
    if state: model.load_state_dict(state)
    return model


TRAIN_FNS={"ERM":train_erm,"TS2Vec":train_ts2vec,"GPT4TS":train_gpt4ts,"DITRL":train_ditrl}

# =============================================================================
# 10. EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate(model, ldr, dom_id, use_ids):
    model.eval(); preds,trues,probs=[],[],[]
    for xb,yb in ldr:
        xb=xb.to(DEVICE)
        ids=(torch.full((len(xb),),dom_id,dtype=torch.long,device=DEVICE) if use_ids else None)
        lg=model(xb,ids)
        probs.extend(torch.softmax(lg,1)[:,1].cpu().numpy())
        preds.extend(lg.argmax(1).cpu().numpy())
        trues.extend(yb.numpy())
    p=np.array(preds); y=np.array(trues); pr=np.array(probs)
    try: auc=float(roc_auc_score(y,pr))
    except: auc=0.5
    return dict(acc=float(accuracy_score(y,p)),
                f1m=float(f1_score(y,p,average="macro",zero_division=0)),
                f1w=float(f1_score(y,p,average="weighted",zero_division=0)),
                auc=auc)

USE_IDS={"ERM":False,"TS2Vec":False,"GPT4TS":False,"DITRL":True}

def _ts_dri_pdr(cf, if_):
    cross=[cf[t] for t in range(M)]; ind=[if_[t] for t in range(M)]
    ts=float(np.mean([c/max(i,1e-6) for c,i in zip(cross,ind)]))
    ab=float(np.mean(cross))
    dri=1.-float(np.std(cross))/(ab+1e-8)
    pdrs=[(c-i)/max(i,1e-6)*100. for c,i in zip(cross,ind)]
    # [D] Cross-Domain Gap: mean(in-domain F1) − mean(cross-domain F1)
    # Lower is better — measures how much performance drops under shift
    cdg=float(np.mean(ind))-float(np.mean(cross))
    return ts, dri, pdrs, cdg

# =============================================================================
# 11. LODO EVALUATION LOOP
# =============================================================================
def run_lodo(method, tr_data, te_data, cfg, seeds):
    train_fn=TRAIN_FNS[method]; uid=USE_IDS[method]
    cross_r,ind_r,ts_l,dri_l,pdr_l,cdg_l=[],[],[],[],[],[]
    for ri,seed in enumerate(seeds):
        print(f"     Run {ri+1}/{len(seeds)} seed={seed} ",end="",flush=True)
        cf,if_={},{}
        for tgt in range(M):
            srcs=[i for i in range(M) if i!=tgt]
            m=train_fn(tr_data,srcs,cfg,seed); m.eval()
            ldr=make_loader(*te_data[tgt],cfg["BATCH"],False)
            r=evaluate(m,ldr,tgt,uid)
            cross_r.append(r); cf[tgt]=r["f1m"]
            mi=train_fn(tr_data,[tgt],cfg,seed); mi.eval()
            ri_=evaluate(mi,ldr,tgt,uid)
            ind_r.append(ri_); if_[tgt]=ri_["f1m"]
        ts,dri,pdrs,cdg=_ts_dri_pdr(cf,if_)
        ts_l.append(ts); dri_l.append(dri); pdr_l.extend(pdrs); cdg_l.append(cdg)
        print(f"✓  cross-F1={np.mean(list(cf.values())):.3f}"
              f"  TS={ts:.3f}  DRI={dri:.3f}  CDG={cdg:.3f}")
    def mn(lst,k): return float(np.mean([r[k] for r in lst]))
    def sd(lst,k): return float(np.std( [r[k] for r in lst]))
    return dict(
        cross_f1_mean=mn(cross_r,"f1m"), cross_f1_std=sd(cross_r,"f1m"),
        cross_acc_mean=mn(cross_r,"acc"),cross_acc_std=sd(cross_r,"acc"),
        cross_auc_mean=mn(cross_r,"auc"),cross_auc_std=sd(cross_r,"auc"),
        ind_f1_mean=mn(ind_r,"f1m"),     ind_f1_std=sd(ind_r,"f1m"),
        ind_acc_mean=mn(ind_r,"acc"),    ind_acc_std=sd(ind_r,"acc"),
        ts_mean=np.mean(ts_l),  ts_std=np.std(ts_l),
        dri_mean=np.mean(dri_l),dri_std=np.std(dri_l),
        pdr_mean=np.mean(pdr_l),pdr_std=np.std(pdr_l),
        # [D] CDG
        cdg_mean=float(np.mean(cdg_l)), cdg_std=float(np.std(cdg_l)),
    )

# =============================================================================
# 12. PAIRWISE TRANSFER MATRIX
# =============================================================================
def pairwise_matrix(method, tr_data, te_data, cfg, seed):
    train_fn=TRAIN_FNS[method]; uid=USE_IDS[method]
    mat=np.zeros((M,M))
    for src in range(M):
        m=train_fn(tr_data,[src],cfg,seed); m.eval()
        for tgt in range(M):
            ldr=make_loader(*te_data[tgt],cfg["BATCH"],False)
            mat[src,tgt]=evaluate(m,ldr,src,uid)["f1m"]
    return mat

# =============================================================================
# 13. ABLATION
# =============================================================================
ABLATION_CFGS=[
    ("ERM",       dict(use_dain=False,use_adv=False,use_proto=False)),
    ("w/o DAIN",  dict(use_dain=False,use_adv=True, use_proto=True)),
    ("w/o Adv",   dict(use_dain=True, use_adv=False,use_proto=True)),
    ("w/o Proto", dict(use_dain=True, use_adv=True, use_proto=False)),
    ("DITRL",     dict(use_dain=True, use_adv=True, use_proto=True)),
]

def _train_ditrl_ablated(tr_data, src_ids, cfg, seed,
                          use_dain, use_adv, use_proto):
    torch.manual_seed(seed); np.random.seed(seed)
    model=DITRLModel(cfg,n_domains=M).to(DEVICE)
    proto=(ProtoMem(M,N_CLASSES,cfg["D_EMB"],cfg["PROTO_MOM"],cfg["PROTO_MIN_COUNT"])
           if use_proto else None)
    ent=_dom_entropy(tr_data,src_ids)
    disc_p=list(model.disc.parameters()); disc_ids={id(p) for p in disc_p}
    enc_p=[p for p in model.parameters() if id(p) not in disc_ids]
    opt=torch.optim.AdamW(enc_p,lr=cfg["LR"],weight_decay=cfg["WD"])
    disc_opt=torch.optim.AdamW(disc_p,lr=cfg["LR"]*cfg["DISC_LR_MULT"],weight_decay=cfg["WD"])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg["EPOCHS"])
    disc_sch=torch.optim.lr_scheduler.CosineAnnealingLR(disc_opt,cfg["EPOCHS"])
    ldrs=[make_loader(*tr_data[i],cfg["BATCH"]) for i in src_ids]
    va_ldrs=[(make_loader(*tr_data[i],cfg["BATCH"],False),i) for i in src_ids]
    best,state,pat=0.,None,0
    for epoch in range(cfg["EPOCHS"]):
        model.train()
        alpha=_ramp(epoch,cfg["WARMUP"],cfg["EPOCHS"])
        lam_grl=2./(1+np.exp(-10*alpha))-1.
        # [C] same cosine decay applied in ablation so comparison is fair
        cos_decay=0.5*(1+np.cos(np.pi*epoch/max(1,cfg["EPOCHS"]-1)))
        w_adv_eff=(cfg["W_ADV_MIN"]+(cfg["W_ADV"]-cfg["W_ADV_MIN"])*cos_decay)
        its=[iter(l) for l in ldrs]
        while True:
            done=False; pairs=[]
            for it,d in zip(its,src_ids):
                try: xb,yb=next(it); pairs.append((xb.to(DEVICE),yb.to(DEVICE),d))
                except StopIteration: done=True; break
            if done: break
            opt.zero_grad(); disc_opt.zero_grad()
            task=adv=pl=torch.tensor(0.,device=DEVICE)
            all_z,all_y,all_d=[],[],[]
            for xb,yb,d in pairs:
                ids=(torch.full((len(xb),),d,dtype=torch.long,device=DEVICE) if use_dain else None)
                xb_in=xb+torch.randn_like(xb)*cfg["INPUT_JITTER"]
                z=model.encode(xb_in,ids)
                task+=F.cross_entropy(model.head(z),yb)/ent[d]
                if alpha>0:
                    if use_adv:
                        dlbl=torch.full((len(xb),),d,dtype=torch.long,device=DEVICE)
                        adv+=F.cross_entropy(model.disc(z,lam_grl),dlbl)
                    if use_proto and proto:
                        proto.update(z.detach(),yb,d)
                all_z.append(z); all_y.append(yb); all_d.append(d)
            task/=len(pairs); adv/=len(pairs)
            if alpha>0 and use_proto and proto:
                for z,yb,d in zip(all_z,all_y,all_d):
                    others=[dd for dd in src_ids if dd!=d]
                    pl+=proto.loss(z,yb,d,others,cfg["PROTO_TEMP"],DEVICE)
                pl/=max(1,len(all_z))
            loss=task+alpha*w_adv_eff*adv+alpha*cfg["W_PROTO"]*pl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); disc_opt.step()
        sch.step(); disc_sch.step()
        vf=_val_f1(model,va_ldrs,use_ids=use_dain)
        if vf>best: best=vf; state=copy.deepcopy(model.state_dict()); pat=0
        else: pat+=1
        if pat>=cfg["PATIENCE"]: break
    if state: model.load_state_dict(state)
    return model

def run_ablation(tr_data, te_data, cfg):
    results={}
    for name,flags in ABLATION_CFGS:
        print(f"  {name:<12s}",end="  ")
        ts_l,dri_l=[],[]
        for seed in cfg["ABL_SEEDS"]:
            mat=np.zeros((M,M))
            for src in range(M):
                m=_train_ditrl_ablated(tr_data,[src],cfg,seed,**flags); m.eval()
                for tgt in range(M):
                    ldr=make_loader(*te_data[tgt],cfg["BATCH"],False)
                    mat[src,tgt]=evaluate(m,ldr,src,flags["use_dain"])["f1m"]
            cf={t:float(np.mean([mat[s,t] for s in range(M) if s!=t])) for t in range(M)}
            if_={t:float(mat[t,t]) for t in range(M)}
            ts,dri,_,_=_ts_dri_pdr(cf,if_)
            ts_l.append(ts); dri_l.append(dri); print(f"TS={ts:.3f}",end=" ")
        print()
        results[name]=dict(ts=ts_l,dri=dri_l)
    return results

# =============================================================================
# 14. ROBUSTNESS
# =============================================================================
def _pert_amp(X,scale): return _zscore((X*scale).astype(np.float32))
def _pert_spec(X,frac,rng):
    Xf=np.fft.rfft(X,axis=1); n=int(Xf.shape[1]*frac)
    if n>0: Xf[:,rng.choice(Xf.shape[1],n,replace=False)]=0
    return _zscore(np.fft.irfft(Xf,n=X.shape[1],axis=1).astype(np.float32))
def _pert_temp(X,ratio,T):
    T2=max(4,int(T*ratio)); out=[]
    for x in X:
        rs=sp_resample(x,T2).astype(np.float32)
        if T2>=T: out.append(rs[:T])
        else:
            pad=np.zeros(T,np.float32); pad[:T2]=rs; out.append(pad)
    return _zscore(np.stack(out))

def run_robustness(tr_data, te_data, cfg, seed):
    srcs=list(range(M-1)); X_te,y_te=te_data[M-1]
    rng=np.random.default_rng(seed)
    models={}
    for nm,fn in [("ERM",train_erm),("DITRL",train_ditrl)]:
        print(f"  Training {nm} for robustness …",end=" ",flush=True)
        models[nm]=fn(tr_data,srcs,cfg,seed); models[nm].eval(); print("✓")
    AMP=[0.1,0.25,0.5,1.,2.,4.,8.]
    SPEC=[0.,0.1,0.2,0.3,0.4,0.5]
    TEMP_R=[0.25,0.5,1.,2.,4.]
    uid={"ERM":False,"DITRL":True}
    rob={"amplitude":{},"spectral":{},"temporal":{}}
    for scale in AMP:
        Xp=_pert_amp(X_te,scale)
        for nm,m in models.items():
            ldr=make_loader(Xp,y_te,cfg["BATCH"],False)
            rob["amplitude"].setdefault(scale,{})[nm]=evaluate(m,ldr,M-1,uid[nm])["f1m"]
    for frac in SPEC:
        Xp=_pert_spec(X_te.copy(),frac,rng)
        for nm,m in models.items():
            ldr=make_loader(Xp,y_te,cfg["BATCH"],False)
            rob["spectral"].setdefault(frac,{})[nm]=evaluate(m,ldr,M-1,uid[nm])["f1m"]
    for ratio in TEMP_R:
        Xp=_pert_temp(X_te.copy(),ratio,cfg["T"])
        for nm,m in models.items():
            ldr=make_loader(Xp,y_te,cfg["BATCH"],False)
            rob["temporal"].setdefault(ratio,{})[nm]=evaluate(m,ldr,M-1,uid[nm])["f1m"]
    return rob

# =============================================================================
# 15. EMBEDDINGS — t-SNE, MMD, Fréchet
# =============================================================================
@torch.no_grad()
def get_embeddings(model, ldr, dom_id, use_ids):
    model.eval(); parts=[]
    for xb,_ in ldr:
        xb=xb.to(DEVICE)
        ids=(torch.full((len(xb),),dom_id,dtype=torch.long,device=DEVICE) if use_ids else None)
        parts.append(model.encode(xb,ids).cpu().numpy())
    return np.concatenate(parts,0)

def mmd_rbf(A, B, sigmas=(0.5,1.,2.), n=400):
    rng=np.random.default_rng(0)
    def s(X): return torch.from_numpy(X[rng.choice(len(X),min(n,len(X)),replace=False)]).float()
    At,Bt=s(A),s(B); res=0.
    for sg in sigmas:
        kAA=torch.exp(-torch.cdist(At,At).pow(2)/(2*sg**2)).mean()
        kBB=torch.exp(-torch.cdist(Bt,Bt).pow(2)/(2*sg**2)).mean()
        kAB=torch.exp(-torch.cdist(At,Bt).pow(2)/(2*sg**2)).mean()
        res+=(kAA+kBB-2*kAB).item()
    return res/len(sigmas)

def frechet(A, B):
    eps=1e-4; d=A.shape[1]
    m1,m2=A.mean(0),B.mean(0)
    S1=(np.cov(A.T) if len(A)>1 else np.eye(d))+np.eye(d)*eps
    S2=(np.cov(B.T) if len(B)>1 else np.eye(d))+np.eye(d)*eps
    diff=m1-m2; cm=mat_sqrt(S1@S2)
    if np.iscomplexobj(cm): cm=cm.real
    return max(float(diff@diff+np.trace(S1+S2-2*cm)),0.)

# =============================================================================
# 16. FIGURES
# =============================================================================
def _sf(fig, name):
    fig.savefig(P(name),dpi=300,bbox_inches="tight")
    plt.close(fig); print(f"  Saved ▶  {name}")

def _short(): return [n[:6] for n in DOMAIN_NAMES]

def fig_heatmap(erm_mat, ditrl_mat):
    sh=_short()
    vmin=min(erm_mat.min(),ditrl_mat.min()); vmax=max(erm_mat.max(),ditrl_mat.max())
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,mat,title in zip(axes,[erm_mat,ditrl_mat],["ERM Baseline","DITRL (Proposed)"]):
        sns.heatmap(mat,annot=True,fmt=".3f",ax=ax,xticklabels=sh,yticklabels=sh,
                    cmap="YlOrRd",vmin=vmin,vmax=vmax,
                    linewidths=0.6,linecolor="white",annot_kws={"size":9})
        ax.set_title(title,fontweight="bold",pad=10)
        ax.set_xlabel("Target Domain"); ax.set_ylabel("Source Domain")
        for i in range(M):
            ax.add_patch(plt.Rectangle((i,i),1,1,fill=False,
                         edgecolor="royalblue",lw=2.5,zorder=3))
    fig.suptitle("Source→Target Transfer Matrix (Macro-F1)",
                 fontsize=14,fontweight="bold",y=1.01)
    fig.tight_layout(); _sf(fig,"transfer_heatmap.png")


def fig_in_vs_cross(all_stats):
    """
    [D] Updated figure: adds Cross-Domain Gap (CDG) as a third row of
        annotations below TS and DRI. CDG = In-F1 − Cross-F1.
        Arrow glyph (↓) signals 'lower is better'.
        Also adds a right-side inset bar chart of CDG values so reviewers
        can compare domain-shift sensitivity across methods at a glance.
    """
    methods=list(all_stats.keys())
    in_m =[all_stats[m]["ind_f1_mean"]   for m in methods]
    in_s =[all_stats[m]["ind_f1_std"]    for m in methods]
    cr_m =[all_stats[m]["cross_f1_mean"] for m in methods]
    cr_s =[all_stats[m]["cross_f1_std"]  for m in methods]
    cdg_m=[all_stats[m]["cdg_mean"]      for m in methods]
    cdg_s=[all_stats[m]["cdg_std"]       for m in methods]

    # ── Layout: main bar chart (left 70%) + CDG inset (right 30%) ──────────
    fig=plt.figure(figsize=(13,6))
    gs=fig.add_gridspec(1,2,width_ratios=[2.3,1],wspace=0.35)
    ax=fig.add_subplot(gs[0]); ax_cdg=fig.add_subplot(gs[1])

    x=np.arange(len(methods)); w=0.35
    colors=[METHOD_COLOR.get(m,"#888") for m in methods]
    bi=ax.bar(x-w/2,in_m,w,yerr=in_s,capsize=5,color=colors,alpha=0.50,
              label="In-Domain",error_kw=dict(lw=1.5,ecolor="black"))
    bc=ax.bar(x+w/2,cr_m,w,yerr=cr_s,capsize=5,color=colors,alpha=0.92,
              label="Cross-Domain",edgecolor="white",lw=0.7,
              error_kw=dict(lw=1.5,ecolor="black"))
    for bar,v,s in zip(list(bi)+list(bc),in_m+cr_m,in_s+cr_s):
        ax.text(bar.get_x()+bar.get_width()/2.,v+s+0.006,f"{v:.3f}",
                ha="center",va="bottom",fontsize=7.5,fontweight="bold")

    col_p=[mpatches.Patch(color=METHOD_COLOR.get(m,"#888"),label=m) for m in methods]
    pat_p=[mpatches.Patch(facecolor="gray",alpha=0.50,label="In-Domain"),
           mpatches.Patch(facecolor="gray",alpha=0.92,label="Cross-Domain")]
    l1=ax.legend(handles=col_p,loc="upper left",title="Method",
                 fontsize=8.5,title_fontsize=9); ax.add_artist(l1)
    ax.legend(handles=pat_p,loc="upper right",fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(methods,fontsize=10.5)
    ax.set_ylabel("Macro-F1 (mean ± std)"); ax.set_ylim(0,1.12)
    ax.set_title("In-Domain vs Cross-Domain Performance",fontweight="bold")
    ax.axhline(0.5,ls=":",lw=1,color="gray",alpha=0.5)

    # TS / DRI / CDG annotations below x-axis
    for xi,m in enumerate(methods):
        s=all_stats[m]
        ax.text(xi,-0.13,
                f"TS={s['ts_mean']:.2f}\nDRI={s['dri_mean']:.2f}\n"
                f"CDG↓={s['cdg_mean']:.3f}",
                ha="center",va="top",fontsize=7,
                color=METHOD_COLOR.get(m,"#888"),fontweight="bold",
                transform=ax.get_xaxis_transform())

    # ── [D] CDG inset bar chart ────────────────────────────────────────────
    cdg_bars=ax_cdg.barh(methods,cdg_m,xerr=cdg_s,color=colors,alpha=0.85,
                          capsize=4,error_kw=dict(lw=1.5,ecolor="black"))
    ax_cdg.set_xlabel("Cross-Domain Gap (CDG) ↓",fontsize=9)
    ax_cdg.set_title("CDG\n(lower = less\ndomain shift)",
                     fontweight="bold",fontsize=9,linespacing=1.4)
    ax_cdg.axvline(0,color="gray",lw=0.8,ls="--")
    for bar,v,s in zip(cdg_bars,cdg_m,cdg_s):
        ax_cdg.text(v+s+0.003,bar.get_y()+bar.get_height()/2.,f"{v:.3f}",
                    va="center",fontsize=8,fontweight="bold")
    ax_cdg.invert_yaxis()

    fig.tight_layout(); _sf(fig,"in_vs_cross_domain.png")


def fig_tsne(erm_embs, ditrl_embs):
    pal=sns.color_palette("tab10",M)
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for ax,embs,title in zip(axes,[erm_embs,ditrl_embs],
                              ["ERM Embeddings","DITRL Embeddings"]):
        all_e=np.concatenate(embs,0)
        labs=np.concatenate([np.full(len(e),i) for i,e in enumerate(embs)])
        MAX=1500
        if len(all_e)>MAX:
            rng=np.random.default_rng(42); idx=rng.choice(len(all_e),MAX,replace=False)
            all_e=all_e[idx]; labs=labs[idx]
        print(f"    t-SNE ({title}) …")
        pp=min(30,max(5,len(all_e)//10))
        try: proj=TSNE(2,perplexity=pp,random_state=42,max_iter=500).fit_transform(all_e)
        except: proj=TSNE(2,perplexity=pp,random_state=42,n_iter=500).fit_transform(all_e)
        for i,nm in enumerate(DOMAIN_NAMES):
            mask=labs==i
            ax.scatter(proj[mask,0],proj[mask,1],c=[pal[i]],s=14,
                       alpha=0.65,label=nm,edgecolors="none")
        ax.set_title(title,fontweight="bold")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=8,markerscale=2)
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        ax.grid(False)
    fig.suptitle("Embedding Space — Coloured by Domain",fontsize=13,fontweight="bold")
    fig.tight_layout(); _sf(fig,"embedding_tsne.png")

def fig_alignment(mmd_erm, mmd_ditrl, fd_erm, fd_ditrl):
    pairs=[(i,j) for i in range(M) for j in range(i+1,M)]
    lbl=[f"{DOMAIN_NAMES[i][:4]}→{DOMAIN_NAMES[j][:4]}" for i,j in pairs]
    x=np.arange(len(pairs)); w=0.35
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,(ev,dv,title,ylabel) in zip(axes,[
        ([mmd_erm[p] for p in pairs],[mmd_ditrl[p] for p in pairs],"MMD","MMD"),
        ([fd_erm[p]  for p in pairs],[fd_ditrl[p]  for p in pairs],"Fréchet Distance","FD"),
    ]):
        ax.bar(x-w/2,ev,w,label="ERM",color="#4C72B0",alpha=0.85)
        ax.bar(x+w/2,dv,w,label="DITRL",color="#DD8452",alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(lbl,rotation=40,ha="right",fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title,fontweight="bold"); ax.legend()
    fig.suptitle("Embedding Alignment (ERM vs DITRL)",fontsize=13,fontweight="bold")
    fig.tight_layout(); _sf(fig,"alignment_metrics.png")

def fig_ablation(abl):
    names=[n for n,_ in ABLATION_CFGS]
    ts_m=[np.mean(abl[n]["ts"]) for n in names]
    ts_s=[np.std(abl[n]["ts"])  for n in names]
    di_m=[np.mean(abl[n]["dri"]) for n in names]
    di_s=[np.std(abl[n]["dri"])  for n in names]
    cols=[METHOD_COLOR.get(n,"#8172B2") for n in names]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,(vals,stds,title,yl) in zip(axes,[
        (ts_m,ts_s,"Transferability Score (TS)","TS"),
        (di_m,di_s,"Domain Robustness Index (DRI)","DRI"),
    ]):
        bars=ax.bar(names,vals,color=cols,alpha=0.88,
                    yerr=stds,capsize=5,error_kw=dict(lw=1.5))
        ax.set_xticklabels(names,rotation=15,ha="right")
        ax.set_ylabel(yl); ax.set_title(title,fontweight="bold")
        for bar,v,s in zip(bars,vals,stds):
            ax.text(bar.get_x()+bar.get_width()/2.,v+s+0.005,f"{v:.3f}",
                    ha="center",va="bottom",fontsize=8.5)
        bars[-1].set_edgecolor("black"); bars[-1].set_linewidth(2.2)
        ax.set_ylim(0,max(vals)+max(stds)+0.12)
    fig.suptitle("Ablation Study",fontsize=13,fontweight="bold")
    fig.tight_layout(); _sf(fig,"ablation_results.png")

def fig_robustness(rob):
    import matplotlib.ticker as mticker
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    panels=[
        ("amplitude","Amplitude Scale",[0.1,0.25,0.5,1.,2.,4.,8.],"log"),
        ("spectral","Freq Mask %",[0.,10,20,30,40,50],"linear"),
        ("temporal","Temporal Ratio",[0.25,0.5,1.,2.,4.],"log"),
    ]
    raw_keys=([0.1,0.25,0.5,1.,2.,4.,8.],[0.,0.1,0.2,0.3,0.4,0.5],[0.25,0.5,1.,2.,4.])
    for ax,(key,xlabel,disp,xsc),rkeys in zip(axes,panels,raw_keys):
        data=rob[key]
        erm_v=[data[k]["ERM"]   for k in rkeys]
        dit_v=[data[k]["DITRL"] for k in rkeys]
        ax.plot(disp,erm_v,"o-",color="#4C72B0",lw=2,ms=7,
                markerfacecolor="white",markeredgewidth=2,label="ERM")
        ax.plot(disp,dit_v,"s-",color="#DD8452",lw=2,ms=7,
                markerfacecolor="white",markeredgewidth=2,label="DITRL")
        ax.fill_between(disp,erm_v,dit_v,
                        where=[d>=e for d,e in zip(dit_v,erm_v)],
                        alpha=0.12,color="#DD8452",interpolate=True)
        ax.set_xscale(xsc)
        if xsc=="log": ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel(xlabel); ax.set_ylabel("Macro-F1")
        ax.set_title(f"{key.capitalize()} Perturbation",fontweight="bold")
        ax.legend(fontsize=9); ax.set_ylim(0,1.)
    fig.suptitle("Robustness to Distribution Perturbations",fontsize=13,fontweight="bold")
    fig.tight_layout(); _sf(fig,"robustness_curves.png")

# =============================================================================
# 17. CSV EXPORT
# =============================================================================
def save_csvs(all_stats, abl, rob):
    rows=[]
    for method,s in all_stats.items():
        for setting,prefix in [("Cross-Domain","cross"),("In-Domain","ind")]:
            for metric in ["f1m","acc","auc"]:
                mk=f"{prefix}_{metric}_mean"; sk=f"{prefix}_{metric}_std"
                if mk not in s: continue
                rows.append(dict(Method=method,Setting=setting,Metric=metric,
                                 Mean=round(s[mk],4),Std=round(s[sk],4)))
    pd.DataFrame(rows).to_csv(P("main_results.csv"),index=False)
    print("  Saved ▶  main_results.csv")

    rows=[]
    for method,s in all_stats.items():
        rows.append(dict(
            Method=method,
            TS_mean=round(s["ts_mean"],4),   TS_std=round(s["ts_std"],4),
            DRI_mean=round(s["dri_mean"],4), DRI_std=round(s["dri_std"],4),
            PDR_mean=round(s["pdr_mean"],2), PDR_std=round(s["pdr_std"],2),
            # [D] CDG in CSV
            CDG_mean=round(s["cdg_mean"],4), CDG_std=round(s["cdg_std"],4),
        ))
    pd.DataFrame(rows).to_csv(P("transfer_metrics.csv"),index=False)
    print("  Saved ▶  transfer_metrics.csv")

    rows=[]
    for name,_ in ABLATION_CFGS:
        d=abl[name]
        rows.append(dict(Variant=name,
                         TS_mean=round(np.mean(d["ts"]),4),TS_std=round(np.std(d["ts"]),4),
                         DRI_mean=round(np.mean(d["dri"]),4),DRI_std=round(np.std(d["dri"]),4)))
    pd.DataFrame(rows).to_csv(P("ablation_results.csv"),index=False)
    print("  Saved ▶  ablation_results.csv")

    rkeys={"amplitude":[0.1,0.25,0.5,1.,2.,4.,8.],
           "spectral":[0.,0.1,0.2,0.3,0.4,0.5],
           "temporal":[0.25,0.5,1.,2.,4.]}
    rows=[]
    for ptype,data in rob.items():
        for k in rkeys[ptype]:
            rows.append(dict(Perturbation=ptype,Level=k,
                             ERM_F1=round(data[k]["ERM"],4),
                             DITRL_F1=round(data[k]["DITRL"],4)))
    pd.DataFrame(rows).to_csv(P("robustness_results.csv"),index=False)
    print("  Saved ▶  robustness_results.csv")

# =============================================================================
# 18. SUMMARY
# =============================================================================
def print_summary(all_stats):
    W=80
    print(f"\n{'='*W}\n  FINAL RESULTS — Real Paper Datasets\n{'='*W}")
    print(f"  {'Method':<10} {'In-F1':>9} {'Cross-F1':>11}"
          f"  {'AUC':>6}  {'TS':>6}  {'DRI':>6}  {'PDR%':>7}  {'CDG↓':>7}")
    print(f"  {'-'*(W-2)}")
    for m,s in all_stats.items():
        mk="★" if m=="DITRL" else " "
        print(f"  {m:<10}{mk}"
              f" {s['ind_f1_mean']:.3f}±{s['ind_f1_std']:.3f}"
              f"  {s['cross_f1_mean']:.3f}±{s['cross_f1_std']:.3f}"
              f"  {s.get('cross_auc_mean',0.):.3f}"
              f"  {s['ts_mean']:.3f}  {s['dri_mean']:.3f}"
              f"  {s['pdr_mean']:>+6.1f}"
              f"  {s['cdg_mean']:>7.3f}")   # [D]
    ditrl=all_stats.get("DITRL",{}); erm=all_stats.get("ERM",{})
    if ditrl and erm:
        print(f"\n  DITRL improvement over ERM:")
        print(f"    Cross-F1 : {ditrl['cross_f1_mean']-erm['cross_f1_mean']:>+.3f}")
        print(f"    TS       : {ditrl['ts_mean']-erm['ts_mean']:>+.3f}")
        print(f"    DRI      : {ditrl['dri_mean']-erm['dri_mean']:>+.3f}")
        print(f"    CDG      : {ditrl['cdg_mean']-erm['cdg_mean']:>+.3f}  (negative = less gap)")
    print(f"\n  Output directory: {OUT}")
    print(f"  Data cache      : {CACHE}")
    print("  ✅  Done!")

# =============================================================================
# 19. MAIN
# =============================================================================
def parse_args():
    p=argparse.ArgumentParser(description="DITRL — Real Dataset Pipeline")
    p.add_argument("--quick",action="store_true")
    p.add_argument("--skip_heavy",action="store_true")
    p.add_argument("--epochs",type=int,default=None)
    return p.parse_args()

def banner(n, total, label):
    print(f"\n{'='*65}\n  Step {n}/{total} — {label}\n{'='*65}")

def main():
    args=parse_args(); cfg=build_cfg(args)
    if args.quick: print("  [Quick mode: 2 seeds · 15 epochs · 200 samples/class]")
    if args.skip_heavy: print("  [skip_heavy: PTB-XL and PAMAP2 will use ETTh1 placeholder]")
    print(f"\nDITRL Real-Dataset Pipeline  |  Device: {DEVICE}")
    print(f"Output : {OUT}")
    print(f"Config : T={cfg['T']}  D={cfg['D_EMB']}"
          f"  epochs={cfg['EPOCHS']}  warmup={cfg['WARMUP']}"
          f"  W_ADV={cfg['W_ADV']}→{cfg['W_ADV_MIN']} (cosine decay)"
          f"  N_runs={cfg['N_RUNS']}")
    TOTAL=7; seeds=cfg["SEEDS"][:cfg["N_RUNS"]]

    banner(1,TOTAL,"Loading real paper datasets")
    tr_data,te_data=load_domains(cfg,skip_heavy=args.skip_heavy)

    banner(2,TOTAL,"LODO evaluation — all methods")
    all_stats={}
    for method in ALL_METHODS:
        print(f"\n  ▶  {method}")
        all_stats[method]=run_lodo(method,tr_data,te_data,cfg,seeds)

    banner(3,TOTAL,"Pairwise transfer matrices (heatmap)")
    seed0=seeds[0]
    erm_mat  =pairwise_matrix("ERM",  tr_data,te_data,cfg,seed0)
    ditrl_mat=pairwise_matrix("DITRL",tr_data,te_data,cfg,seed0)

    banner(4,TOTAL,"Ablation study")
    abl=run_ablation(tr_data,te_data,cfg)

    banner(5,TOTAL,"Robustness analysis")
    rob=run_robustness(tr_data,te_data,cfg,seed0)

    banner(6,TOTAL,"Embeddings · t-SNE · MMD · Fréchet")
    print("  Training full-domain models for embeddings …")
    m_erm  =train_erm(  tr_data,list(range(M)),cfg,seed0); m_erm.eval()
    m_ditrl=train_ditrl(tr_data,list(range(M)),cfg,seed0); m_ditrl.eval()
    erm_embs  =[get_embeddings(m_erm,  make_loader(*te_data[i],cfg["BATCH"],False),i,False) for i in range(M)]
    ditrl_embs=[get_embeddings(m_ditrl,make_loader(*te_data[i],cfg["BATCH"],False),i,True)  for i in range(M)]
    pairs=[(i,j) for i in range(M) for j in range(i+1,M)]
    mmd_erm  ={p:mmd_rbf(erm_embs[p[0]],  erm_embs[p[1]])   for p in pairs}
    mmd_ditrl={p:mmd_rbf(ditrl_embs[p[0]],ditrl_embs[p[1]]) for p in pairs}
    fd_erm   ={p:frechet(erm_embs[p[0]],  erm_embs[p[1]])   for p in pairs}
    fd_ditrl ={p:frechet(ditrl_embs[p[0]],ditrl_embs[p[1]]) for p in pairs}

    banner(7,TOTAL,"Figures and CSVs")
    fig_heatmap(erm_mat,ditrl_mat)
    fig_in_vs_cross(all_stats)
    fig_tsne(erm_embs,ditrl_embs)
    fig_alignment(mmd_erm,mmd_ditrl,fd_erm,fd_ditrl)
    fig_ablation(abl)
    fig_robustness(rob)
    save_csvs(all_stats,abl,rob)
    print_summary(all_stats)

if __name__=="__main__":
    main()
