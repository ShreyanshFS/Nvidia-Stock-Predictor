import os 
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data. timeframe import TimeFrame
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

root=Path(__file__).resolve().parent
out= root/"outputs"
out.mkdir(exist_ok=True)

symbol,bench ="NVDA" ,"SPY"
horizon =63
years=10
test_frac= 00.2
def fetch_bars(symbol):                                                          #data downloadin' fnc
    cache =out/F"{symbol}_bars.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    load_dotenv(root/".env")
    client=StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    )
    now =datetime.now(timezone.utc)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=now - timedelta(days=365* years),
        end=now - timedelta(days=1),
        adjustment= Adjustment.ALL,
        feed=DataFeed.IEX
    )
    df=client.get_stock_bars(req).df.reset_index(level=0,drop=True)
    df.index= pd.to_datetime(df.index).tz_localize(None).normalize()
    df=df[["open","high","low","close","volume"]].sort_index()
    df.to_csv(cache)
    return df

nvda= fetch_bars(symbol)
spy =fetch_bars(bench)

def rsi (close, n=14):                                                           #momentum indicator 
    delta =close.diff()
    gain= delta.clip(lower=0).rolling(n).mean()
    loss=(-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100/(1+ gain/loss)

def build_dataset(bars,bench):                                                   #format data for ip
    c,v=bars["close"],bars["volume"]
    b = bench["close"].reindex(bars.index).ffill()
    f=pd.DataFrame(index=bars.index)

    for w in (5,21,63):
        f[f"ret_{w}"] = c.pct_change(w)
        f[f"spy_ret_{w}"]=b.pct_change(w)
    f["rel_21"]=f["ret_21"]-f["spy_ret_21"]

    r1 = c.pct_change()
    f["vol_21"]=r1.rolling(21).std()
    f["vol_63"]= r1.rolling(63).std()
    f["drawdown_63"]=c/c.rolling(63).max()-1

    f["rsi_14"]=rsi(c)
    f["dist_ma50"]= c/ c.rolling(50).mean()-1
    f["macd"]=(c.ewm(span=12).mean()-c.ewm(span=26).mean())/c
    f["vol_ratio"]=v/v.rolling(21).mean()
    f["close"]=c
    f["target_ret"]= c.shift(-horizon)/c-1
    f["target_date"]=pd.Series(c.index,index =c.index).shift(-horizon)
    return f.replace([np.inf,-np.inf],np.nan)

data =build_dataset( nvda,spy)
print(data.shape)
print(data.tail(3).T)
print("Rows w Target: ",data["target_ret"].notna().sum())

feature_cols = [c for c in data.columns if c not in(                             #feature columns
    "close","target_ret","target_date")]
labeled = data.dropna(subset=feature_cols + ["target_ret"])
latest= data.iloc[[-1]]

split = int(len(labeled)*(1-test_frac))                                          #time split
train = labeled.iloc[:split-horizon]
test = labeled.iloc[split:]
print("train:", len(train),"| test:", len(test))

x_tr,y_tr= train[feature_cols], train["target_ret"]                             #train 
x_te,y_te= test[feature_cols], test["target_ret"]                               #test  

preds = {"No-change baseline": np.zeros(len(test))}                             #models
models={
    "Linear model": make_pipeline(StandardScaler(),LinearRegression()),
    "Gradient boosting":HistGradientBoostingRegressor(
        max_iter=200,learning_rate=0.05,max_depth=3,random_state=42
    ),
}
for name, m in models.items():
    m.fit(x_tr,y_tr)
    preds[name]=m.predict(x_te)

print("\nMAE (pp,lower is better):")                                             #error check
for name, p in preds.items():
    print(f"{name}: {100* mean_absolute_error(y_te,p):.2f}")

last_close = float(data["close"].iloc[-1])                                       #month forecast
last_date = data.index[-1]
target_day = last_date +pd.offsets.BDay(horizon)
print(f"\nLast close({last_date.date()}):{last_close:.2f}")
print(f"Forecast for ~{target_day.date()}")
print(f" No-Change baseline: {last_close:.2f}")
for name, m in models.items():
    m.fit(labeled[feature_cols], labeled["target_ret"])
    ret = float(m.predict(latest[feature_cols])[0])
    print(f" {name}:{last_close*(1+ret):>8.2f} ({ret*100:.1f}%)")

pred_rows= data.tail(3).copy()                                                   # fill empty rows with prediction 
m = models["Gradient boosting"]
pred_rows["target_ret"] = m.predict(pred_rows[feature_cols])
pred_rows["target_date"] = pred_rows.index + pd.offsets.BDay(horizon)
print("\nPredicted 3-month return:")
print(pred_rows.T)