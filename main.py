import json
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
import matplotlib.pyplot as plt

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

feature_cols = [c for c in data.columns if c not in(                             #feature columns
    "close","target_ret","target_date")]
labeled = data.dropna(subset=feature_cols + ["target_ret"])
latest= data.iloc[[-1]]

split = int(len(labeled)*(1-test_frac))                                          #time split
train = labeled.iloc[:split-horizon]
test = labeled.iloc[split:]

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

def show(title,df):                                                            #reports 
    print(f"/n=== {title} ===")
    print(df.to_string())

def feat_table(df):
    t=df[feature_cols+["close"]].copy()
    t.index = t.index.strftime("%y-%m-%d")
    return t.T.apply(lambda c: c.map("{:.4f}".format))

overview =pd.DataFrame(                                                        #data Summary
    {"value":[data.shape[0],data.shape[1], int(data["target_ret"].notna().sum()),len(train),len(test)]},
    index= ["Total rows", "Columns","Rows w Target","Train rows","Test row"])
show("Dataset",overview)

show("Latest 3 days", feat_table(data.tail(3)))                                #latest feature

mae= pd.DataFrame({"MAE (pp)":{n: round(100 * mean_absolute_error(y_te,p),2)   #error table
                                for n,p in preds.items()}})

last_close = float(data["close"].iloc[-1])                                      #3 month forecast
last_date = data.index[-1]
target_day = last_date + pd.offsets.BDay(horizon)
forecasts = {"No-Change baseline": last_close}
for name,m in models.items():
    m.fit(labeled[feature_cols],labeled["target_ret"])
    ret = float(m.predict(latest[feature_cols])[0])
    forecasts[name] = last_close * (1+ret)
fc = pd.DataFrame({"Price ($)": forecasts})
fc["change (%)"]=(fc["Price ($)"] / last_close -1)*100
show(f"Forecast for ~{target_day.date()} (last close {last_close:.2f})" , fc.round(2))

pred_rows= data.tail(3).copy()                                                   #predicted rows
pred_rows["target_ret"] = models["Gradient boosting"].predict(pred_rows[feature_cols])
pred_rows["target_date"] = pred_rows.index + pd.offsets.BDay(horizon)
tbl= feat_table(pred_rows)
tbl.loc["pred_return"] = [f"{x*100:+.1f}%" for x in pred_rows["target_ret"]]
tbl.loc["target_date"] = pred_rows["target_date"].dt.strftime("%Y-%m-%d").tolist()
show("Predicted 3-Month return (Gradient boosting)", tbl)

report={                                                                            #saved Report
    "last_close": round(last_close,2),
    "forecast_date":str(target_day.date()),
    "mae_pp":mae["MAE (pp)"].to_dict(),
    "forecast_price": fc["Price ($)"].round(2).to_dict(),
}
with open(out/"report.json","w") as f:
    json.dump(report,f,indent=2)
print("\nSaved",out/"report.json")

choice = input ("\nShow Charts?(y/n): ").strip().lower()                            #ask for chart
if choice == 'y':
    colors = ["tab:green","tab:blue","tab:orange"]
    fig = plt.figure(figsize=(14,8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,2,3)
    ax3 = fig.add_subplot(2,2,4)

    dates = pd.to_datetime(test["target_date"])                                     #chart 1 : actual vs predicted
    ax1.plot(dates,test["close"]*(1+y_te),"k",lw=2,label="Actual")
    for name , p in preds.items():
        ax1.plot(dates,test["close"]*(1+p),lw=1.2,alpha=0.85,label =name)
    ax1.set_title("Predicted v/s actual price by date")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(alpha=.3)

    bars=ax2.barh(mae.index,mae["MAE (pp)"], color = colors)                         #chart 2: error
    ax2.bar_label(bars, fmt="%.2f",padding=3)
    ax2.set_title("Avg error (pp), lower is better")
    ax2.invert_yaxis()

    bars=ax3.bar(fc.index,fc["Price ($)"],color=colors)                              #Chart 3: 3 Month Prediction
    ax3.bar_label(bars,fmt="%.2f",padding=3)
    ax3.axhline(last_close,color="grey",ls="--",lw=1)
    ax3.set_title("3-month price forecast")
    ax3.tick_params(axis="x",labelrotation=15)

    fig.tight_layout()
    plt.show()