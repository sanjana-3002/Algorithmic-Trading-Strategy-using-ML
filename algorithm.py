import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use("dark_background")

apple = pd.read_csv("/Users/sanjanawaghray/Downloads/AAPL.csv")
apple.head(5)

# visualizing the data 
plt.figure(figsize=(12, 5))
plt.plot(apple['Adj Close Price'], label='Apple')
plt.title('Apple Adj Close Price History')
plt.xlabel("May 27,2014 - May 25,2020 ")
plt.ylabel("Adj Close Price USD ($)")
plt.legend(loc="upper left")
plt.show()

# creating moving averages for 30-day window
sma30 = pd.DataFrame()
sma30['Adj Close Price'] = apple['Adj Close Price'].rolling(window=30).mean()
sma30

# creating moving averages for 100-day window
sma100 = pd.DataFrame()
sma100['Adj Close Price'] = apple['Adj Close Price'].rolling(window=100).mean()
sma100

# visualizing the new data 
plt.figure(figsize=(12,5))
plt.plot(apple['Adj Close Price'], label='Apple')
plt.plot(sma30['Adj Close Price'], label='SMA30')
plt.plot(sma100['Adj Close Price'], label='SMA100')
plt.title("Apple Adj. Close Price History")
plt.xlabel('May 27,2014 - May 25,2020')
plt.ylabel('Adj. Close Price USD($)')
plt.legend(loc='upper left')
plt.show()

data = pd.DataFrame()
data['apple'] = apple['Adj Close Price']
data['SMA30'] = sma30['Adj Close Price']
data['SMA100'] = sma100['Adj Close Price']
data

# Trying to create a function to really understand when to buy and sell the stock
def buySell(data):
  sigPriceBuy = []
  sigPriceSell = []
  flag = -1
  for i in range(len(data)):
    if data ['SMA30'][i] > data['SMA100'][i]:
      if flag != 1:
        sigPriceBuy.append(data['apple'][i])
        sigPriceSell.append(np.nan)
        flag = 1
      else:
        sigPriceBuy.append(np.nan)
        sigPriceSell.append(np.nan)
    elif data['SMA30'][i] < data['SMA100'][i]:
      if flag != 0:
        sigPriceBuy.append(np.nan)
        sigPriceSell.append(data['apple'][i])
        flag = 0
      else:
        sigPriceBuy.append(np.nan)
        sigPriceSell.append(np.nan)
    else:
      sigPriceBuy.append(np.nan)
      sigPriceSell.append(np.nan)
  return(sigPriceBuy, sigPriceSell)

buySell = buySell(data)
data['Buy Signal Price'] = buySell[0]
data['Sell Signal Price'] = buySell[1]
# To show the data
data

# Finally, understanding when to buy and sell it visually
plt.style.use('classic')
plt.figure(figsize=(12,5))
plt.plot(data['apple'], label='Apple', alpha=0.35)
plt.plot(data['SMA30'], label='SMA30', alpha=0.35)
plt.plot(data['SMA100'],label='SMA100', alpha=0.35)
plt.scatter(data.index, data['Buy Signal Price'], label ='Buy', marker='^',color='green')
plt.scatter(data.index, data['Sell Signal Price'],label='Sell', marker='v', color='red')
plt.title('Apple Adj. Close Price History Buy and Sell Signals')
plt.xlabel("May 27,2014 - May 25,2020")
plt.ylabel("Adj Close Price USD($)")
plt.legend(loc='upper left')
plt.show()


# Building Metrics

def compute_metrics(equity, rf_annual=0.0, freq=252):
    equity = equity.dropna()
    if equity.size < 2:
        return {"Total Return": np.nan, "CAGR": np.nan, "Vol (ann.)": np.nan,
                "Sharpe": np.nan, "MaxDD": np.nan, "Years": 0.0}
    rets = equity.pct_change().dropna()
    n_years = max(1e-9, len(equity) / freq)
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = float((equity.iloc[-1] / equity.iloc[0])**(1/n_years) - 1)
    vol_ann = float(rets.std() * np.sqrt(freq))
    sharpe = np.nan if vol_ann == 0 else float((rets.mean()*freq - rf_annual) / vol_ann)
    dd = equity / equity.cummax() - 1
    max_dd = float(dd.min())
    return {"Total Return": total_ret, "CAGR": cagr, "Vol (ann.)": vol_ann,
            "Sharpe": sharpe, "MaxDD": max_dd, "Years": float(n_years)}

def trade_stats(signals, prices):
    # signals should be the traded (shifted) 0/1 exposure
    flips = signals.diff().fillna(0)
    entries = flips[flips > 0].index
    exits   = flips[flips < 0].index
    exits_all = list(exits)
    if len(exits_all) < len(entries):
        exits_all += [prices.index[-1]]
    tr = []
    for en, ex in zip(entries, exits_all):
        if en < ex:
            tr.append(float(prices.loc[ex] / prices.loc[en] - 1))
    if len(tr) == 0:
        return {"#Trades": 0, "WinRate": np.nan, "AvgGain": np.nan, "AvgLoss": np.nan}
    tr = pd.Series(tr, dtype="float64")
    return {
        "#Trades": int(len(tr)),
        "WinRate": float((tr > 0).mean()),
        "AvgGain": float(tr[tr > 0].mean()) if (tr > 0).any() else np.nan,
        "AvgLoss": float(tr[tr <= 0].mean()) if (tr <= 0).any() else np.nan
    }

# Indicators
def sma(s, w): 
    return s.rolling(int(w)).mean()

def ema(s, span): 
    return s.ewm(span=int(span), adjust=False).mean()

def rsi(s, period=14):
    d = s.diff()
    up = d.clip(lower=0); down = -d.clip(upper=0)
    rs = up.ewm(alpha=1/period, adjust=False).mean() / (down.ewm(alpha=1/period, adjust=False).mean() + 1e-12)
    return 100 - 100/(1+rs)

def vol_lookback(prices, window=20):
    return prices.pct_change().rolling(int(window)).std()

# ---------- STRATEGY ----------
def crossover_strategy(
    df, price_col,
    fast=30, slow=100, use_ema=False,
    use_regime_200=False,          # only trade when price > 200SMA
    use_rsi_filter=False, rsi_period=14,  # require RSI > 50 to enter
    vol_target=False, vol_window=20,      # inverse-vol sizing
    cost_bps=5
):
    px = df[price_col].astype(float).copy()
    out = pd.DataFrame(index=df.index, data={"price": px})

    fast_ma = ema(px, fast) if use_ema else sma(px, fast)
    slow_ma = ema(px, slow) if use_ema else sma(px, slow)
    out["fast"], out["slow"] = fast_ma, slow_ma

    # Base signal: long when fast > slow
    signal_raw = (out["fast"] > out["slow"]).astype(int)

    if use_regime_200:
        regime = (sma(px, 200) < px).astype(int)  # risk-on when price above 200SMA
        signal_raw = signal_raw * regime

    if use_rsi_filter:
        out["rsi"] = rsi(px, rsi_period)
        signal_raw = signal_raw * (out["rsi"] > 50).astype(int)

    # Trade next bar (no look-ahead)
    signal_traded = signal_raw.shift(1).fillna(0).astype(float)

    # Position sizing
    if vol_target:
        vol = vol_lookback(px, vol_window)
        inv_vol = 1 / vol.replace(0, np.nan)
        # Normalize & cap; if too few non-nans, fallback to 1.0
        if inv_vol.notna().sum() > 5:
            inv_vol = (inv_vol / inv_vol.quantile(0.9)).clip(upper=1.0)
        else:
            inv_vol = pd.Series(1.0, index=inv_vol.index)
        inv_vol = inv_vol.fillna(0)
        position = signal_traded * inv_vol
    else:
        position = signal_traded

    # Returns + costs
    rets = px.pct_change().fillna(0)
    gross = position * rets
    turns = position.diff().abs().fillna(0)            # change in exposure
    costs = turns * (cost_bps / 10_000)                # 5 bps per side by default
    net = gross - costs

    equity = (1 + net).cumprod()

    out["signal_raw"] = signal_raw
    out["position"] = position
    out["strategy_ret"] = net
    out["equity"] = equity

    metrics = compute_metrics(equity)
    trades  = trade_stats(signal_traded, px)
    return {"df": out, "metrics": metrics, "trades": trades}

# ---------- RUN BASELINE & ADVANCED ----------
price_col = "Adj Close Price"
assert isinstance(price_col, str) and price_col in apple.columns
res = crossover_strategy(
    apple, price_col=price_col,
    fast=30, slow=100, use_ema=False,
    use_regime_200=False, use_rsi_filter=False,
    vol_target=False, cost_bps=5
)

res_adv = crossover_strategy(
    apple, price_col=price_col,
    fast=21, slow=84, use_ema=True,
    use_regime_200=True, use_rsi_filter=True, rsi_period=14,
    vol_target=True, vol_window=20, cost_bps=5
)

# Buy & Hold
bh = (1 + apple[price_col].pct_change().fillna(0)).cumprod()
bh_metrics = compute_metrics(bh)

# ---------- PRINT RESULTS ----------
def pretty(tag, d):
    print(f"\n== {tag} ==")
    s = pd.Series(d)
    print(s.round(4))

pretty("SMA(30/100) Metrics", res["metrics"])
pretty("Advanced Variant Metrics", res_adv["metrics"])
pretty("Buy & Hold Metrics", bh_metrics)

pretty("Trade Stats (SMA 30/100)", res["trades"])
pretty("Trade Stats (Advanced)", res_adv["trades"])

# visualize final one
plt.figure(figsize=(12,5))
plt.plot(res["df"].index, res["df"]["equity"], label="Strategy")
plt.plot(bh.index, bh, label="Buy & Hold", alpha=0.6)
plt.title("Equity Curve: Strategy vs Buy & Hold"); plt.legend(); plt.tight_layout(); plt.show()