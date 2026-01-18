"""
macd_massive.py
Základní MACD strategie používající Massive.com REST klienta
Požadavky: massive, pandas, numpy, matplotlib
Instalace: pip install -U massive pandas numpy matplotlib
Před spuštěním nastavte MASSIVE_API_KEY jako proměnnou prostředí.
Příklad použití:
    python macd_massive.py --ticker AAPL --timeframe 1day --from 2015-01-01 --to 2024-12-31
"""

import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from massive import RESTClient

# ---------------------
# Pomocné funkce: parsování timeframe a načtení agregátů
# ---------------------
def parse_timeframe(tf_str: str):
    """
    Akceptuje řetězce jako '1day', 'day', '1d', '1min', '5min', '1hour'.
    Vrací (násobitel:int, jednotka_str) očekávané Massive klientem, např. (1, 'day') nebo (5, 'minute').
    """
    s = tf_str.lower().strip()
    s = s.replace(' ', '')
    if s in ('day','1day','1d','daily'):
        return 1, 'day'
    if s in ('hour','1hour','1h','60min'):
        return 1, 'hour'
    if s.endswith('hour') or s.endswith('h'):
        num = ''.join([c for c in s if c.isdigit()])
        return int(num) if num else 1, 'hour'
    if s in ('minute','min','1min','1m'):
        return 1, 'minute'
    if 'min' in s:
        num = int(''.join([c for c in s if c.isdigit()]) or 1)
        return num, 'minute'
    raise ValueError(f"Nepodporovaný timeframe '{tf_str}'. Použijte např. '1day', '1min', '5min', '1hour'.")

def fetch_aggregates(client, ticker: str, multiplier: int, timespan: str, start_date: str, end_date: str, adjusted=True):
    """
    Používá client.list_aggs(...) pro stažení OHLCV agregátů.
    Vrací pandas DataFrame s indexem datumu/času.
    """
    aggs = []
    for a in client.list_aggs(ticker, multiplier, timespan, start_date, end_date, adjusted=adjusted, limit=50000):
        aggs.append({
            "timestamp": pd.to_datetime(a.timestamp, unit="ms", utc=True),
            "open": a.open,
            "high": a.high,
            "low": a.low,
            "close": a.close,
            "volume": a.volume,
        })
    if not aggs:
        return pd.DataFrame()
    df = pd.DataFrame(aggs).set_index('timestamp').sort_index()
    df.index = df.index.tz_convert(None)
    return df

# ---------------------
# Technické indikátory a backtest
# ---------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_macd(df, fast, slow, signal):
    """
    Vypočítá MACD linii, signální linii a histogram a přidá je do df.
    """
    close = df['close']
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    df['macd'] = macd_line
    df['signal'] = signal_line
    df['hist'] = hist
    return df

def run_backtest(df, slippage_pct=0.0, fee_pct=0.0):
    """
    Jednoduchý backtest:
      - Vstup long pozice, když MACD kříží nad signál
      - Výstup (flat) když MACD kříží pod signál
      - Při vstupu investujeme veškerý kapitál (1.0) pro základní scénář
      - slippage_pct a fee_pct aplikovány na provedené obchody (relativně)
    """
    df = df.copy()
    # signály: 1=long, 0=flat, -1=short-signal (používáme pouze exit)
    df['signal_raw'] = 0
    df.loc[(df['macd'].shift(1) < df['signal'].shift(1)) & (df['macd'] > df['signal']), 'signal_raw'] = 1
    df.loc[(df['macd'].shift(1) > df['signal'].shift(1)) & (df['macd'] < df['signal']), 'signal_raw'] = -1

    position = 0 
    cash = 1.0  # normalizovaný kapitál
    shares = 0.0 
    equity_list = []
    trades = []
    last_price = None

    for idx, row in df.iterrows():
        price = row['close']
        last_price = price
        sig = row['signal_raw']
        if sig == 1 and position == 0:
            # nákup za cenu + slippage
            exec_price = price * (1 + slippage_pct)
            # investujeme veškerý cash
            shares = cash / exec_price
            cash = 0.0
            position = 1
            trades.append(('BUY', idx, exec_price))
            # poplatek snížený z cashe (aproximace)
            cash -= cash * fee_pct
        elif sig == -1 and position == 1:
            # prodej za cenu - slippage
            exec_price = price * (1 - slippage_pct)
            cash = shares * exec_price
            shares = 0.0
            position = 0
            trades.append(('SELL', idx, exec_price))
            cash -= cash * fee_pct

        # mark-to-market equity
        equity = cash + shares * price
        equity_list.append((idx, equity))

    # výstupní DataFrame s grafem kapitálu
    res_df = pd.DataFrame(equity_list, columns=['timestamp','equity']).set_index('timestamp')
    # metriky
    returns = res_df['equity'].pct_change().fillna(0)
    total_return = res_df['equity'].iloc[-1] - 1.0
    annualized_return = (res_df['equity'].iloc[-1]) ** (252 / len(res_df)) - 1 if len(res_df) > 0 else 0
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else np.nan
    max_dd = max_drawdown(res_df['equity'])

    results = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_trades': len(trades),
        'trades': trades,
        'equity_curve': res_df
    }
    return results

def max_drawdown(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

# ---------------------
# CLI / hlavní část
# ---------------------
def main(args):
    api_key = os.getenv('MASSIVE_API_KEY')
    if not api_key:
        raise RuntimeError("Nastavte prosím proměnnou prostředí MASSIVE_API_KEY s vaším API klíčem.")
    client = RESTClient(api_key)

    mult, timespan = parse_timeframe(args.timeframe)
    df = fetch_aggregates(client, args.ticker.upper(), mult, timespan, args.from_date, args.to_date, adjusted=False)
    if df.empty:
        print("Nebyla vrácena žádná data. Zkontrolujte ticker, timeframe, rozsah dat a oprávnění vašeho plánu.")
        return

    # MACD
    df = compute_macd(df, fast=8, slow=17, signal=9)

    # křivky kapitálu
    equity_steps = {}

    # 1) Základní MACD
    results_baseline = run_backtest(df, slippage_pct=0.0, fee_pct=0.0)
    equity_steps['Základní strategie MACD'] = results_baseline['equity_curve']['equity']

    # 2) MACD + slippage a poplatky
    results_slip = run_backtest(df, slippage_pct=0.0005, fee_pct=0.01)
    equity_steps['MACD + slippage / poplatky'] = results_slip['equity_curve']['equity']

    # 3) MACD + rozdělení podle volatility (alokace dle realizované volatility)
    sig_series = pd.Series(0, index=df.index)
    sig_series.loc[(df['macd'].shift(1) < df['signal'].shift(1)) & (df['macd'] > df['signal'])] = 1
    sig_series.loc[(df['macd'].shift(1) > df['signal'].shift(1)) & (df['macd'] < df['signal'])] = -1

    rets = df['close'].pct_change()
    rol_std = rets.rolling(21).std()  # 21-periodní realizovaná volatilita
    vol_med = rol_std.median() if not rol_std.dropna().empty else 0.0
    alloc = (vol_med / rol_std).replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(lower=0.1, upper=1.0) 

    cash = 1.0
    shares = 0.0
    position = 0
    equity_list = []
    trades = []
    slippage_pct=0.0005

    for idx, row in df.iterrows(): 
        price = row['close']
        sig = sig_series.loc[idx]
        a = float(alloc.loc[idx]) if idx in alloc.index else 1.0

        if sig == 1 and position == 0:
            exec_price = price * (1 - slippage_pct)
            invest = cash * a
            if invest > 0:
                shares = invest / exec_price
                cash -= invest
                position = 1
                trades.append(('BUY', idx, exec_price, a))
        elif sig == -1 and position == 1:
            exec_price = price
            cash += shares * exec_price
            shares = 0.0
            position = 0
            trades.append(('SELL', idx, exec_price, a))

        equity = cash + shares * price
        equity_list.append((idx, equity))

    res_df = pd.DataFrame(equity_list, columns=['timestamp', 'equity']).set_index('timestamp') 
    results_volsplit = {
        'equity_curve': res_df,
        'n_trades': len(trades),
        'trades': trades
    }
    equity_steps['MACD + váha dle volatility'] = results_volsplit['equity_curve']['equity']

    # 4) Buy and Hold
    bh_equity = (df['close'] / df['close'].iloc[0]).to_frame('equity')
    equity_steps['Buy and Hold'] = bh_equity['equity']

    # -----------------------
    # Plotting všech grafů
    # -----------------------
    plt.figure(figsize=(12,7))
    for label, equity in equity_steps.items():
        equity.plot(label=label)

    plt.title(f"{args.ticker} - MACD strategie: postupné křivky portfolia")
    plt.xlabel("Datum")
    plt.ylabel("Normalizovaný kapitál")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker (např. AAPL)")
    parser.add_argument("--timeframe", type=str, default="1day", help="např. 1day, 1min, 5min, 1hour")
    parser.add_argument("--from", dest="from_date", type=str, default="2015-01-01", help="Počáteční datum YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", type=str, default=datetime.today().strftime("%Y-%m-%d"), help="Koncové datum YYYY-MM-DD")
    args = parser.parse_args()
    main(args)
