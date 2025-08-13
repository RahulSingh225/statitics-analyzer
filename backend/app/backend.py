from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from textblob import TextBlob
import requests
import os
from datetime import datetime, timedelta
import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import io
import base64
import time
from apscheduler.schedulers.background import BackgroundScheduler
from drizzle_orm import Drizzle, pgTable, serial, text, jsonb, float64, timestamp
from drizzle_orm.query import select, insert, delete
import asyncpg

# Logging setup
logging.basicConfig(filename='stock_analysis.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load env
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# DB tables
stocks = pgTable('stocks', [
    serial('id').primaryKey(),
    text('symbol').notNull().unique(),
    timestamp('created_at').defaultNow().notNull()
])

daily_data = pgTable('daily_data', [
    serial('id').primaryKey(),
    text('symbol').notNull(),
    timestamp('date').notNull(),
    float64('open').notNull(),
    float64('high').notNull(),
    float64('low').notNull(),
    float64('close').notNull(),
    float64('volume').notNull(),
    timestamp('created_at').defaultNow().notNull()
])

financials = pgTable('financials', [
    serial('id').primaryKey(),
    text('symbol').notNull(),
    timestamp('date').notNull(),
    float64('pe_ratio').notNull(),
    float64('eps').notNull(),
    float64('revenue').notNull(),
    float64('debt_to_equity').notNull(),
    timestamp('created_at').defaultNow().notNull()
])

indicators = pgTable('indicators', [
    serial('id').primaryKey(),
    text('symbol').notNull(),
    timestamp('date').notNull(),
    float64('sma5').notNull(),
    float64('ema5').notNull(),
    float64('rsi').notNull(),
    float64('macd').notNull(),
    float64('macd_signal').notNull(),
    float64('upper_bb').notNull(),
    float64('middle_bb').notNull(),
    float64('lower_bb').notNull(),
    float64('vwap').notNull(),
    float64('atr').notNull(),  # Volatility
    timestamp('created_at').defaultNow().notNull()
])

sentiments = pgTable('sentiments', [
    serial('id').primaryKey(),
    text('symbol').notNull(),
    timestamp('date').notNull(),
    float64('score').notNull(),
    timestamp('created_at').defaultNow().notNull()
])

predictions = pgTable('predictions', [
    serial('id').primaryKey(),
    text('symbol').notNull(),
    timestamp('date').notNull(),
    float64('rf_prediction').notNull(),
    float64('lstm_prediction').notNull(),
    text('entry_exit_signal').notNull(),  # 'buy', 'sell', 'hold'
    timestamp('created_at').defaultNow().notNull()
])

reports = pgTable('reports', [
    serial('id').primaryKey(),
    text('symbol').notNull(),
    timestamp('date').notNull(),
    text('report_text').notNull(),
    timestamp('created_at').defaultNow().notNull()
])

async def init_db():
    pool = await asyncpg.create_pool(DATABASE_URL)
    return Drizzle(pool)

PENNY_STOCKS = [
    'GTLINFRA.NS', 'DHARAN.NS', 'FILATFASH.NS', 'GODHA.NS', 'GATECH.NS',
    'EXCEL.NS', 'VIKASLIFE.NS', 'NDL.NS', 'VIKASECO.NS', 'GATECHDVR.NS',
    'FCSSOFT.NS', 'RHFL.NS', 'TPHQ.NS', 'DAVANGERE.NS', 'ESSENTIA.NS',
    'JPASSOCIAT.NS', 'NAVKARURB.NS', 'RCOM.NS', 'INVENTURE.NS', 'SHAH.NS',
    'MITTAL.NS', 'AKSHAR.NS', 'GLOBE.NS', 'SAKUMA.NS', 'SHRENIK.NS',
    'GANGAFORGE.NS', 'AGSTRA.NS', 'VCL.NS', 'SUVIDHAA.NS', 'GVKPIL.NS',
    'SUNDARAM.NS', 'RAJMET.NS', 'ANTGRAPHIC.NS', 'SANWARIA.NS'
]
PRICE_THRESHOLD = 5
VOLATILITY_THRESHOLD = 0.2  # ATR > 0.2 for high volatility

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fetch price data (daily or 5m)
def fetch_price_data(ticker, interval='1d'):
    try:
        stock = yf.Ticker(ticker)
        start = (datetime.now() - timedelta(days=10 if interval == '5m' else 30)).strftime("%Y-%m-%d")
        df = stock.history(start=start, interval=interval)
        if not df.empty:
            df = df.reset_index().rename(columns={
                'Date' if interval == '1d' else 'Datetime': 'date',
                'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'
            })
            return df
        return None
    except Exception as e:
        logging.error(f"Error fetching price data for {ticker}: {e}")
        return None

# Other functions (fetch_financials, fetch_news, analyze_sentiment, calculate_technical_indicators, prepare_lstm_data, train_lstm_model, generate_plot) remain the same as previous, but add ATR to indicators
def calculate_technical_indicators(df):
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
    df['upper'], df['middle'], df['lower'] = talib.BBANDS(df['close'])
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    return df

# Generate signal
def generate_signal(rsi, macd, macd_signal, sentiment, predicted, pe_ratio, atr):
    if rsi < 30 and macd > macd_signal and sentiment > 0 and predicted == 1 and pe_ratio < 15 and atr > VOLATILITY_THRESHOLD:
        return 'buy'
    elif rsi > 70 and macd < macd_signal and sentiment < 0 and predicted == 0:
        return 'sell'
    else:
        return 'hold'

# Daily pipeline
async def daily_pipeline():
    drizzle = await init_db()
    filtered = [symbol for symbol in PENNY_STOCKS if yf.Ticker(symbol).history(period='1d')['Close'].iloc[-1] < PRICE_THRESHOLD]
    
    for symbol in filtered:
        # Daily for swing
        df_daily = fetch_price_data(symbol, '1d')
        if df_daily is None:
            continue
        financials_data = fetch_financials(symbol)
        df_daily = calculate_technical_indicators(df_daily)
        news_data = fetch_news(symbol)
        df_daily = analyze_sentiment(news_data, df_daily)
        sentiment_score = df_daily['sentiment'].mean()
        
        features = ['SMA_5', 'RSI', 'MACD', 'volume', 'upper', 'lower', 'sentiment', 'ATR']
        X = df_daily[features].dropna()
        y = (df_daily['close'].shift(-1) > df_daily['close']).astype(int)[X.index]
        
        if len(X) >= 2:
            train_size = int(0.8 * len(X))
            X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
            
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X_train, y_train)
            rf_prediction = rf_model.predict(X_val[-1:])
            rf_accuracy = accuracy_score(y_val, rf_model.predict(X_val))
            
            X_lstm, y_lstm, _ = prepare_lstm_data(df_daily, features)
            if len(X_lstm) > 0:
                lstm_model = train_lstm_model(X_lstm, y_lstm)
                lstm_prediction = (lstm_model.predict(X_lstm[-1:]) > 0.5).astype(int)
                lstm_accuracy = accuracy_score(y_lstm, lstm_model.predict(X_lstm))
            else:
                lstm_prediction, lstm_accuracy = 0, 0.0
            
            predicted = 1 if rf_prediction[0] + lstm_prediction[0] >= 1 else 0
            latest = df_daily.iloc[-1]
            signal = generate_signal(latest['RSI'], latest['MACD'], latest['MACD_signal'], sentiment_score, predicted, financials_data['PE_Ratio'], latest['ATR'])
            
            df_daily['predictions'] = pd.Series(rf_model.predict(X), index=X.index)
            df_daily['returns'] = df_daily['close'].pct_change().shift(-1) * df_daily['predictions']
            cumulative_return = (1 + df_daily['returns'].fillna(0)).cumprod() - 1.iloc[-1]
            
            report = generate_report_summary(symbol, df_daily, financials_data, sentiment_score, rf_accuracy, lstm_accuracy, cumulative_return) + f"\n**Signal**: {signal}"
            
            # Store
            await insert(daily_data).values([{
                'symbol': symbol,
                'date': row['date'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            } for index, row in df_daily.iterrows()]).execute(drizzle)
            
            await insert(financials).values({
                'symbol': symbol,
                'date': datetime.now(),
                'pe_ratio': financials_data['PE_Ratio'],
                'eps': financials_data['EPS'],
                'revenue': financials_data['Revenue'],
                'debt_to_equity': financials_data['DebtToEquity']
            }).execute(drizzle)
            
            await insert(indicators).values({
                'symbol': symbol,
                'date': datetime.now(),
                'sma5': latest['SMA_5'],
                'ema5': latest['EMA_5'],
                'rsi': latest['RSI'],
                'macd': latest['MACD'],
                'macd_signal': latest['MACD_signal'],
                'upper_bb': latest['upper'],
                'middle_bb': latest['middle'],
                'lower_bb': latest['lower'],
                'vwap': latest['VWAP'],
                'atr': latest['ATR']
            }).execute(drizzle)
            
            await insert(sentiments).values({
                'symbol': symbol,
                'date': datetime.now(),
                'score': sentiment_score
            }).execute(drizzle)
            
            await insert(predictions).values({
                'symbol': symbol,
                'date': datetime.now(),
                'rf_prediction': float(rf_prediction[0]),
                'lstm_prediction': float(lstm_prediction[0]),
                'entry_exit_signal': signal
            }).execute(drizzle)
            
            await insert(reports).values({
                'symbol': symbol,
                'date': datetime.now(),
                'report_text': report
            }).execute(drizzle)
        
        # Intraday (5m) similar, but omit for brevity; add if needed
        
    await drizzle.pool.close()

# Weekly retrain
async def weekly_retrain():
    drizzle = await init_db()
    # Fetch historical data from DB, retrain models with grid search
    historical = await select(daily_data).execute(drizzle)
    df = pd.DataFrame(historical)
    if not df.empty:
        # Group by symbol, retrain per stock
        for symbol, group in df.groupby('symbol'):
            # Reprocess indicators, retrain RF with grid search
            param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
            rf = RandomForestClassifier(random_state=42)
            grid = GridSearchCV(rf, param_grid)
            features = group[['open', 'high', 'low', 'close', 'volume']]  # Example
            X = features.iloc[:-1]
            y = (group['close'].shift(-1) > group['close']).astype(int).iloc[:-1]
            if len(X) > 10:
                grid.fit(X, y)
                # Update model (save best params or serialized model in DB if needed)
                logging.info(f"Retrained for {symbol} with best params {grid.best_params_}")
    await drizzle.pool.close()

# Scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(daily_pipeline, 'cron', hour=0)  # Daily at midnight
scheduler.add_job(weekly_retrain, 'cron', day_of_week='sun', hour=0)  # Weekly Sunday
scheduler.start()

# API to get reports
@app.get("/reports")
async def get_reports():
    drizzle = await init_db()
    results = await select(reports).execute(drizzle)
    await drizzle.pool.close()
    return {"reports": results}

# Frontend fetches from /reports