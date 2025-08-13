import asyncio
import yfinance as yf
import pandas as pd
import requests
import os
from datetime import datetime
import logging
import time
from dotenv import load_dotenv
from drizzle_orm import Drizzle
import asyncpg
from .daily_pipeline import fetch_financials, fetch_news, analyze_sentiment, calculate_technical_indicators, prepare_lstm_data, train_lstm_model, generate_signal, generate_report_summary, generate_plot, PENNY_STOCKS, PRICE_THRESHOLD, VOLATILITY_THRESHOLD
from .models import stocks, daily_data, financials, indicators, sentiments, predictions, reports

logging.basicConfig(filename='/app/stock_analysis.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

async def init_db():
    pool = await asyncpg.create_pool(DATABASE_URL)
    return Drizzle(pool)

async def load_historical_data():
    drizzle = await init_db()
    start_date = "2025-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    filtered = [symbol for symbol in PENNY_STOCKS if yf.Ticker(symbol).history(period='1d')['Close'].iloc[-1] < PRICE_THRESHOLD]
    
    for symbol in filtered:
        logging.info(f"Loading historical data for {symbol}")
        df = yf.Ticker(symbol).history(start=start_date, end=end_date, interval='1d')
        if df.empty:
            logging.warning(f"No historical data for {symbol}")
            continue
        
        df = df.reset_index().rename(columns={
            'Date': 'date', 'Close': 'close', 'Open': 'open',
            'High': 'high', 'Low': 'low', 'Volume': 'volume'
        })
        
        financials_data = fetch_financials(symbol)
        df = calculate_technical_indicators(df)
        news_data = fetch_news(symbol)
        df = analyze_sentiment(news_data, df)
        sentiment_score = float(df['sentiment'].mean())
        
        features = ['SMA_5', 'RSI', 'MACD', 'volume', 'upper', 'lower', 'sentiment', 'ATR']
        X = df[features].dropna()
        y = (df['close'].shift(-1) > df['close']).astype(int)[X.index]
        
        if len(X) < 2:
            logging.warning(f"Insufficient data for {symbol}")
            continue
        
        train_size = int(0.8 * len(X))
        X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
        
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        rf_prediction = rf_model.predict(X_val[-1:])
        rf_accuracy = accuracy_score(y_val, rf_model.predict(X_val))
        
        X_lstm, y_lstm, _ = prepare_lstm_data(df, features)
        lstm_prediction, lstm_accuracy = 0, 0.0
        if len(X_lstm) > 0:
            lstm_model = train_lstm_model(X_lstm, y_lstm)
            lstm_prediction = (lstm_model.predict(X_lstm[-1:]) > 0.5).astype(int)[0]
            lstm_accuracy = accuracy_score(y_lstm, lstm_model.predict(X_lstm))
        
        predicted = 1 if rf_prediction[0] + lstm_prediction >= 1 else 0
        latest = df.iloc[-1]
        signal = generate_signal(latest['RSI'], latest['MACD'], latest['MACD_signal'], sentiment_score, predicted, financials_data['pe_ratio'], latest['ATR'])
        
        df['predictions'] = pd.Series(rf_model.predict(X), index=X.index)
        df['returns'] = df['close'].pct_change().shift(-1) * df['predictions']
        cumulative_return = float((1 + df['returns'].fillna(0)).cumprod() - 1).iloc[-1]
        
        report_text = generate_report_summary(symbol, df, financials_data, sentiment_score, rf_accuracy, lstm_accuracy, cumulative_return, signal)
        plot_base64 = generate_plot(df)
        
        now = datetime.now()
        await insert(stocks).values(symbol=symbol).on_conflict_do_nothing().execute(drizzle)
        await insert(daily_data).values([{
            'symbol': symbol,
            'date': row['date'],
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        } for index, row in df.iterrows()]).execute(drizzle)
        
        await insert(financials).values({
            'symbol': symbol,
            'date': now,
            'pe_ratio': float(financials_data['pe_ratio']),
            'eps': float(financials_data['eps']),
            'revenue': float(financials_data['revenue']),
            'debt_to_equity': float(financials_data['debt_to_equity'])
        }).execute(drizzle)
        
        await insert(indicators).values({
            'symbol': symbol,
            'date': now,
            'sma5': float(latest['SMA_5']),
            'ema5': float(latest['EMA_5']),
            'rsi': float(latest['RSI']),
            'macd': float(latest['MACD']),
            'macd_signal': float(latest['MACD_signal']),
            'upper_bb': float(latest['upper']),
            'middle_bb': float(latest['middle']),
            'lower_bb': float(latest['lower']),
            'vwap': float(latest['VWAP']),
            'atr': float(latest['ATR'])
        }).execute(drizzle)
        
        await insert(sentiments).values({
            'symbol': symbol,
            'date': now,
            'score': sentiment_score
        }).execute(drizzle)
        
        await insert(predictions).values({
            'symbol': symbol,
            'date': now,
            'rf_prediction': float(rf_prediction[0]),
            'lstm_prediction': float(lstm_prediction),
            'entry_exit_signal': signal
        }).execute(drizzle)
        
        await insert(reports).values({
            'symbol': symbol,
            'date': now,
            'report_text': report_text
        }).execute(drizzle)
    
    await drizzle.pool.close()

if __name__ == '__main__':
    asyncio.run(load_historical_data())