from apscheduler.schedulers.background import BackgroundScheduler
import yfinance as yf
import pandas as pd
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import requests
import os
from datetime import datetime, timedelta
import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64
import time
from drizzle_orm.query import insert, select
from .models import stocks, daily_data, financials, indicators, sentiments, predictions, reports

logging.basicConfig(filename='/app/stock_analysis.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
VOLATILITY_THRESHOLD = 0.2
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_price_data(ticker, interval='1d', days=10):
    try:
        stock = yf.Ticker(ticker)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
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

def fetch_financials(ticker):
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        time.sleep(12)
        return {
            'pe_ratio': float(data.get('PERatio', 0) or 0),
            'eps': float(data.get('EPS', 0) or 0),
            'revenue': float(data.get('RevenueTTM', 0) or 0),
            'debt_to_equity': float(data.get('DebtToEquityRatio', 0) or 0)
        }
    except Exception as e:
        logging.error(f"Error fetching financials for {ticker}: {e}")
        return {'pe_ratio': 0, 'eps': 0, 'revenue': 0, 'debt_to_equity': 0}

def fetch_news(ticker):
    try:
        start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
        url = f"https://newsapi.org/v2/everything?q={ticker}&from={start}&to={end}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt"
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        time.sleep(1)
        return [(article['publishedAt'], article['description']) for article in articles if article['description']]
    except Exception as e:
        logging.error(f"Error fetching news for {ticker}: {e}")
        return []

def analyze_sentiment(news_data, df):
    sentiment_scores = []
    for date, text in news_data:
        sentiment = TextBlob(text).sentiment.polarity
        sentiment_scores.append((pd.to_datetime(date), sentiment))
    
    sentiment_df = pd.DataFrame(sentiment_scores, columns=['date', 'sentiment'])
    
    df['sentiment'] = 0.0
    for _, row in sentiment_df.iterrows():
        closest_idx = df['date'].sub(row['date']).abs().idxmin()
        df.loc[closest_idx, 'sentiment'] = row['sentiment']
    
    df['sentiment'] = df['sentiment'].replace(0.0, method='ffill').fillna(0.0)
    return df

def calculate_technical_indicators(df):
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
    df['upper'], df['middle'], df['lower'] = talib.BBANDS(df['close'])
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    return df

def prepare_lstm_data(df, features, lookback=10):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features].dropna())
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(1 if df['close'].iloc[i] > df['close'].iloc[i-1] else 0)
    return np.array(X), np.array(y), scaler

def train_lstm_model(X, y):
    model = Sequential([
        LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
    return model

def generate_signal(rsi, macd, macd_signal, sentiment, predicted, pe_ratio, atr):
    if rsi < 30 and macd > macd_signal and sentiment > 0 and predicted == 1 and pe_ratio < 15 and atr > VOLATILITY_THRESHOLD:
        return 'buy'
    elif rsi > 70 and macd < macd_signal and sentiment < 0 and predicted == 0:
        return 'sell'
    else:
        return 'hold'

def generate_report_summary(symbol, df, financials, sentiment_score, rf_accuracy, lstm_accuracy, cumulative_return, signal):
    summary = f"# Stock Analysis for {symbol}\n\n"
    summary += f"**Latest Close**: ₹{df['close'].iloc[-1]:.2f}\n"
    summary += f"**Average Volume**: {df['volume'].mean():,.0f}\n"
    summary += f"**10-Day High**: ₹{df['high'].max():.2f}\n"
    summary += f"**10-Day Low**: ₹{df['low'].min():.2f}\n"
    summary += f"**Volatility (ATR)**: {df['ATR'].iloc[-1]:.2f}\n\n"
    summary += "## Financials\n"
    summary += f"- **P/E Ratio**: {financials['pe_ratio']:.2f}\n"
    summary += f"- **EPS**: {financials['eps']:.2f}\n"
    summary += f"- **Revenue (TTM)**: ₹{financials['revenue']:,.0f}\n"
    summary += f"- **Debt-to-Equity**: {financials['debt_to_equity']:.2f}\n\n"
    summary += f"**Sentiment Score**: {sentiment_score:.2f}\n"
    summary += f"**Random Forest Accuracy**: {rf_accuracy:.2%}\n"
    summary += f"**LSTM Accuracy**: {lstm_accuracy:.2%}\n"
    summary += f"**Cumulative Return**: {cumulative_return:.2%}\n"
    summary += f"**Trading Signal**: {signal}\n"
    return summary

def generate_plot(df):
    buf = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['close'], label='Close Price')
    plt.plot(df['date'], df['SMA_5'], label='SMA 5')
    plt.plot(df['date'], df['EMA_5'], label='EMA 5')
    plt.title('Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode('utf-8')

async def process_stock(symbol, drizzle, interval='1d', days=10):
    try:
        df = fetch_price_data(symbol, interval, days)
        if df is None or df.empty:
            logging.warning(f"No data for {symbol}")
            return None
        
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
            return None
        
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
        
        return {
            "symbol": symbol,
            "date": now.isoformat(),
            "report_text": report_text,
            "indicators": [{
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
            }],
            "financials": [financials_data],
            "sentiment": [{'score': sentiment_score}],
            "prediction": [{
                'rf_prediction': float(rf_prediction[0]),
                'lstm_prediction': float(lstm_prediction),
                'entry_exit_signal': signal
            }],
            "data": df.to_dict(orient='records'),
            "plot_base64": plot_base64
        }
    except Exception as e:
        logging.error(f"Error processing {symbol}: {e}")
        return None

async def daily_pipeline(drizzle):
    filtered = [symbol for symbol in PENNY_STOCKS if yf.Ticker(symbol).history(period='1d')['Close'].iloc[-1] < PRICE_THRESHOLD]
    for symbol in filtered:
        await process_stock(symbol, drizzle, '1d', 10)  # Daily for swing
        await process_stock(symbol, drizzle, '5m', 2)  # Intraday for scalping

scheduler = BackgroundScheduler()
scheduler.add_job(lambda: asyncio.run(daily_pipeline(init_db())), 'cron', hour=0)
scheduler.start()