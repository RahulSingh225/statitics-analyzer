import yfinance as yf
import pandas as pd
import talib
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert
from .models import Stock, DailyData, Financial, Indicator, Sentiment, Prediction, Report
import logging
import os
from dotenv import load_dotenv

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

logging.basicConfig(filename='/app/stock_analysis.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def daily_pipeline(init_db):
    logging.info("Starting daily pipeline")
    
    async_session = await init_db()
    async with async_session() as session:
        # Fetch stock symbols
        stocks = (await session.execute(select(Stock))).scalars().all()
        symbols = [stock.symbol for stock in stocks]
        
        for symbol in symbols:
            try:
                # Fetch price data
                stock = yf.Ticker(symbol)
                df = stock.history(period="1d")
                if df.empty:
                    logging.warning(f"No price data for {symbol}")
                    continue
                
                # Fetch financials
                url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
                response = requests.get(url)
                financial_data = response.json()
                
                # Fetch news sentiment
                news_url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
                news_response = requests.get(news_url)
                articles = news_response.json().get('articles', [])
                sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles) if articles else 0
                
                # Calculate indicators
                df['SMA_5'] = talib.SMA(df['Close'], timeperiod=5)
                df['EMA_5'] = talib.EMA(df['Close'], timeperiod=5)
                df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
                macd, macd_signal, _ = talib.MACD(df['Close'])
                df['MACD'] = macd
                df['MACD_signal'] = macd_signal
                upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20)
                df['upper_bb'] = upper
                df['middle_bb'] = middle
                df['lower_bb'] = lower
                df['VWAP'] = talib.WMA((df['High'] + df['Low'] + df['Close']) / 3, timeperiod=14)
                df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
                
                # Store data
                date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                await session.execute(insert(DailyData).values(
                    symbol=symbol,
                    date=date,
                    open=float(df['Open'].iloc[-1]),
                    high=float(df['High'].iloc[-1]),
                    low=float(df['Low'].iloc[-1]),
                    close=float(df['Close'].iloc[-1]),
                    volume=float(df['Volume'].iloc[-1])
                ))
                
                await session.execute(insert(Financial).values(
                    symbol=symbol,
                    date=date,
                    pe_ratio=float(financial_data.get('PERatio', 0)),
                    eps=float(financial_data.get('EPS', 0)),
                    revenue=float(financial_data.get('RevenueTTM', 0)),
                    debt_to_equity=float(financial_data.get('DebtToEquityRatio', 0))
                ))
                
                await session.execute(insert(Indicator).values(
                    symbol=symbol,
                    date=date,
                    sma5=float(df['SMA_5'].iloc[-1]),
                    ema5=float(df['EMA_5'].iloc[-1]),
                    rsi=float(df['RSI'].iloc[-1]),
                    macd=float(df['MACD'].iloc[-1]),
                    macd_signal=float(df['MACD_signal'].iloc[-1]),
                    upper_bb=float(df['upper_bb'].iloc[-1]),
                    middle_bb=float(df['middle_bb'].iloc[-1]),
                    lower_bb=float(df['lower_bb'].iloc[-1]),
                    vwap=float(df['VWAP'].iloc[-1]),
                    atr=float(df['ATR'].iloc[-1])
                ))
                
                await session.execute(insert(Sentiment).values(
                    symbol=symbol,
                    date=date,
                    score=float(sentiment_score)
                ))
                
                # Dummy predictions (replace with actual ML model calls)
                rf_pred = 0.6  # Placeholder
                lstm_pred = 0.7  # Placeholder
                signal = "Buy" if rf_pred > 0.5 and lstm_pred > 0.5 else "Hold"
                
                await session.execute(insert(Prediction).values(
                    symbol=symbol,
                    date=date,
                    rf_prediction=float(rf_pred),
                    lstm_prediction=float(lstm_pred),
                    entry_exit_signal=signal
                ))
                
                report_text = f"Analysis for {symbol} on {date}: Sentiment {sentiment_score:.2f}, RSI {df['RSI'].iloc[-1]:.2f}"
                await session.execute(insert(Report).values(
                    symbol=symbol,
                    date=date,
                    report_text=report_text
                ))
                
                await session.commit()
                logging.info(f"Processed data for {symbol}")
            
            except Exception as e:
                logging.error(f"Error processing {symbol}: {e}")
                await session.rollback()
    
    logging.info("Daily pipeline completed")