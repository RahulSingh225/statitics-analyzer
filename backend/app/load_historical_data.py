import yfinance as yf
import pandas as pd
import talib
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import select, insert
from sqlalchemy.orm import sessionmaker
from app.models import Stock, DailyData, Financial, Indicator, Sentiment, Prediction, Report
import logging
import os
from dotenv import load_dotenv
import asyncio
import asyncpg
import socket
import time
import yfinance.shared as shared

# Configure logging
logging.basicConfig(
    filename='/app/stock_analysis.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable yfinance debug logging
yf.enable_debug_mode()

# Load environment variables
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Validate environment variables
if not all([ALPHA_VANTAGE_API_KEY, NEWS_API_KEY, DATABASE_URL]):
    logger.error("Missing environment variables: ALPHA_VANTAGE_API_KEY=%s, NEWS_API_KEY=%s, DATABASE_URL=%s",
                 ALPHA_VANTAGE_API_KEY, NEWS_API_KEY, DATABASE_URL)
    raise ValueError("Required environment variables are missing")

# Ensure DATABASE_URL uses asyncpg
if not DATABASE_URL.startswith("postgresql+asyncpg://"):
    logger.error("DATABASE_URL must use asyncpg, found: %s", DATABASE_URL)
    raise ValueError("DATABASE_URL must start with 'postgresql+asyncpg://'")

# Verify asyncpg is available
try:
    logger.debug("Asyncpg version: %s", asyncpg.__version__)
except ImportError:
    logger.error("asyncpg module not found")
    raise ImportError("asyncpg is required for async database operations")

# Test DNS resolution for Yahoo Finance domains
for domain in ["finance.yahoo.com", "query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
    try:
        ip = socket.gethostbyname(domain)
        logger.debug("DNS resolution for %s successful: %s", domain, ip)
    except socket.gaierror as e:
        logger.error("DNS resolution failed for %s: %s", domain, e)
        raise

async def load_historical_data(init_db):
    logger.debug("Starting historical data load")
    
    # Initialize database
    try:
        async_session = await init_db()
        logger.debug("Database session initialized successfully with asyncpg")
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)
        raise
    
    async with async_session() as session:
        # Define penny stocks (NSE, price < ₹5)
        symbols = ['GANGAFORGE.NS', 'AGSTRA.NS', 'VCL.NS', 'SUVIDHAA.NS', 'GVKPIL.NS',
    'SUNDARAM.NS', 'RAJMET.NS', 'ANTGRAPHIC.NS', 'SANWARIA.NS']
        logger.debug("Processing symbols: %s", symbols)
        
        for symbol in symbols:
            try:
                # Ensure stock exists
                result = await session.execute(select(Stock).where(Stock.symbol == symbol))
                if not result.scalars().first():
                    await session.execute(insert(Stock).values(symbol=symbol))
                    await session.commit()
                    logger.debug("Inserted stock: %s", symbol)
                
                # Fetch historical price data (past 10 days to align with example)
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        stock = yf.Ticker(symbol)
                        df = stock.history(period="5d", interval="1d",raise_errors=True)
                        if df.empty:
                            logger.warning("No historical data for %s", symbol)
                            continue
                        logger.debug("Fetched %d rows of price data for %s", len(df), symbol)
                        break
                    except Exception as e:
                        logger.error("Attempt %d/%d failed to fetch historical data for %s: %s", 
                                    attempt + 1, max_retries, symbol, e)
                        if shared._ERRORS.get(symbol):
                            logger.error("yfinance error details for %s: %s", symbol, shared._ERRORS[symbol])
                        if attempt < max_retries - 1:
                            time.sleep(5)  # Longer delay for retries
                        else:
                            logger.error("Max retries reached for %s", symbol)
                            continue
                
                if df.empty:
                    continue
                
                df = df.reset_index().rename(columns={
                    'Date': 'Datetime', 'Close': 'close', 'Open': 'open',
                    'High': 'high', 'Low': 'low', 'Volume': 'volume'
                })
                
                # Fetch financials
                url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    financial_data = response.json()
                    if not financial_data:
                        logger.warning("Empty financial data for %s", symbol)
                        financial_data = {}
                    time.sleep(12)  # Respect Alpha Vantage rate limit
                except requests.RequestException as e:
                    logger.error("Failed to fetch financials for %s: %s", symbol, e)
                    financial_data = {}
                
                # Fetch news sentiment (last 10 days)
                news_url = f"https://newsapi.org/v2/everything?q={symbol}&from={(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')}&to={(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}&apiKey={NEWS_API_KEY}"
                try:
                    news_response = requests.get(news_url, timeout=10)
                    news_response.raise_for_status()
                    articles = news_response.json().get('articles', [])
                    sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles) if articles else 0
                    logger.debug("Calculated sentiment score for %s: %.2f", symbol, sentiment_score)
                    time.sleep(1)  # Respect NewsAPI rate limit
                except requests.RequestException as e:
                    logger.error("Failed to fetch news for %s: %s", symbol, e)
                    sentiment_score = 0
                
                # Calculate indicators
                try:
                    df['SMA_5'] = talib.SMA(df['close'], timeperiod=5)
                    df['EMA_5'] = talib.EMA(df['close'], timeperiod=5)
                    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
                    macd, macd_signal, _ = talib.MACD(df['close'])
                    df['MACD'] = macd
                    df['MACD_signal'] = macd_signal
                    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
                    df['upper_bb'] = upper
                    df['middle_bb'] = middle
                    df['lower_bb'] = lower
                    df['VWAP'] = talib.WMA((df['high'] + df['low'] + df['close']) / 3, timeperiod=14)
                    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
                    logger.debug("Calculated indicators for %s", symbol)
                except Exception as e:
                    logger.error("Failed to calculate indicators for %s: %s", symbol, e)
                    continue
                
                # Store data
                for _, row in df.iterrows():
                    try:
                        date = pd.to_datetime(row['Datetime']).replace(hour=0, minute=0, second=0, microsecond=0)
                        await session.execute(insert(DailyData).values(
                            symbol=symbol,
                            date=date,
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=float(row['volume'])
                        ))
                        
                        await session.execute(insert(Indicator).values(
                            symbol=symbol,
                            date=date,
                            sma5=float(row['SMA_5']) if pd.notna(row['SMA_5']) else 0,
                            ema5=float(row['EMA_5']) if pd.notna(row['EMA_5']) else 0,
                            rsi=float(row['RSI']) if pd.notna(row['RSI']) else 0,
                            macd=float(row['MACD']) if pd.notna(row['MACD']) else 0,
                            macd_signal=float(row['MACD_signal']) if pd.notna(row['MACD_signal']) else 0,
                            upper_bb=float(row['upper_bb']) if pd.notna(row['upper_bb']) else 0,
                            middle_bb=float(row['middle_bb']) if pd.notna(row['middle_bb']) else 0,
                            lower_bb=float(row['lower_bb']) if pd.notna(row['lower_bb']) else 0,
                            vwap=float(row['VWAP']) if pd.notna(row['VWAP']) else 0,
                            atr=float(row['ATR']) if pd.notna(row['ATR']) else 0
                        ))
                    except Exception as e:
                        logger.error("Failed to store daily data or indicators for %s on %s: %s", symbol, date, e)
                        continue
                
                # Store financials and sentiment
                try:
                    await session.execute(insert(Financial).values(
                        symbol=symbol,
                        date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                        pe_ratio=float(financial_data.get('PERatio', 0)),
                        eps=float(financial_data.get('EPS', 0)),
                        revenue=float(financial_data.get('RevenueTTM', 0)),
                        debt_to_equity=float(financial_data.get('DebtToEquityRatio', 0))
                    ))
                    
                    await session.execute(insert(Sentiment).values(
                        symbol=symbol,
                        date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                        score=float(sentiment_score)
                    ))
                    
                    # Dummy predictions
                    rf_pred = 0.6  # Placeholder
                    lstm_pred = 0.7  # Placeholder
                    signal = "Buy" if rf_pred > 0.5 and lstm_pred > 0.5 else "Hold"
                    
                    await session.execute(insert(Prediction).values(
                        symbol=symbol,
                        date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                        rf_prediction=float(rf_pred),
                        lstm_prediction=float(lstm_pred),
                        entry_exit_signal=signal
                    ))
                    
                    report_text = f"Historical analysis for {symbol}: Sentiment {sentiment_score:.2f}, RSI {df['RSI'].iloc[-1]:.2f if pd.notna(df['RSI'].iloc[-1]) else 0}"
                    await session.execute(insert(Report).values(
                        symbol=symbol,
                        date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                        report_text=report_text
                    ))
                    
                    await session.commit()
                    logger.info("Loaded historical data for %s", symbol)
                except Exception as e:
                    logger.error("Failed to store financials, sentiment, predictions, or report for %s: %s", symbol, e)
                    await session.rollback()
                    continue
            
            except Exception as e:
                logger.error("Error processing %s: %s", symbol, e)
                await session.rollback()
    
    logger.info("Historical data load completed")

if __name__ == "__main__":
    async def init_db():
        try:
            logger.debug("Creating async engine with DATABASE_URL: %s", DATABASE_URL)
            engine = create_async_engine(DATABASE_URL, echo=True)
            async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            logger.debug("Database engine created successfully with asyncpg")
            return async_session
        except Exception as e:
            logger.error("Failed to create database engine: %s", e)
            raise
    
    asyncio.run(load_historical_data(init_db))