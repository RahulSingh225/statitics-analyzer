from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from dotenv import load_dotenv
import os
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from .daily_pipeline import daily_pipeline
from .weekly_retrain import weekly_retrain
from .models import Stock, Report, Indicator, Financial, Sentiment, Prediction, DailyData

# Logging setup
logging.basicConfig(filename='/app/stock_analysis.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def init_db():
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return async_session

@app.get("/reports")
async def get_reports():
    async_session = await init_db()
    async with async_session() as session:
        reports = (await session.execute(select(Report))).scalars().all()
        reports_data = []
        for report in reports:
            indicators = (await session.execute(
                select(Indicator).where(Indicator.symbol == report.symbol).where(Indicator.date == report.date)
            )).scalars().all()
            financials = (await session.execute(
                select(Financial).where(Financial.symbol == report.symbol).where(Financial.date == report.date)
            )).scalars().all()
            sentiments = (await session.execute(
                select(Sentiment).where(Sentiment.symbol == report.symbol).where(Sentiment.date == report.date)
            )).scalars().all()
            predictions = (await session.execute(
                select(Prediction).where(Prediction.symbol == report.symbol).where(Prediction.date == report.date)
            )).scalars().all()
            daily_data = (await session.execute(
                select(DailyData).where(DailyData.symbol == report.symbol)
            )).scalars().all()
            reports_data.append({
                "symbol": report.symbol,
                "date": report.date.isoformat(),
                "report_text": report.report_text,
                "indicators": [{"sma5": i.sma5, "ema5": i.ema5, "rsi": i.rsi, "macd": i.macd, "macd_signal": i.macd_signal,
                               "upper_bb": i.upper_bb, "middle_bb": i.middle_bb, "lower_bb": i.lower_bb, "vwap": i.vwap, "atr": i.atr,
                               "date": i.date.isoformat()} for i in indicators],
                "financials": [{"pe_ratio": f.pe_ratio, "eps": f.eps, "revenue": f.revenue, "debt_to_equity": f.debt_to_equity,
                               "date": f.date.isoformat()} for f in financials],
                "sentiments": [{"score": s.score, "date": s.date.isoformat()} for s in sentiments],
                "predictions": [{"rf_prediction": p.rf_prediction, "lstm_prediction": p.lstm_prediction,
                               "entry_exit_signal": p.entry_exit_signal, "date": p.date.isoformat()} for p in predictions],
                "daily_data": [{"open": d.open, "high": d.high, "low": d.low, "close": d.close, "volume": d.volume,
                               "date": d.date.isoformat()} for d in daily_data]
            })
        return {"reports": reports_data}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Scheduler
scheduler = AsyncIOScheduler()
scheduler.add_job(daily_pipeline, 'cron', hour=0, args=[init_db])  # Daily at midnight IST
scheduler.add_job(weekly_retrain, 'cron', day_of_week='sun', hour=0, args=[init_db])  # Weekly Sunday at midnight IST
scheduler.start()