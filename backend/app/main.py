from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from drizzle_orm import Drizzle
import asyncpg
from dotenv import load_dotenv
import os
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from .daily_pipeline import daily_pipeline
from .weekly_retrain import weekly_retrain
from .models import reports, indicators, financials, sentiments, predictions, daily_data

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
    pool = await asyncpg.create_pool(DATABASE_URL)
    return Drizzle(pool)

@app.get("/reports")
async def get_reports():
    drizzle = await init_db()
    results = await reports.select().execute(drizzle)
    reports_data = [
        {
            "symbol": row['symbol'],
            "date": row['date'].isoformat(),
            "report_text": row['report_text'],
            "indicators": await indicators.select().where(indicators.symbol == row['symbol']).where(indicators.date == row['date']).execute(drizzle),
            "financials": await financials.select().where(financials.symbol == row['symbol']).where(financials.date == row['date']).execute(drizzle),
            "sentiment": await sentiments.select().where(sentiments.symbol == row['symbol']).where(sentiments.date == row['date']).execute(drizzle),
            "prediction": await predictions.select().where(predictions.symbol == row['symbol']).where(predictions.date == row['date']).execute(drizzle),
            "data": await daily_data.select().where(daily_data.symbol == row['symbol']).execute(drizzle)
        } for row in results
    ]
    await drizzle.pool.close()
    return {"reports": reports_data}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Scheduler
scheduler = AsyncIOScheduler()
scheduler.add_job(daily_pipeline, 'cron', hour=0, args=[init_db()])  # Daily at midnight IST
scheduler.add_job(weekly_retrain, 'cron', day_of_week='sun', hour=0, args=[init_db()])  # Weekly Sunday at midnight IST
scheduler.start()