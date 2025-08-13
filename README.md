Penny Stock Analysis
A dockerized monorepo for analyzing NSE penny stocks (< ₹5) with high volatility, supporting swing and intraday trading. Uses FastAPI, React, and PostgreSQL with Drizzle ORM.
Features

Daily data fetch (yfinance: prices, Alpha Vantage: financials, NewsAPI: sentiment).
Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, VWAP, ATR).
ML predictions (Random Forest, LSTM) with weekly retraining.
Entry/exit signals for swing and intraday trading.
Persists data/reports in PostgreSQL.
React SPA with charts, gauges, and tables.
Historical data loader for 2025.

Project Structure
penny-stock-analysis/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── daily_pipeline.py
│   │   ├── load_historical_data.py
│   │   └── models.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.js
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── README.md
├── .gitignore
└── LICENSE

Setup

Clone Repository:
git clone https://github.com/your-username/penny-stock-analysis.git
cd penny-stock-analysis


Environment:

Create backend/.env:ALPHA_VANTAGE_API_KEY=your_key
NEWS_API_KEY=your_key
DATABASE_URL=postgresql://user:password@postgres:5432/stock_db




Docker:

Install Docker and Docker Compose.
Build and run:docker-compose up --build


Access:
Frontend: http://localhost:3000
Backend API: http://localhost:8000/reports




Database Setup:

Connect to PostgreSQL:docker exec -it penny-stock-analysis-postgres-1 psql -U user -d stock_db


Create tables:CREATE TABLE stocks (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL UNIQUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE daily_data (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, date TIMESTAMP NOT NULL, open DOUBLE PRECISION NOT NULL, high DOUBLE PRECISION NOT NULL, low DOUBLE PRECISION NOT NULL, close DOUBLE PRECISION NOT NULL, volume DOUBLE PRECISION NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE financials (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, date TIMESTAMP NOT NULL, pe_ratio DOUBLE PRECISION NOT NULL, eps DOUBLE PRECISION NOT NULL, revenue DOUBLE PRECISION NOT NULL, debt_to_equity DOUBLE PRECISION NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE indicators (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, date TIMESTAMP NOT NULL, sma5 DOUBLE PRECISION NOT NULL, ema5 DOUBLE PRECISION NOT NULL, rsi DOUBLE PRECISION NOT NULL, macd DOUBLE PRECISION NOT NULL, macd_signal DOUBLE PRECISION NOT NULL, upper_bb DOUBLE PRECISION NOT NULL, middle_bb DOUBLE PRECISION NOT NULL, lower_bb DOUBLE PRECISION NOT NULL, vwap DOUBLE PRECISION NOT NULL, atr DOUBLE PRECISION NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE sentiments (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, date TIMESTAMP NOT NULL, score DOUBLE PRECISION NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE predictions (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, date TIMESTAMP NOT NULL, rf_prediction DOUBLE PRECISION NOT NULL, lstm_prediction DOUBLE PRECISION NOT NULL, entry_exit_signal TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE reports (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, date TIMESTAMP NOT NULL, report_text TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    date TIMESTAMP NOT NULL,
    model_type TEXT NOT NULL,
    accuracy DOUBLE PRECISION NOT NULL,
    parameters JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);




Load Historical Data:

Run:docker exec -it penny-stock-analysis-backend-1 python app/load_historical_data.py





Deployment

Local: Use docker-compose up.
Cloud:
Push to AWS ECS or Kubernetes.
Use managed PostgreSQL (e.g., AWS RDS).
Deploy frontend to Netlify/Vercel, backend to Heroku/EC2.



Notes

Rate Limits: Alpha Vantage (5 calls/min) may delay historical load (~400s for 34 stocks). Use premium key for faster processing.
TA-Lib: Dockerfile includes libta-lib0 and libta-lib-dev.
Daily Run: APScheduler runs pipeline at midnight IST.

License
MIT