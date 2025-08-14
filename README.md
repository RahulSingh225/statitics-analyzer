Penny Stock Analysis
A dockerized monorepo for analyzing NSE penny stocks (< ₹5) with high volatility, supporting swing and intraday trading. Uses FastAPI, React, and PostgreSQL with SQLAlchemy ORM.
Features

Daily data fetch (yfinance: prices, Alpha Vantage: financials, NewsAPI: sentiment).
Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, VWAP, ATR).
ML predictions (Random Forest, LSTM) with weekly retraining.
Entry/exit signals for swing and intraday trading.
Persists data/reports in PostgreSQL.
React SPA with charts, gauges, and tables.
Historical data loader for 2025.
Weekly ML model retraining with performance tracking.

Project Structure
statistics-analyzer/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── daily_pipeline.py
│   │   ├── load_historical_data.py
│   │   ├── weekly_retrain.py
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
git clone https://github.com/your-username/statistics-analyzer.git
cd statistics-analyzer


Environment:

Create backend/.env:ALPHA_VANTAGE_API_KEY=your_key
NEWS_API_KEY=your_key
DATABASE_URL=postgresql+asyncpg://user:password@postgres:5432/stock_db




Docker:

Install Docker and Docker Compose.
Build and run:docker-compose up --build


Access:
Frontend: http://localhost:3000
Backend API: http://localhost:8000/reports




Database Setup:

Connect to PostgreSQL:docker exec -it statistics-analyzer-postgres-1 psql -U user -d stock_db


Create tables:CREATE TABLE stocks (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL UNIQUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE daily_data (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, date TIMESTAMP NOT NULL, open DOUBLE PRECISION NOT NULL, high DOUBLE PRECISION NOT NULL, low DOUBLE PRECISION NOT NULL, close DOUBLE PRECISION NOT NULL, volume DOUBLE PRECISION NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE financials (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, date TIMESTAMP NOT NULL, pe_ratio DOUBLE PRECISION NOT NULL, eps DOUBLE PRECISION NOT NULL, revenue DOUBLE PRECISION NOT NULL, debt_to_equity DOUBLE PRECISION NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE indicators (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, date TIMESTAMP NOT NULL, sma5 DOUBLE PRECISION NOT NULL, ema5 DOUBLE PRECISION NOT NULL, rsi DOUBLE PRECISION NOT NULL, macd DOUBLE PRECISION NOT NULL, macd_signal DOUBLE PRECISION NOT NULL, upper_bb DOUBLE PRECISION NOT NULL, middle_bb DOUBLE PRECISION NOT NULL, lower_bb DOUBLE PRECISION NOT NULL, vwap DOUBLE PRECISION NOT NULL, atr DOUBLE PRECISION NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE sentiments (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, date TIMESTAMP NOT NULL, score DOUBLE PRECISION NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE predictions (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, date TIMESTAMP NOT NULL, rf_prediction DOUBLE PRECISION NOT NULL, lstm_prediction DOUBLE PRECISION NOT NULL, entry_exit_signal TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE reports (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, date TIMESTAMP NOT NULL, report_text TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);
CREATE TABLE model_metrics (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, date TIMESTAMP NOT NULL, model_type TEXT NOT NULL, accuracy DOUBLE PRECISION NOT NULL, parameters JSONB NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL);




Load Historical Data:

Run:docker exec -it statitics-analyzer-backend-1 python -m app.load_historical_data




Weekly Retraining:

Runs automatically every Sunday at midnight IST via APScheduler.
Check logs in backend/stock_analysis.log for retraining results.
Metrics stored in model_metrics table.
Manually test:docker exec -it statitics-analyzer-backend-1 python -m app.weekly_retrain





Deployment

Local: Use docker-compose up.
Cloud:
Push to AWS ECS or Kubernetes.
Use managed PostgreSQL (e.g., AWS RDS).
Deploy frontend to Netlify/Vercel, backend to Heroku/EC2.



Troubleshooting

TA-Lib Installation:

If pip install TA-Lib fails or you see config.guess: unable to guess system type, the issue is due to an outdated config.guess script in TA-Lib 0.4.0, especially on aarch64 systems (e.g., ARM-based Macs or servers).
The Dockerfile downloads updated config.guess and config.sub from GNU Savannah to fix this.
Check logs if the build fails:docker logs statitics-analyzer-backend-1


Verify TA-Lib installation:docker exec -it statitics-analyzer-backend-1 python -c "import talib; print(talib.__version__)"




Python Version Errors:

If you see errors like Could not find a version that satisfies the requirement for pandas==2.2.3 or other packages, ensure the Docker image uses Python >=3.10 (the Dockerfile uses python:3.10-slim).
Check Python version:docker exec -it statitics-analyzer-backend-1 python --version


If using a local environment, ensure Python 3.10+:python3 --version
python3 -m pip install --upgrade pip




Import Errors:

If you see ImportError: attempted relative import with no known parent package when running load_historical_data.py or weekly_retrain.py, use python -m app.<script_name> to run the script as a module:docker exec -it statitics-analyzer-backend-1 python -m app.load_historical_data


This ensures the app package context is preserved for relative imports.


Rate Limits: Alpha Vantage (5 calls/min) may delay historical load (~400s for 34 stocks). Use premium key for faster processing.


Notes

Python Version: The backend requires Python 3.10+ due to pandas==2.2.3 and other dependencies.
ORM: Uses SQLAlchemy with asyncpg for PostgreSQL database operations.
TA-Lib: Dockerfile compiles TA-Lib from source with updated config.guess and config.sub to support aarch64 architectures.
Daily Run: APScheduler runs pipeline at midnight IST.
Weekly Retraining: Runs Sundays at midnight IST, logs to backend/stock_analysis.log, stores metrics in model_metrics.

License
MIT