from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Stock(Base):
    __tablename__ = 'stocks'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

class DailyData(Base):
    __tablename__ = 'daily_data'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

class Financial(Base):
    __tablename__ = 'financials'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    pe_ratio = Column(Float, nullable=False)
    eps = Column(Float, nullable=False)
    revenue = Column(Float, nullable=False)
    debt_to_equity = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

class Indicator(Base):
    __tablename__ = 'indicators'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    sma5 = Column(Float, nullable=False)
    ema5 = Column(Float, nullable=False)
    rsi = Column(Float, nullable=False)
    macd = Column(Float, nullable=False)
    macd_signal = Column(Float, nullable=False)
    upper_bb = Column(Float, nullable=False)
    middle_bb = Column(Float, nullable=False)
    lower_bb = Column(Float, nullable=False)
    vwap = Column(Float, nullable=False)
    atr = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

class Sentiment(Base):
    __tablename__ = 'sentiments'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    rf_prediction = Column(Float, nullable=False)
    lstm_prediction = Column(Float, nullable=False)
    entry_exit_signal = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

class Report(Base):
    __tablename__ = 'reports'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    report_text = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

class ModelMetric(Base):
    __tablename__ = 'model_metrics'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    model_type = Column(String, nullable=False)
    accuracy = Column(Float, nullable=False)
    parameters = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())