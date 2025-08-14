import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert
from .models import Stock, DailyData, Indicator, ModelMetric
import logging

logging.basicConfig(filename='/app/stock_analysis.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def prepare_lstm_data(df, features, lookback=10):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features].dropna())
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(1 if df['close'].iloc[i] > df['close'].iloc[i-1] else 0)
    return np.array(X), np.array(y), scaler

async def train_lstm_model(X, y):
    model = Sequential([
        LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
    return model

async def weekly_retrain(init_db):
    logging.info("Starting weekly retraining")
    
    async_session = await init_db()
    async with async_session() as session:
        # Fetch all symbols
        stocks = (await session.execute(select(Stock))).scalars().all()
        symbols = [stock.symbol for stock in stocks]
        
        for symbol in symbols:
            try:
                # Fetch historical data
                data_rows = (await session.execute(
                    select(DailyData).where(DailyData.symbol == symbol)
                )).scalars().all()
                indicator_rows = (await session.execute(
                    select(Indicator).where(Indicator.symbol == symbol)
                )).scalars().all()
                
                if not data_rows or not indicator_rows:
                    logging.warning(f"No data for {symbol}")
                    continue
                
                df_data = pd.DataFrame([
                    {'symbol': d.symbol, 'date': d.date, 'open': d.open, 'high': d.high, 'low': d.low,
                     'close': d.close, 'volume': d.volume} for d in data_rows
                ])
                df_indicators = pd.DataFrame([
                    {'symbol': i.symbol, 'date': i.date, 'sma5': i.sma5, 'ema5': i.ema5, 'rsi': i.rsi,
                     'macd': i.macd, 'macd_signal': i.macd_signal, 'upper_bb': i.upper_bb,
                     'middle_bb': i.middle_bb, 'lower_bb': i.lower_bb, 'vwap': i.vwap, 'atr': i.atr}
                    for i in indicator_rows
                ])
                
                # Merge on date
                df = df_data.merge(df_indicators, on=['symbol', 'date'], how='inner')
                if df.empty:
                    logging.warning(f"No merged data for {symbol}")
                    continue
                
                # Prepare features
                features = ['sma5', 'ema5', 'rsi', 'macd', 'volume', 'upper_bb', 'lower_bb', 'atr']
                X = df[features].dropna()
                y = (df['close'].shift(-1) > df['close']).astype(int)[X.index]
                
                if len(X) < 10:
                    logging.warning(f"Insufficient data for {symbol}")
                    continue
                
                train_size = int(0.8 * len(X))
                X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
                y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
                
                # Retrain Random Forest with grid search
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None]
                }
                rf_model = RandomForestClassifier(random_state=42)
                grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='accuracy')
                grid_search.fit(X_train, y_train)
                rf_accuracy = accuracy_score(y_val, grid_search.predict(X_val))
                
                # Store RF metrics
                await session.execute(insert(ModelMetric).values(
                    symbol=symbol,
                    date=datetime.now(),
                    model_type='rf',
                    accuracy=float(rf_accuracy),
                    parameters=grid_search.best_params_
                ))
                logging.info(f"Retrained RF for {symbol}: accuracy={rf_accuracy:.2%}, params={grid_search.best_params_}")
                
                # Retrain LSTM
                X_lstm, y_lstm, _ = await prepare_lstm_data(df, features)
                if len(X_lstm) > 0:
                    lstm_model = await train_lstm_model(X_lstm, y_lstm)
                    lstm_predictions = (lstm_model.predict(X_lstm) > 0.5).astype(int)
                    lstm_accuracy = accuracy_score(y_lstm, lstm_predictions)
                    
                    # Store LSTM metrics
                    await session.execute(insert(ModelMetric).values(
                        symbol=symbol,
                        date=datetime.now(),
                        model_type='lstm',
                        accuracy=float(lstm_accuracy),
                        parameters={'units': 50, 'epochs': 10, 'batch_size': 32}
                    ))
                    logging.info(f"Retrained LSTM for {symbol}: accuracy={lstm_accuracy:.2%}")
                else:
                    logging.warning(f"Insufficient LSTM data for {symbol}")
                
                await session.commit()
            
            except Exception as e:
                logging.error(f"Error retraining for {symbol}: {e}")
                await session.rollback()
    
    logging.info("Weekly retraining completed")