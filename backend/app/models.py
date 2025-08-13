from drizzle_orm import pgTable, serial, text, jsonb, float64, timestamp

stocks = pgTable('stocks', [
    serial('id').primaryKey(),
    text('symbol').notNull().unique(),
    timestamp('created_at').defaultNow().notNull()
])

daily_data = pgTable('daily_data', [
    serial('id').primaryKey(),
    text('symbol').notNull(),
    timestamp('date').notNull(),
    float64('open').notNull(),
    float64('high').notNull(),
    float64('low').notNull(),
    float64('close').notNull(),
    float64('volume').notNull(),
    timestamp('created_at').defaultNow().notNull()
])

financials = pgTable('financials', [
    serial('id').primaryKey(),
    text('symbol').notNull(),
    timestamp('date').notNull(),
    float64('pe_ratio').notNull(),
    float64('eps').notNull(),
    float64('revenue').notNull(),
    float64('debt_to_equity').notNull(),
    timestamp('created_at').defaultNow().notNull()
])

indicators = pgTable('indicators', [
    serial('id').primaryKey(),
    text('symbol').notNull(),
    timestamp('date').notNull(),
    float64('sma5').notNull(),
    float64('ema5').notNull(),
    float64('rsi').notNull(),
    float64('macd').notNull(),
    float64('macd_signal').notNull(),
    float64('upper_bb').notNull(),
    float64('middle_bb').notNull(),
    float64('lower_bb').notNull(),
    float64('vwap').notNull(),
    float64('atr').notNull(),
    timestamp('created_at').defaultNow().notNull()
])

sentiments = pgTable('sentiments', [
    serial('id').primaryKey(),
    text('symbol').notNull(),
    timestamp('date').notNull(),
    float64('score').notNull(),
    timestamp('created_at').defaultNow().notNull()
])

predictions = pgTable('predictions', [
    serial('id').primaryKey(),
    text('symbol').notNull(),
    timestamp('date').notNull(),
    float64('rf_prediction').notNull(),
    float64('lstm_prediction').notNull(),
    text('entry_exit_signal').notNull(),
    timestamp('created_at').defaultNow().notNull()
])

reports = pgTable('reports', [
    serial('id').primaryKey(),
    text('symbol').notNull(),
    timestamp('date').notNull(),
    text('report_text').notNull(),
    timestamp('created_at').defaultNow().notNull()
])

model_metrics = pgTable('model_metrics', [
    serial('id').primaryKey(),
    text('symbol').notNull(),
    timestamp('date').notNull(),
    text('model_type').notNull(),  # 'rf' or 'lstm'
    float64('accuracy').notNull(),
    jsonb('parameters').notNull(),  # Best params for RF or LSTM config
    timestamp('created_at').defaultNow().notNull()
])