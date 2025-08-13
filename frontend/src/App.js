import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import ReactMarkdown from 'react-markdown';
import GaugeChart from 'react-gauge-chart';
import './App.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    axios.get('http://backend:8000/reports')
      .then(response => {
        setReports(response.data.reports);
        setLoading(false);
      })
      .catch(err => {
        setError('Failed to load reports');
        setLoading(false);
      });
  }, []);

  const renderPriceChart = (data) => {
    if (!data) return null;
    const labels = data.map(item => item.date);
    const closePrices = data.map(item => item.close);
    const sma5 = data.map(item => item.SMA_5);
    const ema5 = data.map(item => item.EMA_5);

    const chartData = {
      labels,
      datasets: [
        { label: 'Close Price', data: closePrices, borderColor: 'blue', fill: false },
        { label: 'SMA 5', data: sma5, borderColor: 'green', fill: false },
        { label: 'EMA 5', data: ema5, borderColor: 'red', fill: false },
      ],
    };

    return <Line data={chartData} options={{ responsive: true, scales: { x: { type: 'time' } } }} />;
  };

  const renderRSIChart = (data) => {
    if (!data) return null;
    const labels = data.map(item => item.date);
    const rsi = data.map(item => item.rsi);

    const chartData = {
      labels,
      datasets: [{ label: 'RSI', data: rsi, borderColor: 'purple', fill: false }],
    };

    return <Line data={chartData} options={{ responsive: true, scales: { x: { type: 'time' } } }} />;
  };

  const renderSentimentGauge = (score) => {
    return (
      <GaugeChart id={`sentiment-gauge-${Math.random()}`}
        nrOfLevels={20}
        percent={(score + 1) / 2}
        textColor="#000"
        style={{ width: '200px' }}
      />
    );
  };

  const renderFinancialTable = (financials) => {
    if (!financials || !financials.length) return null;
    const latest = financials[0];
    return (
      <table className="metrics-table">
        <thead>
          <tr><th>Metric</th><th>Value</th></tr>
        </thead>
        <tbody>
          <tr><td>P/E Ratio</td><td>{latest.pe_ratio.toFixed(2)}</td></tr>
          <tr><td>EPS</td><td>{latest.eps.toFixed(2)}</td></tr>
          <tr><td>Revenue (TTM)</td><td>{latest.revenue.toLocaleString()}</td></tr>
          <tr><td>Debt-to-Equity</td><td>{latest.debt_to_equity.toFixed(2)}</td></tr>
        </tbody>
      </table>
    );
  };

  const renderIndicatorsTable = (indicators) => {
    if (!indicators || !indicators.length) return null;
    const latest = indicators[0];
    return (
      <table className="metrics-table">
        <thead>
          <tr><th>Indicator</th><th>Value</th></tr>
        </thead>
        <tbody>
          <tr><td>SMA 5</td><td>{latest.sma5 ? latest.sma5.toFixed(2) : 'N/A'}</td></tr>
          <tr><td>EMA 5</td><td>{latest.ema5 ? latest.ema5.toFixed(2) : 'N/A'}</td></tr>
          <tr><td>RSI</td><td>{latest.rsi ? latest.rsi.toFixed(2) : 'N/A'}</td></tr>
          <tr><td>MACD</td><td>{latest.macd ? latest.macd.toFixed(2) : 'N/A'}</td></tr>
          <tr><td>VWAP</td><td>{latest.vwap ? latest.vwap.toFixed(2) : 'N/A'}</td></tr>
          <tr><td>ATR</td><td>{latest.atr ? latest.atr.toFixed(2) : 'N/A'}</td></tr>
        </tbody>
      </table>
    );
  };

  return (
    <div className="app-container">
      <h1>Penny Stock Analysis Dashboard</h1>
      {loading && <p>Loading reports...</p>}
      {error && <p className="error">{error}</p>}
      
      <div className="stock-grid">
        {reports.map(report => (
          <div key={report.symbol} className="stock-card">
            <h2>{report.symbol}</h2>
            <ReactMarkdown>{report.report_text}</ReactMarkdown>
            
            <h3>Financial Metrics</h3>
            {renderFinancialTable(report.financials)}
            
            <h3>Technical Indicators (Latest)</h3>
            {renderIndicatorsTable(report.indicators)}
            
            <h3>Sentiment Gauge</h3>
            {renderSentimentGauge(report.sentiment[0]?.score || 0)}
            
            <h3>Price Chart</h3>
            {renderPriceChart(report.data)}
            
            <h3>RSI Chart</h3>
            {renderRSIChart(report.indicators)}
            
            <h3>Static Plot from Backend</h3>
            <img src={`data:image/png;base64,${report.plot_base64}`} alt={`${report.symbol} Plot`} className="static-plot" />
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;