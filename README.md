# Stock Market Prediction Experiment

An experimental project exploring deep learning approaches for stock market prediction. This project implements various neural network architectures (LSTM, GRU, and Transformer) and combines technical indicators to study their effectiveness in market prediction.

## 🔍 Features

- **Neural Network Models**
  - LSTM Networks
  - GRU Networks
  - Transformer Models
  - Ensemble Methods

- **Technical Indicators**
  - Moving Averages (SMA, EMA)
  - Relative Strength Index (RSI)
  - MACD
  - Volume Analysis

- **Data Processing**
  - Data fetching from Yahoo Finance
  - Basic data cleaning
  - Feature engineering
  - Data normalization

- **Model Training**
  - Cross-validation
  - Hyperparameter tuning with Optuna
  - Performance metrics tracking
  - MLflow experiment logging

## 📊 Metrics

- Mean Absolute Percentage Error (MAPE)
- Root Mean Square Error (RMSE)
- Directional Accuracy
- Returns Analysis

## 🛠️ Tech Stack

- Python 3.8+
- TensorFlow 2.x
- Pandas & NumPy
- Scikit-learn
- YFinance API
- Technical Analysis Library (TA-Lib)
- MLflow
- Optuna

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stonks.git
   cd stonks
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

4. Run the experiment:
   ```bash
   python src/main.py
   ```

## 📁 Project Structure

```
stonks/
├── src/                  # Source code
│   ├── data/            # Data collection and preprocessing
│   ├── models/          # Neural network models
│   ├── utils/           # Helper functions
│   └── visualization/   # Plotting tools
├── data/                # Data directory (created on first run)
│   ├── raw/            # Raw data (will be downloaded)
│   └── processed/      # Processed data
├── models/              # Saved models directory
├── visualizations/      # Output visualizations
├── tests/              # Unit tests
├── LICENSE             # MIT License
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```

## 📥 Data Handling

The project automatically downloads stock data using the Yahoo Finance API when you run the experiment. The data will be stored in:
- `data/raw/`: Raw downloaded stock data
- `data/processed/`: Processed and feature-engineered data

These directories are git-ignored to keep the repository light. The data will be downloaded fresh when you run the experiment.

## ⚠️ Disclaimer

This project is purely experimental and should not be used for actual trading decisions. Stock market prediction is inherently difficult and no model can guarantee accurate predictions.

## 🤝 Contributing

Feel free to contribute to this experiment! Open an issue or submit a pull request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 