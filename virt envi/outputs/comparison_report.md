# AP Research Strategy Comparison

## Study Design
- Assets: SPY, QQQ, AAPL
- Full data window: 2005-01-01 to 2025-12-31
- Training window: 2005-01-01 to 2018-12-31
- Test window: 2019-01-01 to 2025-12-31
- Strategy A: long/cash rule-based system using moving averages, momentum, and RSI.
- Strategy B: Logistic classifier predicting next-day direction.
- ML features: return_5d, return_10d, volatility_10d, close_vs_sma_20, sma_20_vs_sma_50, rsi_14, volume_change_5d
- Win rate is calculated at the trade level, not the daily level.

## Average Test Metrics Across Assets

```text
        strategy  cumulative_return  sharpe_ratio  maximum_drawdown  win_rate  number_of_trades
    Buy and Hold             3.8058        0.9780           -0.3407    1.0000            1.0000
Machine Learning             0.8773        0.5813           -0.3305    0.5749          167.0000
      Rule-Based             2.3477        1.0894           -0.2496    0.3950           81.6667
```

## Asset-Level Test Metrics

```text
 cumulative_return  sharpe_ratio  maximum_drawdown  win_rate  number_of_trades ticker         strategy test_start   test_end  probability_threshold
            0.9484        0.8945           -0.2187    0.4368           87.0000    SPY       Rule-Based 2019-01-02 2025-12-30                    NaN
            1.2065        0.8248           -0.2637    0.7456          228.0000    SPY Machine Learning 2019-01-02 2025-12-30                 0.5000
            2.0308        0.9027           -0.3372    1.0000            1.0000    SPY     Buy and Hold 2019-01-02 2025-12-30                    NaN
            1.9329        1.1049           -0.2864    0.3614           83.0000    QQQ       Rule-Based 2019-01-02 2025-12-30                    NaN
            0.1489        0.2112           -0.3752    0.5153          163.0000    QQQ Machine Learning 2019-01-02 2025-12-30                 0.5000
            3.1444        0.9628           -0.3512    1.0000            1.0000    QQQ     Buy and Hold 2019-01-02 2025-12-30                    NaN
            4.1617        1.2690           -0.2436    0.3867           75.0000   AAPL       Rule-Based 2019-01-02 2025-12-30                    NaN
            1.2765        0.7080           -0.3525    0.4636          110.0000   AAPL Machine Learning 2019-01-02 2025-12-30                 0.5000
            6.2421        1.0686           -0.3336    1.0000            1.0000   AAPL     Buy and Hold 2019-01-02 2025-12-30                    NaN
```

## ML Probability Thresholds

```text
ticker model_type  probability_threshold
   SPY   logistic                 0.5000
   QQQ   logistic                 0.5000
  AAPL   logistic                 0.5000
```