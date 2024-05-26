# Forecasting-Cryptocurrency-Prices-Using-Machine-Learning-Techniques
There have been many projects focusing on the forecasting of crypto prices but this project aims with a new methodology shown below.The Project is aimed to check whether fear & greed index, whale transactions have any significant impact and could they help in understanding spikes and downs of crypto on the 7 day forecasting of cryptocurrencies like Bitcoin, Ethereum, Tether, Binance Coin and US dollar coin using Machine Learning models like ARIMA, SARIMAX, LSTM and Bi-LSTM.

Predicting prices for any asset worldwide is never an easy job, but the highly volatile world of cryptocurrencies has garnered special attention in recent years with massive adoption. With the market open 24 hours a day and factor in everything from investor mood to macroeconomic events, whale transactions, fear and greed, price prediction is essential for investors as well as traders. Implementing various machine learning models, including LSTM (long short-term memory), Bi-LSTM ( bi directional long short term memory), ARIMA and SARIMAX to make forecasts at different granular levels for highly volatile prices for bitcoin, Ethereum, binance, coins like tether and USD coin.

Our study suggests that LSTM and bi-directional LSTM models display considerable potential for cryptocurrency forecasting, as they are able to accommodate both short-term and long-term correlations in the data. These deep learning models adapt to the particular nature of cryptocurrency markets, which makes them strong weapons in traders and investors hands. Nevertheless, it is important to keep in mind that such methods are sensitive hyperparameters and they require enormous amounts of training data.

On the other hand, autoregressive integrated moving average models (ARIMAs) are relatively good at predictions about all coin prices. This is especially true when it comes to stable coins or less volatile materials. Outside influences The potential of SARIMAX forecasting lies in the way it can account for all sorts of external factors.

Research Questions:
1•	Which ML models exhibit superior predictive capabilities among ARIMA, SARIMAX, LSTM and Bi-LSTM?
Among Statistical Models ARIMA model showed better accuracy than SARIMAX with the external factors. Among Neural Network Models LSTM showed slightly better results than Bi-LSTM. Overall ARIMA model showed better results in MAE scores.
2•	To what extent external factors like fear & Greed, Whale transactions improve accuracy of ML models.
External Factors like Fear & greed, Whale transactions did not improve MAE scores on a significant level, however plot results of actual & predicted curves show that they have improved models ability to capture trends & patterns.
3•	Does SARIMAX show any improved accuracy after adding external factors compared to ARIMA?
SARIMAX did not have any significant improved accuracy compared to ARIMA model as per MAE score but SARIMAX plots have shown its ability to capture trend based on fear & greed and whale transactions data.

