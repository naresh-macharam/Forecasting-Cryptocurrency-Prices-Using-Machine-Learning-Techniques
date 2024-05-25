import streamlit as streamlit
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
# evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

from statsmodels.tsa.statespace.sarimax import SARIMAX  # for SARIMAX model implementation
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense
from keras.optimizers import Adam

# Title
streamlit.title("Crypto currency price forecasting")


def arima_forecast(train, test, crypto_name):
    # Train ARIMA model
    model = ARIMA(train, order=(2, 0, 3))
    model_fit = model.fit()

    forecast_steps = len(test)
    # Make prediction
    prediction = model_fit.forecast(steps=forecast_steps)

    df_forecast = pd.DataFrame(prediction.values, columns=['Forecasted Close'], index=test.index)
    print("ARIMA VALUES")
    print(df_forecast.head())
    #plt.figure(figsize=(12, 6))
    #plt.plot(test, color='red', label='Actual')
    #plt.plot(df_forecast['Forecasted Close'], ls='--', color='blue', label='predicted')
    #plt.xlabel('Date')
    #plt.ylabel('Seasonal difference close price')
    #plt.title(f'{crypto_name} Actual vs Predicted ARIMA Close values')
    #plt.legend()
    #plt.show()

    #streamlit.set_option('deprecation.showPyplotGlobalUse', False)
    #streamlit.pyplot()
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(test, df_forecast['Forecasted Close'])

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(test, df_forecast['Forecasted Close'])

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # R2score
    r2 = r2_score(test, df_forecast['Forecasted Close'])

    # Print the metrics
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R2 score: {r2}")

    return df_forecast

def merged_data_and_plot(df,df_forecast):
    df3 = df[['Close', 'box_diff_seasonal_1']]
    merged_df = pd.merge(df3, df_forecast, left_index=True, right_index=True, how='inner')
    merged_df['Forecasted Close'].iloc[0] = 0  # Assign first close value as 0
    merged_df['Forecasted Close'] = merged_df['Forecasted Close'].shift(-1)  # move values up
    merged_df['Forecasted Close'].iloc[0] = merged_df['Close'].iloc[0]
    merged_df['cumulative_sum_close'] = merged_df['Forecasted Close'].cumsum()  # make cummulative sum to find actual predictions

    plt.figure(figsize=(12, 6))
    plt.plot(merged_df['Close'], color='red', label='Actual')
    plt.plot(merged_df['cumulative_sum_close'], ls='--', color='blue', label='predicted')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Close Price')
    plt.legend()
    plt.show()
    streamlit.set_option('deprecation.showPyplotGlobalUse', False)
    streamlit.pyplot()

def sarimax_forecast(train, test, exog, days, crypto_name):
    model = sm.tsa.SARIMAX(train, exog=exog[:-24 * days], order=(p, d, q), seasonal_order=(P, D, Q, s))
    model_fit = model.fit()
    forecast_steps = len(test)
    forecast = model_fit.get_forecast(steps=forecast_steps, exog=exog[-24 * days:])

    df_forecast = pd.DataFrame(forecast.predicted_mean, columns=['Forecasted'], index=test.index)
    forecast = forecast.predicted_mean
    forecast = pd.DataFrame(forecast)
    forecast.reset_index(inplace=True)

    df_forecast.reset_index(inplace=True)
    df_forecast['Forecasted Close'] = forecast.predicted_mean
    df_forecast.set_index('Date', inplace=True)
    df_forecast.drop('Forecasted', axis=1, inplace=True)

    print("SARIMAX VALUES")
    print(df_forecast.head())

    #plt.figure(figsize=(12, 6))
    #plt.plot(test, color='red', label='Actual')
    #plt.plot(df_forecast, ls='--', color='blue', label='predicted')
    #plt.xlabel('Date')
    #plt.ylabel('Close Price')
    #plt.title(f'{crypto_name} Actual vs Predicted SARIMAX close values')
    #plt.legend()
    #plt.show()

    #streamlit.set_option('deprecation.showPyplotGlobalUse', False)
    #streamlit.pyplot()
    return df_forecast


def apply_lstm_sequences(data, company_name, features, target, test_size=0.2, random_state=False, num_epochs=50,
                         batch_size=32, sequence_length=10):
    X = data[features].values
    y = data[target].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Split data into training and testing sets
    # train_size = int((1 - test_size) * len(X_scaled))
    train_size = len(X_scaled) - test_size
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

    # Create sequences
    X_train_seq, y_train_seq = [], []
    X_test_seq, y_test_seq = [], []

    for i in range(len(X_train) - sequence_length):
        X_train_seq.append(X_train[i:i + sequence_length])
        y_train_seq.append(y_train[i + sequence_length])

    for i in range(len(X_test) - sequence_length):
        X_test_seq.append(X_test[i:i + sequence_length])
        y_test_seq.append(y_test[i + sequence_length])

    X_train_seq, y_train_seq = np.array(X_train_seq), np.array(y_train_seq)
    X_test_seq, y_test_seq = np.array(X_test_seq), np.array(y_test_seq)

    # Build and train LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train_seq, y_train_seq, epochs=num_epochs, batch_size=batch_size, validation_split=0.2,
                        verbose=1)
    model.summary()
    # Plot validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{company_name} Training and Validation Loss')
    plt.legend()
    plt.show()
    streamlit.set_option('deprecation.showPyplotGlobalUse', False)
    streamlit.pyplot()

    # Evaluate the model on training data
    y_train_pred_scaled = model.predict(X_train_seq)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    y_train_orig = scaler_y.inverse_transform(y_train_seq)

    # Calculate RMSE, MSE, and MAE for training set
    train_rmse = np.sqrt(mean_squared_error(y_train_orig, y_train_pred))
    train_mse = mean_squared_error(y_train_orig, y_train_pred)
    train_mae = mean_absolute_error(y_train_orig, y_train_pred)
    # R2score
    train_r2 = r2_score(y_train_orig, y_train_pred)

    print(f"{company_name} Training Set:")
    print(f"RMSE: {train_rmse:}")
    print(f"MSE: {train_mse:}")
    print(f"MAE: {train_mae:}")
    print(f"R2 score: {train_r2:.2f}")

    # Plot actual vs. predicted values for training set
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-len(y_train_orig):], y_train_orig, label='Actual Train')
    plt.plot(data.index[-len(y_train_orig):], y_train_pred, label='Predicted Train')
    plt.xlabel('Date Time')
    plt.ylabel('crypto Price')
    plt.title(f'{company_name} Training Set: Actual vs. Predicted crypto Prices')
    plt.legend()
    plt.show()
    streamlit.set_option('deprecation.showPyplotGlobalUse', False)
    streamlit.pyplot()

    # Evaluate the model on test data
    y_pred_scaled = model.predict(X_test_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test_seq)

    # Calculate RMSE, MSE, and MAE for test set
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    mse = mean_squared_error(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    test_r2 = r2_score(y_test_orig, y_pred)

    print(f"{company_name} Test Set:")
    print(f"RMSE: {rmse:}")
    print(f"MSE: {mse:}")
    print(f"MAE: {mae:}")
    print(f"test R2 score: {test_r2:.2f}")
    # scaled results
    # Plot actual vs. predicted values for test scaled set
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-len(y_test_orig):], y_test_seq, label='Actual Test')
    plt.plot(data.index[-len(y_test_orig):], y_pred_scaled, label='Predicted Test')
    plt.xlabel('Date Time')
    plt.ylabel('crypto Price')
    plt.title(f'{company_name} Test Set: Actual vs. Predicted scaled Prices')
    plt.legend()
    plt.show()
    streamlit.set_option('deprecation.showPyplotGlobalUse', False)
    streamlit.pyplot()

    # Plot actual vs. predicted values for test set
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-len(y_test_orig):], y_test_orig, label='Actual Test')
    plt.plot(data.index[-len(y_test_orig):], y_pred, label='Predicted Test')
    plt.xlabel('Date Time')
    plt.ylabel('crypto Price')
    plt.title(f'{company_name} Test Set: Actual vs. Predicted crypto Prices')
    plt.legend()
    plt.show()
    streamlit.set_option('deprecation.showPyplotGlobalUse', False)
    streamlit.pyplot()


def apply_bi_lstm_sequences(data, company_name, features, target, test_size=0.2, random_state=42, num_epochs=50,
                            batch_size=32, sequence_length=10):
    # Preprocessing

    X = data[features].values
    y = data[target].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Split data into training and testing sets
    # train_size = int((1 - test_size) * len(X_scaled))
    train_size = len(X_scaled) - test_size
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

    # Create sequences
    X_train_seq, y_train_seq = [], []
    X_test_seq, y_test_seq = [], []

    for i in range(len(X_train) - sequence_length):
        X_train_seq.append(X_train[i:i + sequence_length])
        y_train_seq.append(y_train[i + sequence_length])

    for i in range(len(X_test) - sequence_length):
        X_test_seq.append(X_test[i:i + sequence_length])
        y_test_seq.append(y_test[i + sequence_length])

    X_train_seq, y_train_seq = np.array(X_train_seq), np.array(y_train_seq)
    X_test_seq, y_test_seq = np.array(X_test_seq), np.array(y_test_seq)

    # Build and train Bi-LSTM model
    model = Sequential()
    model.add(
        Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Bidirectional(LSTM(units=50)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train_seq, y_train_seq, epochs=num_epochs, batch_size=batch_size, validation_split=0.2,
                        verbose=1)
    model.summary()
    # Plot validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{company_name} Training and Validation Loss')
    plt.legend()
    plt.show()
    streamlit.set_option('deprecation.showPyplotGlobalUse', False)
    streamlit.pyplot()

    # Evaluate the model on training data
    y_train_pred_scaled = model.predict(X_train_seq)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    y_train_orig = scaler_y.inverse_transform(y_train_seq)

    # Calculate RMSE, MSE, and MAE for training set
    train_rmse = np.sqrt(mean_squared_error(y_train_orig, y_train_pred))
    train_mse = mean_squared_error(y_train_orig, y_train_pred)
    train_mae = mean_absolute_error(y_train_orig, y_train_pred)
    # R2score
    train_r2 = r2_score(y_train_orig, y_train_pred)

    print(f"{company_name} Training Set:")
    print(f"RMSE: {train_rmse:.2f}")
    print(f"MSE: {train_mse:.2f}")
    print(f"MAE: {train_mae:.2f}")
    print(f"R2: {train_r2:.2f}")
    # Plot actual vs. predicted values for training set
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-len(y_train_orig):], y_train_orig, label='Actual Train')
    plt.plot(data.index[-len(y_train_orig):], y_train_pred, label='Predicted Train')
    plt.xlabel('Date Time')
    plt.ylabel('crypto Price')
    plt.title(f'{company_name} Training Set: Actual vs. Predicted crypto Prices')
    plt.legend()
    plt.show()
    streamlit.set_option('deprecation.showPyplotGlobalUse', False)
    streamlit.pyplot()

    # Evaluate the model on test data
    y_pred_scaled = model.predict(X_test_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test_seq)

    # Calculate RMSE, MSE, and MAE for test set
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    mse = mean_squared_error(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    test_r2 = r2_score(y_test_orig, y_pred)

    print(f"{company_name} Test Set:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {test_r2:.2f}")

    # scaled results
    # Plot actual vs. predicted values for test set
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-len(y_test_orig):], y_test_seq, label='Actual Test')
    plt.plot(data.index[-len(y_test_orig):], y_pred_scaled, label='Predicted Test')
    plt.xlabel('Date Time')
    plt.ylabel('crypto Price')
    plt.title(f'{company_name} Test Set: Actual vs. Predicted scaled Prices')
    plt.legend()
    plt.show()
    streamlit.set_option('deprecation.showPyplotGlobalUse', False)
    streamlit.pyplot()

    # Plot actual vs. predicted values for test set
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-len(y_test_orig):], y_test_orig, label='Actual Test')
    plt.plot(data.index[-len(y_test_orig):], y_pred, label='Predicted Test')
    plt.xlabel('Date Time')
    plt.ylabel('crypto Price')
    plt.title(f'{company_name} Test Set: Actual vs. Predicted crypto Prices')
    plt.legend()
    plt.show()
    streamlit.set_option('deprecation.showPyplotGlobalUse', False)
    streamlit.pyplot()


import streamlit as st

#st.title("Crypto Data Exploration")

    # Create tabs
tabs = ["Fear and Greed Data", "Fear and Greed & Whale Transactions Data"]
selected_tab = st.sidebar.radio("Select Data", tabs)

    # Display the selected tab
if selected_tab == "Fear and Greed Data":

    import streamlit as st
    import time
    import threading
    # Function that simulates a long-running task
    def long_running_task():
        for i in range(10):
            time.sleep(1)
            st.write(f"Task is running... {i}")
    # Button to interrupt the code
    if st.button("Interrupt Code"):
            # Create a thread for the long-running task
            task_thread = threading.Thread(target=long_running_task)

            # Start the thread
            task_thread.start()

            # Wait for the thread to finish
            task_thread.join()

            st.write("Code interrupted successfully!")
    # Main button

    option = st.selectbox("Select an option", ["None","Ethereum", "Bitcoin","Binance", "Tether", "USD Coin"])

        # Check which option is selected and trigger code accordingly
    if option == "None":
            st.write("Please select an option")
    elif option == "Ethereum":
            st.write("Ethereum")
            # Add your code for Option 1
            streamlit.title('Ethereum Fear & Greed Influence')
            streamlit.header('ARIMA prediction')
            data = pd.read_csv('df_eth_csv.csv')
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            df = data
            data_arima = df
            data_arima = pd.DataFrame(data_arima)
            data_arima['box_diff_seasonal_1'] = (data['Close'] - data['Close'].shift(1))
            data = data['Close']
            print(data)

            train = data_arima[:-24 * 7]
            test = data_arima[-24 * 7:]


            #st.header("ARIMA Forecast")
            df_forecast = arima_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], 'Ethereum')

            merged_data_and_plot(df, df_forecast)

            p, d, q = 1, 0, 1  # Non-seasonal order
            P, D, Q, s = 2, 1, 2, 7  # Seasonal order with daily seasonality
            #train = data[:-24 * 7]
            #test = data[-24 * 7:]
            # df['box_diff_seasonal_1'] = (df['Close'] - df['Close'].shift(1))
            days = 7
            exog = df['value']
            print(df.columns)
            three_exog = df[['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']]
            st.header("SARIMAX Forecast")

            df_forecast = sarimax_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], exog, days, 'Ethereum')
            merged_data_and_plot(df, df_forecast)
            st.header("LSTM Forecast")
            data = df
            features = ['value']
            target = 'Close'
            test_size = 168
            random_state = False
            num_epochs = 15
            batch_size = 32
            sequence_length = 10

            # Call the function
            apply_lstm_sequences(data, 'Ethereum', features, target, test_size, random_state, num_epochs, batch_size,
                                 sequence_length)

            st.header("Bi-LSTM Forecast")
            data = df
            features = ['value']
            target = 'Close'
            test_size = 168
            random_state = False
            num_epochs = 15
            batch_size = 32
            sequence_length = 10

            # Call the function
            apply_bi_lstm_sequences(data, 'Ethereum', features, target, test_size, random_state, num_epochs, batch_size,
                                    sequence_length)

    elif option == "Bitcoin":
            st.write("You selected Bitcoin.")
            # Add your code for Option 2
            streamlit.title('Bitcoin Fear & Greed Influence')

            data = pd.read_csv('df_btc.csv')
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            df = data
            data_arima = df
            data_arima = pd.DataFrame(data_arima)
            data_arima['box_diff_seasonal_1'] = (data['Close'] - data['Close'].shift(1))
            data = data['Close']
            print(data)

            train = data_arima[:-24 * 7]
            test = data_arima[-24 * 7:]
            st.header("ARIMA Forecast")
            #arima_forecast(train, test, 'Bitcoin')

            df_forecast = arima_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], 'Bitcoin')

            merged_data_and_plot(df,df_forecast)
            p, d, q = 1, 0, 1  # Non-seasonal order
            P, D, Q, s = 2, 1, 2, 7  # Seasonal order with daily seasonality

            # df['box_diff_seasonal_1'] = (df['Close'] - df['Close'].shift(1))
            days = 7
            exog = df['value']
            print(df.columns)
            three_exog = df[['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']]
            st.header("SARIMAX Forecast")
            df_forecast = sarimax_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], exog, days,
                                           'Bitcoin')
            merged_data_and_plot(df, df_forecast)
            st.header("LSTM Forecast")
            data = df
            features = ['value']
            target = 'Close'
            test_size = 168
            random_state = False
            num_epochs = 15
            batch_size = 32
            sequence_length = 10

            # Call the function
            apply_lstm_sequences(data, 'Bitcoin', features, target, test_size, random_state, num_epochs, batch_size,
                                 sequence_length)

            st.header("Bi-LSTM Forecast")
            data = df
            features = ['value']
            target = 'Close'
            test_size = 168
            random_state = False
            num_epochs = 15
            batch_size = 32
            sequence_length = 10

            # Call the function
            apply_bi_lstm_sequences(data, 'Bitcoin', features, target, test_size, random_state, num_epochs, batch_size,
                                    sequence_length)

    elif option == "Binance":
        st.write("You selected Binance.")
        # Add your code for Option 2
        streamlit.title('Binance Fear & Greed Influence')

        # streamlit.header('BNB ARIMA prediction')
        data = pd.read_csv('df_bnb_csv.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        df = data
        data_arima = df
        data_arima = pd.DataFrame(data_arima)
        data_arima['box_diff_seasonal_1'] = (data['Close'] - data['Close'].shift(1))
        data = data['Close']
        print(data)

        train = data_arima[:-24 * 7]
        test = data_arima[-24 * 7:]
        st.header("ARIMA Forecast")
       # arima_forecast(train, test, 'Binance')
        df_forecast = arima_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], 'Binance')

        merged_data_and_plot(df,df_forecast)
        p, d, q = 1, 0, 1  # Non-seasonal order
        P, D, Q, s = 2, 1, 2, 7  # Seasonal order with daily seasonality

        # df['box_diff_seasonal_1'] = (df['Close'] - df['Close'].shift(1))
        days = 7
        exog = df['value']
        print(df.columns)
        three_exog = df[['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']]
        st.header("SARIMAX Forecast")
        df_forecast = sarimax_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], exog, days,
                                       'Binance')
        merged_data_and_plot(df,df_forecast)
        st.header("LSTM Forecast")
        data = df
        features = ['value']
        target = 'Close'
        test_size = 168
        random_state = False
        num_epochs = 15
        batch_size = 32
        sequence_length = 10

        # Call the function
        apply_lstm_sequences(data, 'Binance', features, target, test_size, random_state, num_epochs, batch_size,
                             sequence_length)

        st.header("Bi-LSTM Forecast")
        data = df
        features = ['value']
        target = 'Close'
        test_size = 168
        random_state = False
        num_epochs = 15
        batch_size = 32
        sequence_length = 10

        # Call the function
        apply_bi_lstm_sequences(data, 'Binance', features, target, test_size, random_state, num_epochs, batch_size,
                                sequence_length)
    elif option == "Tether":
        st.write("You selected Tether.")
        # Add your code for Option 2
        streamlit.title('Tether Fear & Greed Influence')

        data = pd.read_csv('df_tether_csv.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        df = data
        data_arima = df
        data_arima = pd.DataFrame(data_arima)
        data_arima['box_diff_seasonal_1'] = (data['Close'] - data['Close'].shift(1))
        data = data['Close']
        print(data)

        train = data_arima[:-24 * 7]
        test = data_arima[-24 * 7:]
        st.header("ARIMA Forecast")
        #arima_forecast(train, test, 'Tether')
        df_forecast = arima_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], 'Tether')

        merged_data_and_plot(df,df_forecast)
        p, d, q = 1, 0, 1  # Non-seasonal order
        P, D, Q, s = 2, 1, 2, 7  # Seasonal order with daily seasonality

        # df['box_diff_seasonal_1'] = (df['Close'] - df['Close'].shift(1))
        days = 7
        exog = df['value']
        print(df.columns)
        three_exog = df[['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']]
        st.header("SARIMAX Forecast")
        df_forecast = sarimax_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], exog, days,
                                       'Tether')
        merged_data_and_plot(df,df_forecast)
        st.header("LSTM Forecast")
        data = df
        features = ['value']
        target = 'Close'
        test_size = 168
        random_state = False
        num_epochs = 15
        batch_size = 32
        sequence_length = 10

        # Call the function
        apply_lstm_sequences(data, 'Tether', features, target, test_size, random_state, num_epochs, batch_size,
                             sequence_length)

        st.header("Bi-LSTM Forecast")
        data = df
        features = ['value']
        target = 'Close'
        test_size = 168
        random_state = False
        num_epochs = 15
        batch_size = 32
        sequence_length = 10

        # Call the function
        apply_bi_lstm_sequences(data, 'Tether', features, target, test_size, random_state, num_epochs, batch_size,
                                sequence_length)

    elif option == "USD Coin":
        st.write("You selected USD Coin.")
        # Add your code for Option 2
        streamlit.title('USD Coin Fear & Greed Influence')

        # streamlit.header('BNB ARIMA prediction')
        data = pd.read_csv('df_usdc_csv.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        df = data
        data_arima = df
        data_arima = pd.DataFrame(data_arima)
        data_arima['box_diff_seasonal_1'] = (data['Close'] - data['Close'].shift(1))
        data = data['Close']
        print(data)

        train = data_arima[:-24 * 7]
        test = data_arima[-24 * 7:]
        st.header("ARIMA Forecast")
        #arima_forecast(train, test, 'USD Coin')
        df_forecast = arima_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], 'USD Coin')

        merged_data_and_plot(df_forecast)
        p, d, q = 1, 0, 1  # Non-seasonal order
        P, D, Q, s = 2, 1, 2, 7  # Seasonal order with daily seasonality

        # df['box_diff_seasonal_1'] = (df['Close'] - df['Close'].shift(1))
        days = 7
        exog = df['value']
        print(df.columns)
        three_exog = df[['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']]
        st.header("SARIMAX Forecast")
        df_forecast = sarimax_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], exog, days, 'USD Coin')
        merged_data_and_plot(df,df_forecast)
        st.header("LSTM Forecast")
        data = df
        features = ['value']
        target = 'Close'
        test_size = 168
        random_state = False
        num_epochs = 15
        batch_size = 32
        sequence_length = 10

        # Call the function
        apply_lstm_sequences(data, 'USD Coin', features, target, test_size, random_state, num_epochs, batch_size,
                             sequence_length)

        st.header("Bi-LSTM Forecast")
        data = df
        features = ['value']
        target = 'Close'
        test_size = 168
        random_state = False
        num_epochs = 15
        batch_size = 32
        sequence_length = 10

        # Call the function
        apply_bi_lstm_sequences(data, 'USD Coin', features, target, test_size, random_state, num_epochs, batch_size,
                                sequence_length)
elif selected_tab == "Fear and Greed & Whale Transactions Data":

    import streamlit as st
    import time
    import threading
    # Function that simulates a long-running task
    def long_running_task():
        for i in range(10):
            time.sleep(1)
            st.write(f"Task is running... {i}")
    # Button to interrupt the code
    if st.button("Interrupt Code"):
            # Create a thread for the long-running task
            task_thread = threading.Thread(target=long_running_task)

            # Start the thread
            task_thread.start()

            # Wait for the thread to finish
            task_thread.join()

            st.write("Code interrupted successfully!")
    # Main button

    option = st.selectbox("Select an option", ["None","Ethereum", "Bitcoin","Binance", "Tether", "USD Coin"])

        # Check which option is selected and trigger code accordingly
    if option == "None":
            st.write("Please select an option")
    elif option == "Ethereum":
            st.write("Ethereum")
            # Add your code for Option 1
            streamlit.title('Ethereum Fear and Greed & Whale Transactions Influence')
            streamlit.header('ARIMA prediction')
            data = pd.read_csv('df_eth_csv.csv')
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            df = data
            data_arima = df
            data_arima = pd.DataFrame(data_arima)
            data_arima['box_diff_seasonal_1'] = (data['Close'] - data['Close'].shift(1))
            data = data['Close']
            print(data)

            train = data_arima[:-24 * 7]
            test = data_arima[-24 * 7:]
            st.header("ARIMA Forecast")
            # arima_forecast(train, test, 'USD Coin')
            df_forecast = arima_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], 'USD Coin')

            merged_data_and_plot(df,df_forecast)

            p, d, q = 1, 0, 1  # Non-seasonal order
            P, D, Q, s = 2, 1, 2, 7  # Seasonal order with daily seasonality

            # df['box_diff_seasonal_1'] = (df['Close'] - df['Close'].shift(1))
            days = 7
            exog = df['value']
            print(df.columns)
            three_exog = df[['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']]
            st.header("SARIMAX Forecast")
            #sarimax_forecast(train, test, three_exog, days, 'Ethereum')
            df_forecast = sarimax_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], exog, days,
                                           'USD Coin')
            merged_data_and_plot(df, df_forecast)


            streamlit.header('LSTM')
            data = df
            features = ['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']
            target = 'Close'
            test_size = 168
            random_state = False
            num_epochs = 15
            batch_size = 32
            sequence_length = 10

            # Call the function
            apply_lstm_sequences(data, 'Ethereum', features, target, test_size, random_state, num_epochs, batch_size,
                                 sequence_length)

            streamlit.header('Bi-LSTM')
            data = df
            features = ['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']
            target = 'Close'
            test_size = 168
            random_state = False
            num_epochs = 15
            batch_size = 32
            sequence_length = 10

            # Call the function
            apply_bi_lstm_sequences(data, 'Ethereum', features, target, test_size, random_state, num_epochs, batch_size,
                                    sequence_length)
    elif option == "Bitcoin":
        st.write("Bitcoin")
        # Add your code for Option 1
        streamlit.title('Bitcoin Fear and Greed & Whale Transactions Influence')
        streamlit.header('ARIMA prediction')
        data = pd.read_csv('df_btc.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        df = data
        data_arima = df
        data_arima = pd.DataFrame(data_arima)
        data_arima['box_diff_seasonal_1'] = (data['Close'] - data['Close'].shift(1))
        data = data['Close']
        print(data)

        train = data_arima[:-24 * 7]
        test = data_arima[-24 * 7:]
        st.header("ARIMA Forecast")
        # arima_forecast(train, test, 'USD Coin')
        df_forecast = arima_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], 'USD Coin')

        merged_data_and_plot(df,df_forecast)

        p, d, q = 1, 0, 1  # Non-seasonal order
        P, D, Q, s = 2, 1, 2, 7  # Seasonal order with daily seasonality

        # df['box_diff_seasonal_1'] = (df['Close'] - df['Close'].shift(1))
        days = 7
        exog = df['value']
        print(df.columns)
        three_exog = df[['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']]
        st.header("SARIMAX Forecast")
        #sarimax_forecast(train, test, three_exog, days, 'Bitcoin')
        df_forecast = sarimax_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], exog, days,
                                       'USD Coin')
        merged_data_and_plot(df, df_forecast)
        streamlit.header('LSTM')
        data = df
        features = ['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']
        target = 'Close'
        test_size = 168
        random_state = False
        num_epochs = 15
        batch_size = 32
        sequence_length = 10

        # Call the function
        apply_lstm_sequences(data, 'Bitcoin', features, target, test_size, random_state, num_epochs, batch_size,
                             sequence_length)

        streamlit.header('Bi-LSTM')
        data = df
        features = ['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']
        target = 'Close'
        test_size = 168
        random_state = False
        num_epochs = 15
        batch_size = 32
        sequence_length = 10

        # Call the function
        apply_bi_lstm_sequences(data, 'Bitcoin', features, target, test_size, random_state, num_epochs, batch_size,
                                sequence_length)
    elif option == "Binance":
        st.write("Binance")
        # Add your code for Option 1
        streamlit.title('Binance Fear and Greed & Whale Transactions Influence')
        streamlit.header('ARIMA prediction')
        data = pd.read_csv('df_bnb_csv.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        df = data
        data_arima = df
        data_arima = pd.DataFrame(data_arima)
        data_arima['box_diff_seasonal_1'] = (data['Close'] - data['Close'].shift(1))
        data = data['Close']
        print(data)

        train = data_arima[:-24 * 7]
        test = data_arima[-24 * 7:]
        st.header("ARIMA Forecast")
        # arima_forecast(train, test, 'USD Coin')
        df_forecast = arima_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], 'USD Coin')

        merged_data_and_plot(df,df_forecast)
        p, d, q = 1, 0, 1  # Non-seasonal order
        P, D, Q, s = 2, 1, 2, 7  # Seasonal order with daily seasonality

        # df['box_diff_seasonal_1'] = (df['Close'] - df['Close'].shift(1))
        days = 7
        exog = df['value']
        print(df.columns)
        three_exog = df[['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']]
        st.header("SARIMAX Forecast")
        #sarimax_forecast(train, test, three_exog, days, 'Binance')
        df_forecast = sarimax_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], exog, days,
                                       'USD Coin')
        merged_data_and_plot(df, df_forecast)
        streamlit.header('LSTM')
        data = df
        features = ['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']
        target = 'Close'
        test_size = 168
        random_state = False
        num_epochs = 15
        batch_size = 32
        sequence_length = 10

        # Call the function
        apply_lstm_sequences(data, 'Binance', features, target, test_size, random_state, num_epochs, batch_size,
                             sequence_length)

        streamlit.header('Bi-LSTM')
        data = df
        features = ['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']
        target = 'Close'
        test_size = 168
        random_state = False
        num_epochs = 15
        batch_size = 32
        sequence_length = 10

        # Call the function
        apply_bi_lstm_sequences(data, 'Binance', features, target, test_size, random_state, num_epochs, batch_size,
                                sequence_length)
    elif option == "Tether":
        st.write("Tether")
        # Add your code for Option 1
        streamlit.title('Tether Fear and Greed & Whale Transactions Influence')
        streamlit.header('ARIMA prediction')
        data = pd.read_csv('df_tether_csv.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        df = data
        data_arima = df
        data_arima = pd.DataFrame(data_arima)
        data_arima['box_diff_seasonal_1'] = (data['Close'] - data['Close'].shift(1))
        data = data['Close']
        print(data)

        train = data_arima[:-24 * 7]
        test = data_arima[-24 * 7:]
        st.header("ARIMA Forecast")
        # arima_forecast(train, test, 'USD Coin')
        df_forecast = arima_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], 'USD Coin')

        merged_data_and_plot(df,df_forecast)
        p, d, q = 1, 0, 1  # Non-seasonal order
        P, D, Q, s = 2, 1, 2, 7  # Seasonal order with daily seasonality

        # df['box_diff_seasonal_1'] = (df['Close'] - df['Close'].shift(1))
        days = 7
        exog = df['value']
        print(df.columns)
        three_exog = df[['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']]
        st.header("SARIMAX Forecast")

        #sarimax_forecast(train, test, three_exog, days, 'Tether')
        df_forecast = sarimax_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], exog, days,
                                       'USD Coin')
        merged_data_and_plot(df, df_forecast)
        streamlit.header('LSTM')
        data = df
        features = ['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']
        target = 'Close'
        test_size = 168
        random_state = False
        num_epochs = 15
        batch_size = 32
        sequence_length = 10

        # Call the function
        apply_lstm_sequences(data, 'Tether', features, target, test_size, random_state, num_epochs, batch_size,
                             sequence_length)

        streamlit.header('Bi-LSTM')
        data = df
        features = ['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']
        target = 'Close'
        test_size = 168
        random_state = False
        num_epochs = 15
        batch_size = 32
        sequence_length = 10

        # Call the function
        apply_bi_lstm_sequences(data, 'Tether', features, target, test_size, random_state, num_epochs, batch_size,
                                sequence_length)
    elif option == "USD Coin":
        st.write("USD Coin")
        # Add your code for Option 1
        streamlit.title('USD Coin Fear and Greed & Whale Transactions Influence')
        streamlit.header('ARIMA prediction')
        data = pd.read_csv('df_usdc_csv.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        df = data
        data_arima = df
        data_arima = pd.DataFrame(data_arima)
        data_arima['box_diff_seasonal_1'] = (data['Close'] - data['Close'].shift(1))
        data = data['Close']
        print(data)

        train = data_arima[:-24 * 7]
        test = data_arima[-24 * 7:]
        st.header("ARIMA Forecast")
        # arima_forecast(train, test, 'USD Coin')
        df_forecast = arima_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], 'USD Coin')

        merged_data_and_plot(df,df_forecast)
        p, d, q = 1, 0, 1  # Non-seasonal order
        P, D, Q, s = 2, 1, 2, 7  # Seasonal order with daily seasonality

        # df['box_diff_seasonal_1'] = (df['Close'] - df['Close'].shift(1))
        days = 7
        exog = df['value']
        print(df.columns)
        three_exog = df[['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']]
        st.header("SARIMAX Forecast")
        #sarimax_forecast(train, test, three_exog, days, 'USD Coin')
        df_forecast = sarimax_forecast(train['box_diff_seasonal_1'], test['box_diff_seasonal_1'], exog, days,
                                       'USD Coin')
        merged_data_and_plot(df, df_forecast)
        streamlit.header('LSTM')
        data = df
        features = ['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']
        target = 'Close'
        test_size = 168
        random_state = False
        num_epochs = 15
        batch_size = 32
        sequence_length = 10

        # Call the function
        apply_lstm_sequences(data, 'USD Coin', features, target, test_size, random_state, num_epochs, batch_size,
                             sequence_length)

        streamlit.header('Bi-LSTM')
        data = df
        features = ['value', 'Whale Transaction Count (>100k USD', 'Whale Transaction Count (>1m USD)']
        target = 'Close'
        test_size = 168
        random_state = False
        num_epochs = 15
        batch_size = 32
        sequence_length = 10

        # Call the function
        apply_bi_lstm_sequences(data, 'USD Coin', features, target, test_size, random_state, num_epochs, batch_size,
                                sequence_length)
