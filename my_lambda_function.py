#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import pandas_ta as ta
import alpaca_trade_api as tradeapi
import boto3
from datetime import datetime
from boto3.dynamodb.conditions import Key
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import time
from decimal import Decimal
import json

# starts the algorithm through the lambda_helper
def start_algorithm():
    # Connecting to Alpaca web API
    API_KEY = 'ALPACA API KEY'
    SECRET_KEY = 'ALPACA API SECRET KEY'
    BASE_URL = 'https://paper-api.alpaca.markets'  # or the live URL
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

    # creates a portfolio dataframe with the currently invested stocks and the buy date
    positions = api.list_positions()
    portfolio_data = []
    for position in positions:
        ticker = position.symbol
        portfolio_data.append([ticker])
    portfolio = pd.DataFrame(portfolio_data, columns=['Ticker'])
    buy_dates = {}
    orders = api.list_orders(status='filled', side='buy')
    for order in orders:
        ticker = order.symbol
        filled_at = order.filled_at.date()
        if ticker not in buy_dates:
            buy_dates[ticker] = filled_at
        else:
            buy_dates[ticker] = min(buy_dates[ticker], filled_at)
    portfolio['Buy Date'] = portfolio['Ticker'].map(buy_dates)
    print(portfolio)


    # reads in the s&p500 stocks + several ETFs
    # Removed KVUE, GEHC, VLTO, SOLV, GEV, CARR, ABNB, OTIS, CEG, WRK due to lack of data
    tickers = pd.read_csv('./sp_500_stocks.csv')['Ticker'].tolist()
    stocks = {}
    data = yf.download(tickers, period='10y', group_by='ticker', threads=True)
    for ticker in tickers:
        ticker_data = data[ticker]
        ticker_data = ticker_data.dropna()
        stocks[ticker] = ticker_data


    # connects to AWS DynamoDB
    session = boto3.Session(
    aws_access_key_id='AWS ACCESS KEY',
    aws_secret_access_key='AWS SECRET ACCESS KEY',
    region_name='us-east-1'
    )
    dynamodb = session.resource('dynamodb')
    table = dynamodb.Table('stock_data')
    def add_to_database(ticker, trade_id, trade_type, quantity, price, valuation, stop_loss, profit):
        """
        Adds the data to the dynamodb database whenever a stock is bought or sold.
        This will keep track of the algorithm's performance over time
        Parameters
            ticker (string): ticker of the stock to be added to the database
            trade_id (string): Alpaca's assigned id for a trade
            trade_type (string): Indicates BUY or SELL order
            quantity (int): The quantity of stock
            price (float): The price of the stock when the function is called
            valuation (float): price * quantity
            stop_loss (float): 0.8 * bought price of the stock to prevent further losses
            profit (float): How much a stock profited/lost
        """
        item={
            'Ticker': ticker,
            'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Strategy': 'meanreversion_rfc',
            'Trade_ID': trade_id,
            'Type': trade_type,
            'Quantity': quantity,
            'Price': price,
            'Valuation': valuation,
            'StopLoss': stop_loss,
            'Profit': profit
        }
        try:
            response = table.put_item(Item=item)
            print('Successful input into table')
        except Exception as e:
            print('Error inserting item:', e)
    def get_most_recent_entry(ticker, strategy='meanreversion_rfc', trade_type='BUY'):
        """
        Gets the most recent entry in the database according to the Date entry
        Only called when a stock is bought
        Parameters
            ticker (string): ticker of the stock to be added to the database
            strategy (string): set to meanreversion_rfc for this algorithm
            trade_type (string): Set to buy due to the function's use
        """
        try:
            # Scan the table to retrieve all items for the given ticker, strategy, and trade_type
            response = table.scan(
                FilterExpression=Key('Ticker').eq(ticker) & Key('Strategy').eq(strategy) & Key('Type').eq(trade_type)
            )

            items = response.get('Items', [])
            if items:
                # Find the most recent entry by sorting items based on Date
                most_recent_item = max(items, key=lambda x: x['Date'])
                return most_recent_item
            else:
                print(f"No entries found for ticker {ticker} with strategy {strategy} and trade type {trade_type}.")
                return None

        except Exception as e:
            print(f"Error retrieving most recent entry: {e}")
            return None

    def buy_order(ticker):
        """
        Buys a stock through Alpaca web API
        Parameters
            ticker (string): ticker of the stock to be added to the database
        """
        # Buys up to $1000 value of shares
        quantity = 1000 // stocks[ticker]['Close'][-1]
        # Place a market order to buy the stock
        order = api.submit_order(
            symbol=ticker,
            qty=quantity,
            side='buy',
            type='market',
            time_in_force='gtc'  # Good 'til canceled
        )
        print(f"Bought {quantity} shares of {ticker} at {stocks[ticker]['Close'][-1]}.")

        # Set a 5-minute timeout to cap runtime and prevent costly infinite loops
        timeout = 300
        start_time = time.time()


        # Wait for the order to be filled
        while True:
            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Fetch the order status
            order_status = api.get_order(order.id)

            # Check if the order is filled and adds it to the database if filled in time
            if order_status.status == 'filled':
                price = float(order_status.filled_avg_price)
                valuation = price * quantity
                stop_loss_db = price * 0.8
                print(f"Successfully bought {quantity} shares of {ticker} at {price}.")
                add_to_database(str(ticker), str(order.id), 'BUY', int(quantity), Decimal(str(price)), Decimal(str(valuation)), Decimal(str(stop_loss_db)), Decimal(str(0)))
                break
            # If order was rejected or canceled, leave the function and do not buy
            elif order_status.status in ['rejected', 'canceled']:
                print(f"Order {order.id} was {order_status.status}.")
                return

            # If the elapsed time exceeds the 5-minute timeout, cancel the order
            if elapsed_time > timeout:
                print(f"Order {order.id} for {ticker} not filled within 5 minutes. Cancelling order...")
                try:
                    api.cancel_order(order.id)
                    print(f"Order {order.id} has been canceled.")
                    return
                except Exception as e:
                    print(f"Failed to cancel order {order.id}: {e}")
                    return

            # Sleep for a short period before checking again
            time.sleep(1)
        # Creates stop loss
        create_stop_loss(ticker)


    def create_stop_loss(ticker):
        """
        Create a stop loss order of 80% of what the stock was bought at to minimize risk
        Parameters
            ticker (string): ticker of the stock to be added to the database
        """
        quantity = 1000 // stocks[ticker]['Close'][-1]
        stop_limit = round(stocks[ticker]['Close'][-1] * 0.8, 2)
        # Place a stop-loss order
        api.submit_order(
            symbol=ticker,
            qty=quantity,
            side='sell',
            type='stop',
            stop_price=stop_limit,
            time_in_force='gtc'  # Good 'til canceled
        )
        print(f"Created Stop Loss for {ticker}.")


    def sell_order(ticker):
        """
        Sells a stock using Alpaca web API
        Parameters
            ticker (string): ticker of the stock to be added to the database
        """
        # Checks if the stock is currently held
        position = api.get_position(ticker)
        if position:
            # Cancel stop-loss orders for the stock
            orders = api.list_orders(status='open')
            for order in orders:
                if order.symbol == ticker and order.side == 'sell' and order.type == 'stop':
                    print(f"Cancelling stop-loss order: {order.id}")
                    api.cancel_order(order.id)
            quantity = int(position.qty)
            # Place a market order to sell the stock
            sell_order = api.submit_order(
                symbol=ticker,
                qty=quantity,
                side='sell',
                type='market',
                time_in_force='gtc'  # Good 'til canceled
            )

            # Set a 5-minute timeout to cap runtime and prevent costly infinite loops
            timeout = 300
            start_time = time.time()
            # Loops while the order is not filled/rejected/canceled
            while True:
                elapsed_time = time.time() - start_time
                order_status = api.get_order(sell_order.id)
                # if the order is filled, sell the stock and add a database entry documenting important data
                if order_status.status == 'filled':
                    stock_data = get_most_recent_entry(ticker)
                    buy_valuation = 0
                    if stock_data:
                        buy_valuation = float(stock_data.get('Valuation'))
                    price =float(order_status.filled_avg_price)
                    valuation = price * quantity
                    profit = valuation - buy_valuation
                    print(f"Sold {quantity} shares of {ticker}.")
                    add_to_database(str(ticker), str(sell_order.id), 'SELL', int(quantity), Decimal(str(price)), Decimal(str(valuation)), int(-1), Decimal(str(profit)))
                    break
                # If order was rejected or canceled, leave the function and do not buy
                elif order_status.status in ['rejected', 'canceled']:
                    print(f"Order {sell_order.id} was {order_status.status}.")
                    return

                # If the elapsed time exceeds the 5-minute timeout, cancel the order
                if elapsed_time > timeout:
                    print(f"Order {sell_order.id} for {ticker} not filled within 5 minutes. Cancelling order...")
                    try:
                        api.cancel_order(sell_order.id)
                        print(f"Order {sell_order.id} has been canceled.")
                    except Exception as e:
                        print(f"Failed to cancel order {sell_order.id}: {e}")
                    break
                # Sleep for a short period before checking again
                time.sleep(1)
        else:
            print(f"No position found for {ticker}.")


    warnings.filterwarnings("ignore")
    # list of stocks to be bought. Filled by the mean_reversion function
    buy_stocks = []
    def mean_reversion(ticker):
        """
        Mean reversion strategy used to buy/sell a stock at a certain threshold
        Parameters
            ticker (string): ticker of the stock to be added to the database
        """
        # sets a 21 day moving average
        ma = 21
        # percentiles to calculate when a stock should be bought and sold
        percentiles = [5, 10, 50, 90, 95]
        # finds the logarithmic returns of the close price for each stock
        stocks[ticker]['returns'] = np.log(stocks[ticker]['Adj Close']).diff()
        # sets the moving average of ma days at the current day
        stocks[ticker]['ma'] = stocks[ticker]['Adj Close'].rolling(ma).mean()
        # sets the ratio of the adj cose to the moving average to determine if the stock is overvalued or undervalued
        stocks[ticker]['ratio'] = stocks[ticker]['Adj Close'] / stocks[ticker]['ma']
        # finds the percentile of each ratio
        p = np.percentile(stocks[ticker]['ratio'].dropna(), percentiles)

        # buys when the ratio is at the 5th percentile and sells at 95th percentile
        buy_condition = stocks[ticker]['ratio'] <= p[0]
        sell_condition = stocks[ticker]['ratio'] >= p[-1]
        stocks[ticker]['position'] = np.nan
        stocks[ticker]['position'] = np.where(buy_condition, 1, np.nan)
        stocks[ticker]['position'] = np.where(sell_condition, 0, stocks[ticker]['position'])

        # forward fills position column to fill any nan vallues
        stocks[ticker]['position'] = stocks[ticker]['position'].ffill()

        # If the stock is already in the portfolio, maintain position = 1
        # This is done since previous days may reflect a position of 0 when a stock is held
        # This issue is caused when a stock's ratio falls below and then above the 5th percentile in a single day
        if ticker in portfolio['Ticker'].values:
            buy_date = pd.to_datetime(portfolio.loc[portfolio['Ticker'] == ticker, 'Buy Date'].values[0])
            # If `buy_date` is timezone-naive, make it timezone-aware by converting to UTC (or another appropriate timezone)
            if buy_date.tzinfo is None:
                buy_date = buy_date.tz_localize('UTC')

            # Ensure the stock index is also timezone-aware
            if stocks[ticker].index.tzinfo is None:
                stocks[ticker].index = stocks[ticker].index.tz_localize('UTC')
            last_date = stocks[ticker].index[-1]
            # Ensure position stays at 1 from the buy date onward, up to and including the last date
            stocks[ticker].loc[stocks[ticker].index >= buy_date, 'position'] = 1
            # Check the sell condition for the last day
            sell_condition_last_day = stocks[ticker].loc[last_date, 'ratio'] >= p[-1]
            # If the sell condition is true, set position for the last day to 0
            if sell_condition_last_day:
                stocks[ticker].loc[last_date, 'position'] = 0

        # finds strategy returns
        stocks[ticker]['strat_returns'] = stocks[ticker]['returns'] * stocks[ticker]['position'].shift()

        # sets a column for the previous day's position to make it easier to distinguish when to buy/sell
        stocks[ticker]['previous_position'] = stocks[ticker]['position'].shift(1)
        # Finds whether to buy or sell in the current day
        last_index = stocks[ticker].index[-1]
        buy_condition_today = (stocks[ticker].loc[last_index, 'previous_position'] == 0) & (stocks[ticker].loc[last_index, 'position'] == 1)
        sell_condition_today = (stocks[ticker].loc[last_index, 'previous_position'] == 1) & (stocks[ticker].loc[last_index, 'position'] == 0)
        # adds the stock to a list if the mean reversion algorithm indicates to buy. Will buy the stock after future analysis
        if buy_condition_today and (ticker not in portfolio['Ticker'].values):
            print("Considering:", ticker)
            buy_stocks.append(ticker)
        # immediately sells if mean reversion indicates sell
        elif sell_condition_today and (ticker in portfolio['Ticker'].values):
            print("Sold:", ticker)
            sell_order(ticker)

    # runs mean reversion on all stocks in the dataset
    for ticker in tickers:
        mean_reversion(ticker)


    # Initialize the RandomForestClassifier model
    model = None
    def train_model():
        """
        Train the machine learning model using historical data from all tickers.
        This will be run every time a stock is signaled to be bought to keep the model updated
        All data will be combined into a single DataFrame, each feature will be scaled independently of one another
        and then data will be extracted from when a stock is bought, and classified on whether it was profitable
        """
        global model
        all_data = []

        # Loop through all tickers in the stocks dictionary
        for ticker in stocks:
            df = stocks[ticker].copy()
            df['Ticker'] = ticker
            # Add technical indicators using pandas_ta
            # Relative Strength Index (RSI): Identifies overbought/oversold conditions
            df['RSI'] = ta.rsi(df['Adj Close'], length=21)
            # Simple Moving Average for 21 days
            df['SMA_21'] = ta.sma(df['Adj Close'], length=21)
            # Exponential Moving Average for 21 days (more weight to recent prices)
            df['EMA_21'] = ta.ema(df['Adj Close'], length=21)
            # Ratio of closing price to SMA_21 to assess whether the price is above or below its average
            df['price_to_SMA_21'] = df['Adj Close'] / df['SMA_21']
            # Average True Range (ATR): measures market volatility based on recent price movements
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=21)
            # On-Balance Volume (OBV): momentum indicator that relates volume to price change
            df['OBV'] = ta.obv(df['Adj Close'], df['Volume'])
            # Relative Volume: compares current volume to its 21 day moving average, identifying outliers
            df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=21).mean()
            # Month, Year, and Quarter to capture seasonality of data
            df['Month'] = df.index.month
            df['Year'] = df.index.year
            df['Quarter'] = df['Month'].apply(lambda x: (x - 1) // 3 + 1)
            # Calculate rolling volatility and momentum
            df['Volatility'] = df['returns'].rolling(21).std()
            df['Momentum_5'] = df['Adj Close'].pct_change(5)
            df['Momentum_21'] = df['Adj Close'].pct_change(21)
            # Applies sine and cosine transformation of the month to calculate cyclical patters
            df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)

            # Initialize the 'Trade_Result' column with 0s (no action taken by default)
            df['Trade_Result'] = 0

            # Identify buy (1) and sell (0) signals
            df['Buy_Signal'] = (df['previous_position'] == 0) & (df['position'] == 1)
            df['Sell_Signal'] = (df['previous_position'] == 1) & (df['position'] == 0)

            # Loop through each row and assign 1 if the trade was profitable, -1 if it was not
            open_trade_idx = None  # to track the index where the buy occurred
            for i in range(len(df)):
                if df['Buy_Signal'].iloc[i]:
                    open_trade_idx = i
                # When a sell signal is triggered
                if df['Sell_Signal'].iloc[i] and open_trade_idx is not None:
                    # Calculate profit/loss between the buy and sell
                    buy_price = df['Adj Close'].iloc[open_trade_idx]
                    sell_price = df['Adj Close'].iloc[i]
                    trade_return = (sell_price - buy_price) / buy_price
                    # Update 'Trade_Result' for the entire trade period (between buy and sell)
                    df.iloc[open_trade_idx, df.columns.get_loc('Trade_Result')] = 1 if trade_return > 0 else -1
                    # Reset the open trade index after trade is closed
                    open_trade_idx = None
            df.drop(columns=['Buy_Signal', 'Sell_Signal'], inplace=True)
            all_data.append(df)
        # combines all the data into a single DataFrame
        combined_data = pd.concat(all_data)
        # Reset the index to avoid overlapping indices
        combined_data.reset_index(drop=True, inplace=True)

        # Identify columns to scale and then scale them
        columns_to_scale = ['RSI', 'returns', 'Momentum_5', 'Momentum_21',
                            'price_to_SMA_21', 'ATR', 'OBV', 'Relative_Volume', 'Volatility',
                           'Month_sin', 'Month_cos']
        scaler = StandardScaler()
        combined_data[columns_to_scale] = scaler.fit_transform(combined_data[columns_to_scale])
        # add features that should not be scaled
        columns_to_scale.append('Month')
        columns_to_scale.append('Year')
        columns_to_scale.append('Quarter')
        filtered_data = combined_data[combined_data['Trade_Result'].isin([-1, 1])]
        # Extract the features (X) using the selected columns
        X = filtered_data[columns_to_scale]

        # Extract the labels (y), which is the Trade_Result
        y = filtered_data['Trade_Result']
        y = y.map({-1: 0, 1: 1})
        # Split data into training and testing sets using stratification since data is combined (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Initialize the RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model - Confusion Matrix and Classification Report
        conf_matrix = confusion_matrix(y_test, y_pred)
        # Plot the Confusion Matrix using a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Profit'], yticklabels=['Loss', 'Profit'])
        plt.title('Confusion Matrix - RandomForestClassifier')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # Print the classification report and accuracy
        class_report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print("\nClassification Report:")
        print(class_report)

        print(f"\nAccuracy: {accuracy:.2f}")

        return model, columns_to_scale, combined_data


    def predict_from_combined(stocks, model, columns_to_scale, combined_data):
        """
        Predict whether to buy or hold/sell based on the most recent data from combined_data.

        Parameters
            stocks (list): List of stock tickers to predict.
            model (sklearn model): The trained RandomForestClassifier model.
            columns_to_scale (list): List of feature columns that need to be used for prediction.
            combined_data (DataFrame): The data containing all the stock information (already scaled).

        Returns:
        DataFrame: A DataFrame with tickers, predictions, and probabilities.
        """

        # Filter combined_data for only the rows corresponding to the most recent data for each stock
        predictions_list = []

        for ticker in stocks:
            # Get the last row for this ticker
            stock_data = combined_data[combined_data['Ticker'] == ticker].iloc[-1:]

            # Ensure the data has the same features used during training
            X_new = stock_data[columns_to_scale]  # No need to scale again

            # Predict using the trained model
            prediction = model.predict(X_new)[0]
            probability = model.predict_proba(X_new)[0][1] # Probability for 'profit' class (1)

            # Convert prediction to label
            label = 'Buy' if prediction == 1 else 'Avoid'

            # Store the results
            predictions_list.append({
                'Stock': ticker,
                'Prediction': prediction,
                'Label': label,
                'Probability of Profit': probability
            })

        # Return the results as a DataFrame
        return pd.DataFrame(predictions_list)

    # Runs the model if mean_reversion indicates 1 or more stocks to be bought
    if len(buy_stocks) > 0:
        model, columns_to_scale, combined_df = train_model()
        results = predict_from_combined(buy_stocks, model, columns_to_scale, combined_df)
        # sets a threshold of 0.7 to only buy stocks that are strongly expected to be profitable via the model
        filtered_results = results[(results['Prediction'] == 1) & (results['Probability of Profit'] >= 0.8)]
        sorted_results = filtered_results.sort_values(by='Probability of Profit', ascending=False)
        print(sorted_results)
        # buys the stocks most likely to be profitable
        for index, row in sorted_results.iterrows():
            print('Bought', row['Stock'])
            buy_order(row['Stock'])
    else:
        print("Finished Algorithm Without Errors And Did Not Buy Anything")


def lambda_handler(event, context):
    start_algorithm()
    return {
        'statusCode': 200,
        'body': json.dumps('Completed Algorithm')
    }