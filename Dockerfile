FROM public.ecr.aws/lambda/python:3.9
RUN pip install numpy==1.26.4 pandas yfinance pandas_ta alpaca_trade_api boto3 matplotlib seaborn scikit-learn
COPY my_lambda_function.py ./
COPY sp_500_stocks.csv ./
CMD ["my_lambda_function.lambda_handler"]

