# Automatic Stock Algorithm
## Description
This project is an automated stock trading algorithm based on Mean Reversion and RandomForestClassifier, running on AWS Lambda with Docker. The algorithm uses Alpaca API for trading and DynamoDB to store stock data.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Features](#features)
- [Usage](#usage)
- [Future Improvements](#future-improvements)

## Prerequisites
1. Have AWS account with admin access and connect it to terminal
2. Have Docker Desktop open
3. Create DynamoDB table named stock_data

## Setup
1. Replace Alpaca and AWS access keys with your own keys as well as the AWS region in the code (where you replace AWS access keys) 
2. Navigate to correct folder and run the following while replacing anything with <>
```bash
docker build -t stock-algorithm-image .
aws ecr get-login-password --region <YOUR AWS REGION (us-east-1)> | docker login --username AWS --password-stdin <YOUR AWS USER ID>.dkr.ecr.<YOUR AWS REGION (us-east-1)>.amazonaws.com
docker tag stock-algorithm-image:latest <YOUR AWS USER ID>.dkr.ecr.<YOUR AWS REGION (us-east-1)>.amazonaws.com/stock-algorithm:latest
docker push <YOUR AWS USER ID>.dkr.ecr.<YOUR AWS REGION (us-east-1)>.amazonaws.com/stock-algorithm:latest
```
3. Create Lambda function and set the container image. Image -> Deploy New Image -> Select Container -> Select Latest
4. Add EventBridge for CloudWatch with command cron(35 13-20 ? * MON-FRI *) to automatically run the program once an hour during market hours subject to requirements (optional extra hour to account for daylight savings)

## Features
- Mean Reversion strategy optimized with RandomForestClassifier for profitability.
- Fully deployed on AWS Lambda with Docker containers.
- Automatic trading via the Alpaca API.
- Data storage in AWS DynamoDB for stock trading records.

## Usage
- Ensure the Lambda function is running as expected, either through EventBridge (automated) or manually triggering it in the AWS Console.
- You can monitor logs in AWS CloudWatch to check the status and output of your trading algorithm.
- Program costs ~$0.10 per month to run subjet to change based on how often it is run

## Future Improvements
- Use AWS Secrets Manager to hold the access keys (currently not done due to extra costs)
- Improve features for the RandomForestClassifier Model
- Performance (yfinance downloads are threaded but can be done faster for optimization)
- Implement data analytics as more DynamoDB data is entered for further optimization
- Adjust hyperparameters (moving average) for optimization
