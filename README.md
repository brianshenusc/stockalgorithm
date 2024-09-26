# stockalgorithm
## Description
Mean Reversion &amp; RandomForestClassifier Automatic stock algorithm running on AWS

## Table of Contents
- [Prerequisites](#Prerequisites)
- [Setup](#Setup)
- [Features](#Features)

## Prerequisites
1. Have AWS account with admin access and connect it to terminal
2. Have Docker Desktop open
3. Create DynamoDB table named stock_data

## Setup
1. Replace Alpaca and AWS access keys with your own keys
2. Navigate to correct folder and run the following while replacing anything with <>
- docker build -t stock-algorithm-image .
- aws ecr get-login-password --region <YOUR AWS REGION (us-east-1)> | docker login --username AWS --password-stdin <YOUR AWS USER ID>.dkr.ecr.<YOUR AWS REGION (us-east-1)>.amazonaws.com
- docker tag stock-algorithm-image:latest <YOUR AWS USER ID>.dkr.ecr.<YOUR AWS REGION (us-east-1)>.amazonaws.com/stock-algorithm:latest
- docker push <YOUR AWS USER ID>.dkr.ecr.<YOUR AWS REGION (us-east-1)>.amazonaws.com/stock-algorithm:latest
3. Create Lambda function and set the container image. Image -> Deploy New Image -> Select Container -> Select Latest
4. Add EventBridge for CloudWatch with command cron(35 13-20 ? * MON-FRI *) to automatically run the program once an hour during market hours subject to requirements (optional extra hour to account for daylight savings)

## Features
