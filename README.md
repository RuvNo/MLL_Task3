# Welcome to "The one and only crystal ball".

## Description
This project is designed as a python application that is started from the terminal.
It consists of three statistical models that can be used to forecast the stock price of a selected stock some time into the future.
The supported models are as follows:
- **AR(p)**: A simple auto-regressive model. Conditionally, the stock-price time series is differenced to make it stationary.
- **ARIMA(p,d,q)**: An ARIMA-model, that consists of an autoregressive (AR), integrating (I) and moving average (MA) component.
- **LSTM**: A Long Short-Term Memory-model, that consists of a layer of lstm-nodes followed by two dense layers of nodes with "relu"-activation functions.

One can either use one of the three models by itself, or use all three combined to compare their predictions.
The models make their predictions based on a recursive forecasting, where the prediction of t+1 is featured as input for the prediction for t+2 (Islam et al., 2022).
Keep in mind that this can lead to accumulating errors.
Additionally, one has to remember that the predictive power of these models is limited due to the "random walk" properties exhibited by stock prices (Fama, 1995).

## Data
Generally, it is possible to use data from *https://www.nasdaq.com/market-activity/quotes/historical*.
However, for your convenience, the stock market data for Apple (**AAPL**) and Advanced Micro Devices (**AMD**) is already available in this repo.

## Setup
Make sure, that you have python installed (preferably, python 3.13.0 to avoid any issues).
To run the project do the following:
1. Go to the root of this project
2. Set up the Virtual Environment: ```python -m venv .venv``` or ```python3 -m venv .venv```
3. Activate the Virtual Environment: ```source .venv/bin/activate``` # On Windows use ```'.venv\Scripts\activate'```
4. Install dependencies: ```pip install -r requirements.txt```
5. Run the tool (this might take a while the first time around): ```python main.py``` or ```python3 main.py```

The tool will guide you through the whole process.
Keep in mind that for every selection there is a default, so if you do not want to bother with any special settings, just proceed by pressing **ENTER** every time.
Additionally, some selections provide a hint that can be accessed by typing **?**.
If the options of the selection are enumerated, you can make your choice by entering the corresponding number.


## Sources
- Fama, Eugene F. "Random walks in stock market prices." Financial analysts journal 51.1 (1995): 75-80.
- Rashedul Islam, Md, Momotaz Begum, and Md Nasim Akhtar. "Recursive approach for multiple step-ahead software fault prediction through long short-term memory (LSTM)." Journal of Discrete Mathematical Sciences and Cryptography 25.7 (2022): 2129-2138.

