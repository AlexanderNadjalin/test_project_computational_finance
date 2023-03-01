# test_project_computational_finance
Test project for The Python Quant Platinum Certificate Program, January 2023.

### Instructions

Write a pricing library for European plain vanilla and exotic options making use of geometric Brownian motion as the stochastic process. The pricing shall be based on Monte Carlo simulation.

The pricing library shall have:

* a class for the risk-neutral discounting (taking the short rate as parameter, generating discount factors as required, etc.)
* a class to model the parameters of the underyling risk factor and a European option (initial value, volatility, time-to-maturity, dividends, strike price, option type etc.)
* a class to value European vanilla options based on Monte Carlo simulation 
* a class to value European lookback options — with either fixed or floating strike (see https://en.wikipedia.org/wiki/Lookback_option)
* a class to value European Asian options — with either fixed price or fixed strike (see https://en.wikipedia.org/wiki/Asian_option)