import lseg.data as ld
import datetime as dt
import numpy as np
import pandas as pd

class DataFromSource():    
    def __init__(self, RIC, interval_period='1D', initial = 365):
        self.RIC = RIC
        self.interval_period = interval_period
        self.initial = initial

    def __str__(self):
        return f"{self.RIC}"

    def get_session(self):
        try:
            ld.open_session()
        except Exception as e:
            print(f"Error opening session: {e}")
        return ld
    
    def get_timeframe(self):
        today = dt.date.today()
        initial_day = today - dt.timedelta(days=self.initial)
        return today, initial_day
    
    def get_prices(self):
        ld = self.get_session()
        today, initial_day = self.get_timeframe()
        try:
            data = ld.get_history(universe=[self.RIC], fields=['TR.PriceClose'], interval = self.interval_period,
               start = initial_day, end = today)
            data['close'] = data['Price Close'].astype(float)
            data = data.drop(['Price Close'], axis=1)
            return data
        except Exception as e:
            print(f"Error retrieving data for {self.RIC}: {e}")
            return None 
        
    def get_daily_returns(self):
        prices = self.get_prices()
        if prices is not None:
            try:
                prices[self.RIC] = np.log(prices['close'].div(prices['close'].shift(1)))
                daily_returns = prices.drop(['close'], axis=1)
                return daily_returns
            except Exception as e:
                print(f"Error calculating daily returns for {self.RIC}: {e}")
                return None
        else:
            return None

    def get_returns(self):
        prices = self.get_prices()
        if prices is not None:
            try:
                prices['returns'] = np.log(prices[f"{self.RIC}"].div(prices[f"{self.RIC}"].shift(1)))
                returns = prices.drop([f"{self.RIC}"], axis=1).rename(columns={'returns': f"{self.RIC}"})
                return returns
            except Exception as e:
                print(f"Error calculating returns for {self.RIC}: {e}")
                return None
        else:
            return None
