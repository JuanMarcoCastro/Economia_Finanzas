from datafromsource import DataFromSource

class PortfolioData():
    def __init__(self, assets, interval_period="1D", initial = 365):
        self.assets = assets
        self.interval_period = interval_period
        self.initial = initial

    def get_portfolio_prices(self):  
        portfolio = self.assets
        Stock_0 = DataFromSource(portfolio[0], interval_period=self.interval_period, initial = self.initial)
        portfolio_prices = Stock_0.get_prices()
        print(portfolio[0])
        
        for stock in portfolio[1:]:
            try:
                stock_info = DataFromSource(stock, interval_period=self.interval_period, initial = self.initial)
                w = stock_info.get_prices()
                w1 = w.rename(columns={'Price Close': stock})
                portfolio_prices = portfolio_prices.join(w1)
                print(stock)
            except:
                print("An exception occurred for stock: "+ stock)

        return portfolio_prices

    def get_portfolio_returns(self):  
        portfolio = self.assets
        Stock_0 = DataFromSource(portfolio[0], interval_period=self.interval_period, initial = self.initial)
        portfolio_returns = Stock_0.get_returns()
        print(portfolio[0])

        for stock in portfolio[1:]:
            try:
                stock_info = DataFromSource(stock, interval_period=self.interval_period, initial = self.initial)
                w = stock_info.get_returns()
                portfolio_returns = portfolio_returns.join(w)
                print(stock)
            except:
                print("An exception occurred for stock: "+ stock)
    
        return portfolio_returns