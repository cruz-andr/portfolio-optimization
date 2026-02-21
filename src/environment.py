import pandas as pd
import numpy as np

## this is the gym used for simulating the actual trading environment 
class TransactionEnvironment: 

    def __init__(self, stockData, featureColumns, transactionCost=0.0005): 
        self.stockData = stockData
        self.featureColumns = featureColumns
        self.transactionCost = transactionCost
        ## for quick accessing of a random stock
        self.stockList = self.stockData['ticker'].unique()
        self.currentStock = None
        self.currentStockData = None
        self.currentDay = None
        self.currentPosition = None
        ## 0 for cash 1 for invested


    def reset(self):
        rng = np.random.default_rng()
        newStock = rng.choice(self.stockList)
        self.currentStock = newStock
        self.currentStockData = self.stockData[self.stockData['ticker'] == self.currentStock]
        self.currentStockData = self.currentStockData.sort_values('date').reset_index(drop=True)
        self.currentDay = 0
        self.currentPosition = 0
        ## must be cash because you cant hold stock when starting to analyze it 
        return self.getState()


    def getState(self): 
        features = self.currentStockData[self.featureColumns].iloc[self.currentDay].values
        state = np.append(features, self.currentPosition)
        ## this is the dummy that indicates most recent action 
        return state
    

    def step(self, action): 
        stockReturnNextDay = self.currentStockData['ret'].iloc[self.currentDay + 1]
        avgMarketReturn = self.currentStockData['vw_mkt_ret'].iloc[self.currentDay + 1]
        if action == 1:
            reward = stockReturnNextDay - (1 - self.currentPosition) * self.transactionCost
        else:
            reward = avgMarketReturn
        
        self.currentDay += 1
        self.currentPosition = action 
        done = self.currentDay >= len(self.currentStockData) - 1
        if done:
            next_state = None
            ## this is when there is no more data on the stock 
        else:
            next_state = self.getState()

        return next_state, reward, done
    
