# region imports
from AlgorithmImports import *
from helpers import (compute_features_from_daily_data, select_option_contract)
import pickle
import base64
# endregion

class TenDayOnePercent(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2020,1,1)
        self.set_end_date(2023,1,1)
        self.set_cash(100000)
        self.ticker = 'QQQ'
        self.symbol = self.AddEquity(self.ticker, Resolution.DAILY).Symbol
        self.call_contract = None
        self.put_contract = None

        # Load lightgbm model
        model_filename = 'SK_10_1.0.txt'

        # Add options for SPY
        self.option = self.AddOption(self.ticker, Resolution.Daily)
        self.option.SetFilter(-10,+10, timedelta(0), timedelta(30))

        if self.ObjectStore.ContainsKey(model_filename):

            # Load the model using joblib
            str_data = self.ObjectStore.Read(model_filename)
            model_bytes = base64.b64decode(str_data)

            # Deserialize the model from bytes
            self.model = pickle.loads(model_bytes)

        self.contract = None

    def on_data(self, data: Slice):

        # Get the last 100 days
        history = self.History(self.symbol, 100, Resolution.Daily)
        history_close_hundred_day = history['close'].to_frame()

        # Compute features
        features_array = compute_features_from_daily_data(history_close_hundred_day).iloc[-1:,:]

        # Predict using the model
        prediction = self.model.predict(features_array)

        # # If the model predicts 1, then buy; if 0, then sell; else do nothing.
        # # In: False, Out: True
        # # Select an option contract if there's none
        # if self.contract is None or not self.Securities[self.contract].Invested:
        #     self.contract = select_option_contract(self, data, prediction[0])

        # # Exit the position if the prediction changes
        # if self.contract is not None and self.Securities[self.contract].Invested:
        #     if not prediction[0]:  # If prediction changes to within ±1%, exit the position
        #         self.Liquidate(self.contract)

        # Select option contracts if none are held or the prediction changes
        if (self.call_contract is None or not self.Securities[self.call_contract].Invested) and \
           (self.put_contract is None or not self.Securities[self.put_contract].Invested):
            result = select_option_contract(self, data, prediction)
            
            if result is not None:
                self.call_contract, self.put_contract = result
            else:
                return

        # Exit the position if the prediction changes to within ±1%
        if self.call_contract is not None and self.put_contract is not None:
            if not prediction:  # If prediction changes to within ±1%, expect stability
                if self.Securities[self.call_contract].Invested:
                    self.Liquidate(self.call_contract)
                if self.Securities[self.put_contract].Invested:
                    self.Liquidate(self.put_contract)

    

