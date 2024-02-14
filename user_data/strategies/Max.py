# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (IStrategy, informative)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class Max(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.1
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.1

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    plot_config = {
        'main_plot': {
            'max1': {'color': 'green'},
            'max2': {'color': 'green'},
            'max3': {'color': 'green'},
            'max4': {'color': 'green'},
        },
        'subplots': {
            "ST": {
                "supertrend_3": {'color': 'purple'},
            },
            "PROFIT": {
                "profit_high": {'color': 'red'},
                "profit_close": {'color': 'blue'},
            },
            "RSI": {
                "rsi": {'color': 'orange'},
            },
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['supertrend_3'] = self.supertrend(dataframe, 1, 10)['STX']

        dataframe['max1'] = dataframe['high'].rolling(9).max()
        dataframe['max2'] = dataframe['high'].rolling(21).max()
        dataframe['max3'] = dataframe['high'].rolling(50).max()
        dataframe['max4'] = dataframe['high'].rolling(100).max()

        dataframe['profit_high'] = np.where(dataframe['high'] < dataframe['high'].shift(1), 0, (dataframe['high'] - dataframe['high'].shift(1)) / dataframe['high'].shift(1) * 100)
        dataframe['profit_close'] = np.where(dataframe['high'] < dataframe['high'].shift(1), 0, (dataframe['close'] - dataframe['high'].shift(1)) / dataframe['high'].shift(1) * 100)

        dataframe['rsi'] = ta.RSI(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['close'], dataframe['max3'].shift(1)) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[
            (
                # (
                #     (qtpylib.crossed_below(dataframe['supertrend_1'], 1)) |
                #     (qtpylib.crossed_below(dataframe['supertrend_2'], 1)) |
                #     (qtpylib.crossed_below(dataframe['supertrend_3'], 1))
                # ) &

                # (dataframe['close'] <= dataframe['min']) &

                (dataframe['volume'] < 0)  # Make sure Volume is not 0
            ),

            'exit_long'] = 1

        return dataframe

    def supertrend(self, dataframe: DataFrame, multiplier, period):
        df = dataframe.copy()

        df['TR'] = ta.TRANGE(df)
        df['ATR'] = ta.SMA(df['TR'], period)

        st = 'ST_' + str(period) + '_' + str(multiplier)
        stx = 'STX_' + str(period) + '_' + str(multiplier)

        # Compute basic upper and lower bands
        df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
        df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']

        # Compute final upper and lower bands
        df['final_ub'] = 0.00
        df['final_lb'] = 0.00
        for i in range(period, len(df)):
            df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
            df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]

        # Set the Supertrend value
        df[st] = 0.00
        for i in range(period, len(df)):
            df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i] else \
                            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] >  df['final_ub'].iat[i] else \
                            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i] else \
                            df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] <  df['final_lb'].iat[i] else 0.00
        # Mark the trend direction up/down
        df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 0, 1), np.NaN)

        # Remove basic and final bands from the columns
        df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

        df.fillna(0, inplace=True)

        return DataFrame(index=df.index, data={
            'ST' : df[st],
            'STX' : df[stx]
        })
    
    def pivots_points(self, dataframe: pd.DataFrame, timeperiod=30, levels=3) -> pd.DataFrame:
        """
        Pivots Points

        https://www.tradingview.com/support/solutions/43000521824-pivot-points-standard/

        Formula:
        Pivot = (Previous High + Previous Low + Previous Close)/3

        Resistance #1 = (2 x Pivot) - Previous Low
        Support #1 = (2 x Pivot) - Previous High

        Resistance #2 = (Pivot - Support #1) + Resistance #1
        Support #2 = Pivot - (Resistance #1 - Support #1)

        Resistance #3 = (Pivot - Support #2) + Resistance #2
        Support #3 = Pivot - (Resistance #2 - Support #2)
        ...

        :param dataframe:
        :param timeperiod: Period to compare (in ticker)
        :param levels: Num of support/resistance desired
        :return: dataframe
        """

        data = {}

        low = qtpylib.rolling_mean(
            series=pd.Series(index=dataframe.index, data=dataframe["low"]), window=timeperiod
        )

        high = qtpylib.rolling_mean(
            series=pd.Series(index=dataframe.index, data=dataframe["high"]), window=timeperiod
        )

        # Pivot
        data["pivot"] = qtpylib.rolling_mean(series=qtpylib.typical_price(dataframe), window=timeperiod)

        # Resistance #1
        data["r1"] = (2 * data["pivot"]) - low

        # Resistance #2
        data["s1"] = (2 * data["pivot"]) - high

        # Calculate Resistances and Supports >1
        for i in range(2, levels + 1):
            prev_support = data["s" + str(i - 1)]
            prev_resistance = data["r" + str(i - 1)]

            # Resitance
            data["r" + str(i)] = (data["pivot"] - prev_support) + prev_resistance

            # Support
            data["s" + str(i)] = data["pivot"] - (prev_resistance - prev_support)

        return pd.DataFrame(index=dataframe.index, data=data)
