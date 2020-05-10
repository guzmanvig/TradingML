from math import pi, sin

import duka.app.app as import_ticks_method
from duka.core.utils import TimeFrame
import datetime
import pandas as pd


def import_data():
    start_date = datetime.date(2020,1,1)
    end_date = datetime.date(2020,5,1)
    Assets = ["EURUSD"]
    import_ticks_method(Assets,start_date,end_date,1,TimeFrame.H1,"./data",True)


def get_data(type):
    if type == "CSV":
        data = pd.read_csv("./data/EURUSD-2020_01_01-2020_05_01.csv")  # Time, open, close, high, low
        exchange = [float(data["open"][i]) for i in range(len(data["open"]))]  # Use the open price
        hour = [int(data["time"][i][11:13]) for i in range(len(data["time"]))]  # Save only the hour
        return exchange, hour
    if type == "SIN":
        return create_sin_data()
    if type == "MOCK":
        return create_mock_data()


def create_mock_data():
    exchanges = [1, 3, 5, 7, 9, 2, 4, 6, 8]
    hours = [15, 23, 0, 15, 23, 0, 15, 23, 0]
    return exchanges, hours


def create_sin_data():
    exchanges = []
    hours = []
    for i in range(200000):
        hour = i % 24
        x = hour * 2 * pi / 24
        exchange = 1 + sin(x)
        exchanges.append(exchange)
        hours.append(hour)
    return exchanges, hours


# get_data()
# import_data()







