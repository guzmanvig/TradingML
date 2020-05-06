import duka.app.app as import_ticks_method
from duka.core.utils import TimeFrame
import datetime
import pandas as pd
import matplotlib.pyplot as plt


def import_data():
    start_date = datetime.date(2020,1,1)
    end_date = datetime.date(2020,5,1)
    Assets = ["EURUSD"]
    import_ticks_method(Assets,start_date,end_date,1,TimeFrame.H1,"./data",True)


def get_data():
    data = pd.read_csv("./data/EURUSD-2020_01_01-2020_05_01.csv")  # Time, open, close, high, low
    open = [float(data["open"][i]) for i in range(len(data["open"]))]  # Use the open price
    hour = [int(data["time"][i][11:13]) for i in range(len(data["time"]))]  # Save only the hour
    return open, hour

# get_data()
# import_data()







