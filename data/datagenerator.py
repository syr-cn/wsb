import pandas as pd
from datetime import datetime, timedelta
import random as rd

t = datetime(2020, 12, 1, 15, 30)
end_t = datetime(2021, 2, 1, 15, 30)
dt = timedelta(1)

data = []

while t <= end_t:
    data.append([str(datetime.timestamp(t)), rd.random()])
    t = t+dt

df = pd.DataFrame(data)
df.to_csv('testy.csv')
