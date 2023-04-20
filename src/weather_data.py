import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
# %% Load data
data_folder = Path("data")
data_train = pd.read_csv(data_folder / "illinois_basing_train.csv")
data_test = pd.read_csv(data_folder / "illinois_basing_test.csv")
# %% Dates
data_train["SampleTimeUTC"] = pd.to_datetime(
    data_train["SampleTimeUTC"], format="%d/%m/%Y %H:%M"
)
data_test["SampleTimeUTC"] = pd.to_datetime(
    data_test["SampleTimeUTC"], format="%m/%d/%Y %H:%M"
)
# %% Set index
data_train.set_index("SampleTimeUTC", inplace=True)
data_test.set_index("SampleTimeUTC", inplace=True)
#%% Get min and max dates
# start_date = data_train.index.min().to_pydatetime()
# end_date = data_test.index.max().to_pydatetime()
#%% Get weather data
api_key = "727842ce9ac8ca86e0a725008ddb191d"

latitude = "39.876963"
longitude = "-88.893410"

start_date = int(datetime(2019, 1, 1).timestamp())
end_date = int(datetime(2019, 1, 3).timestamp())
date_range = pd.date_range(start_date, end_date, freq="H")

hourly_weather_data = []

# Fetch hourly weather data for each hour in the date range

url = f"https://history.openweathermap.org/data/2.5/history/city?lat=" \
      f"{latitude}&lon={longitude}&type=hour&start={start_date}&end={end_date}" \
      f"&appid={api_key}"
response = requests.get(url)
data = response.json()

# Convert the fetched data to a pandas DataFrame
df = pd.DataFrame(hourly_weather_data)

