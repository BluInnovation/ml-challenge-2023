import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.utils import remove_outliers_iqr
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

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
# %% Exclude columns
cols_to_exclude = [
    "Avg_PLT_CO2InjRate_TPH",
    "Avg_VW1_Z03D6945Ps_psi",
    "Avg_VW1_Z03D6945Tp_F",
]
data_train.drop(cols_to_exclude, axis=1, inplace=True)
data_test.drop(cols_to_exclude[1:], axis=1, inplace=True)
# %% Plot inj_diff column with time  using seaborn
sns.set()
sns.set_style("whitegrid")
sns.lineplot(x="SampleTimeUTC", y="inj_diff", data=data_train, linewidth=0.5)
plt.gcf().autofmt_xdate()
plt.ylim(-50, 50)
plt.show()
# %% Create a heatmap of the correlation matrix
corr = data_train.corr()
fig_1, ax_1 = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, vmin=-1, vmax=1, cmap="coolwarm", ax=ax_1)
plt.suptitle("Correlation matrix")
plt.tight_layout()
plt.show()
# %% Remove all the columns that contain the word "VW1"
cols_2 = [col for col in data_train.columns if "VW1" in col]
data_train_2 = data_train.drop(cols_2, axis=1)
data_test_2 = data_test.drop(cols_2, axis=1)
# %% Create the heatmap of the correlation matrix again with the new data
corr_2 = data_train_2.corr()
fig_2, ax_2 = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_2, vmin=-1, vmax=1, cmap="coolwarm", ax=ax_2)
plt.suptitle("Correlation matrix")
plt.tight_layout()
plt.show()
# %% Remove outliers using IQ range for each column
for col in data_train_2.columns:
    data_train_2[col] = remove_outliers_iqr(data_train_2[col], multiplier=5)
# %% Interpolate missing values for each column
for col in data_train_2.columns:
    data_train_2[col].interpolate(method="linear", inplace=True)
# %% Drop nan values
data_train_2.dropna(inplace=True)
# %% Plot lineplot of all the columns in the data_train_2 in a different axes
data_train_2.plot(linewidth=0.5, subplots=True, figsize=(10, 10))
plt.tight_layout()
plt.show()
# %% Create boxplot for each column in the data_train_2 in 6 month intervals with
# different colors
fig_3, ax_3 = plt.subplots(7, 1, figsize=(20, 20))
colors = [
    "lightblue",
    "lightgreen",
    "lightpink",
    "tab:orange",
    "lightgray",
    "lightcoral",
    "tab:purple",
]
axs = ax_3.flatten()
for i, col in enumerate(data_train_2.columns):
    sns.boxplot(
        data=data_train_2,
        x=data_train_2.index.to_period("M"),
        y=col,
        ax=axs[i],
        color=colors[i],
    )
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()
#%% Select the values from 2012 onwards
data_train_cut = data_train_2.loc["2012-01-01":]
# %% Visualize the time series
pm.tsdisplay(data_train_cut["inj_diff"], lag_max=100, figsize=(15, 10))
plt.show()
# %% Plot ACF and PACF
fig_4, ax_4 = plt.subplots(2, 1)
plot_acf(data_train_cut["inj_diff"], lags=50, ax=ax_4[0])
plot_pacf(data_train_cut["inj_diff"], lags=50, ax=ax_4[1])
plt.tight_layout()
plt.show()
# %% Decompose the time series
decomposition = seasonal_decompose(data_train_cut["inj_diff"], period=24 * 7 * 4)
fig_5 = decomposition.plot()
fig_5.set_size_inches(15, 10)
plt.tight_layout()
plt.show()
# %% Calculate difference between consecutive values
D = pm.arima.utils.nsdiffs(
    data_train_cut["inj_diff"], m=24, max_D=24, test="ch"
)
# %% Train with SARIMAX using pmdarima
model = pm.auto_arima(
    data_train_cut["inj_diff"],
    X=data_train_cut.drop("inj_diff", axis=1),
    start_p=0,
    start_q=0,
    max_p=5,
    max_q=5,
    m=24,
    start_P=0,
    seasonal=True,
    D=D,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)
