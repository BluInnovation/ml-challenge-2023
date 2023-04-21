import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.utils import remove_outliers_iqr
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    data_train_2[col] = remove_outliers_iqr(data_train_2[col], multiplier=300)
# %% Interpolate missing values for each column
for col in data_train_2.columns:
    data_train_2[col].interpolate(method="linear", inplace=True)
# %% Drop nan values
data_train_2.dropna(inplace=True)
#%% Plot the new inj diff column
sns.set()
sns.set_style("whitegrid")
sns.lineplot(x="SampleTimeUTC", y="inj_diff", data=data_train_2, linewidth=0.5)
plt.gcf().autofmt_xdate()
plt.ylim(-50, 50)
plt.show()
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
data_train_cut = data_train_2.loc["2011-01-01":]

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
decomposition = seasonal_decompose(data_train_cut["inj_diff"], period=24 * 7)
fig_5 = decomposition.plot()
fig_5.set_size_inches(15, 10)
plt.tight_layout()
plt.show()
# %% Calculate difference between consecutive values
D = pm.arima.utils.nsdiffs(
    data_train_cut["inj_diff"], m=24, max_D=24, test="ch"
)
#%% Load original test data
y_test = pd.read_csv("data/illinois_basing_test_original.csv")
y_test = y_test["inj_diff_Calculated"]
#%% Train auto ARIMA model
model_auto = pm.auto_arima(
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
#%% Train SARIMAX model
model = pm.ARIMA(
    order=(2, 2, 1),
    seasonal_order=(0, 1, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
#%%

model.fit(data_train_cut["inj_diff"], data_train_cut.drop("inj_diff", axis=1))
#%% Save model to pickle file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
#%% Load model from pickle file
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
#%% Print model summary
model = model_auto
print(model.summary())
#%% Plot model diagnostics
model.plot_diagnostics(figsize=(15, 10))
plt.tight_layout()
plt.show()
#%% Calculate the predictions
pred = model.predict(n_periods=len(data_test_2), X=data_test_2)
#%% Plot the predictions for comparison
plt.plot(data_test_2.index, pred, label="prediction")
plt.plot(data_test_2.index, y_test, label="actual")
plt.gcf().autofmt_xdate()
# plt.ylim(3, -3)
plt.legend()
plt.show()
#%% Calculate the mean absolute error
rmse = np.sqrt(mean_squared_error(y_test[1:], pred[1:]))
#%% Export pred as csv with column name "inj_diff"
pred = pd.DataFrame(pred, columns=["inj_diff"])
pred.to_csv("data/pred.csv", index=False)
#%% Put y_test and pred in one dataframe
df_y_test = pd.DataFrame(y_test)
df_y_test.index = data_test_2.index
df_y_test.columns = ["inj_diff"]
df_y_test["pred"] = pred.values
