# project libraries
from db_api import SIATE_DB_API
import utils
from hyperparameters import Config

# internal libraries
import datetime

# external libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import statsmodels.formula.api as smf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.api import OLS



def timeseres_visualization(features:list[str],
                            output_file_path:str,
                            patient_idx:int=0,
                            min_allowed_samples:int=35,
                            hparams:dict=None,
                            verbose:bool=False,
                            just_get_timeseries_with_no_plot:bool=False,
                            from_db:bool=True,
                            del_nans:bool=True,
                            as_dataframe:bool=False,
                           ) -> np.ndarray|pd.DataFrame:
    '''
    Plots the timeseries and returns it as a Numpy NDArray with shape `(n_samples, n_features)`
    '''
    if verbose:
        print("Getting data ", end="")
    i:int = 0
    oldest_date = datetime.datetime(year=1800, month=1, day=1)
    newest_date = datetime.datetime(year=3000, month=1, day=1)
    if from_db:
        DB = SIATE_DB_API()
        patients_list = DB.get_patients_list()
    else:
        patients_list = sorted([f[:-4] for f in utils.get_files_in_dir(directory=hparams.ICOT_csv_sessions_with_triage)])
    while True:
        if not from_db:
            X_HD = utils.get_data(features=features,
                                sane_threshold=Config().sane_threshold,
                                patients_list=[patients_list[patient_idx+i]] if patient_idx > -1 else patients_list,
                                verbose=False,
                                del_nans=del_nans,
                                del_sick=False,
                                get_sick=False,
                                csv_folder_path=Config().ICOT_csv_sessions_with_triage,
                                db_api=None,
                                as_dataframe=True,
                                )
        else:
            if patient_idx > -1: # get single patient
                patient = patients_list[patient_idx+i]
                X_HD = DB.get_patient_dataframe(patient=patient,
                                                oldest_date=oldest_date,
                                                newest_date=newest_date,
                                                features=features,
                                               )
            else:
                X_HD = DB.get_all_patients_dataframe(oldest_date=oldest_date,
                                                     newest_date=newest_date, 
                                                     features=features
                                                    )
        if del_nans:
            X_HD = X_HD.dropna()
        if len(X_HD) >= min_allowed_samples:
            break
        if verbose:
            print(".", end="")
            i += 1
    X_HD.reset_index(inplace=True)
    if verbose:
        print(" done.")
        print(f"{len(X_HD)} sessions found for given parameters.") 
    # plot timeseries
    plt.figure(figsize=(12, 6))

    for feature in features:
        plt.plot(list(X_HD[feature]), label=feature)

    if not just_get_timeseries_with_no_plot:
        plt.grid()
        plt.legend()
        plt.title(f"Time Series Data of patient '{patients_list[patient_idx]}'")
        plt.savefig(output_file_path, dpi=200)
        plt.clf()

    if as_dataframe:
        return X_HD
    return X_HD.to_numpy()



def plot_ACF_PACF(timeseries:np.ndarray, output_folder:str, series_name:str, max_lag:int=10, verbose:bool=False) -> None:

    if verbose:
        print("Generating diagnostic check plots ... ", end="")


    lags = min(max_lag, len(timeseries))

    # Plot the ACF of the residuals
    plt.figure(figsize=(6, 6))
    sm.graphics.tsa.plot_acf(timeseries, lags=lags)
    plt.title(f"'{series_name}' ACF Plot")
    plt.grid()
    plt.savefig(f"{output_folder}{series_name.replace('/', '-')}-ACF.png")
    plt.clf()

    # Plot the PACF of the residuals
    plt.figure(figsize=(6, 6))
    sm.graphics.tsa.plot_pacf(timeseries, lags=lags)
    plt.title(f"'{series_name}' PACF Plot")
    plt.grid()
    plt.savefig(f"{output_folder}{series_name.replace('/', '-')}-PACF.png")
    plt.clf()

    if verbose:
        print("done.")



def diagnostic_check(model, output_folder:str, lag=10, model_name:str="", verbose:bool=False) -> None:

    if verbose:
        print("Generating residual diagnostic check plots ... ", end="")
    # Plot the residuals
    residuals = model.resid
    plt.figure(figsize=(10, 6))
    plt.plot(residuals)
    plt.title(f'Residuals of the {model_name} Model')
    plt.savefig(f"{output_folder}residuals.png")
    plt.clf()

    # Plot the ACF of the residuals
    plt.figure(figsize=(10, 6))
    sm.graphics.tsa.plot_acf(residuals, lags=lag)
    plt.savefig(f"{output_folder}residuals_ACF.png")
    plt.clf()

    # Plot the PACF of the residuals
    plt.figure(figsize=(10, 6))
    sm.graphics.tsa.plot_pacf(residuals, lags=lag)
    plt.savefig(f"{output_folder}residuals_PACF.png")
    plt.clf()

    if verbose:
        print("done.")
        print(f"\tResiduals Expected Value = {np.average(residuals)}")



def forecast_expected_value(model, timeseries:pd.Series|np.ndarray, n_periods:int) -> pd.Series:
    return model.predict(n_periods=n_periods, 
                         exogenous=timeseries,
                         return_conf_int=False)



def plot_forecast(time_series:pd.Series, forecast:pd.Series, output_folder:str, verbose:bool=False, model_name:str="ARIMA") -> None:
    # Plot the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, label='Original')
    if len(time_series) != len(forecast):
        plt.plot(np.arange(len(time_series), len(time_series) + len(forecast)), forecast, label='Forecast')
    else:
        plt.plot(forecast, label='Forecast')
    # plt.fill_between(np.arange(len(time_series), len(time_series) + forecast_steps), conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
    plt.title(f'{model_name} Model Forecast')
    plt.legend()
    plt.savefig(f"{output_folder}forecast.png")
    plt.clf()



def arima_model(p:int, i:int, q:int, train_series:pd.Series, verbose:bool=False):
    model = tsa.ARIMA(endog=train_series, order=(p, i, q)).fit()
    if verbose:
        print(model.summary())
    return model 



def var_model(p:int, train_series:pd.Series, verbose:bool=False) -> VAR:
    '''
    `train_series` shape must be `(n_samples, n_features)`
    '''
    model = VAR(train_series).fit(p)
    if verbose:
        print(model.summary())
    return model



def check_stationarity(univariate_series:pd.Series|np.ndarray, significance_level:float=0.05, detailed_info:bool=False, verbose:bool=False, print_full_analysis:bool=False) -> bool|int:
    '''
    ADF and KPSS tests for stationarity.
    
    Returns `True` if the series is stationary, `False` otherwise.  The `univariate_series` timeseries must be univariate.

    If `detailed_info` is true it returns the type of (non)stationarity as well:
    - `0`: The series is **stationary**.
    - `1`: The series is **UNIT ROOT** non-stationary.
    - `2`: The series is stationary around a **DETERMINISTIC TREND**
    - `3`: The series is **DIFFERENCE** stationary.
    '''
    adf = adf_test(timeseries=univariate_series, verbose=print_full_analysis)
    adf_stationarity:bool = False

    # ADF checks
    if (adf['p-value'] > adf['Critical Value (5%)']) and (adf['Test Statistic'] > adf['Critical Value (1%)']) and (adf['Test Statistic'] > adf['Critical Value (5%)']) and (adf['Test Statistic'] > adf['Critical Value (10%)']):
        # NON STATIONARY
        adf_stationarity = False
    elif (adf['p-value'] < significance_level):
        # STATIONARY
        adf_stationarity = True
    else:
        adf_stationarity = False
    
    if print_full_analysis:
        print('\n')
    kpss = kpss_test(timeseries=univariate_series, verbose=print_full_analysis)
    kpss_stationarity:bool = False

    # KPSS checks
    if (kpss['p-value'] < significance_level):
        # NON STATIONARY
        kpss_stationarity = False
    else:
        # STATIONARY
        kpss_stationarity = True

    # COMPARE
    if kpss_stationarity and adf_stationarity:
        if verbose:
            print("The series is stationary.")
        return True if not detailed_info else 0
    elif (not kpss_stationarity) and (not adf_stationarity):
        if verbose:
            utils.print_colored("The series is UNIT ROOT, thus not stationary.", 'red')
        return False if not detailed_info else 1
    elif (kpss_stationarity) and (not adf_stationarity):
        if verbose:
            utils.print_colored("The series is stationary around a DETERMINISTIC TREND, use detrending techniques to make it stationary.", 'red')
        return False if not detailed_info else 2
    elif (not kpss_stationarity) and (adf_stationarity):
        if verbose:
            utils.print_colored("The series is DIFFERENCE stationary, use differencing techniques to make it stationary.", 'red')
        return False if not detailed_info else 3
    else:
        raise SyntaxError
    


def diff_timeseries(timeseries:np.ndarray) -> np.ndarray:
    return np.diff(timeseries, axis=0)



def ith_diff_timeseries(timeseries:np.ndarray, i:int=1) -> pd.Series:
    assert i >= 0
    new_series:np.ndarray = timeseries.copy()
    for _ in range(i):
        new_series = diff_timeseries(timeseries=new_series)
    return new_series



def non_stationary_features_list(multivariate_timeseries:pd.Series|np.ndarray, features:list[str], detailed_info:bool=False, verbose:bool=False) -> list[str]|list[tuple[str,int]]:
    '''
    If `detailed_info` returns a list of tuples `(feature_name, nonstationarity_type)`
    '''
    non_stationary_features:list[str] = list()
    if verbose:
            print("Stationarity check:")
    for i in range(len(features)):
        feature = features[i]
        if verbose:
            print(f"\t{feature}: ", end="")
        is_stationary = check_stationarity(univariate_series=multivariate_timeseries[:,i], detailed_info=detailed_info, verbose=verbose)
        if detailed_info:
            if not is_stationary > 0:
                non_stationary_features.append((feature,is_stationary))
        else:
            if not is_stationary:
                non_stationary_features.append(feature)
    return non_stationary_features



def adf_test(timeseries:pd.Series|np.ndarray, verbose:bool=False) -> pd.Series:
    '''
    Check stationarity metrics.
    '''
    if verbose:
        print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries)#, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    if verbose:
        print(dfoutput)
    return dfoutput



def kpss_test(timeseries:pd.Series|np.ndarray, verbose:bool=False) -> pd.Series:
    '''
    CHeck for stationarities around deterministic trends
    '''
    if verbose:
        print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c")#, nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    if verbose:
        print(kpss_output)
    return kpss_output



def check_cointegration(series_1:pd.Series|np.ndarray, series_2:pd.Series|np.ndarray, verbose:bool=False) -> bool:
    '''
    Check if the pair of tiemseries can be cointegrated.
    '''
    old_test = OLS(series_1, series_2).fit()
    result = adfuller(old_test.resid)
    t_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    if (t_stat <= critical_values['10%']) and (p_value <= 0.1):
        if verbose:
            print("Pair is co-integrated")
        return True
    else:
        if verbose:
            print("Pair is NOT co-integrated")
        return False
    

