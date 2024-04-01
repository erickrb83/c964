import os
import numpy as np
import streamlit as st
import pandas as pd
import pickle
import math

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Interactive graphs
import plotly.express as px
from plotly import graph_objs as go
import matplotlib.pyplot as plt


################################################################
# 
# Helper Functions
#
################################################################
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df



def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_and_eval_model(model, X_train, y_train, cv = 5):
    cv_results = cross_validate(model, X_train, y_train, cv = cv, scoring = ('r2', 'neg_root_mean_squared_error'),)
    print('Model:', model)
    r2_scores = cv_results['test_r2']
    print('R2 CV Scores:', r2_scores)
    print('R2 CV scores mean / stdev:', np.mean(r2_scores), '/', np.std(r2_scores)) 
    
    rmse_scores = cv_results['test_neg_root_mean_squared_error']
    rmse_scores = [-1*score for score in rmse_scores]
    print('RMSE CV scores:', rmse_scores)
    print('RMSE CV scores mean / stdev:', np.mean(rmse_scores), '/', np.std(rmse_scores))
    
    
def create_features(dt, lags = [28], wins = [7,28]):
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id", observed=False)["sales"].shift(lag).fillna(-1)

    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id", observed=False)[lag_col].transform(lambda x : x.rolling(win).mean()).fillna(-1)
        
    return dt

################################################################
# 
#   IMPORT DATA
#
################################################################
cal = pd.read_csv('calendar.csv')
cal = reduce_mem_usage(cal)
prices = pd.read_csv('sell_prices.csv')
prices = reduce_mem_usage(prices)
val = pd.read_csv('sales_train_validation.csv')
val = reduce_mem_usage(val)

cal.sort_values(by = 'date', inplace = True, ascending = True)
cal = cal.drop(['snap_TX', 'snap_WI'], axis = 1)

prices = prices.drop(['store_id'], axis = 1)

val = val.drop(['store_id', 'state_id'], axis = 1)
if val.isnull().values.any():
    val.fillna(0)

################################################################
# 
# MAIN PAGE
#
################################################################
def show_page():
    st.title("Demand Forecasting App")
    st.write("""
             #### Demand Forecasting App comparing a traditional forecasting tool, Croston Method, to a machine learning model, LightGBM.
            """)

    st.sidebar.header('User Filters')

show_page()

@st.cache_data
def get_df():
    cal = pd.read_csv('calendar.csv')
    prices = pd.read_csv('sell_prices.csv')
    val = pd.read_csv('sales_train_validation.csv')

        # Just in case sort dates from oldest to newest
    cal.sort_values(by = 'date', inplace = True, ascending = True)
    cal = cal.drop(['snap_TX', 'snap_WI'], axis = 1)
    prices = prices.drop(['store_id'], axis = 1)
    # Only keep relevant columns
    val = val.drop(['store_id', 'state_id'], axis = 1)
    # Find any Null Values
    if val.isnull().values.any():
        val.fillna(0)

    df = pd.melt(val, id_vars=['id', 'item_id', 'dept_id', 'cat_id'], var_name='d', value_name='sales')
    df = pd.merge(df, cal, on='d', how='left')
    df["d"] = df["d"].apply(lambda x: int(x.split("_")[1]))
    return df

del prices
df = get_df()
df = reduce_mem_usage(df)

def get_dataset(name):
    if name in datasets:
        data = model.load(name)
        return data
    return print('Dataset not found')

################################################################
# 
# SIDEBAR
#
################################################################
def user_filters():
    sorted_categories = sorted(df['cat_id'].unique())
    category_id = st.sidebar.selectbox(
        'Category', 
        list(sorted_categories))

    sorted_departments = sorted(df['dept_id'].unique())
    sorted_departments = [sorted_departments for sorted_departments in sorted_departments if category_id in sorted_departments]
    department_id = st.sidebar.selectbox(
        'Department', 
        sorted_departments)
    
    sorted_items = sorted(df['item_id'].unique())
    sorted_items = [sorted_items for sorted_items in sorted_items if department_id in sorted_items]
    item_id =st.sidebar.selectbox(
        'Item ID', 
        (sorted_items))

    return item_id

item_id = user_filters()

ok = st.sidebar.button('Forecast')
if ok: 
    print(True)

################################################################
#
# Feature Engineering
#
################################################################
# Drop unimportant columns
df = df.drop(['wm_yr_wk', 'weekday'], axis =1)

# Create Lags
df = create_features(df)

item_id_list = df['item_id'].unique().tolist()

# Create Features
cat_feat = ['item_id', 'dept_id', 'cat_id', 'wday', 'month', 'year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
for cc in cat_feat:
    le = LabelEncoder()
    df[cc] = le.fit_transform(df[cc])

item_id_list = dict(zip(item_id_list, df['item_id'].unique().tolist()))

total_days = df.d.max()
test_days = 31
train_days = total_days - test_days

################################################################
# 
# SPLIT THE DATA
#
################################################################
# Training Data Sets
X_train = df[df['d'] < train_days].drop(['id', 'sales', 'date'], axis = 1) 
y_train = df[df['d'] < train_days]['sales']

# Testing Data Sets
X_test = df[df['d'].between(train_days, total_days)].drop(['id','sales', 'date'], axis = 1)
y_test = df[df['d'].between(train_days, total_days)]['sales']

################################################################
#
# CROSTON METHOD
#
################################################################
def Croston(ts,extra_periods=0,alpha=0.4):
    d = np.array(ts) # Transform the input into a numpy array
    cols = len(d) # Historical period length
    d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods
    
    #level (a), periodicity(p) and forecast (f)
    a,p,f = np.full((3,cols+extra_periods),np.nan)
    q = 1 #periods since last demand observation
    
    # Initialization
    first_occurence = np.argmax(d[:cols]>0)
    a[0] = d[first_occurence]
    p[0] = 1 + first_occurence
    f[0] = a[0]/p[0]
# Create all the t+1 forecasts
    for t in range(0,cols):        
        if d[t] > 0:
            a[t+1] = alpha*d[t] + (1-alpha)*a[t] 
            p[t+1] = alpha*q + (1-alpha)*p[t]
            f[t+1] = a[t+1]/p[t+1]
            q = 1           
        else:
            a[t+1] = a[t]
            p[t+1] = p[t]
            f[t+1] = f[t]
            q += 1
       
    # Future Forecast 
    a[cols+1:cols+extra_periods] = a[cols]
    p[cols+1:cols+extra_periods] = p[cols]
    f[cols+1:cols+extra_periods] = f[cols]
                      
    df = pd.DataFrame.from_dict({"Demand":d,"Forecast":f,"Period":p,"Level":a,"Error":d-f})
    return df

preds = []

@st.cache_data
def croston_model(preds):
    item_list = df.item_id.unique().tolist()

    for item in item_list:
        data = df.loc[df['item_id'] == item]
        temp = Croston(data.sales, extra_periods= 1)
        temp = temp.drop(temp.tail(1).index)
        temp = round(temp.Forecast, 1)
        preds.append(temp)

    preds = pd.DataFrame(preds).T

    preds.columns = item_list
    preds = preds.T
    return preds
    
preds = croston_model(preds)
del df
################################################################
# 
# Evaluate Croston
#
################################################################
@st.cache_data
def eval_c(error_df):
    val.drop(['id', 'dept_id', 'cat_id'], axis = 1, inplace= True)
    y_true_sum = 0
    y_pred_sum = 0
    for i in range(len(val)):
        y_true = val.loc[i]
        y_true = y_true.iloc[1:]
        y_pred = preds.loc[i]
        error_df.append(rmse(y_true, y_pred))
        y_true_sum = y_true_sum + y_true.sum()
        y_pred_sum = y_pred_sum + y_pred.sum()
    

    numerator = len(val.loc[0]) * len(val)

    top = (((y_true_sum - y_pred_sum) **2)/ len(val))

    croston_rmse_total = math.sqrt(top/numerator)
    return croston_rmse_total, error_df

error_df = []
croston_rmse_total, error_df = eval_c(error_df)
del val
################################################################
# 
# LIGHTGBM MODEL
#
################################################################
@st.cache_data
def lgbm_model():
    dtrain = lgb.Dataset(X_train , label = y_train,  free_raw_data=False)
    dvalid = lgb.Dataset(X_test, label = y_test,   free_raw_data=False)

    params = {
        "objective" : "poisson",
        "metric" : "rmse",
        "learning_rate" : 0.075,
        "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        'verbosity': 1,        
        'num_leaves': 128,
        "min_data_in_leaf": 50,
    }

    model_lgb = lgb.train(params, dtrain, valid_sets = [dtrain, dvalid])
    return model_lgb  

model_lgb = lgbm_model()
del X_train, y_train
################################################################
# 
# Evaluate LGBM
#
################################################################
rmse_lgb_each = []

@st.cache_data
def evaluate_lgbm():
    feature_importance = lgb.plot_importance(model_lgb,height = 0.5)

    st.pyplot(feature_importance.figure)

    lgb_preds = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)

    y_real = y_test

    y_real = pd.concat([X_test, y_test], axis = 1)

    y_real['y_preds'] = lgb_preds

    y_real['r_value'] = y_real['sales'] - y_real['y_preds']

    overall_rmse_lgb = rmse(y_real['sales'], y_real['y_preds'])

    item_list = y_real.item_id.unique().tolist()

    for item in item_list:
        data = y_real.loc[y_real['item_id'] == item]
        rmse_lgb_each.append(rmse(data.sales, data.y_preds))

    return overall_rmse_lgb, rmse_lgb_each, y_real

overall_rmse_lgb, rmse_lgb_each, y_real = evaluate_lgbm()

del X_test, y_test
################################################################
# 
# Compare LGBM to Croston
#
################################################################
@st.cache_data
def comparison(error_df, rmse_lgb_each):

    error_df.columns = ['Croston']
    rmse_lgb_each = pd.DataFrame(rmse_lgb_each)
    rmse_lgb_each.columns = ['lgb']
    error_df = pd.concat([error_df, rmse_lgb_each], axis = 1)

    N = 3
    ind = np.arange(N)
    width = 0.25
    fig, ax = plt.subplots()

    lgbm_sum = np.round([np.median(error_df['lgb']), np.mean(error_df['lgb']), overall_rmse_lgb], 3)
    croston_sum = np.round([np.median(error_df['Croston']), np.mean(error_df['Croston']), croston_rmse_total], 3)

    bar1 = plt.bar(ind, lgbm_sum, width, label= 'LightGBM')
    bar2 = plt.bar(ind + width, croston_sum, width, label = "Croston")

    plt.xticks(ind+width/2,['Median RMSE', 'Mean RMSE', 'RMSE']) 
    ax.set_ylim([0, 11])
    plt.bar_label(bar1, label_type='edge', padding=4)
    plt.bar_label(bar2, label_type='edge', padding=4)
    ax.legend(handles = [bar1, bar2])

    st.pyplot(fig)

    return croston_rmse_total, overall_rmse_lgb
    
error_df = pd.DataFrame(error_df)  

croston_rmse_total, overall_rmse_lgb = comparison(error_df, rmse_lgb_each) 

@st.cache_data
def plot_data(item_id, preds):
    item_id_num = item_id_list[item_id]
    cal['d'] = cal["d"].apply(lambda x: int(x.split("_")[1]))
    dates = cal['date'].unique().tolist()
    dates = dict(zip(cal['d'], dates)) 
    
    
    preds = preds.T
    preds['date'] = dates
    preds['date'] = preds['date'].shift(-1)
    preds['date'][1912] = '2106-04-24'
    preds = preds[-32:]
    
    filtered_df = y_real.loc[y_real['item_id'] == item_id_num]
     
    selected_df = pd.concat([filtered_df[['sales', 'y_preds']].reset_index(drop=True), preds[[item_id_num, 'date']].reset_index(drop=True)], axis = 1)
   
    rmse_lgb = np.round(rmse(selected_df['sales'], selected_df['y_preds']), 3)
    rmse_cros = np.round(rmse(selected_df['sales'], selected_df[item_id_num]), 3)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = selected_df.index,
            y = selected_df[item_id_num],
            name = 'Croston'
        )
    )
    fig.add_trace(
        go.Scatter(
            x = selected_df.index,
            y = selected_df['y_preds'],
            name = 'LightGBM'
        )
    )
    fig.add_trace(
        go.Scatter(
            x = selected_df.index,
            y = selected_df['sales'],
            name = 'Sales'
        )
    )
    fig.update_layout(
        xaxis = dict(
        tickmode = 'array',
        tickvals = selected_df.index,
        ticktext = selected_df["date"]
    )
)
    fig.layout.update(title_text=f'Demand Model Comparison for {item_id}', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

        
    st.write(f'LGBM RMSE for {item_id} = {rmse_lgb}')
    st.write(f'Croston RMSE for {item_id} = {rmse_cros}')
    del dates

plot_data(item_id, preds)


