#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 10:32:09 2018

@author: m.jones
"""

#%%
#!pip install git+https://github.com/MarcusJones/kaggle_utils.git

#%% ===========================================================================
# Logging
# =============================================================================
import sys
import logging

#Delete Jupyter notebook root logger handler
logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.INFO)

# Create formatter
#FORMAT = "%(asctime)s - %(levelno)-3s - %(module)-10s  %(funcName)-10s: %(message)s"
#FORMAT = "%(asctime)s - %(levelno)-3s - %(funcName)-10s: %(message)s"
#FORMAT = "%(asctime)s - %(funcName)-10s: %(message)s"
FORMAT = "%(asctime)s : %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
#DATE_FMT = "%H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FMT)

# Create handler and assign
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.handlers = [handler]
logging.info("Logging started")



#%%
import os
from pathlib import Path
# %% Globals
#
# LANDSCAPE_A3 = (16.53, 11.69)
# PORTRAIT_A3 = (11.69, 16.53)
# LANDSCAPE_A4 = (11.69, 8.27)
if 'KAGGLE_WORKING_DIR' in os.environ:
    DEPLOYMENT = 'Kaggle'
else:
    DEPLOYMENT = 'Local'
logging.info("Deployment: {}".format(DEPLOYMENT))
if DEPLOYMENT=='Kaggle':
    # PATH_DATA_ROOT = Path.cwd() / '..' / 'input'
    SAMPLE_FRACTION = 1
    # import transformers as trf
if DEPLOYMENT == 'Local':
    # PATH_DATA_ROOT = r"~/DATA/petfinder_adoption"
    SAMPLE_FRACTION = 1
    # import kaggle_utils.transformers as trf


# PATH_OUT = r"/home/batman/git/hack_sfpd1/Out"
# PATH_OUT_KDE = r"/home/batman/git/hack_sfpd1/out_kde"
# PATH_REPORTING = r"/home/batman/git/hack_sfpd1/Reporting"
# PATH_MODELS = r"/home/batman/git/hack_sfpd4/models"
# TITLE_FONT = {'fontname': 'helvetica'}


# TITLE_FONT_NAME = "Arial"
# plt.rc('font', family='Helvetica')

#%% ===========================================================================
# Standard imports
# =============================================================================
import os
from pathlib import Path
import sys
import zipfile
from datetime import datetime
import gc
import time
from pprint import pprint

#%% ===========================================================================
# ML imports
# =============================================================================
import numpy as np
print('numpy', np.__version__)
import pandas as pd
print('pandas', pd.__version__)
import sklearn as sk
print('sklearn', sk.__version__)

import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.pipeline
import sklearn.model_selection

from sklearn_pandas import DataFrameMapper

# Models
import lightgbm as lgb
print("lightgbm", lgb.__version__)
import xgboost as xgb
print("xgboost", xgb.__version__)
from catboost import CatBoostClassifier
import catboost as catb
print("catboost", catb.__version__)

# Metric
from sklearn.metrics import cohen_kappa_score
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

#%% ===========================================================================
# Custom imports
# =============================================================================

# %% {"_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5", "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19"}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings

from fancyimpute import SimpleFill

from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

warnings.simplefilter(action='ignore', category=FutureWarning)

def rmsle(real, predicted):
   sum=0.0
   for x in range(len(predicted)):
       if predicted[x]<0 or real[x]<0: #check for negative values
           continue
       p = np.log(predicted[x]+1)
       r = np.log(real[x]+1)
       sum = sum + (p - r)**2
   return (sum/len(predicted))**0.5


#%%
def timeit(method):
    """ Decorator to time execution of transformers
    :param method:
    :return:
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print("\t {} {:2.1f}s".format(method.__name__, (te - ts)))
        return result
    return timed

#%%

class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, dtype=None, inverse=False,
                 return_vector=True, name=None):
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        self.return_vector = return_vector
        self.name = name

        if isinstance(self.columns, str):
            self.columns = [self.columns]

        logging.info("Init {} on cols: {}".format(name, columns))

    def check_condition(self, x, col):
        cond = (self.dtype is not None and x[col].dtype == self.dtype) or \
               (self.columns is not None and col in self.columns)
        return self.inverse ^ cond

    def fit(self, x, y=None):
        return self

    def _check_if_all_columns_present(self, x):
        if not self.inverse and self.columns is not None:
            missing_columns = set(self.columns) - set(x.columns)
            if len(missing_columns) > 0:
                missing_columns_ = ','.join(col for col in missing_columns)
                raise KeyError("Keys are missing in the record: {}, columns required:{}".format( missing_columns_, self.columns))

    def transform(self, x):
        logging.info("{} is transforming...".format(self.name))
        # check if x is a pandas DataFrame
        if not isinstance(x, pd.DataFrame):
            raise KeyError("Input is not a pandas DataFrame it's a {}".format(type(x)))

        selected_cols = []
        for col in x.columns:
            if self.check_condition(x, col):
                selected_cols.append(col)

        # if the column was selected and inversed = False make sure the column
        # is in the DataFrame
        self._check_if_all_columns_present(x)

        # if only 1 column is returned return a vector instead of a dataframe
        if len(selected_cols) == 1 and self.return_vector:
            return list(x[selected_cols[0]])
        else:
            return x[selected_cols]


#%%
class TransformerLog():
    """Add a .log attribute for logging
    """
    @property
    def log(self):
        return "Transformer: {}".format(type(self).__name__)

# %%==============================================================================
# Imputer1D - Simple Imputer wrapper
# ===============================================================================
class Imputer1D(sk.preprocessing.Imputer):
    """
    A simple wrapper class on Imputer to avoid having to make a single column 2D.
    """
    def fit(self, X, y=None):
        if X.ndim == 1:
            X = np.expand_dims(X, axis=1)
        # Call the Imputer as normal, return result
        return super(Imputer1D, self).fit(X, y=None)

    def transform(self, X, y=None):
        if X.ndim == 1:
            X = np.expand_dims(X, axis=1)
            # Call the Imputer as normal, return result
        return super(Imputer1D, self).transform(X)

#%%
class MultipleToNewFeature(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """
    def __init__(self, selected_cols, new_col_name,func):
        self.selected_cols = selected_cols
        self.new_col_name = new_col_name
        self.func = func

    def fit(self, X, y=None):
        return self
    @timeit
    def transform(self, df, y=None):
        # print(df)
        df[self.new_col_name] = df.apply(self.func, axis=1)
        print(self.log, "{}({}) -> ['{}']".format(self.func.__name__,self.selected_cols,self.new_col_name))
        return df


# %%
class NumericalToCat(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """Convert numeric indexed column into dtype category with labels
    Convert a column which has a category, presented as an Integer
    Initialize with a dict of ALL mappings for this session, keyed by column name
    (This could be easily refactored to have only the required mapping)
    """

    def __init__(self, label_map_dict, allow_more_labels=False):
        self.label_map_dict = label_map_dict
        self.allow_more_labels = allow_more_labels

    def fit(self, X, y=None):
        return self

    def get_unique_values(self, this_series):
        return list(this_series.value_counts().index)

    def transform(self, this_series):
        if not self.allow_more_labels:
            if len(self.label_map_dict) > len(this_series.value_counts()):
                msg = "{} labels provided, but {} values in column!\nLabels:{}\nValues:{}".format(
                    len(self.label_map_dict), len(this_series.value_counts()), self.label_map_dict,
                    self.get_unique_values(this_series), )
                raise ValueError(msg)

        if len(self.label_map_dict) < len(this_series.value_counts()):
            raise ValueError

        assert type(this_series) == pd.Series
        # assert this_series.name in self.label_map_dict, "{} not in label map!".format(this_series.name)
        return_series = this_series.copy()
        # return_series = pd.Series(pd.Categorical.from_codes(this_series, self.label_map_dict))
        return_series = return_series.astype('category')
        return_series.cat.rename_categories(self.label_map_dict, inplace=True)
        # print(return_series.cat.categories)

        assert return_series.dtype == 'category'
        return return_series

#%%==============================================================================
# WordCounter
# ===============================================================================
class WordCounter(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """ Count the words in the input column
    """
    def __init__(self, col_name, new_col_name):
        self.col_name = col_name
        self.new_col_name = new_col_name

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        new_col = df[self.col_name].apply(lambda x: len(x.split(" ")))
        df[self.new_col_name] = new_col
        print(self.log, self.new_col_name)
        return df

#%% =============================================================================
# ConvertToDatetime
# ===============================================================================
class ConvertToDatetime(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """

    def __init__(self, time_col_name, unit='s'):
        self.time_col_name = time_col_name
        self.unit = unit

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        df[self.time_col_name] = pd.to_datetime(df[self.time_col_name], unit=self.unit)
        print("Transformer:", type(self).__name__, "converted", self.time_col_name, "to dt")
        return df

#%% =============================================================================
# TimeProperty
# ===============================================================================
class TimeProperty(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """
    def __init__(self, time_col_name, new_col_name, time_property):
        """

        :param time_col_name: Source column, MUST BE A datetime TYPE!
        :param new_col_name: New column name
        :param time_property: hour, month, dayofweek
        """
        self.time_col_name = time_col_name
        self.new_col_name = new_col_name
        self.time_property = time_property

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        original_shape = df.shape
        if self.time_property == 'hour':
            df[self.new_col_name] = df[self.time_col_name].dt.hour
        elif self.time_property == 'month':
            df[self.new_col_name] = df[self.time_col_name].dt.month
        elif self.time_property == 'dayofweek':
            df[self.new_col_name] = df[self.time_col_name].dayofweek
        else:
            raise
        print("Transformer:", type(self).__name__, original_shape, "->", df.shape, vars(self))
        return df
# Debug:
# df = X_train
# time_col_name = 'question_utc'
# new_col_name = 'question_hour'
# time_property = 'hour'
# time_col_name = 'question_utc'
# new_col_name = 'question_month'
# time_property = 'month'
# time_adder = TimeProperty(time_col_name,new_col_name,time_property)
# res=time_adder.transform(df)

#%% =============================================================================
# DEPRECIATED - AnswerDelay
# ===============================================================================
class AnswerDelay(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """ Used once, not general, gets time elapsed
    """

    def __init__(self, new_col_name, divisor=1):
        self.new_col_name = new_col_name
        self.divisor = divisor

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        df[self.new_col_name] = df['answer_utc'] - df['question_utc']
        df[self.new_col_name] = df[self.new_col_name].dt.seconds / self.divisor
        print(self.log)
        return df


# Debug:
# df = X_train
# new_col_name = 'answer_delay_seconds'
# answer_delay_adder = AnswerDelay(new_col_name)
# res=answer_delay_adder.transform(df)

#%% =============================================================================
# ValueCounter
# ===============================================================================
class ValueCounter(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """??
    """

    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        # Count the number of unique entries in a column
        # reset_index() is used to maintain the DataFrame for merging
        selected_df_col = df[self.col_name].value_counts().reset_index()
        # Create a new name for this column
        selected_df_col.columns = [self.col_name, self.col_name + '_counts']
        print(self.log)
        return pd.merge(selected_df_col, df, on=self.col_name)

# %%=============================================================================
# DEPRECIATED ConvertDoubleColToDatetime
# ===============================================================================
class ConvertDoubleColToDatetime(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """
    """

    # pd.options.mode.chained_assignment = None  # default='warn'
    def __init__(self, new_col_name, name_col1, name_col2, this_format):
        self.new_col_name = new_col_name
        self.name_col1 = name_col1
        self.name_col2 = name_col2
        self.this_format = this_format

    def fit(self, X, y=None):
        return self

    @timeit
    def transform(self, df, y=None):
        combined_date_string_series = df.loc[:, self.name_col1] + " " + df.loc[:, self.name_col2]
        with ChainedAssignment():
            df.loc[:, self.new_col_name] = pd.to_datetime(combined_date_string_series, format=self.this_format)
        #        pd.options.mode.chained_assignment = 'warn'  # default='warn'

        # print("Transformer:", type(self).__name__, "converted", self.new_col_name, "to dt")
        print(self.log)
        return df

# Debug:
# df = sfpd_head
# new_col_name = 'dt'
# time_adder = ConvertDoubleColToDatetime(new_col_name,name_col1="Date", name_col2="Time",this_format=r'%m/%d/%Y %H:%M')
# res=time_adder.transform(df)

#%% DEBUG TRF
#
# class TransformerLog():
#     """Add a .log attribute for logging
#     """
#     @property
#     def log(self):
#         return "Transformer: {}".format(type(self).__name__)
# class NumericalToCat(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
#     """Convert numeric indexed column into dtype category with labels
#     Convert a column which has a category, presented as an Integer
#     Initialize with a dict of ALL mappings for this session, keyed by column name
#     (This could be easily refactored to have only the required mapping)
#     """
#     def __init__(self,label_map):
#         self.label_map = label_map
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, this_series):
#         assert type(this_series) == pd.Series
#         mapped_labels = list(self.label_map.values())
#         # assert this_series.name in self.label_map_dict, "{} not in label map!".format(this_series.Name)
#         return_series = this_series.copy()
#         return_series = pd.Series(pd.Categorical.from_codes(this_series, mapped_labels))
#         # return_series = return_series.astype('category')
#         # return_series.cat.rename_categories(self.label_map_dict[return_series.name], inplace=True)
#         print(self.log, mapped_labels, return_series.cat.categories, )
#         assert return_series.dtype == 'category'
#         return return_series
#
# # this_series = df_all['Vaccinated'].copy()
# # this_series.value_counts()
# # label_map = label_maps['Vaccinated']
# # mapped_labels = list(label_map.values())
# # my_labels = pd.Index(mapped_labels)
# # pd.Series(pd.Categorical.from_codes(this_series, my_labels))
#
# for col_name in label_maps:
#     df_all[col_name].value_counts().index
#     print(col_name)
#     label_maps[col_name]
#     df_all.replace({col_name: label_maps[col_name]},inplace=True)
#
#
#
# df_all['Vaccinated'] = df_all['Vaccinated'] - 1
#
# pandas.CategoricalIndex.reorder_categories
#
# # To return the original integer mapping!
# ivd = {v: k for k, v in label_maps['State'].items()}
# df_all['State'].astype('object').replace(ivd)
#%% ===========================================================================
# Data source and paths
# =============================================================================
# path_data = Path(PATH_DATA_ROOT, r"").expanduser()
# assert path_data.exists(), "Data path does not exist: {}".format(path_data)
# logging.info("Data path {}".format(PATH_DATA_ROOT))

#%% ===========================================================================
# Load data
# =============================================================================
logging.info(f"Loading files into memory")
#product data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

logging.info("Loaded train {}".format(df_train.shape))
logging.info("Loaded test {}".format(df_test.shape))

# Add a column to label the source of the data
df_train['dataset_type'] = 'train'
df_test['dataset_type'] = 'test'

logging.info("Added dataset_type column for origin".format())

# Set this aside for debugging
#TODO: Remove later
original_y_train = df_train['target'].copy()

df_all = pd.concat([df_train, df_test], sort=False)
index_col = 'ID'
df_all.set_index(index_col, inplace=True)

logging.info("Concatenated dataset on {}, with origin column 'dataset_type', shape: {}".format(index_col, df_all.shape))
del df_train, df_test, index_col


#%% OTHER DATA

#sales, exchange rates, social network data
df_sales = pd.read_csv('../input/sales.csv')
logging.info("df_sales {}".format(df_sales.shape))
series_sales_null_count = df_sales.isnull().sum(axis=0)
# r = series_sales_null_count[0]
for row in [row for row in series_sales_null_count.iteritems() if row[1]]:
    print(row, row[1]/df_sales.shape[0])

# df_sales.describe()
# df_sales.info()
# sales_counts = df_sales.apply(pd.value_counts(dropna=False))

#website navigation data
df_navigation = pd.read_csv('../input/navigation.csv')
logging.info("df_navigation {}".format(df_navigation.shape))

#product images vectorized with ResNet50
df_vimages = pd.read_csv('../input/vimages.csv')
logging.info("df_vimages {}".format(df_vimages.shape))
logging.info("Loaded other data ".format())

#%% IMPUTATION
#
#%%
# Impute missing sales
sales_float_columns = df_sales.dtypes[df_sales.dtypes == 'float64'].index.tolist()
df_sales.loc[:, sales_float_columns] = SimpleFill(fill_method='random').fit_transform(df_sales.loc[:, sales_float_columns])
logging.info("Sales columns filled".format())
# %%

navigation_null_count = df_navigation.isnull().sum(axis=0)
# r = series_sales_null_count[0]
for row in [row for row in navigation_null_count.iteritems() if row[1]]:
    print(row, row[1]/df_sales.shape[0])

# Impute missing website_version_zone_number
df_navigation.loc[df_navigation.website_version_zone_number.isna(), 'website_version_zone_number'] = 'unknown'
# Impute missing website_version_country_number
df_navigation.loc[df_navigation.website_version_country_number.isna(), 'website_version_country_number'] = 'unknown'
# %%
color_null_count = df_all.color.isnull().sum(axis=0)
# Impute missing colors
df_all.loc[df_all.color.isna(), 'color'] = 'unknown'
logging.info("Filled {} colors".format(color_null_count))

import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def extract_number_from_sent_strg(x):
    number_find = re.compile(r'.*(\d).*')
    return int(number_find.match(str(x)).group(1))

def return_lr_coeff(y):

    # Reshape data
    X = np.arange(1,8).reshape(-1,1)
    y = y.values.reshape(-1,1)

    # Fit regression model
    l_model = LinearRegression(n_jobs=-1)
    l_model.fit(X,y)

    return l_model.coef_[0][0]


def extract_sentiment_lin_reg_coefficients(enriched_sales_df, sentiment):

    """

       Extract linear regression coefficients (slope and intercept)
       from the enriched_sales_df.
       Sentiment is a string that specifies which values to use. You should use
       one of the following:
           - BuzzPost
           - Buzz_
           - NetSent
           - Positive
           - Negative
           - Impressions

       Args:
           enriched_sales_df: pandas DataFrame
           sentiment: string

    """

    # Extract sentiment columns
    sentiment_compiler = re.compile(r'.*{}.*before'.format(sentiment))
    sentiment_columns = [c for c in enriched_sales_df if sentiment_compiler.match(c)]

    # Simple check
    if sentiment_columns == []:
        print("Wrong sentiment string")
        raise ValueError()

    # Group by product
    time_series_df = enriched_sales_df.groupby(['sku_hash'])[sentiment_columns].mean()

    # Melt
    time_series_df = time_series_df.reset_index().melt(id_vars=['sku_hash'], value_vars = sentiment_columns)

    # Extract number from sent string
    time_series_df.loc[:, 'days_before'] = (time_series_df.variable.transform(extract_number_from_sent_strg) * -1)+8

    # Drop variable column
    time_series_df.drop(columns=['variable'], inplace=True)

    # Pivot back
    time_series_df = time_series_df.pivot(index='sku_hash', columns='days_before', values='value')

    # Backward fill NA
    time_series_df.fillna(axis=1, method='backfill', inplace=True)

    # Apply and calculate regression coefficient
    reg_coeff_df = time_series_df.apply(return_lr_coeff, axis=1).reset_index()

    # Rescale beta
    scaler = StandardScaler()

    reg_coeff_df.iloc[:,1] = scaler.fit_transform(reg_coeff_df.iloc[:,1].values.reshape(-1,1)).reshape(-1)

    # Rename columns
    reg_coeff_df.columns = ['sku_hash', sentiment + '_beta']

    return reg_coeff_df



# %% [markdown]
# # Data preparation

# %% [markdown]
# # Currency and social columns
# %%
# Sales data breakdown:
#
# Currency:
# 5 Currency rates: USD, GBP, CNY, JPY, KRW x 8 days = 40
#
# Social:
# TotalBuzzPost
# TotalBuzz
# PositiveSentiment
# NegativeSentiment
# NetSentiment
# Impressions
# TOTAL = 6
# And for each of the 7 days = 42
# 40 + 48 = 88 columns
def log_df(df,name):
    logging.info("DataFrame {} {}".format(name,df.shape))
log_df(df_all,'df_all')

# %% INJECT NEW SALES TIME TREND FEATURE

trend_columns = ['BuzzPost', 'Buzz_', 'NetSent', 'Positive', 'Negative', 'Impressions']

for feature_col in trend_columns:
    extracted = extract_sentiment_lin_reg_coefficients(df_sales,feature_col)
    df_sales = df_sales.merge(extracted,on='sku_hash')
    logging.info("Appended {}".format(feature_col))
    log_df(df_sales, 'df_sales')



#%%
logging.info("SALES DATA".format())

cols_currency_and_social = df_sales.columns[9:].tolist()
print(cols_currency_and_social)
logging.info("{} currency_and_social_columns columns".format(len(cols_currency_and_social)))

# %% [markdown]
# This is a dataframe of ONLY the first day of activity
# %%
df_first_day = df_sales.loc[df_sales.Date == 'Day_1', :]
log_df(df_first_day,'df_first_day')
df_second_day = df_sales.loc[df_sales.Date == 'Day_2', :]
log_df(df_first_day,'df_second_day')
# print(first_day)

# %%
# Get the mean of ALL columns for ALL sku
df_all_currency_and_social = df_sales.groupby('sku_hash').mean()[cols_currency_and_social]
#
df_first_day_currency_and_social = df_first_day.groupby('sku_hash').mean()[cols_currency_and_social]
df_first_day_currency_and_social.columns = ['first_day_' + col for col in df_first_day_currency_and_social.columns]

# %%
df_total_sales = df_sales.groupby('sku_hash').sum()['sales_quantity']
df_total_sales = pd.DataFrame(df_total_sales)
df_first_day_sales = df_first_day.groupby(['sku_hash', 'day_transaction_date', 'Month_transaction']).sum()['sales_quantity']
df_first_day_sales = pd.DataFrame(df_first_day_sales)
df_first_day_sales.columns = ['first_day_sales']
df_first_day_sales.reset_index(inplace=True)
df_first_day_sales.set_index('sku_hash', inplace=True)

# %%
# Merge total with first day
df_sales_data = pd.merge(df_total_sales, df_first_day_sales, left_index=True, right_index=True)
logging.info("Merged df_total_sales {} df_first_day_sales {} = {}".format(df_total_sales.shape, df_first_day_sales.shape, df_sales_data.shape))

# Merge all currency/social
df_sales_data = pd.merge(df_sales_data, df_all_currency_and_social, left_index=True, right_index=True)
logging.info("Merged df_all_currency_and_social {} = {}".format(df_sales_data.shape, df_sales_data.shape))

# Merge first day curr/soc
df_sales_data = pd.merge(df_sales_data, df_first_day_currency_and_social, left_index=True, right_index=True)
logging.info("Merged df_first_day_currency_and_social {} {} = {}".format(df_first_day_currency_and_social.shape, df_first_day_sales.shape, df_sales_data.shape))
log_df(df_sales_data,'df_sales_data')




# %%
monthDict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
df_sales_data.Month_transaction = df_sales_data.Month_transaction.astype('object').map(monthDict)
logging.info("Mapped months in df_sales_data".format())

# %%
logging.info("NAVIGATION DATA".format())
df_first_day_navigation = df_navigation.loc[df_navigation.Date == 'Day 1', :]
df_first_day_views = df_first_day_navigation.groupby('sku_hash').sum()[['page_views', 'addtocart']]
df_first_day_views.columns = ['first_day_page_views', 'first_day_addtocart']
df_views = df_navigation.groupby('sku_hash').sum()[['page_views', 'addtocart']]
df_navigation_data = pd.merge(df_views, df_first_day_views, left_index=True, right_index=True)


# %%
# Convert to float
df_sales_data.sales_quantity = df_sales_data.sales_quantity.astype('float64')
df_sales_data.first_day_sales = df_sales_data.first_day_sales.astype('float64')

# %%
# Log transform the sales
df_sales_data['sales_quantity_log'] = (df_sales_data.sales_quantity + 1).apply(np.log)
df_sales_data['first_day_sales_log'] = (df_sales_data.first_day_sales + 1).apply(np.log)


#%% [markdown]
# # Merge to main table!
# %%

df_all_merged = pd.merge(df_all, df_sales_data, left_on='sku_hash', right_index=True)
df_all_merged = pd.merge(df_all_merged, df_navigation_data, how='left', left_on='sku_hash', right_index=True)
df_all_merged = pd.merge(df_all_merged, df_vimages, left_on='sku_hash', right_on='sku_hash')

# %%
df_all_merged[df_navigation_data.columns] = df_all_merged[df_navigation_data.columns].fillna(0)
log_df(df_all_merged,'df_all_merged')

# %% [markdown] {"_uuid": "107def94e3025396616fa08411e8260d6e7b4fb6"}
# ## separate models for each prediction month
df_all_merged = df_all_merged
# %%
# Seperate

dfs_monthly_list = dict()
for i in range(1, 4):
    dfs_monthly_list[i] = dict()
    dfs_monthly_list[i]['df'] = df_all_merged.loc[df_all_merged.month == i, :].copy()
    log_df(dfs_monthly_list[i]['df'], 'Month 1 DF, indexed by ID')
    # dfs_monthly_list[i].set_index('ID', inplace=True)
    # logging.info("Set index ID".format())
    dfs_monthly_list[i]['df'].drop(['month', 'sku_hash'], axis=1, inplace=True)
    logging.info("Dropped month, sku_hash".format())
    assert 'month' not in dfs_monthly_list[i]['df'].columns
    assert 'sku_hash' not in dfs_monthly_list[i]['df'].columns
    assert 'ID' not in dfs_monthly_list[i]['df'].columns

# %%
# SPLIT
for i in range(1, 4):
    df = dfs_monthly_list[i]['df']

    # Training df
    dfs_monthly_list[i]['df_tr'] = df[df['dataset_type'] == 'train'].copy()
    dfs_monthly_list[i]['df_tr'].drop('dataset_type', axis=1, inplace=True)
    log_df(dfs_monthly_list[i]['df_tr'], 'df_tr ' + str(i))

    # X_tr
    dfs_monthly_list[i]['X_tr'] = dfs_monthly_list[i]['df_tr'].drop('target', axis=1)
    log_df(dfs_monthly_list[i]['X_tr'], 'X_tr ' + str(i))

    # y_tr
    dfs_monthly_list[i]['y_tr'] = pd.DataFrame(dfs_monthly_list[i]['df_tr']['target'])
    log_df(dfs_monthly_list[i]['y_tr'] , 'y_tr ' + str(i))

    logging.info("Applying log transform".format())
    dfs_monthly_list[i]['y_tr']['target'] = (dfs_monthly_list[i]['y_tr']['target'] + 1).apply(np.log)

    # Test df
    dfs_monthly_list[i]['df_te'] = df[df['dataset_type'] == 'test'].copy()
    dfs_monthly_list[i]['df_te'].drop('dataset_type', axis=1, inplace=True)
    log_df(dfs_monthly_list[i]['df_te'], 'df_te ' + str(i))

    # X_te
    dfs_monthly_list[i]['X_te'] = dfs_monthly_list[i]['df_te'].drop('target', axis=1)
    log_df(dfs_monthly_list[i]['X_te'], 'X_te ' + str(i))


a = dfs_monthly_list[i]['y_tr']

# %%
images_cols = list()
images_cols = df_vimages.columns[1:].tolist()
float_cols = dfs_monthly_list[1]['df'].dtypes[dfs_monthly_list[1]['df'].dtypes == 'float64'].index.tolist()
logging.info("{} float cols".format(len(float_cols)))
float_cols = list(set(float_cols) - set(images_cols))
logging.info("{} float cols".format(len(float_cols)))
float_cols.remove('sales_quantity_log')
float_cols.remove('first_day_sales_log')
float_cols.remove('sales_quantity')
float_cols.remove('first_day_sales')
logging.info("{} float cols".format(len(float_cols)))

# %%
categorical_cols = list()
logging.info("{} categorical cols".format(len(float_cols)))
categorical_cols = dfs_monthly_list[1]['df'].dtypes[dfs_monthly_list[1]['df'].dtypes == 'object'].index.tolist()
categorical_cols.remove('en_US_description')
categorical_cols.remove('color')
logging.info("{} categorical cols".format(len(float_cols)))
dfs_monthly_list = dfs_monthly_list
#%%
if 'target' in float_cols: float_cols.remove('target')
if 'dataset_type' in float_cols: float_cols.remove('dataset_type')
assert 'target' not in float_cols
assert 'dataset_type' not in float_cols


if 'target' in categorical_cols: categorical_cols.remove('target')
if 'dataset_type' in categorical_cols: categorical_cols.remove('dataset_type')
assert 'target' not in categorical_cols
assert 'dataset_type' not in categorical_cols
# %% {"_uuid": "66bd00938a631cb806de3f1ade45ddd25a0119ec"}
my_sales_cols =['sales_quantity_log', 'first_day_sales_log', 'sales_quantity', 'first_day_sales']
pipeline = make_pipeline(
    make_union(
        make_pipeline(
            PandasSelector(columns='en_US_description',name='Description'),
            CountVectorizer(stop_words='english'),
            LatentDirichletAllocation(n_components=10)),
        make_pipeline(
            PandasSelector(columns='color',name='color'),
            CountVectorizer() ),
        make_pipeline(
            PandasSelector(columns=images_cols,name='Images_cols'),
            PCA(10)),
        make_pipeline(
            PandasSelector(columns=float_cols, name='Floats'),
            PCA(10)),
        make_pipeline(
            PandasSelector(columns=my_sales_cols,name='Sales stuff')),
        make_pipeline(
            PandasSelector(columns=categorical_cols, name ='Categoricals'),
            OneHotEncoder(handle_unknown='ignore'),
            LatentDirichletAllocation(n_components=10))
    ),
    SelectFromModel(RandomForestRegressor(n_estimators=100)),
    DecisionTreeRegressor(),
)


#%%

params = {'decisiontreeregressor__min_samples_split': [40, 60, 80],
          'decisiontreeregressor__max_depth': [4, 6, 8]}


grid_search_list = dict()

for i in range(1,4):
    logging.info('Month {}'.format(i))
    grid_search_list[i] = None

    grid_search_list[i] = GridSearchCV(pipeline, param_grid=params, cv=4, verbose=3, n_jobs=-1)

    grid_search_list[i].fit(dfs_monthly_list[i]['X_tr'], dfs_monthly_list[i]['y_tr'])

    dfs_monthly_list[i]['y_te'] = grid_search_list[i].predict(dfs_monthly_list[i]['X_te'])

    dfs_monthly_list[i]['y_tr_pred'] = grid_search_list[i].predict(dfs_monthly_list[i]['X_tr'])

    this_y_tr_pred = pd.Series(dfs_monthly_list[i]['y_tr_pred'])
    this_y_tr = dfs_monthly_list[i]['y_tr']
    this_y_tr = this_y_tr.iloc[:,0]
    # compare = pd.DataFrame([dfs_monthly_list[i]['y_tr'],dfs_monthly_list[i]['y_tr']])
    dfs_monthly_list[i]['compare'] = pd.DataFrame.from_records(
        {'y_tr': this_y_tr,
         'y_tr_pred': this_y_tr_pred}
    ).reset_index()
    logging.info('metric cv: {}'.format( np.round(np.sqrt(grid_search_list[i].best_score_), 4)))
    logging.info('metric train: {}'.format(np.round(np.sqrt(mean_squared_error(this_y_tr, this_y_tr_pred)), 4)))
    logging.info('params: {}'.format(grid_search_list[i].best_params_))

#%% SUBMISSION

# %% {"_uuid": "4d9d4dc8361c4dc285d2283bd58cd7465a0b0e61"}
for i in range(1,4):
    logging.info('Month {}'.format(i))
    grid_search_list[i]
    dfs_monthly_list[i]['y_submit'] = (pd.Series(dfs_monthly_list[i]['y_te'])).apply(np.exp)  - 1
    a = dfs_monthly_list[i]['df']

    dfs_monthly_list[i]['y_submit'].index = dfs_monthly_list[i]['X_te'].index

submission = pd.DataFrame(pd.concat([dfs_monthly_list[1]['y_submit'],
                                     dfs_monthly_list[2]['y_submit'],
                                     dfs_monthly_list[3]['y_submit']]))

submission.index = df_all[df_all['dataset_type'] == 'test'].copy().index
submission.columns = ['target']


# %% {"_uuid": "e886a8013dcef7729789dffdbe5bb32270934c5b"}
submission.describe()
submission.to_csv('submission.csv', index_label='ID')


