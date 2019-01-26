# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

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

# %% {"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"}
#product data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#sales, exchange rates, social network data
sales = pd.read_csv('../input/sales.csv')

#website navigation data
navigation = pd.read_csv('../input/navigation.csv')

#product images vectorized with ResNet50
vimages = pd.read_csv('../input/vimages.csv')

# %% {"_uuid": "3c2242cb5ca18f9be44f8a5b367ffd8b65bef2cd"}
sales_float_columns = sales.dtypes[sales.dtypes == 'float64'].index.tolist()
sales.loc[:,sales_float_columns] = SimpleFill(fill_method='random').fit_transform(sales.loc[:,sales_float_columns])

# %% {"_uuid": "9dd6a7ee61107ec4d1ea42f20353b968684d9525"}
navigation.loc[navigation.website_version_zone_number.isna(), 'website_version_zone_number'] = 'unknown'
navigation.loc[navigation.website_version_country_number.isna(), 'website_version_country_number'] = 'unknown'

# %% {"_uuid": "f6691458a8e7a2b3aa9e4afe30130790850900e7"}
train.loc[train.color.isna(), 'color'] = 'unknown'
test.loc[test.color.isna(), 'color'] = 'unknown'

# %% [markdown] {"_uuid": "bda1637e7052353ab9acad765f0e870f8078b066"}
# # Data preparation

# %% {"_uuid": "66ca5a8fc9b5449d4417a85d5b13e71aaf85b924"}
currency_and_social_columns = sales.columns[9:].tolist()

# %% {"_uuid": "3037dd2f0cf7ecd05b0b0e60648464573e18c277"}
first_day = sales.loc[sales.Date == 'Day_1',:]

# %% {"_uuid": "8f75e5524d09fe87b0a5c4db1576d75e7b26e4bf"}
all_currency_and_social = sales.groupby('sku_hash').mean()[currency_and_social_columns]
first_day_currency_and_social = first_day.groupby('sku_hash').mean()[currency_and_social_columns]
first_day_currency_and_social.columns = ['first_day_' + col for col in first_day_currency_and_social.columns]

# %% {"_uuid": "1b4a7b1f23753fef7f6f08f225406933b8dd0cc8"}
all_sales = sales.groupby('sku_hash').sum()['sales_quantity']
all_sales = pd.DataFrame(all_sales)
first_day_sales = first_day.groupby(['sku_hash', 'day_transaction_date', 'Month_transaction']).sum()['sales_quantity']
first_day_sales = pd.DataFrame(first_day_sales)
first_day_sales.columns = ['first_day_sales']
first_day_sales.reset_index(inplace=True)
first_day_sales.set_index('sku_hash', inplace=True)

# %% {"_uuid": "d8f2ab70598dbb879912b15f29b5ac259d377f29"}
sales_data = pd.merge(all_sales, first_day_sales, left_index=True, right_index=True)
sales_data = pd.merge(sales_data, all_currency_and_social, left_index=True, right_index=True)
sales_data = pd.merge(sales_data, first_day_currency_and_social, left_index=True, right_index=True)

# %% {"_uuid": "6ffaf9b7a26c4b59cce5cf0b2558c1af1dc8b40e"}
monthDict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
sales_data.Month_transaction = sales_data.Month_transaction.astype('object').map(monthDict)

# %% {"_uuid": "47555ae4986bb0076458b16ba79c128e185aaeb1"}
sales_data.head()

# %% {"_uuid": "c079fa8c7e5fbf025992978fda132052d7b2c86b"}
first_day_navigation = navigation.loc[navigation.Date == 'Day 1',:]
first_day_views = first_day_navigation.groupby('sku_hash').sum()[['page_views', 'addtocart']]
first_day_views.columns = ['first_day_page_views', 'first_day_addtocart']
views = navigation.groupby('sku_hash').sum()[['page_views', 'addtocart']]
navigation_data = pd.merge(views, first_day_views, left_index=True, right_index=True)

# %% {"_uuid": "e1e9390dd178b3027e67c6d014f06b4384ca7c4a"}
sales_data.sales_quantity = sales_data.sales_quantity.astype('float64')
sales_data.first_day_sales = sales_data.first_day_sales.astype('float64')

# %% {"_uuid": "8ac1d88343fc901fab020553bf35089283c86500"}
sales_data['sales_quantity_log'] = (sales_data.sales_quantity + 1).apply(np.log)
sales_data['first_day_sales_log'] = (sales_data.first_day_sales + 1).apply(np.log)

# %% {"_uuid": "c3a8340e1694ce077c8e9d28dd46ca451f0e0ea9"}
train_data = pd.merge(train, sales_data, left_on='sku_hash', right_index=True)
train_data = pd.merge(train_data, navigation_data, how='left', left_on='sku_hash', right_index=True)
train_data = pd.merge(train_data, vimages, left_on='sku_hash', right_on='sku_hash')

# %% {"_uuid": "4a90d1563fb6bd221613c23ea1c499177f7fe8d1"}
test_data = pd.merge(test, sales_data, left_on='sku_hash', right_index=True)
test_data = pd.merge(test_data, navigation_data, how='left', left_on='sku_hash', right_index=True)
test_data = pd.merge(test_data, vimages, left_on='sku_hash', right_on='sku_hash')

# %% {"_uuid": "f4f73442ff66d44fd5a27b1e9c85b97933f96465"}
train_data[navigation_data.columns] = train_data[navigation_data.columns].fillna(0)
test_data[navigation_data.columns] = test_data[navigation_data.columns].fillna(0)

# %% [markdown] {"_uuid": "e92ecde8be29c103afbef5861a6a5b55751697ec"}
# # Modeling
# ## utils
# from https://github.com/pjankiewicz/PandasSelector

# %% {"_uuid": "5aa824605618927c2bff72a54db96f7a66c572d7"}


class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, dtype=None, inverse=False,
                 return_vector=True):
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        self.return_vector = return_vector

        if isinstance(self.columns, str):
            self.columns = [self.columns]

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
                raise KeyError('Keys are missing in the record: %s' %
                               missing_columns_)

    def transform(self, x):
        # check if x is a pandas DataFrame
        if not isinstance(x, pd.DataFrame):
            raise KeyError('Input is not a pandas DataFrame')

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

# %% [markdown] {"_uuid": "107def94e3025396616fa08411e8260d6e7b4fb6"}
# ## separate models for each prediction month

# %% {"_uuid": "816b8f0cb0fbadf9328c28df8795e3ac1d69b582"}
train_data1 = train_data.loc[train_data.month == 1, :].copy()
train_data1.drop(['month', 'sku_hash', 'ID'], axis=1, inplace=True)

X_test1 = test_data.loc[test_data.month == 1, :].copy()
X_test1.drop(['month', 'sku_hash'], axis=1, inplace=True)
X_test1.set_index('ID', inplace=True)

y_train1 = (train_data1.target + 1).apply(np.log)
X_train1 = train_data1.drop('target', axis=1)

train_data2 = train_data.loc[train_data.month == 2, :].copy()
train_data2.drop(['month', 'sku_hash', 'ID'], axis=1, inplace=True)

X_test2 = test_data.loc[test_data.month == 2, :].copy()
X_test2.drop(['month', 'sku_hash'], axis=1, inplace=True)
X_test2.set_index('ID', inplace=True)

y_train2 = (train_data2.target + 1).apply(np.log)
X_train2 = train_data2.drop('target', axis=1)


train_data3 = train_data.loc[train_data.month == 3, :].copy()
train_data3.drop(['month', 'sku_hash', 'ID'], axis=1, inplace=True)

X_test3 = test_data.loc[test_data.month == 3, :].copy()
X_test3.drop(['month', 'sku_hash'], axis=1, inplace=True)
X_test3.set_index('ID', inplace=True)

y_train3 = (train_data3.target + 1).apply(np.log)
X_train3 = train_data3.drop('target', axis=1)

# %% {"_uuid": "614637c1e10a48604fedb614c472815b08954642"}
images_cols = vimages.columns[1:].tolist()
float_cols = X_train1.dtypes[X_train1.dtypes == 'float64'].index.tolist()
float_cols = list(set(float_cols) - set(images_cols))
float_cols.remove('sales_quantity_log')
float_cols.remove('first_day_sales_log')
float_cols.remove('sales_quantity')
float_cols.remove('first_day_sales')

# %% {"_uuid": "6f9e5a39469aec53f970042015788144ab4432d8"}
categorical_cols = X_train1.dtypes[X_train1.dtypes == 'object'].index.tolist()
categorical_cols.remove('en_US_description')
categorical_cols.remove('color')

# %% {"_uuid": "66bd00938a631cb806de3f1ade45ddd25a0119ec"}
model = make_pipeline(
    make_union(
        make_pipeline(PandasSelector(columns='en_US_description'), 
                      CountVectorizer(stop_words='english'),
                      LatentDirichletAllocation(n_components=10)),
        make_pipeline(PandasSelector(columns='color'), 
                      CountVectorizer()
                     ),
        make_pipeline(PandasSelector(columns=images_cols), 
                      PCA(10)),
        make_pipeline(PandasSelector(columns=float_cols),
                      PCA(10)),
        make_pipeline(PandasSelector(columns=['sales_quantity_log', 
                                              'first_day_sales_log', 
                                              'sales_quantity', 
                                              'first_day_sales'])),
        make_pipeline(PandasSelector(columns=categorical_cols), 
                      OneHotEncoder(handle_unknown='ignore'),
                      LatentDirichletAllocation(n_components=10))
    ),
    SelectFromModel(RandomForestRegressor(n_estimators=100)),
    DecisionTreeRegressor()
)

# %% {"_uuid": "51417d3cce26142e3d21df7d35d75d1130a6acf1"}
params = {'decisiontreeregressor__min_samples_split': [40, 60, 80],
          'decisiontreeregressor__max_depth': [4, 6, 8]}

# %% {"_uuid": "3a967109b4bbd8c56f29df0a2bc3904f48157412"}
gs = GridSearchCV(model, param_grid=params, cv=4, verbose=3, n_jobs=-1)

gs.fit(X_train1, y_train1)
y_test1 = gs.predict(X_test1)

print('metric cv: ', np.round(np.sqrt(gs.best_score_),4))
print('metric train: ', np.round(np.sqrt(mean_squared_error(y_train1, gs.predict(X_train1))),4))
print('params: ', gs.best_params_)

# %% {"_uuid": "c6acc1649b67dc5bd2b81731f2e15ed835574e00"}
gs.fit(X_train2, y_train2)
y_test2 = gs.predict(X_test2)

print('metric cv: ', np.round(np.sqrt(gs.best_score_),4))
print('metric train: ', np.round(np.sqrt(mean_squared_error(y_train2, gs.predict(X_train2))),4))
print('params: ', gs.best_params_)

# %% {"_uuid": "1741c7d10249ee022db5dd345b65f9ac3c06ab62"}
gs.fit(X_train3, y_train3)
y_test3 = gs.predict(X_test3)

print('metric cv: ', np.round(np.sqrt(gs.best_score_),4))
print('metric train: ', np.round(np.sqrt(mean_squared_error(y_train3, gs.predict(X_train3))),4))
print('params: ', gs.best_params_)

# %% {"_uuid": "4d9d4dc8361c4dc285d2283bd58cd7465a0b0e61"}
y_test1 = pd.Series(y_test1)
y_test1 = (y_test1).apply(np.exp)  - 1
y_test1.index = X_test1.index

y_test2 = pd.Series(y_test2)
y_test2 = (y_test2).apply(np.exp)  - 1
y_test2.index = X_test2.index

y_test3 = pd.Series(y_test3)
y_test3 = (y_test3).apply(np.exp) - 1
y_test3.index = X_test3.index

submission = pd.DataFrame(pd.concat([y_test1, y_test2, y_test3]))
submission.columns = ['target']

# %% {"_uuid": "e886a8013dcef7729789dffdbe5bb32270934c5b"}
submission.to_csv('submission.csv')

# %% {"_uuid": "64c46d264f4c14828e399da1e5c01ead65ee1ec6"}

