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

# %%
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