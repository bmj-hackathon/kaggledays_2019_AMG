
# %% [markdown] {"_uuid": "107def94e3025396616fa08411e8260d6e7b4fb6"}
# ## separate models for each prediction month
df_all_merged = df_all_merged
# %%

dfs_monthly_list = dict()
for i in range(1,4):
    dfs_monthly_list[i] = df_all_merged.loc[df_all_merged.month == i, :].copy()
    log_df(dfs_monthly_list[i], 'Month 1 DF, indexed by ID')
    # dfs_monthly_list[i].set_index('ID', inplace=True)
    # logging.info("Set index ID".format())
    dfs_monthly_list[i].drop(['month', 'sku_hash'], axis=1, inplace=True)
    logging.info("Dropped month, sku_hash".format())

# a = dfs_monthly_list[i]
# b = df_all_merged.columns

#%%


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