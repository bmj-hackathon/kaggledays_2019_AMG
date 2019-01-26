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

    # Test df
    dfs_monthly_list[i]['df_te'] = df[df['dataset_type'] == 'test'].copy()
    dfs_monthly_list[i]['df_te'].drop('dataset_type', axis=1, inplace=True)
    log_df(dfs_monthly_list[i]['df_te'], 'df_te ' + str(i))

    # X_te
    dfs_monthly_list[i]['X_te'] = dfs_monthly_list[i]['df_te'].drop('target', axis=1)
    log_df(dfs_monthly_list[i]['X_te'], 'X_te ' + str(i))


