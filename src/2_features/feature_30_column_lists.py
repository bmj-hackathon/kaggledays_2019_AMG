

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
