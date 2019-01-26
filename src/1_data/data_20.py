
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

