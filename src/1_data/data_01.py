
# %% [markdown]
# # Data preparation

# %% [markdown]
# # Currency and social columns
# %%
currency_and_social_columns = df_sales.columns[9:].tolist()
print(currency_and_social_columns)
logging.info("{} currency_and_social_columns columns".format(len( currency_and_social_columns )))

# %%
first_day = df_sales.loc[df_sales.Date == 'Day_1', :]
# print(first_day)

# %%

#
all_currency_and_social = df_sales.groupby('sku_hash').mean()[currency_and_social_columns]
first_day_currency_and_social = first_day.groupby('sku_hash').mean()[currency_and_social_columns]
first_day_currency_and_social.columns = ['first_day_' + col for col in first_day_currency_and_social.columns]

# %%
all_sales = df_sales.groupby('sku_hash').sum()['sales_quantity']
all_sales = pd.DataFrame(all_sales)
first_day_sales = first_day.groupby(['sku_hash', 'day_transaction_date', 'Month_transaction']).sum()['sales_quantity']
first_day_sales = pd.DataFrame(first_day_sales)
first_day_sales.columns = ['first_day_sales']
first_day_sales.reset_index(inplace=True)
first_day_sales.set_index('sku_hash', inplace=True)

# %%
sales_data = pd.merge(all_sales, first_day_sales, left_index=True, right_index=True)
sales_data = pd.merge(sales_data, all_currency_and_social, left_index=True, right_index=True)
sales_data = pd.merge(sales_data, first_day_currency_and_social, left_index=True, right_index=True)

# %%
monthDict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
sales_data.Month_transaction = sales_data.Month_transaction.astype('object').map(monthDict)

# %%
sales_data.head()

# %%
first_day_navigation = navigation.loc[navigation.Date == 'Day 1',:]
first_day_views = first_day_navigation.groupby('sku_hash').sum()[['page_views', 'addtocart']]
first_day_views.columns = ['first_day_page_views', 'first_day_addtocart']
views = navigation.groupby('sku_hash').sum()[['page_views', 'addtocart']]
navigation_data = pd.merge(views, first_day_views, left_index=True, right_index=True)

# %%
sales_data.sales_quantity = sales_data.sales_quantity.astype('float64')
sales_data.first_day_sales = sales_data.first_day_sales.astype('float64')

# %%
sales_data['sales_quantity_log'] = (sales_data.sales_quantity + 1).apply(np.log)
sales_data['first_day_sales_log'] = (sales_data.first_day_sales + 1).apply(np.log)

# %%
train_data = pd.merge(train, sales_data, left_on='sku_hash', right_index=True)
train_data = pd.merge(train_data, navigation_data, how='left', left_on='sku_hash', right_index=True)
train_data = pd.merge(train_data, vimages, left_on='sku_hash', right_on='sku_hash')

# %%
test_data = pd.merge(test, sales_data, left_on='sku_hash', right_index=True)
test_data = pd.merge(test_data, navigation_data, how='left', left_on='sku_hash', right_index=True)
test_data = pd.merge(test_data, vimages, left_on='sku_hash', right_on='sku_hash')

# %%
train_data[navigation_data.columns] = train_data[navigation_data.columns].fillna(0)
test_data[navigation_data.columns] = test_data[navigation_data.columns].fillna(0)

