
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

