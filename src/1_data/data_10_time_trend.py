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


