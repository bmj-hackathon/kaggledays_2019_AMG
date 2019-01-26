
# %% {"_uuid": "66bd00938a631cb806de3f1ade45ddd25a0119ec"}
pipeline = make_pipeline(
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

)

#%%
res = pipeline.fit_transform(X_train1, y_train1)
#%%
selector = sklearn.feature_selection.SelectFromModel(RandomForestRegressor(n_estimators=100))
res2 = selector.fit_transform(res)


# gs.fit(
# y_test1 = gs.predict(X_test1)
