dfs_monthly_list = dfs_monthly_list
#%%
if 'target' in float_cols: float_cols.remove('target')
if 'dataset_type' in float_cols: float_cols.remove('dataset_type')
assert 'target' not in float_cols
assert 'dataset_type' not in float_cols


if 'target' in categorical_cols: categorical_cols.remove('target')
if 'dataset_type' in categorical_cols: categorical_cols.remove('dataset_type')
assert 'target' not in categorical_cols
assert 'dataset_type' not in categorical_cols

# %% {"_uuid": "66bd00938a631cb806de3f1ade45ddd25a0119ec"}
my_sales_cols =['sales_quantity_log', 'first_day_sales_log', 'sales_quantity', 'first_day_sales']
pipeline = make_pipeline(
    make_union(
        make_pipeline(
            PandasSelector(columns='en_US_description',name='Description'),
            CountVectorizer(stop_words='english'),
            LatentDirichletAllocation(n_components=10)),
        make_pipeline(
            PandasSelector(columns='color',name='color'),
            CountVectorizer() ),
        make_pipeline(
            PandasSelector(columns=images_cols,name='Images_cols'),
            PCA(5)),
        make_pipeline(
            PandasSelector(columns=float_cols, name='Floats'),
            PCA(5)),
        make_pipeline(
            PandasSelector(columns=my_sales_cols,name='Sales stuff')),
        make_pipeline(
            PandasSelector(columns=categorical_cols, name ='Categoricals'),
            OneHotEncoder(handle_unknown='ignore'),
        )
            # LatentDirichletAllocation(n_components=10))
    ),
    # sk.feature_selection.SelectFromModel(RandomForestRegressor(n_estimators=100)),
    # xgb.XGBRegressor(),
)



#%%
# Transform DATA
#%%
pipeline
for step in pipeline.steps:
    print(step)

for i in range(1,4):
    logging.info("Transforming month {}".format(i))
    dfs_monthly_list[i]['arr_transformed'] = pipeline.fit_transform(dfs_monthly_list[i]['X_tr'])
    logging.info("Transformed to array {}".format(dfs_monthly_list[i]['arr_transformed'].shape))

    dfs_monthly_list[i]['arr_transformed']

#%%
months = range(1,2)
for i in months:
    selector = sk.feature_selection.SelectFromModel(RandomForestRegressor(n_estimators=100))
    selector.fit(dfs_monthly_list[i]['arr_transformed'], dfs_monthly_list[i]['y_tr'].values.ravel())

    dfs_monthly_list[i]['arr_selected'] = selector.transform(dfs_monthly_list[i]['arr_transformed'])
    n_features = dfs_monthly_list[i]['arr_selected']
    logging.info("n_features {}".format(n_features.shape[1]))

    # n_features = sfm.transform(X).shape[1]
    # selector.fit_transform()

for i in months:
    pass





#%%

#%%
# params = {'decisiontreeregressor__min_samples_split': [40, 60, 80],
#           'decisiontreeregressor__max_depth': [4, 6, 8]}

params = {
       'xgbregressor__min_child_weight': [5, 15,30],
       'xgbregressor__gamma': [0.1,0.5, 1],
       'xgbregressor__subsample': [0.8, 1.0],
       'xgbregressor__colsample_bytree': [0.5,0.8, 1.0],
       'xgbregressor__max_depth': [5,10, 30]
       }

grid_search_list = dict()

for i in range(1,4):
    logging.info('Month {}'.format(i))
    grid_search_list[i] = None

    grid_search_list[i] = GridSearchCV(pipeline, param_grid=params, cv=4, verbose=3, n_jobs=-1)

    grid_search_list[i].fit(dfs_monthly_list[i]['X_tr'], dfs_monthly_list[i]['y_tr'])

    dfs_monthly_list[i]['y_te'] = grid_search_list[i].predict(dfs_monthly_list[i]['X_te'])

    dfs_monthly_list[i]['y_tr_pred'] = grid_search_list[i].predict(dfs_monthly_list[i]['X_tr'])

    this_y_tr_pred = pd.Series(dfs_monthly_list[i]['y_tr_pred'])
    this_y_tr = dfs_monthly_list[i]['y_tr']
    this_y_tr = this_y_tr.iloc[:,0]
    # compare = pd.DataFrame([dfs_monthly_list[i]['y_tr'],dfs_monthly_list[i]['y_tr']])
    dfs_monthly_list[i]['compare'] = pd.DataFrame.from_records(
        {'y_tr': this_y_tr,
         'y_tr_pred': this_y_tr_pred}
    ).reset_index()
    logging.info('metric cv: {}'.format( np.round(np.sqrt(grid_search_list[i].best_score_), 4)))
    logging.info('metric train: {}'.format(np.round(np.sqrt(mean_squared_error(this_y_tr, this_y_tr_pred)), 4)))
    logging.info('params: {}'.format(grid_search_list[i].best_params_))

#%% SUBMISSION

# %% {"_uuid": "4d9d4dc8361c4dc285d2283bd58cd7465a0b0e61"}
for i in range(1,4):
    logging.info('Month {}'.format(i))
    grid_search_list[i]
    dfs_monthly_list[i]['y_submit'] = (pd.Series(dfs_monthly_list[i]['y_te'])).apply(np.exp)  - 1
    a = dfs_monthly_list[i]['df']

    dfs_monthly_list[i]['y_submit'].index = dfs_monthly_list[i]['X_te'].index

submission = pd.DataFrame(pd.concat([dfs_monthly_list[1]['y_submit'],
                                     dfs_monthly_list[2]['y_submit'],
                                     dfs_monthly_list[3]['y_submit']]))

submission.index = df_all[df_all['dataset_type'] == 'test'].copy().index
submission.columns = ['target']


# %% {"_uuid": "e886a8013dcef7729789dffdbe5bb32270934c5b"}
submission.describe()
submission.to_csv('submission.csv', index_label='ID')


