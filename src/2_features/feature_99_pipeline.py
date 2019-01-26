dfs_monthly_list = dfs_monthly_list
#%%
float_cols.remove('target')
float_cols.remove('dataset_type')
assert 'target' not in float_cols
assert 'dataset_type' not in float_cols

categorical_cols.remove('target')
categorical_cols.remove('dataset_type')
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
            PCA(10)),
        make_pipeline(
            PandasSelector(columns=float_cols, name='Floats'),
            PCA(10)),
        make_pipeline(
            PandasSelector(columns=my_sales_cols,name='Sales stuff')),
        make_pipeline(
            PandasSelector(columns=categorical_cols, name ='Categoricals'),
            OneHotEncoder(handle_unknown='ignore'),
            LatentDirichletAllocation(n_components=10))
    ),
    SelectFromModel(RandomForestRegressor(n_estimators=100)),
    DecisionTreeRegressor(),
)


#%%

params = {'decisiontreeregressor__min_samples_split': [40, 60, 80],
          'decisiontreeregressor__max_depth': [4, 6, 8]}


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



