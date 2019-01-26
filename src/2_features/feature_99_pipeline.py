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

for i in range(1,2):
    logging.info('Month {}'.format(i))
    grid_search_list[i] = None

    grid_search_list[i] = GridSearchCV(pipeline, param_grid=params, cv=4, verbose=3, n_jobs=-1)

    grid_search_list[i].fit(dfs_monthly_list[i]['X_tr'], dfs_monthly_list[i]['y_tr'])

    dfs_monthly_list[i]['y_te'] = grid_search_list[i].predict(dfs_monthly_list[i]['X_te'])

    print('metric cv: ', np.round(np.sqrt(grid_search_list[i].best_score_), 4))

    print('metric train: ', np.round(np.sqrt(mean_squared_error(dfs_monthly_list[i]['y_tr'], grid_search_list[i].predict(dfs_monthly_list[i]['X_tr']))), 4))
    print('params: ', grid_search_list[i].best_params_)

