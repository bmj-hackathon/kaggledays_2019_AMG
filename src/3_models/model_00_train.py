# Train 2 seperate models, one for cats, one for dogs!!

# assert y_tr.dtype == np.dtype('int64'), "y_tr must be integer for LGBM!!"

dfs_monthly_list = dfs_monthly_list


#%% Classifier

# clf = xgb.XGBRegressor(
#     max_depth=8,
#     n_estimators=1000,
#     min_child_weight=300,
#     colsample_bytree=0.8,
#     subsample=0.8,
#     eta=0.3,
#     seed=42)
clf = xgb.XGBRegressor(learning_rate=0.02, n_estimators=600, )


# %% Select data
i = 1



# %% Test fit
for i = range(1,4)
    X_tr = dfs_monthly_list[i]['arr_transformed']
    y_tr = dfs_monthly_list[i]['y_tr']
    # clf_grid.fit(X_tr, y_tr)
    gs = GridSearchCV(pipeline, param_grid=params, cv=4, verbose=3, n_jobs=-1)
    model = clf.fit(X_tr,y_tr)
    print(model)
#%%
folds = 3
param_comb = 5

skf = sk.model_selection.StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

grid_params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

random_search = sk.model_selection.RandomizedSearchCV(clf, param_distributions=grid_params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_tr,y_tr), verbose=3, random_state=1001)

# Here we go
# start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X_tr, y_tr)
# timer(start_time) # timing ends here for "start_time" variable


#%%
# Define grid

clf_grid = sk.model_selection.GridSearchCV(clf, params_grid,
                                       verbose=1,
                                       cv=5,
                                       n_jobs=-1)
#%% Fit
clf_grid.fit(
    X_tr,
    y_tr,
    eval_metric="rmse",
    eval_set=[(X_tr, y_tr), (X_valid, Y_valid)],
    verbose=True,
    early_stopping_rounds = 10)

time.time() - ts

#%% Model and params
params_model = dict()
# params['num_class'] = len(y_tr.value_counts())


#%% GridCV
params_grid = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
    }

clf_grid = sk.model_selection.GridSearchCV(clf, params_grid,
                                       verbose=1,
                                       cv=5,
                                       n_jobs=-1)
#%% Fit
i = 1

X_tr = dfs_monthly_list[i]['arr_transformed']
y_tr = dfs_monthly_list[i]['y_tr']
clf_grid.fit(X_tr, y_tr)

# Print the best parameters found
print("Best score:", clf_grid.best_score_)
print("Bast parameters:", clf_grid.best_params_)

clf_grid_BEST = clf_grid.best_estimator_

#%% Do the final fit on the BEST estimator
# start = datetime.datetime.now()
# predicted = clf_grid_BEST.fit(train_X, train_Y)
# logging.info("Elapsed H:m:s: {}".format(datetime.datetime.now()-start))

#%% Predict on Test set
# NB we only want the defaulters column!
predicted = clf_grid_BEST.predict(X_te)

#%% Metric

# kappa(target, train_predictions)
# rmse(target, [r[0] for r in results['train']])
# submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
# submission.head()


#%%
# n_fold = 5
# folds = sk.model_selection.StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=15)
#
# for fold_n, (train_indices, valid_indices) in enumerate(folds.split(X_tr, y_tr)):
#     logging.info("Fold {:<4} {:0.2f}|{:0.2f}% started {}".format(fold_n,
#                                                        100*len(train_indices)/len(X_tr),
#                                                        100*len(valid_indices)/len(X_tr),
#                                                                   time.ctime()))
#     gc.collect()
#     X_tr_fold, X_val_fold = X_tr.iloc[train_indices], X_tr.iloc[valid_indices]
#     y_tr_fold, y_val_fold = y_tr.iloc[train_indices], y_tr.iloc[valid_indices]
#
#     ds_tr_fold = lgb.Dataset(X_tr_fold, label=y_tr_fold)
#     ds_val_data = lgb.Dataset(X_val_fold, label=y_val_fold)
#
#
#     pprint(model.get_params())
#     logging.info("Model instantiated".format())
#
#
#     # model = lgb.train(params,
#     #                   ds_tr_fold,
#     #                   num_boost_round=2000,
#     #                   valid_sets=[ds_tr_fold, ds_val_data],
#     #                   verbose_eval=100,
#     #                   early_stopping_rounds=200)
#
#         grid = sk.model_selection.GridSearchCV(mdl, gridParams,
#                             verbose=0,
#                             cv=4,
#                             n_jobs=2)
