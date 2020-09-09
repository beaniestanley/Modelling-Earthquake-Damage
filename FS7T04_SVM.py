# Chiah Soon
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

def svm():
    train_values = pd.read_csv('train_values.csv').drop('building_id', axis=1)
    train_labels = pd.read_csv('train_labels.csv').drop('building_id', axis=1)

    all_columns = list(train_values)
    continuous_columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'age', 'area_percentage', 'count_floors_pre_eq'
                         'height_percentage', 'count_families']
    categorical_columns = [col for col in all_columns if col not in continuous_columns]
    for column in categorical_columns:
        train_values[column] = train_values[column].astype('category').cat.codes

    # One hot encode train values
    column_mask = []
    for column_name in list(train_values):
        column_mask.append(column_name in categorical_columns)
    value_enc = OneHotEncoder(categorical_features=column_mask)
    value_mat = value_enc.fit_transform(train_values)
    value_res = pd.DataFrame(value_mat.toarray())

    # One hot encode train labels
    # label_enc = OneHotEncoder()
    # label_mat = label_enc.fit_transform(train_labels)
    # label_res = pd.DataFrame(label_mat.toarray())

    X_train, X_test, Y_train, Y_test = train_test_split(value_res, train_labels, test_size=0.20, stratify=train_labels)

    param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
    # param_grid = {'gamma': [0.001], 'C': [], kernel': ['rbf']}

    estimator = SVC()

    grid = GridSearchCV(estimator, param_grid, refit=True, verbose=3, cv=ShuffleSplit(test_size=0.20, n_splits=1), n_jobs=1)
    grid.fit(X_train, Y_train)
    grid_predictions = grid.predict(X_test)
    print("Accuracy is: ", grid.score(X_test, Y_test))

    file_object = open('out.txt', 'a')
    file_object.write("The best params are: " + str(grid.best_params_))
    file_object.write("The best estimator is: " + str(grid.best_estimator_))
    file_object.write(str(classification_report(Y_test, grid_predictions)))

    # svclassifier = SVC(verbose=True, kernel="rbf", C=param_grid['C'][0])
    # svclassifier.fit(X_train, Y_train)
    # Y_pred = svclassifier.predict(X_test)
    # print("Accuracy is: ", svclassifier.score(X_test, Y_test))


    # csv_data = pd.read_csv('train_values.csv', header=None)
    # csv_data.head()
    # csv_data.columns = csv_data.iloc[0]
    # csv_data = csv_data.drop(csv_data.index[0])
    # all_columns = list(csv_data)
    # numerical_columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'count_floors_pre_eq', 'age',
    #                      'area_percentage', 'height_percentage', 'count_families']
    # categorical_columns = [col for col in all_columns if col not in numerical_columns and col != "building_id"]
    # # Converting non-numeric categorical columns to numeric
    # for column in categorical_columns:
    #     csv_data[column] = csv_data[column].astype('category').cat.codes
    # X = csv_data
    # Y_csv = pd.read_csv('train_labels.csv', header=None)
    # Y_csv.columns = Y_csv.iloc[0]
    # Y_csv = Y_csv.drop(Y_csv.index[0])
    # Y_csv = Y_csv.drop(columns=['building_id'])
    # Y = Y_csv['damage_grade']

svm()