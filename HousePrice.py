import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


Dataset_Train = pd.read_csv(
    '/Users/navin.jain/Desktop/Learning 101/Linear Regression/house-prices-advanced-regression-techniques/train.csv')

Dataset_Test = pd.read_csv(
    '/Users/navin.jain/Desktop/Learning 101/Linear Regression/house-prices-advanced-regression-techniques/test.csv')

Combined_data = pd.concat([Dataset_Train, Dataset_Test])


def data_types(dataset):
    Data = dataset
    Continuous_features = []
    Categorical_features = []
    Discrete_feature = []
    Target_variable = []
    for col in Data.columns:
        if Data[col].dtypes != 'O' and len(Data[col].unique()) <= 25:
            Discrete_feature.append(col)
        elif Data[col].dtypes != 'O' and col not in ['SalePrice', 'Id']:
            Continuous_features.append(col)
        elif Data[col].dtypes == 'O':
            Categorical_features.append(col)
        else:
            Target_variable.append(col)
    return Continuous_features, Categorical_features, Discrete_feature, Target_variable


Continuous_features, Categorical_features, Discrete_feature, Target_variable = data_types(Combined_data)


## plots

def Distribution_plot(dataset):
    for i in dataset:
        if i in Continuous_features:
            sns.histplot(dataset, x=i, kde=True)
            plt.show()


## Visualize discrete Variables
def Bar_plot(dataset):
    for cat_feat in Categorical_features:
        dataset.groupby(cat_feat)['SalePrice'].median().plot.bar()
        plt.xlabel(cat_feat)
        plt.ylabel('Median Sale Price')
        plt.show()


## Yr feature plot
def line_plot(dataset):
    for yr_feat in dataset:
        dataset.groupby(yr_feat)['SalePrice'].median().plot.line()
        plt.xlabel(yr_feat)
        plt.ylabel('Median Sale Price')
        plt.show()


def feature_engineering(dataset):
    null_variables = []
    for col in dataset:
        if dataset[col].isnull().sum() > 0:
            null_variables.append(col)

    for feature in null_variables:
        if feature in Categorical_features:
            dataset[feature] = np.where(dataset[feature].isnull(), 'None', dataset[feature])

    ## step 2 - add median for all nulls in continuous due to skewed data
    for feature in null_variables:
        if feature in Continuous_features:
            dataset[feature] = np.where(dataset[feature].isnull(), dataset[feature].median(),
                                        dataset[feature])

    ## Step 3 - add 0 for discrete
    for feature in null_variables:
        if feature in Discrete_feature:
            dataset[feature] = np.where(dataset[feature].isnull(), 0,
                                        dataset[feature])

    # get dummies
    dataset = pd.get_dummies(dataset, columns=Categorical_features)
    # dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='None')))]

    return dataset


Combined_data = feature_engineering(Combined_data)

# Split Combined into train and test dataset

Combined_data = Combined_data.set_index('Id')

Dataset_Test = Combined_data[Combined_data['SalePrice'].isna()]
Dataset_Train = Combined_data[Combined_data['SalePrice'].notna()]


## Scale Data
def Data_scaler(Data, scaler):
    scaler = ColumnTransformer([('minmax', scaler(), Continuous_features + Discrete_feature)], remainder='passthrough')
    data_scaled = scaler.fit_transform(Data)
    data_scaled = pd.DataFrame(data_scaled, index=Data.index, columns=Data.columns)

    return data_scaled


Dataset_Train = Data_scaler(Dataset_Train, StandardScaler)
Dataset_Test = Data_scaler(Dataset_Test, StandardScaler)

X_train = Dataset_Train.drop(['SalePrice'], axis=1)
Y_train = Dataset_Train['SalePrice']

X_test = Dataset_Test.drop(['SalePrice'], axis=1)

# Try Models
lm = LinearRegression()
lm.fit(X_train, Y_train)
lm.score(X_train, Y_train)
Y_pred = lm.predict(X_test)

# Hyper parameter tuning

# step-1: create a cross-validation scheme
folds = KFold(n_splits=5, shuffle=True, random_state=100)

# step-2: specify range of hyperparameters to tune
hyper_params = [{'n_features_to_select': list(range(1, 50))}]

# step-3: perform grid search
# 3.1 specify model
lm = LinearRegression()
lm.fit(X_train, Y_train)
rfe = RFE(lm)

# 3.2 call GridSearchCV()
model_cv = GridSearchCV(estimator=rfe,
                        param_grid=hyper_params,
                        scoring='r2',
                        cv=folds,
                        verbose=1,
                        return_train_score=True)

# fit the model
model_cv.fit(X_train, Y_train)
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results

# final model using the rank 1 n_features to predict on test
n_features_optimal = 12

rfe = RFE(lm, n_features_to_select=n_features_optimal)
rfe = rfe.fit(X_train, Y_train)

# predict prices of X_test
y_pred = rfe.predict(X_test)
print(rfe.score(X_train, Y_train))

print ('coefficients',rfe.estimator_.coef_)
print(rfe.support_)


test_preds = pd.DataFrame.from_dict({'Id': X_test.index, 'SalePrice': (rfe.predict(X_test))})


test_preds.to_csv(
    '/Users/navin.jain/Desktop/Learning 101/Linear Regression/house-prices-advanced-regression-techniques/Output.csv')
