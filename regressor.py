import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


from xgboost import plot_importance
from numpy import sort
from matplotlib import pyplot
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler


def load_data():
    """
    Loads the static "Uber-esque" data
    :return: 3 dataframes containing the training examples/predictionss and the test examples
    """
    train_examples = pd.read_csv('./data/train_examples.csv', index_col='id')
    train_predictions = pd.read_csv('data/train_predictions.csv', index_col='id')
    test_examples = pd.read_csv('./data/test_examples.csv', index_col='id')

    return train_examples, train_predictions, test_examples


def part_of_day(hour):
    """
    Utility function to convert hour of the day to period of the day
    :param hour: hour of the day in 24 hour format
    :return: a string representing the period of the day
    """
    if hour in [2, 3, 4, 5]:
        return "early_morning"
    elif hour in [6, 7, 8, 9]:
        return "morning"
    elif hour in [10, 11, 12, 13]:
        return "noon"
    elif hour in [14, 15, 16, 17]:
        return "afternoon"
    elif hour in [18, 19, 20, 21]:
        return "evening"
    else:
        return "late_evening"


def feature_scaling(examples):
    """
    Uses RobustScaler (handles outliers well) to scale the features
    :param examples: dataframe representing the examples to scale
    :return: an ndarray with features scaled
    """
    return RobustScaler().fit_transform(examples)


def handle_dt_data(examples, dt_feature_name):
    """
    Utlity function to handle datetime features
    :param examples: dataframe representing the examples
    :param dt_feature_name: name of the column in the input dataframe representing the datetime data
    :return: a dataframe with the datetime feature expanded into various useful features (period of day, month, day of month, etc.)
    """
    examples[dt_feature_name] = pd.to_datetime(examples[dt_feature_name], format='%m-%d %X')

    months = pd.DataFrame(examples[dt_feature_name].dt.month).rename(columns={'feature_0': 'months'})
    dom = pd.DataFrame(examples[dt_feature_name].dt.day).rename(columns={'feature_0': 'day_of_month'})
    hours = examples[dt_feature_name].dt.hour

    day_names = examples[dt_feature_name].dt.day_name()
    day_names_df = pd.get_dummies(day_names)

    parts_of_day = hours.apply(part_of_day)
    parts_of_day_df = pd.get_dummies(parts_of_day)
    parts_of_day_df = parts_of_day_df[['early_morning', 'morning', 'noon', 'afternoon', 'evening', 'late_evening']]

    hours = pd.DataFrame(hours).rename(columns={'feature_0': 'hours'})

    partial_df = pd.DataFrame({
        'feature_1': examples.feature_1,
        'feature_2': examples.feature_2,
        'feature_3': examples.feature_3,
        'feature_4': examples.feature_4,
        'feature_5': examples.feature_5,
        'feature_6': examples.feature_6,
        'feature_7': examples.feature_7,
        'feature_8': examples.feature_8,
        'feature_9': examples.feature_9,
        'feature_10': examples.feature_10
    })

    return pd.concat([partial_df, hours, months, dom, day_names_df, parts_of_day_df], axis=1)


def plot_features_vs_pred(examples, predictions):
    """
    Utility function to plot the features versus their predictions to get a sense of the data
    :param examples: dataframe representing the training examples and their features to plot
    :param predictions: dataframe representing the training predictions
    :return:
    """
    fig, axes = plt.subplots(ncols=5, nrows=6, figsize=(30, 30))

    axes = axes.flatten()

    for i, v in enumerate(examples.columns):
        data = examples[v]

        axes[i].scatter(x=data, y=predictions, s=35, ec='white', label='actual')

        axes[i].set(title=f'Feature: {v}', ylabel='duration')

    for v in range(26, 30):
        fig.delaxes(axes[v])

    fig.show()


if __name__ == '__main__':
    # Regression hyperparams
    k_best_features = 18
    obj = 'reg:squarederror'
    learning_rate = 0.1
    max_depth = 5
    n_estimators = 1000

    # Load the data
    training_examples, training_predictions, test_examples = load_data()

    # One-hot encoding for breaking down of datetime feature into multiple numerical and categorical features
    training_examples = handle_dt_data(training_examples, 'feature_0')
    test_examples = handle_dt_data(test_examples, 'feature_0')

    # To visualize the data
    plot_features_vs_pred(training_examples, training_predictions)

    # Feature scaling using RobustScaler
    training_examples = feature_scaling(training_examples)
    test_examples = feature_scaling(test_examples)

    # XGBoost digestable format and create regressor object
    data_dmatrix = xgb.DMatrix(data=training_examples, label=training_predictions)
    xgb_reg = xgb.XGBRegressor(objective=obj, learning_rate=learning_rate,
                               max_depth=max_depth, n_estimators=n_estimators)

    # Cross validation of params to get optimal number of estimators (trees) using 5 folds, MAE (to calculate error) and 50 as early stopping rounds
    optimal_num_estimators = xgb.cv(xgb_reg.get_xgb_params(), data_dmatrix, n_estimators, nfold=5, metrics='mae', early_stopping_rounds=50).shape[0]
    # Modify regressor accordingly
    xgb_reg.set_params(n_estimators=optimal_num_estimators)

    # Fit the data
    xgb_reg.fit(training_examples, training_predictions)

    # Reduce number of features based on feature importance
    plot_importance(xgb_reg)
    pyplot.show()

    thresholds = sort(xgb_reg.feature_importances_)
    selection = SelectFromModel(xgb_reg, threshold=thresholds[-k_best_features], prefit=True)

    select_training_examples = pd.DataFrame(training_examples).loc[:, selection.get_support()].to_numpy()
    select_test_examples = pd.DataFrame(test_examples).loc[:, selection.get_support()].to_numpy()

    # Repeat above process with most important features selected
    select_data_dmatrix = xgb.DMatrix(data=select_training_examples, label=training_predictions)
    select_xgb_reg = xgb.XGBRegressor(objective=obj, learning_rate=learning_rate,
                               max_depth=max_depth, n_estimators=n_estimators)

    select_optimal_num_estimators = xgb.cv(select_xgb_reg.get_xgb_params(), data_dmatrix, n_estimators, nfold=5, metrics='mae',
                                    early_stopping_rounds=50).shape[0]
    select_xgb_reg.set_params(n_estimators=select_optimal_num_estimators)


    select_xgb_reg.fit(select_training_examples, training_predictions)

    # Calculate predictions
    select_preds = select_xgb_reg.predict(select_test_examples)

    # Persist output for Kaggle competition
    outputs = pd.DataFrame(select_preds, columns=['duration'])
    outputs.index.set_names('id', inplace=True)
    outputs.to_csv('./output/test_predictions.csv')