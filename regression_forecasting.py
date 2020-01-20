# regression algorithms lives here, the file is thought to be used with
# any python regression algorithm that uses '.fit' and '.predict' methods

# import necessary modules
import xgboost as xgb

import matplotlib.pyplot as plt

import sklearn.kernel_ridge
from sklearn import preprocessing

from tools import *
from preprocessing import *

####################################################################################################
####################################################################################################


# split an array in a series of examples (X,y) with X having length f_steps_in
# and y length steps_out
def big_array_splitter(f_ar, f_steps_in, f_steps_out):

    f_in_out_length = f_steps_in + f_steps_out
    limit = len(f_ar[0]) - f_in_out_length

    # still big array with 45 columns
    res_X = []
    res_y = []
    for f_s in range(45):  # TODO: this is a magic number, to change
        Xs = []
        Ys = []
        for d in range(limit):
            in_left = d
            in_right = d + f_steps_in
            out_left = d + f_steps_in
            out_right = d + f_steps_in + f_steps_out
            xs = f_ar[f_s][in_left: in_right].copy()
            ys = f_ar[f_s][out_left: out_right].copy()

            Xs.append(xs)
            Ys.append(ys)
        res_X.append(Xs)
        res_y.append(Ys)

    res_X = np.array(res_X)
    res_y = np.array(res_y)

    return res_X, res_y


# explained below
def demerge_dataset(f_dataset, f_steps_out):
    """
    demerge_dataset transform a dataset of examples in several as follows:
    let (x_1, x_2) -> (y_1, y_2) be an example and
    steps_out = 2
    then demerge_dataset will make two new examples:
    -(x_1, x_2) -> (y_1)
    -(x_1, x_2) -> (y_2)
    """

    [f_X_train, f_X_val, f_y_train, f_y_val] = f_dataset
    res_demerged_dataset = []

    for f_i in range(f_steps_out):
        f_train, f_val = f_y_train[:, f_i].copy(), f_y_val[:, f_i].copy()

        res_demerged_dataset.append([f_X_train, f_X_val, f_train, f_val])

    res_demerged_dataset = np.array(res_demerged_dataset)
    return res_demerged_dataset


def train_for_one_step(f_dataset, f_model, f_steps_out, f_params=None):
    """
    train_for_one_step follows demerge_dataset, it creates and train a number
    of steps_out models, one for each coordinate in the target vectors
    """

    if f_params is None:
        f_params = {}

    res_demerged_dataset = demerge_dataset(f_dataset, f_steps_out)

    res_models = []

    for f_i in range(f_steps_out):
        res_m = f_model(**f_params)

        [f_X_train, _, f_t_train, _] = res_demerged_dataset[f_i]

        res_m.fit(f_X_train, f_t_train)

        res_models.append(res_m)

    return res_models


def train_for_one_step_all_series(f_dataset, f_model, f_steps_out, f_params=None):
    """
    train_for_one_step_all_series is a wrapper for the train_for_one_step
    function
    :param f_dataset:
    :param f_model:
    :param f_steps_out:
    :param f_params:
    :return:
    """

    res_models_by_series = []

    for f_i in range(len(f_dataset)):
        res_models = train_for_one_step(f_dataset[f_i], f_model, f_steps_out, f_params)
        res_models_by_series.append(res_models)

    return res_models_by_series


def predict_custom_out(f_data, f_models, f_steps_out):
    """
    take a list of models with the same lenght as steps_out and predict a
    vector of length steps_out
    :param f_data:
    :param f_models:
    :param f_steps_out:
    :return:
    """
    res_data = f_data.copy()
    res_preds = []

    for f_i in range(f_steps_out):
        f_m = f_models[f_i]

        t_preds = f_m.predict(res_data)

        res_preds.append(t_preds)

    res_preds = np.array(res_preds)
    res_preds = np.transpose(res_preds)

    return res_preds


def predict_custom_out_all_series(f_data, f_models_by_series, f_steps_out):
    """
    wrapper for the predict_custom_out function
    :param f_data:
    :param f_models_by_series:
    :param f_steps_out:
    :return:
    """
    res_preds = []

    for f_i in range(len(f_data)):
        res_p = predict_custom_out(f_data[f_i], f_models_by_series[f_i], f_steps_out)

        res_preds.append(res_p)

    res_preds = np.array(res_preds)
    return res_preds

####################################################################################################
####################################################################################################


if __name__ == '__main__':

    ############################################################################
    ############################################################################
    # save data to file or plot
    plot_to_file = False
    show_plot = False

    # custom values for this run of training
    steps_in = 14  # length of the entry vector for the models
    steps_out = 21  # length of the target vector
    test_size = 0.1

    # periods, for using the past in the regression methods
    m_period = 182
    y_period = 365

    path_to_data = 'train.csv'

    scaler_to_use = preprocessing.MinMaxScaler()

    # these are some models that we use for the regression
    # a simple one, a medium one and a complex one
    # you can of course use a single model, but 'models_constructors' has to be
    # list
    models_constructors = [sklearn.linear_model.Ridge,
                           sklearn.kernel_ridge.KernelRidge,
                           xgb.XGBRegressor]

    # prefix to be attached to submissions
    submission_prefix = "my_sub"
    ############################################################################
    ############################################################################

    # read the csv data
    df = pd.read_csv(path_to_data, parse_dates=['Day'], index_col='Day')

    # remove outliers
    dfc = df.copy(deep=True)
    dfc = remove_outliers(dfc, alpha=2, gamma=0.6, beta=3, method='median')

    # scale and keep track of the scaler
    (dfc, scaler) = scale(dfc, scaler_to_use)

    # keep track of the original shape of the dataframe
    (days, series) = tuple(dfc.shape)

    # transform the data to a numpy array
    data = to_array(dfc)

    # split to create examples, this method creates examples of the form:
    # x_i -> y_i
    # with x_i having length steps_in and y_i having length steps_out
    # it is to predict the next y_i days having looked to the last x_i days
    (X, y) = big_array_splitter(data, steps_in, steps_out)

    # this part of the code modifies the examples to take into account the
    # past, we see one year in the past and six months in the past
    length_X = len(X[0])

    # create new array for the examples
    new_X = np.zeros((series, length_X - y_period, steps_in * 3))

    # populate the array with the concatenation of values
    for s in range(series):
        for i in range(len(new_X[0])):
            new_X[s][i] = np.concatenate([X[s][i],
                                          X[s][i + m_period],
                                          X[s][i + y_period]])

    # create a new array for the target examples with the values that
    # correspond to the new_X array
    new_y = np.zeros((series, length_X - y_period, steps_out))

    for s in range(series):
        for i in range(len(new_X[0])):
            new_y[s][i] = y[s][i + y_period]

    # create the datasets, one per serie
    datasets = split_to_test(new_X, new_y, days, series, steps_in, steps_out,
                             test_size)

    # fit models to the data
    models_by_series_list = []
    # train each model
    for model in models_constructors:
        models_by_series = train_for_one_step_all_series(datasets, model, steps_out)
        models_by_series_list.append(models_by_series)

    ###############################
    # create a validation dataframe

    # immediate last steps_in days
    X2_validation_dataframe = dfc.iloc[-steps_in - steps_out: - steps_out].copy()
    # take last six months
    X1_validation_dataframe = dfc.iloc[-steps_in - steps_out - m_period:
                                       -steps_out - m_period].copy()
    # take last year too
    X0_validation_dataframe = dfc.iloc[-steps_in - steps_out - y_period:
                                       -steps_out - y_period].copy()

    y_validation_dataframe = dfc.tail(steps_out).copy()

    # transform to array
    X0_validation = to_array(X0_validation_dataframe)
    X1_validation = to_array(X1_validation_dataframe)
    X2_validation = to_array(X2_validation_dataframe)

    # transform to an example by concatenation
    X_validation = np.zeros((series, steps_in*3))
    for s in range(series):
        X_validation[s] = np.concatenate([X0_validation[s],
                                          X1_validation[s],
                                          X2_validation[s]])

    X_validation = np.array([x.reshape((1, -1)) for x in X_validation])

    # predict the validation dataframe
    validation_futures = []

    # one by model in models_constructors
    for models_by_series in models_by_series_list:
        X_validation_preds = predict_custom_out_all_series(X_validation,
                                                           models_by_series, steps_out)
        X_validation_preds = X_validation_preds.reshape((series, steps_out))
        X_validation_preds = np.transpose(X_validation_preds)

        # create the dataframe
        validation_future = pd.DataFrame(X_validation_preds,
                              index=y_validation_dataframe.index)

        # rename columns
        validation_future = validation_future.rename(columns=create_rename_dic())

        # unscale
        validation_future_unscaled = unscale(validation_future, scaler)
        validation_futures.append(validation_future_unscaled)

    # compute the smape score and put it in an array, one per series
    number_of_models = len(models_constructors)
    scores = np.zeros((number_of_models, series))
    for i in range(number_of_models):
        scores[i] = compute_score(df.copy(), validation_futures[i], steps_out, smape_loss)


    # with the same procedure as the validation prediction, predict the future
    X0_future = dfc.tail(steps_in + y_period).head(steps_in).copy()
    X1_future = dfc.tail(steps_in + m_period).head(steps_in).copy()
    X2_future = dfc.tail(steps_in).copy()

    X0_future = to_array(X0_future)
    X1_future = to_array(X1_future)
    X2_future = to_array(X2_future)

    X_future = np.zeros((series, steps_in * 3))
    for s in range(series):
        X_future[s] = np.concatenate([X0_future[s], X1_future[s], X2_future[s]])

    X_future = np.array([x.reshape((1, -1)) for x in X_future])

    preds_futures = []
    for models_by_series in models_by_series_list:
        preds_future = predict_custom_out_all_series(X_future, models_by_series,
                                                     steps_out)

        preds_future = preds_future.reshape((series, steps_out))
        preds_future = np.transpose(preds_future)

        preds_future = pd.DataFrame(preds_future,
                                    index=create_next_index(dfc, steps_out))
        preds_future = preds_future.rename(columns=create_rename_dic())
        preds_future = unscale(preds_future, scaler)

        preds_futures.append(preds_future)

    # create a new prediction with a weighted mean of all others predictions
    wp = weighted_prediction(scores, preds_futures)

    # plot validation and predictions
    colors = ['red', 'green', 'cyan', 'magenta', 'yellow', 'brown']
    preds_futures_length = len(preds_futures)

    for column in validation_futures[0].columns:
        plt.figure(figsize=(16, 10))
        plt.plot(df.tail(100)[column], color='blue')
        plt.plot(wp[column], color='black')

        for i in range(preds_futures_length):
            plt.plot(validation_futures[i][column],
                     color=colors[i % preds_futures_length])
            plt.plot(preds_futures[i][column],
                     color=colors[i % preds_futures_length])

        legend = ['real', 'weighted']
        legend.extend([str(m) for m in models_constructors])
        plt.legend(legend)

        plt.title('Predictions of ' + str(column))

        if plot_to_file:
            plt.savefig(('plot_pdf/validation_plus_predictions_of_'
                         + str(column) +'.pdf'))

        if show_plot:
            plt.show()


    ############################################################################
    ############################################################################

    # now save submissions to csv files
    # TODO: do this tomorrow