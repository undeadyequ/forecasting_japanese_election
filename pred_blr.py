from boostedLinearRegression import blr

clf = blr(0.1, 10000, 20)

# Import the packages used to load and manipulate the data
import numpy as np # Numpy is a Matlab-like package for array manipulation and linear algebra
import pandas as pd # Pandas is a data-analysis and table-manipulation tool
import urllib.request # Urlib will be used to download the dataset

# Import the function that performs sample splits from scikit-learn
from sklearn.model_selection import train_test_split
from boostedLinearRegression import blr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def select_data(data, data_condition=("N2012")):
    csvdata = pd.read_csv(data, delimiter=",")
    if "U2009" in data_condition:
        csvdata = csvdata[csvdata['Year'] < 2010]
    if "N2012" in data_condition :
        csvdata = csvdata[csvdata['Year'] != 2012]
    if "N2009" in data_condition:
        csvdata = csvdata[csvdata['Year'] != 2009]
    #csvdata = csvdata.reindex(np.random.permutation(csvdata.index))
    x = csvdata.drop(['Year', 'LDP_votes', 'LDP_seats', 'PM_approval2'], axis=1)
    y = csvdata['LDP_seats']
    return x, y


def train_blr(csv_data, random_seeds=1):
    x, y = select_data(csv_data)
    y = y.values.reshape(-1, 1)
    x = x.values

    # Create the training sample
    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=4, random_state=random_seeds)

    # Split the remaining observations into validation and test
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=3, random_state=random_seeds)

    # Print the numerosities of the three samples
    print('Numerosities of training, validation and test samples:')
    print(x_train.shape[0], x_val.shape[0], x_test.shape[0])

    # Import model-evaluation metrics from scikit-learn

    # Create a boosted linear regression object
    lr = blr(0.1, 1000, 20)

    # Train the model
    lr.fit(x_train, y_train, x_val, y_val)

    # Make predictions on the train, validation and test sets
    y_train_pred = lr.predict(x_train)
    y_val_pred = lr.predict(x_val)
    y_test_pred = lr.predict(x_test)
    return y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred




if __name__ == '__main__':
    csv_data = "data_election_2020_correct2012.csv"
    mae_train = []
    mae_val = []
    mae_test = []
    rmse_train = []
    rmse_val = []
    rmse_test = []
    for r in range(100):
        y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred = \
            train_blr(csv_data, r)
        mae_train.append(mean_absolute_error(y_train, y_train_pred))
        mae_test.append(mean_absolute_error(y_test, y_test_pred))
        mae_val.append(mean_absolute_error(y_val, y_val_pred))

        rmse_train.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        rmse_test.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        rmse_val.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))

    print('MAE on training set:')
    print(np.mean(mae_train), np.std(mae_train))
    #print('MAE on validation set:')
    #print(np.mean(mae_val))
    print('MAE on test set:')
    print(np.mean(mae_test), np.std(mae_test))

    print('RMSE on training set:')
    print(np.mean(rmse_train), np.std(rmse_train))
    #print('RMSE on validation set:')
    #print(np.mean(rmse_val))
    print('RMSE on test set:')
    print(np.mean(rmse_test), np.std(rmse_test))

