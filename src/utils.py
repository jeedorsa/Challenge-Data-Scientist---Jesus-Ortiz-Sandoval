from librerias import *

def split_data(data, target, test_size=0.3, random_state=42):
    """
    Split the data into training and testing sets
    
    :param data: the data to split
    :param target: the target variable to predict
    :param test_size: the proportion of data to use for testing (default: 0.3)
    :param random_state: the random state to use for reproducibility (default: 42)
    :return: a tuple containing the training and testing sets of data and target
    """
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """
    Standardize the data using the mean and standard deviation of the training set
    
    :param X_train: the training set to standardize
    :param X_test: the testing set to standardize
    :return: a tuple containing the standardized training and testing sets
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def plot_correlation_matrix(data):
    """
    Plot a correlation matrix of the data using a heatmap
    
    :param data: the data to plot
    """
    corr = data.corr()
    sns.heatmap(corr, cmap="coolwarm", annot=True)
    plt.title("Correlation Matrix")

def fit_linear_regression(X_train, y_train):
    """
    Fit a linear regression model to the data
    
    :param X_train: the training set of data
    :param y_train: the training set of target values
    :return: a fitted LinearRegression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def fit_lasso_regression(X_train, y_train, alpha=1.0):
    """
    Fit a Lasso regression model to the data
    
    :param X_train: the training set of data
    :param y_train: the training set of target values
    :param alpha: the L1 regularization parameter (default: 1.0)
    :return: a fitted Lasso model
    """
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def fit_ridge_regression(X_train, y_train, alpha=1.0):
    """
    Fit a Ridge regression model to the data
    
    :param X_train: the training set of data
    :param y_train: the training set of target values
    :param alpha: the L2 regularization parameter (default: 1.0)
    :return: a fitted Ridge model
    """
    model = Ridge