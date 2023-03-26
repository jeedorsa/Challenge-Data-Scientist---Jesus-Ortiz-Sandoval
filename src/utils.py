from librerias import *
import os
def extract_numeric_part(flight_number):
    """
    Extracts the numeric part from a flight number string.
    
    Args:
        flight_number (str): A flight number string containing alphanumeric characters.
    
    Returns:
        int: The numeric part of the flight number, or None if the flight_number is not a string or has no numeric part.
    """   
    if isinstance(flight_number, str):
        numeric_part = re.findall(r'\d+', flight_number)
        return int(numeric_part[0]) if numeric_part else None
    else:
        return None

def extract_alphabetic_part(flight_number):
    """
    Extracts the alphabetic part from a flight number string.
    
    Args:
        flight_number (str): A flight number string containing alphanumeric characters.
    
    Returns:
        str: The alphabetic part of the flight number, or None if the flight_number is not a string or has no alphabetic part.
    """
    if isinstance(flight_number, str):
        alphabetic_part = re.findall(r'[a-zA-Z]+', flight_number)
        return alphabetic_part[0] if alphabetic_part else None
    else:
        return 0


def plot_flight_analysis(df):
    """
    Plots various graphs for flight data analysis, including:
    1. Share of International and National Departing Flights by Airline
    2. Top 25 Destinations from SCL
    3. Heatmap of Flight Count by Month and Day of the Week
    4. Distribution of flights by the hour of the day
    5. Distribution of international and national flights by month
    
    Args:
    df (pd.DataFrame): The flight data DataFrame with the following columns:
                       DIA, SIGLADES, OPERA, TIPOVUELO, DIANOM, MES
    
    Returns:
    None
    """
    title_font = dict(family="Arial, sans-serif", size=18, color="black")
    axis_font = dict(family="Arial, sans-serif", size=14, color="black")    
    airline_type_pivot = pd.pivot_table(data=df, values='DIA', index=['OPERA'], columns=['TIPOVUELO'], aggfunc='count', fill_value=0)
    airline_type_percentage = airline_type_pivot / airline_type_pivot.sum()
    airline_type_percentage = airline_type_percentage.reset_index()
    flight_types = [('I', 'International'), ('N', 'National')]
    #1. Share of International and National Departing Flights by Airline
    for flight_type, flight_label in flight_types:
        sorted_data = airline_type_percentage.sort_values(flight_type, ascending=False)
        fig1 = px.bar(sorted_data, x='OPERA', y=flight_type, title=f"Share of {flight_label} Departing Flights by Airline",
                     labels={'OPERA': 'Airline', flight_type: f'Share of {flight_label} Flights'},
                     color=flight_type, color_continuous_scale='Viridis')
        fig1.update_yaxes(tickformat=".0%", title='Share of Flights')
        fig1.update_xaxes(title='Airline')
        fig1.update_layout(
            title_font=title_font,
            xaxis_title_font=axis_font,
            yaxis_title_font=axis_font,
            coloraxis_colorscale="Viridis"
        )
        pyo.iplot(fig1)
    #2. Top 25 Destinations from SCL
    top_destinations = df['SIGLADES'].value_counts().iloc[:25]
    fig2 = px.bar(top_destinations, x=top_destinations.index, y=top_destinations.values,
                 title="Top 25 Destinations from SCL",
                 labels={'x': 'Destination City', 'y': 'Number of Flights'},
                 color=top_destinations.values, color_continuous_scale='Viridis')
    fig2.update_xaxes(title='Destination City')
    fig2.update_xaxes(tickangle=90)
    fig2.update_yaxes(title='Number of Flights')
    fig2.update_layout(
    title_font=title_font,
    xaxis_title_font=axis_font,
    yaxis_title_font=axis_font,
    coloraxis_colorscale="Viridis"
    )
    pyo.iplot(fig2)
    
    #3. Heatmap of Flight Count by Month and Day of the Week
    heatmap_data = df.pivot_table(values='DIA', index='DIANOM', columns='MES', aggfunc='count', fill_value=0)
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    heatmap_data.columns = [month_names[col - 1] for col in heatmap_data.columns]
    fig3 = px.imshow(heatmap_data, title='Heatmap of Flight Count by Month and Day of Week',
                     labels=dict(x='Month', y='Day of Week', color='Number of Flights'))
    fig3.update_xaxes(side='bottom', tickangle=-45)
    fig3.update_layout(
        title_font=title_font,
        xaxis_title_font=axis_font,
        yaxis_title_font=axis_font,
        coloraxis_colorscale="Viridis"
    )
    pyo.iplot(fig3)
    
    #4. Distribution of flights by the hour of the day
    df['Hour'] = pd.to_datetime(df['Fecha-O']).dt.hour
    flights_by_hour = df['Hour'].value_counts().sort_index()
    fig4 = px.bar(flights_by_hour, x=flights_by_hour.index, y=flights_by_hour.values,
                 title="Flights Distribution by Hour of the Day",
                 labels={'x': 'Hour of the Day', 'y': 'Number of Flights'},
                 color=flights_by_hour.values, color_continuous_scale='Viridis')
    fig4.update_xaxes(title='Hour of the Day')
    fig4.update_yaxes(title='Number of Flights')
    fig4.update_xaxes(tickangle=0)
    fig4.update_layout(
    title_font=title_font,
    xaxis_title_font=axis_font,
    yaxis_title_font=axis_font,
    coloraxis_colorscale="Viridis"
    )
    pyo.iplot(fig4)

    # 5. Distribution of international and national flights by month
    flights_by_month_type = pd.pivot_table(df, values='DIA', index=['MES'], columns=['TIPOVUELO'], aggfunc='count', fill_value=0)
    flights_by_month_type = flights_by_month_type.reset_index()

    # Convert month numbers to month names
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    flights_by_month_type['MES'] = flights_by_month_type['MES'].apply(lambda x: month_names[x - 1])

    fig5 = px.bar(flights_by_month_type, x='MES', y=['I', 'N'],
                  title="Distribution of International and National Flights by Month",
                  labels={'MES': 'Month', 'value': 'Number of Flights', 'variable': 'Flight Type'},
                  color_discrete_sequence=px.colors.qualitative.Plotly)
    fig5.update_xaxes(title='Month')
    fig5.update_xaxes(tickangle=0)
    fig5.update_yaxes(title='Number of Flights')
    fig5.update_layout(
        title_font=title_font,
        xaxis_title_font=axis_font,
        yaxis_title_font=axis_font
    )
    pyo.iplot(fig5)

def en_temporada_alta(fecha):
    """
    Devuelve 1 si la fecha está en temporada alta, 0 en caso contrario.
    Temporada alta se define como Fecha-I entre 15-Dic y 3-Mar, 15-Jul y 31-Jul, o 11-Sep y 30-Sep.

    Args:
    fecha (datetime): Fecha a comprobar si está en temporada alta.

    Returns:
    int: 1 si la fecha está en temporada alta, 0 en caso contrario.
    """
    return ((fecha.month == 12 and fecha.day >= 15) or
            (fecha.month == 1 or fecha.month == 2) or
            (fecha.month == 3 and fecha.day <= 3) or
            (fecha.month == 7 and fecha.day >= 15 and fecha.day <= 31) or
            (fecha.month == 9 and fecha.day >= 11 and fecha.day <= 30))

def atraso_func(x):
    """
    Devuelve 1 si la diferencia en minutos es mayor que 15, 0 en caso contrario.

    Args:
    x (float): Diferencia en minutos entre dos fechas.

    Returns:
    int: 1 si la diferencia en minutos es mayor que 15, 0 en caso contrario.
    """
    return 1 if x > 15 else 0

def periodo_func(x):
    """
    Devuelve 'mañana', 'tarde' o 'noche' según la hora de la fecha.

    Args:
    x (datetime): Fecha a comprobar.

    Returns:
    str: 'mañana' si la hora de la fecha está entre las 5:00 y las 11:59,
         'tarde' si está entre las 12:00 y las 18:59,
         'noche' si está entre las 19:00 y las 4:59.
    """
    return 'mañana' if 5 <= x.hour <= 11 else ('tarde' if 12 <= x.hour <= 18 else 'noche')

def nuevas_columnas(df):
    """
    Agrega las columnas 'temporada_alta', 'dif_min', 'atraso_15' y 'periodo_dia' al DataFrame.

    Args:
    df (pandas.DataFrame): DataFrame al que se agregarán las nuevas columnas.

    Returns:
    pandas.DataFrame: DataFrame original con las nuevas columnas agregadas.
    """
    # crear columna temporada_alta
    df['temporada_alta'] = df['Fecha-I'].apply(en_temporada_alta).astype(int)
    
    # crear columna dif_min
    df['dif_min'] = (df['Fecha-O'] - df['Fecha-I']).apply(lambda x: x.total_seconds()/60)
    
    # crear columna atraso_15
    df['atraso_15'] = df['dif_min'].apply(atraso_func)
    
    # crear columna periodo_dia
    df['periodo_dia'] = df['Fecha-I'].apply(periodo_func)
    
    return df

def duration(row):
    """
    Calculates the duration of the flight in minutes.

    Parameters:
    -----------
    row : pandas.Series
        A pandas series representing a row of flight data containing the 'Fecha-O'
        and 'Fecha-I' columns.

    Returns:
    --------
    float
        The duration of the flight in minutes.
    """
    time_diff = pd.to_datetime(row['Fecha-O']) - pd.to_datetime(row['Fecha-I'])
    return time_diff.total_seconds() / 60


def time_since_last_departure(data):
    """
    Calculates the time since the last departure from the airport in minutes.

    Parameters:
    -----------
    data : pandas.DataFrame
        A pandas dataframe containing flight data with a 'Fecha-I' column representing
        the departure date and time.

    Returns:
    --------
    pandas.DataFrame
        A new pandas dataframe with an additional 'time_since_last_departure' column
        representing the time in minutes since the last departure from the airport.
    """
    data = data.sort_values(by='Fecha-I')
    data['time_since_last_departure'] = data['Fecha-I'].diff().apply(lambda x: x.total_seconds() / 60)
    return data


def time_since_last_arrival(data):
    """
    Calculates the time since the last arrival at the airport in minutes.

    Parameters:
    -----------
    data : pandas.DataFrame
        A pandas dataframe containing flight data with a 'Fecha-O' column representing
        the arrival date and time.

    Returns:
    --------
    pandas.DataFrame
        A new pandas dataframe with an additional 'time_since_last_arrival' column
        representing the time in minutes since the last arrival at the airport.
    """
    data = data.sort_values(by='Fecha-O')
    data['time_since_last_arrival'] = data['Fecha-O'].diff().apply(lambda x: x.total_seconds() / 60)
    return data

def is_weekend(row):
    """
    Determines whether a flight is on a weekend day or not.

    Parameters:
    -----------
    row : pandas.Series
        A pandas series representing a row of flight data containing the 'Fecha-I'
        column.

    Returns:
    --------
    int
        Returns 1 if the flight is on a weekend day (Saturday or Sunday), and 0 otherwise.
    """
    day_of_week = pd.to_datetime(row['Fecha-I']).weekday()
    if day_of_week >= 5:
        return 1
    else:
        return 0




def plot_delay_rate_by_group(df, grouping_column):
    """
    Generates a bar graph with the delay rate of flights in *df* by *grouping_column*.
    
    :param df: Data to be plotted. Must have "atraso_15" and *grouping_column* among columns.
    :type df: DataFrame
    :param grouping_column: Column of *df* to group delay rates by.
    :type grouping_column: str
    """
    # Calculate the mean delay rate for each group
    group_means = df.groupby(grouping_column, as_index=False).mean()[[grouping_column, 'atraso_15']]
    group_means['atraso_15'] *= 100

    # Calculate the global mean delay rate
    global_mean = np.nanmean(df['atraso_15']) * 100
    if grouping_column == 'MES':
        month_names = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
            7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        group_means[grouping_column] = group_means[grouping_column].map(month_names)


    # Sort the data by delay rate
    sorted_means = group_means.sort_values('atraso_15', ascending=False)

    # Create the barplot with seaborn
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=sorted_means, x=grouping_column, y='atraso_15', ax=ax, order=sorted_means[grouping_column])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.axhline(global_mean, color='r', linestyle='--', label='Media Global')
        ax.set_title(f"Proporción de Vuelos con Retraso por {grouping_column}")
        ax.set_ylabel("Proporción de Vuelos con Retraso")
        ax.set_xlabel(grouping_column)
        ax.legend()
        plt.show()
        
        
def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, title='Confusion Matrix', cmap='Blues'):
    """
    Plots a confusion matrix using seaborn heatmap.
    
    :param y_true: True labels.
    :type y_true: array-like, shape (n_samples,)
    :param y_pred: Predicted labels.
    :type y_pred: array-like, shape (n_samples,)
    :param labels: List of labels to index the matrix.
    :type labels: list of str, optional, default: None
    :param normalize: Whether to normalize the confusion matrix.
    :type normalize: bool, optional, default: False
    :param title: Title for the confusion matrix plot.
    :type title: str, optional, default: 'Confusion Matrix'
    :param cmap: Colormap to be used for the heatmap.
    :type cmap: str, optional, default: 'Blues'
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalize the confusion matrix if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=labels, yticklabels=labels)

    # Set plot title and labels
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Show the plot
    plt.show()
    
    
def find_best_threshold(y_test, y_pred_prob):
    """
    Finds the best threshold for classification based on the highest F1 score.
    
    :param y_test: True target values.
    :type y_test: array-like
    :param y_pred_prob: Predicted probabilities for the positive class.
    :type y_pred_prob: array-like
    :return: Best threshold.
    :rtype: float
    """
    thresholds = np.arange(0, 1.01, 0.01)
    best_threshold = 0
    best_f1_score = 0
    for t in thresholds:
        y_pred_t = (y_pred_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = t
    return best_threshold

def plot_roc_curve(fpr, tpr, roc_auc, classifier_name):
    """
    Plots the ROC curve for a given classifier.
    
    :param fpr: False positive rates.
    :type fpr: array-like
    :param tpr: True positive rates.
    :type tpr: array-like
    :param roc_auc: Area under the ROC curve.
    :type roc_auc: float
    :param classifier_name: Name of the classifier.
    :type classifier_name: str
    """
    plt.plot(fpr, tpr, label=f'{classifier_name} AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

def train_with_cross_validation(clf, X, y, cv=5):
    """
    Trains a classifier using cross-validation and returns the average performance metrics and the trained classifier.

    :param clf: Classifier to be trained.
    :type clf: Classifier object
    :param X: Training data.
    :type X: DataFrame
    :param y: Labels for the training data.
    :type y: Series
    :param cv: Number of cross-validation folds.
    :type cv: int
    :return: A dictionary containing average performance metrics and the trained classifier.
    :rtype: dict
    """

    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    avg_metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred)
    }

    clf.fit(X, y)  # Train the classifier on the entire dataset

    return avg_metrics, clf

def optimize_hyperparameters(classifier, param_grid, X_train, y_train,cv):
    """
    Optimizes hyperparameters for a classifier using GridSearchCV.
    
    :param classifier: The classifier to optimize.
    :type classifier: estimator
    :param param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values.
    :type param_grid: dict
    :param X_train: Training data.
    :type X_train: array-like
    :param y_train: Training target values.
    :type y_train: array-like
    :return: Best estimator found by grid search.
    :rtype: estimator
    """
    grid_search = GridSearchCV(classifier, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1,verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def encode_categorical_columns(data, categorical_columns):
    """
    Encodes categorical columns in *data* using One-Hot Encoding.
    
    :param data: Data to be encoded.
    :type data: DataFrame
    :param categorical_columns: Categorical columns in *data*.
    :type categorical_columns: list of str
    :return: Encoded data.
    :rtype: DataFrame
    """
    data = pd.get_dummies(data, columns=categorical_columns)
    return data

def train_and_evaluate(clf, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a classifier using the specified data.
    
    :param clf: Classifier to be trained.
    :type clf: Classifier object
    :param X_train: Training data.
    :type X_train: DataFrame
    :param y_train: Labels for the training data.
    :type y_train: Series
    :param X_test: Test data.
    :type X_test: DataFrame
    :param y_test: Labels for the test data.
    :type y_test: Series
    """
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    
    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1-score: {f1_score(y_test, y_pred)}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_prob)}")
    print(classification_report(y_test, y_pred))
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_pred_prob):.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='k')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def generate_requirements_txt(filename):
    """
    Generates a `requirements.txt` file containing a list of all installed packages in the current environment.

    Parameters:
    -----------
    filename : str
        The name of the file to be generated, without any path information.

    Returns:
    --------
    None
    """

    # run pip freeze command to get a list of installed packages
    installed_packages = subprocess.check_output(['pip', 'freeze']).decode('utf-8').split('\n')

    # create the data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # open the specified requirements.txt file in write mode within the data directory
    with open(f"data/{filename}", 'w') as f:
        # write each package to the file
        for package in installed_packages:
            f.write(package + '\n')

    print(f"Generated {filename} successfully in the data directory!")
    
def install_requirements(filepath='requirements.txt'):
    """
    Reads the packages listed in a `requirements.txt` file and installs them in the current environment.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """

    try:
        # read the packages listed in the requirements.txt file
        with open(f"data/{filepath}", 'r') as f:
            packages = f.read().splitlines()

        # install the packages using pip
        result = subprocess.run([sys.executable, "-m", "pip", "install", *packages], capture_output=True)

        if result.returncode == 0:
            print("All packages already installed.")
        else:
            print("Installed packages successfully!")
    except FileNotFoundError:
        print("No requirements.txt file found. Continuing with the execution...")

