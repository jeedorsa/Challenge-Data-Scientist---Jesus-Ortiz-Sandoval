from librerias import *

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