import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Conv1D, GlobalMaxPooling1D, Input, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import timedelta

# Set page configuration
st.set_page_config(page_title="Sales Prediction App", layout="wide")

# Function to load and preprocess data
@st.cache_data
def load_and_preprocess_data(file):
    if isinstance(file, str):  # If it's a file path
        if file.endswith('.csv'):
            df = pd.read_csv(file, thousands=',')
        else:
            df = pd.read_excel(file)
    else:  # If it's an uploaded file
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, thousands=',')
        else:
            df = pd.read_excel(file, thousands=',')
    
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df = df.sort_values('date')
    
    # Ensure 'daily_collection' is numeric
    df['daily_collection'] = pd.to_numeric(df['daily_collection'], errors='coerce')
    
    # Create features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_high_sales_day'] = df['day_of_week'].isin([6, 0]).astype(int)
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    # Create lag features
    df['sales_lag_1'] = df['daily_collection'].shift(1)
    df['sales_lag_7'] = df['daily_collection'].shift(7)
    df['sales_lag_14'] = df['daily_collection'].shift(14)
    df['sales_lag_21'] = df['daily_collection'].shift(21)
    df['sales_lag_30'] = df['daily_collection'].shift(30)
    df['rolling_mean_7'] = df['daily_collection'].rolling(window=7).mean()
    df['rolling_mean_30'] = df['daily_collection'].rolling(window=30).mean()
    
    # Create features to capture potential special days
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    
    # Calculate the day of the week average sales
    day_of_week_avg = df.groupby('day_of_week')['daily_collection'].transform('mean')
    df['day_of_week_avg_diff'] = df['daily_collection'] - day_of_week_avg
    
    # Calculate the month average sales
    month_avg = df.groupby('month')['daily_collection'].transform('mean')
    df['month_avg_diff'] = df['daily_collection'] - month_avg

    # Calculate the month average sales for the current year
    # month_avg_current_year = df[df['year'] == df['year'].max()].groupby('month')['daily_collection'].transform('mean')

    # Calculate the month average sales for the previous year
    # month_avg_previous_year = df[df['year'] == df['year'].max() - 1].groupby('month')['daily_collection'].transform('mean')

    # Calculate the difference between current and previous year's month averages
    # df['month_avg_diff_year_over_year'] = month_avg_current_year - month_avg_previous_year
    
    # Fourier features to capture seasonality
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['sin_week'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # Rolling statistics
    df['rolling_std_7'] = df['daily_collection'].rolling(window=7).std()
    df['rolling_std_30'] = df['daily_collection'].rolling(window=30).std()
    df['rolling_max_7'] = df['daily_collection'].rolling(window=7).max()
    df['rolling_min_7'] = df['daily_collection'].rolling(window=7).min()

    # Day of week one-hot encoding
    for i in range(7):
        df[f'day_{i}'] = (df['day_of_week'] == i).astype(int)

    # Month one-hot encoding
    for i in range(1, 13):
        df[f'month_{i}'] = (df['month'] == i).astype(int)
    
    # Drop rows with NaN values
    df = df.dropna()

    
    return df

# Function to create and train models
@st.cache_resource
def create_and_train_model(model_type, X, y, epochs=50, batch_size=32, layers=3, units=[128, 64, 32], timesteps=1):
    if model_type == 'Dense':
        model = Sequential([Input(shape=(X.shape[1],))])
        for i in range(layers):
            model.add(Dense(units[i], activation='relu'))
            model.add(Dropout(0.3))
        model.add(Dense(1))
    elif model_type in ['LSTM', 'GRU', 'SimpleRNN']:
        # Adjust the reshape operation
        if X.shape[1] % timesteps != 0:
            pad_width = timesteps - (X.shape[1] % timesteps)
            X = np.pad(X, ((0, 0), (0, pad_width)), 'constant')
        X = X.reshape((X.shape[0], timesteps, -1))
        
        model = Sequential([Input(shape=(timesteps, X.shape[2]))])
        for i in range(layers - 1):
            if model_type == 'LSTM':
                model.add(LSTM(units[i], return_sequences=True))
            elif model_type == 'GRU':
                model.add(GRU(units[i], return_sequences=True))
            else:
                model.add(SimpleRNN(units[i], return_sequences=True))
            model.add(Dropout(0.3))
        if model_type == 'LSTM':
            model.add(LSTM(units[-1]))
        elif model_type == 'GRU':
            model.add(GRU(units[-1]))
        else:
            model.add(SimpleRNN(units[-1]))
        model.add(Dropout(0.3))
        model.add(Dense(1))
    elif model_type == 'Conv1D':
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = Sequential([Input(shape=(X.shape[1], 1))])
        for i in range(layers - 1):
            model.add(Conv1D(units[i], kernel_size=3, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(units[-1], activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    return model, history

# Main Streamlit app
def main():
    st.title("Sales Prediction Application")
    
    # Use session state to store the dataframe
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # File uploader with default path
    default_path = r"C:\Users\Vinod1.Choudhary\Documents\Python Project\React app\tensorflow\retail_store_data.csv"
    file_path = st.text_input("Enter file path or use default", value=default_path)
    
    if st.button("Load Data"):
        if os.path.exists(file_path):
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension in ['.csv', '.xlsx', '.xls']:
                st.session_state.df = load_and_preprocess_data(file_path)
                st.success("Data loaded successfully!")
            else:
                st.error("Unsupported file format. Please use CSV or Excel files.")
        else:
            st.error("File not found. Please check the path and try again.")
    
    uploaded_file = st.file_uploader("Or upload your CSV/Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        st.session_state.df = load_and_preprocess_data(uploaded_file)
        st.success("Data loaded successfully!")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Date range selector using slider
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        date_range = st.slider(
            "Select date range for training data",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date - timedelta(days=180)),
            format="DD-MM-YYYY"
        )
        train_start_date, train_end_date = date_range
        
        # Feature selection
        all_features = ['day_of_week', 'month', 'year', 'day_of_month', 'is_weekend', 'is_high_sales_day', 
                        'day_of_year', 'week_of_year', 'quarter', 'sales_lag_1', 'sales_lag_7', 'sales_lag_14',
                        'sales_lag_21', 'sales_lag_30', 'rolling_mean_7', 'rolling_mean_30', 'is_month_start', 
                        'is_month_end', 'is_quarter_start', 'is_quarter_end', 'day_of_week_avg_diff', 
                        'month_avg_diff', 'sin_day', 'cos_day', 'sin_week', 'cos_week', 'rolling_std_7', 
                        'rolling_std_30', 'rolling_max_7', 'rolling_min_7', 
                        'day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6',
                        'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 
                        'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12']
        
        selected_features = st.multiselect("Select features for the model", all_features, default=all_features)
        
        # Model selection
        model_type = st.selectbox("Select the model type", ['Dense', 'LSTM', 'GRU', 'SimpleRNN', 'Conv1D'])
        
        # Model hyperparameters
        epochs = st.slider("Number of epochs", min_value=10, max_value=200, value=50)
        batch_size = st.slider("Batch size", min_value=1, max_value=128, value=32)
        layers = st.slider("Number of layers", min_value=1, max_value=10, value=3)
        units = st.text_input("Units per layer (comma-separated)", value="128,64,32")
        units = [int(u.strip()) for u in units.split(',')]
        
        if model_type in ['LSTM', 'GRU', 'SimpleRNN']:
            timesteps = st.slider("Number of timesteps", min_value=1, max_value=30, value=1)
        else:
            timesteps = 1
        
        # Train and predict button
        if st.button("Train Model and Predict"):
            # Split data
            train_data = df[(df['date'].dt.date >= train_start_date) & (df['date'].dt.date <= train_end_date)]
            test_data = df[df['date'].dt.date > train_end_date]
            
            # Prepare features
            X_train = train_data[selected_features]
            y_train = train_data['daily_collection']
            X_test = test_data[selected_features]
            y_test = test_data['daily_collection']
            
            # Scale features
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
            X_test_scaled = scaler_X.transform(X_test)
            
            # Train model
            model, history = create_and_train_model(model_type, X_train_scaled, y_train_scaled, epochs, batch_size, layers, units, timesteps)
            
            # Make predictions
            if model_type in ['LSTM', 'GRU', 'SimpleRNN']:
                if X_test_scaled.shape[1] % timesteps != 0:
                    pad_width = timesteps - (X_test_scaled.shape[1] % timesteps)
                    X_test_scaled = np.pad(X_test_scaled, ((0, 0), (0, pad_width)), 'constant')
                X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], timesteps, -1))
            elif model_type == 'Conv1D':
                X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            else:
                X_test_reshaped = X_test_scaled

            predictions_scaled = model.predict(X_test_reshaped)
            predictions = scaler_y.inverse_transform(predictions_scaled)
            
            # Create DataFrame with predictions
            # Create DataFrame with predictions
            predictions_df = pd.DataFrame({
                'date': test_data['date'],
                'actual_sales': y_test,
                'predicted_sales': predictions.flatten(),
                'difference': y_test - predictions.flatten(),
                'accuracy': (1 - abs((y_test - predictions.flatten()) / y_test)) * 100
            })

            # Format numbers for display
            predictions_df['actual_sales_formatted'] = predictions_df['actual_sales'].map('{:,.2f}'.format)
            predictions_df['predicted_sales_formatted'] = predictions_df['predicted_sales'].map('{:,.2f}'.format)
            predictions_df['difference_formatted'] = predictions_df['difference'].map('{:+,.2f}'.format)
            predictions_df['accuracy_formatted'] = predictions_df['accuracy'].map('{:.2f}%'.format)            


            # Plot actual vs predicted sales
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=predictions_df['date'], y=predictions_df['actual_sales'], name='Actual Sales', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=predictions_df['date'], y=predictions_df['predicted_sales'], name='Predicted Sales', line=dict(color='red')))
            fig.update_layout(
                title='Actual vs Predicted Sales',
                xaxis_title='Date',
                yaxis_title='Sales',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display metrics
            # Calculate and display metrics
            mae = mean_absolute_error(predictions_df['actual_sales'], predictions_df['predicted_sales'])
            rmse = np.sqrt(mean_squared_error(predictions_df['actual_sales'], predictions_df['predicted_sales']))
            r2 = r2_score(predictions_df['actual_sales'], predictions_df['predicted_sales'])   
            mape = np.mean(np.abs((predictions_df['actual_sales'] - predictions_df['predicted_sales']) / predictions_df['actual_sales'])) * 100
            mpe = np.mean((predictions_df['actual_sales'] - predictions_df['predicted_sales']) / predictions_df['actual_sales']) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error", f"{mae:.2f}")
            col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
            col3.metric("R-squared Score", f"{r2:.4f}")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Mean Absolute Percentage Error", f"{mape:.2f}%")
            col5.metric("Mean Percentage Error", f"{mpe:+.2f}%")
            col6.metric("Accuracy", f"{100 - mape:.2f}%")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': np.abs(np.corrcoef(X_train_scaled.T, y_train_scaled.T)[0, 1:])
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h', 
                                    title='Feature Importance')
            fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Display training history
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss', line=dict(color='blue')))
            fig_history.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss', line=dict(color='red')))
            fig_history.update_layout(
                title='Training History',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_history, use_container_width=True)
            
            # Display predictions table
            st.subheader("Predictions Table")
            display_df = predictions_df[['date', 'actual_sales_formatted', 'predicted_sales_formatted', 'difference_formatted', 'accuracy_formatted']]
            display_df.columns = ['date', 'actual_sales', 'predicted_sales', 'difference', 'accuracy']
            st.dataframe(display_df.style.highlight_max(axis=0))

            # Download predictions
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="sales_predictions.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()