from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

app = Flask(__name__)


# Load your dataset
# Assuming df is your DataFrame (as processed in your provided code)
# Make sure to sort the DataFrame by Date to ensure accurate predictions
df = pd.read_csv('c://users//sufi//yahoo_data.csv')
df = df.sort_values(by='Date')

# Assuming df is your DataFrame with the provided dataset
# Convert 'Date' to datetime and set it as the index
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
df.set_index('Date', inplace=True)

# Define features (X) and target variables (y)
X = df[['Close*']]

# Try to get 'MA10', if not found set it to None
y_ma10 = df['MA10'] if 'MA10' in df.columns else None

# Try to get 'MA20', if not found set it to None
y_ma20 = df['MA20'] if 'MA20' in df.columns else None

# Check if 'y_ma10' and 'y_ma20' are not None before proceeding
if y_ma10 is not None and y_ma20 is not None:
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize the KNN regressor for MA10
    knn_ma10 = KNeighborsRegressor(n_neighbors=5)
    knn_ma10.fit(X_scaled, y_ma10)

    # Initialize the KNN regressor for MA20
    knn_ma20 = KNeighborsRegressor(n_neighbors=5)
    knn_ma20.fit(X_scaled, y_ma20)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            date_str = request.form['date']
            close_value = float(request.form['close'])

            # Convert the input date to datetime
            input_date = pd.to_datetime(date_str, format='%Y-%m-%d')

            # Prepare input data for prediction
            input_data = pd.DataFrame({'Close*': [close_value]}, index=[input_date])
            
            # Check if 'y_ma10' and 'y_ma20' are not None before proceeding
            if y_ma10 is not None and y_ma20 is not None:
                input_data_scaled = scaler.transform(input_data)

                # Make predictions for MA10 and MA20
                predicted_ma10 = knn_ma10.predict(input_data_scaled)[0]
                predicted_ma20 = knn_ma20.predict(input_data_scaled)[0]

                return render_template('index.html', date=date_str, close=close_value, ma10=predicted_ma10, ma20=predicted_ma20)
            else:
                return render_template('index.html', error="Error: 'MA10' or 'MA20' column not found in DataFrame.")
        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', error=f"Internal error: {e}")

if __name__ == '__main__':
    app.run(debug=True)





