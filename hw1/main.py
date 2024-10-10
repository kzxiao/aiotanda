import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
import pandas as pd
import altair as alt


# Streamlit app starts here
st.title("Interactive Linear Regression with Streamlit")

# Create two columns: left for sliders, right for scatter plot
col1, col2 = st.columns([1, 2])

# In the first column (left), add the sliders for user input
with col1:
    st.header("Adjust Parameters")
    
    # 1. Add sliders for user inputs (a, c, and n)
    a = st.slider("Slope (a):", -10.0, 10.0, 2.5)  # Slope of the line
    c = st.slider("Noise level (C):", 0.0, 100.0, 2.0)  # Level of noise
    n = st.slider("Number of data points (n):", 10, 500, 100)  # Number of data points

# In the second column (right), show the scatter plot with regression line
with col2:
    # 2. Generate random linear data based on user input
    np.random.seed(42)
    X = np.random.rand(n, 1) * 10  # n random points between 0 and 10
    y = a * X + 50 + c * np.random.randn(n, 1)  # y = a * X + 50 + noise

    # 3. Convert to DataFrame for visualization
    data = pd.DataFrame(np.hstack((X, y)), columns=['X', 'y'])

    # 4. Fit the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # 5. Create the regression line using the full range of X
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)  # Full range of X values for plotting the regression line
    y_range_pred = model.predict(X_range)  # Corresponding predictions (y values)

    # 6. Prepare data for the plot
    # Data for scatter plot
    scatter_data = pd.DataFrame({
        'X': X.flatten(), 
        'y': y.flatten()
    })

    # Data for regression line
    regression_line = pd.DataFrame({
        'X': X_range.flatten(),
        'y': y_range_pred.flatten()
    })

    # 7. Create Altair chart: Scatter plot with regression line
    st.write("### Scatter Plot with Regression Line")

    # Scatter plot for data points
    scatter_chart = alt.Chart(scatter_data).mark_circle(color='blue', size=60).encode(
        x='X',
        y='y'
    )

    # Line plot for regression line
    regression_chart = alt.Chart(regression_line).mark_line(color='red', size=2).encode(
        x='X',
        y='y'
    )

    # Combine both charts: scatter + regression line
    combined_chart = scatter_chart + regression_chart

    # Display the combined chart in Streamlit
    st.altair_chart(combined_chart, use_container_width=True)
