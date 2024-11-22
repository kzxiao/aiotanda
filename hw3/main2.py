import streamlit as st
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@st.cache_data
def generate_data():
    # Parameters
    num_points = 600
    mean = 0
    variance = 10

    # Data generation
    x1 = np.random.normal(mean, np.sqrt(variance), num_points)
    x2 = np.random.normal(mean, np.sqrt(variance), num_points)
    distances = np.sqrt(x1**2 + x2**2)

    # Gaussian function
    x3 = np.exp(-(x1**2 + x2**2) / (2 * variance))

    return x1, x2, distances, x3

x1, x2, distances, x3 = generate_data();

# Streamlit app title
st.title("3D Scatter Plot with Linear Hyperplane")

col1, col2 = st.columns([1, 2])

with col1:
    # Slider for the user to adjust the classification distance threshold
    threshold = st.slider("Classification Distance Threshold (Radius)", min_value=1.0, max_value=10.0, step=0.5, value=4.0)

with col2:
    # Assign labels based on the user-defined threshold
    y = np.where(distances < threshold, 0, 1)


    # Train the linear classifier
    X = np.column_stack((x1, x2, x3))
    clf = LinearSVC(random_state=0, max_iter=10000)
    clf.fit(X, y)
    w = clf.coef_[0]
    b = clf.intercept_

    # Create the hyperplane
    x1_mesh, x2_mesh = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                                np.linspace(min(x2), max(x2), 10))
    x3_plane = (-w[0] * x1_mesh - w[1] * x2_mesh - b) / w[2]

    # 3D Scatter Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot classified points
    ax.scatter(x1[y == 0], x2[y == 0], x3[y == 0], c='blue', marker='o', label='y=0')
    ax.scatter(x1[y == 1], x2[y == 1], x3[y == 1], c='red', marker='s', label='y=1')

    # Plot the hyperplane
    ax.plot_surface(x1_mesh, x2_mesh, x3_plane, color='gray', alpha=0.3)

    # Add labels and title
    ax.set_title("3D Scatter Plot with Linear Hyperplane")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3 = f(x1, x2)")
    ax.set_box_aspect(None, zoom=0.85)
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)
