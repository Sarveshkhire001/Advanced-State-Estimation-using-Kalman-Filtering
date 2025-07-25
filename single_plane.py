import numpy as np
import plotly.graph_objects as go
from kalman_filter import KalmanFilter
from plotting_utils import plot_ellipse # Assuming this exists
from simulation_utils import initialize_plane_data, append_plane_data, process_plane_data, plot_single_plane_results

def single_plane(vel_flag=False, ellipse_flag=False, traj_type=1):
    # vel_flag = False
    # ellipse_flag = False
    
    X1 = np.array([[0],[0],[0],[0],[0],[0]])
    mu1 = np.copy(X1)
    sigma1 = np.diag([0.0001,0.0001,0.0001,0.0001,0.0001,0.0001])
    
    plane1 = KalmanFilter(X1,mu1,sigma1,1.2,0.01,7,1,traj_type=traj_type)
    
    # Initialize data lists using the utility function
    actual_trajectory1, gps_measurement1, predicted_measurement1_raw, predicted_velocities1, sigma1_array = initialize_plane_data()
    
    # Append initial state
    append_plane_data(
        (actual_trajectory1, gps_measurement1, predicted_measurement1_raw, predicted_velocities1, sigma1_array),
        plane1.state, plane1.mu, plane1.sigma, plane1.initial_measurement
    )
    
    iterations = 500
    for i in range(1,iterations):
        action = plane1.get_action(i)
        new_state = plane1.get_actual_state(action)
        mu_hat,sigma_hat = plane1.predict(action)
        new_measurement = plane1.get_gps_measurement(new_state)
        new_mu,new_sigma = plane1.update(mu_hat,sigma_hat,new_measurement)
        
        # Append data in the loop using the utility function
        append_plane_data(
            (actual_trajectory1, gps_measurement1, predicted_measurement1_raw, predicted_velocities1, sigma1_array),
            new_state, new_mu, new_sigma, new_measurement
        )
        
        plane1.state = new_state
        plane1.mu = new_mu
        plane1.sigma = new_sigma
        
    # Process data using the utility function
    processed_data = process_plane_data(actual_trajectory1, gps_measurement1, predicted_measurement1_raw, predicted_velocities1)

    # Plot results using the utility function
    plot_single_plane_results(processed_data, sigma1_array, vel_flag, ellipse_flag)
