import numpy as np
import plotly.graph_objects as go
from kalman_filter import KalmanFilter
from plotting_utils import plot_ellipse
from simulation_utils import initialize_plane_data, append_plane_data, process_plane_data, plot_single_plane_results

def landmark_data(vel_flag=False, ellipse_flag=False, traj_type=1):
    
    # Initialize data lists using the utility function
    actual_trajectory, gps_measurement, predicted_trajectory_raw, predicted_velocities, sigma1_array = initialize_plane_data()

    landmarks_array = np.array([[150,0,100],[-150,0,100],[0,150,100],[0,-150,100]])
    
    state = np.array([[100],[0],[0],[0],[4],[0]])
    initial_measurement = np.array([[100],[0],[0]])
    mu = np.copy(state)
    sigma = np.diag([0.0001]*6) # More concise way to create diagonal matrix
    
    plane = KalmanFilter(state,mu,sigma,0.01,0.01,10,0.1, traj_type=traj_type)
    
    # Append initial state
    append_plane_data(
        (actual_trajectory, gps_measurement, predicted_trajectory_raw, predicted_velocities, sigma1_array),
        plane.state, plane.mu, plane.sigma, plane.initial_measurement # Pass initial measurement
    )
    
    for i in range(1,201):
        action = plane.get_action(i)
        new_state = plane.get_actual_state(action)
        new_measurement = plane.get_gps_measurement(new_state) # GPS measurement always available
        
        mu_hat,sigma_hat = plane.predict(action)
        
        d,n,H = plane.get_landmark_distance(new_state)
        
        if (d <= 90): # If within landmark influence sphere
            new_combined_z = plane.get_gps_with_landmark(new_state,d)
            new_mu,new_sigma = plane.update_with_landmark(mu_hat,sigma_hat,new_combined_z,H,n)
            # For plotting purposes, we'll still record the GPS part of the combined measurement
            # as the 'gps_measurement' for consistency with other files.
            # The landmark data affects the update, but not the raw GPS measurement itself.
            gps_to_record = new_combined_z[:3,:] 
        else: # Otherwise, use only GPS
            new_mu,new_sigma = plane.update(mu_hat,sigma_hat,new_measurement)
            gps_to_record = new_measurement
        
        plane.state = new_state
        plane.mu = new_mu
        plane.sigma = new_sigma
        
        # Append data using the utility function, passing the GPS measurement (or the GPS part of combined)
        append_plane_data(
            (actual_trajectory, gps_measurement, predicted_trajectory_raw, predicted_velocities, sigma1_array),
            new_state, new_mu, new_sigma, gps_to_record
        )
        
    # Process data using the utility function
    processed_data = process_plane_data(actual_trajectory, gps_measurement, predicted_trajectory_raw, predicted_velocities)

    ellipse_flag = False
    vel_flag = False

    # Plot results using the utility function, passing landmarks_array
    plot_single_plane_results(processed_data, sigma1_array, vel_flag, ellipse_flag, landmarks_array=landmarks_array)

