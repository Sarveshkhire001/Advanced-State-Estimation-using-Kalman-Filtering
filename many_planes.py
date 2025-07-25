import numpy as np
import plotly.graph_objects as go
from kalman_filter import KalmanFilter
from plotting_utils import plot_ellipse
from brute_force import brute_force # Assuming this exists
from simulation_utils import initialize_plane_data, append_plane_data, process_plane_data, initialize_multiple_plane_data, plot_many_planes_results

def many_planes(vel_flag=False, ellipse_flag=False, traj_type=1):
    # vel_flag = False
    # ellipse_flag = False
    
    # Initial states for four planes
    X_initials = [
        np.array([[0],[0],[0],[0],[0],[0]]),
        np.array([[200],[0],[0],[0],[2],[0]]),
        np.array([[0],[300],[0],[4],[0],[0]]),
        np.array([[-150],[250],[0],[1],[1],[0]])
    ]
    
    # Initial mu (estimated state) for each plane, copy from initial X
    mu_initials = [np.copy(X) for X in X_initials]
    
    # Initial sigma (covariance) for each plane
    sigma_initials = [
        np.diag([0.0001]*6),
        np.diag([0.0002]*6),
        np.diag([0.0003]*6),
        np.diag([0.0004]*6)
    ]
    
    # Create KalmanFilter instances for each plane
    planes = [
        KalmanFilter(X_initials[0], mu_initials[0], sigma_initials[0], 1.2, 0.01, 7, 1, traj_type=traj_type),
        KalmanFilter(X_initials[1], mu_initials[1], sigma_initials[1], 1.2, 0.01, 7, 1, traj_type=traj_type),
        KalmanFilter(X_initials[2], mu_initials[2], sigma_initials[2], 1.2, 0.01, 7, 1, traj_type=traj_type),
        KalmanFilter(X_initials[3], mu_initials[3], sigma_initials[3], 1.2, 0.01, 7, 1, traj_type=traj_type)
    ]
    
    num_planes = len(planes)
    
    # Initialize data lists for all planes using the utility function
    all_planes_data = initialize_multiple_plane_data(num_planes)
    
    # Separate lists for easier access (references to the lists within all_planes_data)
    actual_trajectories = [data[0] for data in all_planes_data]
    gps_measurements = [data[1] for data in all_planes_data]
    predicted_measurements_raw = [data[2] for data in all_planes_data]
    predicted_velocities = [data[3] for data in all_planes_data]
    sigma_arrays = [data[4] for data in all_planes_data]

    # Append initial states for all planes
    for i, plane in enumerate(planes):
        append_plane_data(
            all_planes_data[i],
            plane.state, plane.mu, plane.sigma, plane.initial_measurement # Initial measurement is just position
        )
    
    brute_force_flag = True # Keep original flag
    
    iterations = 500
    for i in range(1, iterations):
        current_gps_data = []
        new_states = []
        mu_hats = [] # Store mu_hat for potential update later
        sigma_hats = [] # Store sigma_hat for potential update later

        # Predict and get actual GPS measurements for all planes
        for j, plane in enumerate(planes):
            action = plane.get_action(i)
            new_state = plane.get_actual_state(action)
            mu_hat, sigma_hat = plane.predict(action)
            gps_measurement = plane.get_gps_measurement(new_state) # Always get GPS measurement
            
            new_states.append(new_state)
            mu_hats.append(mu_hat)
            sigma_hats.append(sigma_hat)
            current_gps_data.append(gps_measurement)

        # Apply brute force (or other assignment) if enabled
        if brute_force_flag:
            # brute_force expects planes (KalmanFilter objects) and gps_data (list of measurements)
            # It returns reassigned gps_data
            reassigned_gps_data = brute_force(planes, np.array(current_gps_data))
        else:
            reassigned_gps_data = current_gps_data # No reassignment

        # Update each plane with its assigned measurement and append data
        for j, plane in enumerate(planes):
            assigned_gps = reassigned_gps_data[j]
            # Update the plane's state using the assigned GPS measurement
            updated_mu, updated_sigma = plane.update(mu_hats[j], sigma_hats[j], assigned_gps)
            
            plane.state = new_states[j] # Update true state
            plane.mu = updated_mu # Update filter's estimated mean
            plane.sigma = updated_sigma # Update filter's estimated covariance
            
            # Append data for this plane
            append_plane_data(
                all_planes_data[j],
                new_states[j], updated_mu, updated_sigma, assigned_gps
            )
    
    # Process data for all planes
    all_processed_data = []
    for i in range(num_planes):
        processed = process_plane_data(
            actual_trajectories[i],
            gps_measurements[i],
            predicted_measurements_raw[i],
            predicted_velocities[i]
        )
        all_processed_data.append(processed)

    # Plot results for multiple planes
    plot_many_planes_results(all_processed_data, sigma_arrays, vel_flag, ellipse_flag)

