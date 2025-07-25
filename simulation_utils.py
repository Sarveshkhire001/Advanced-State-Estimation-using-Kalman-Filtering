# simulation_utils.py
import numpy as np
import plotly.graph_objects as go
from plotting_utils import plot_ellipse # Assuming this exists and is correct

def initialize_plane_data():
    """Initializes and returns empty lists for storing simulation data for a single plane."""
    actual_trajectory = []
    gps_measurement = [] # This will store actual measurements or None/NaN
    predicted_measurement = []
    predicted_velocities = []
    sigma_array = []
    return actual_trajectory, gps_measurement, predicted_measurement, predicted_velocities, sigma_array

def initialize_multiple_plane_data(num_planes):
    """
    Initializes and returns a list of data list tuples for multiple planes.
    Each element in the list corresponds to a plane and contains:
    (actual_trajectory, gps_measurement, predicted_measurement, predicted_velocities, sigma_array)
    """
    all_planes_data = []
    for _ in range(num_planes):
        all_planes_data.append(initialize_plane_data())
    return all_planes_data

def append_plane_data(data_lists, new_state, new_mu, new_sigma, new_measurement=None):
    """
    Appends new simulation data to the provided lists for a single plane.
    new_measurement can be None if no measurement is available for the step.
    """
    actual_trajectory, gps_measurement, predicted_measurement, predicted_velocities, sigma_array = data_lists
    actual_trajectory.append(new_state)
    predicted_measurement.append(new_mu[:3,:])
    predicted_velocities.append(new_mu[3:,:])
    sigma_array.append(new_sigma)
    
    # Handle GPS measurement conditionally
    if new_measurement is not None:
        gps_measurement.append(new_measurement)
    else:
        # Append a placeholder for missing measurements to maintain list length
        gps_measurement.append(np.full((3, 1), np.nan)) # Use NaN for missing data

def process_plane_data(actual_trajectory_list, gps_measurement_list, predicted_measurement_list, predicted_velocities_list):
    """
    Converts lists of simulation data for a single plane into NumPy arrays
    and extracts position and velocity components.
    Returns a dictionary containing these processed arrays.
    Handles NaN values in gps_measurement_list by filtering them out for plotting.
    """
    actual_trajectory = np.array(actual_trajectory_list)
    predicted_measurement = np.array(predicted_measurement_list)
    predicted_velocities = np.array(predicted_velocities_list)

    # Extract x, y, z values for actual trajectory
    x_values = actual_trajectory[:, 0, 0]
    y_values = actual_trajectory[:, 1, 0]
    z_values = actual_trajectory[:, 2, 0]
    
    # Extract true velocities
    xd_values = actual_trajectory[:, 3, 0]
    yd_values = actual_trajectory[:, 4, 0]
    zd_values = actual_trajectory[:, 5, 0]
    
    # Process GPS measurement values, filtering out NaNs
    gps_measurement_arr = np.array(gps_measurement_list)
    # Create boolean mask for non-NaN rows (where actual measurement was taken)
    valid_gps_mask = ~np.isnan(gps_measurement_arr[:, 0, 0])
    
    z_x_values = gps_measurement_arr[valid_gps_mask, 0, 0]
    z_y_values = gps_measurement_arr[valid_gps_mask, 1, 0]
    z_z_values = gps_measurement_arr[valid_gps_mask, 2, 0]
    
    # Extract predicted position values
    pred_x_values = predicted_measurement[:,0,0]
    pred_y_values = predicted_measurement[:,1,0]
    pred_z_values = predicted_measurement[:,2,0]
    
    # Extract predicted velocity values
    pred_xd_values = predicted_velocities[:,0,0]
    pred_yd_values = predicted_velocities[:,1,0]
    pred_zd_values = predicted_velocities[:,2,0]

    return {
        'actual_x': x_values, 'actual_y': y_values, 'actual_z': z_values,
        'actual_xd': xd_values, 'actual_yd': yd_values, 'actual_zd': zd_values,
        'gps_x': z_x_values, 'gps_y': z_y_values, 'gps_z': z_z_values,
        'pred_x': pred_x_values, 'pred_y': pred_y_values, 'pred_z': pred_z_values,
        'pred_xd': pred_xd_values, 'pred_yd': pred_yd_values, 'pred_zd': pred_zd_values,
        'predicted_measurement_raw': predicted_measurement_list # Keep raw for ellipse plotting
    }

def plot_single_plane_results(processed_data, sigma_array, vel_flag, ellipse_flag, landmarks_array=None):
    """
    Generates and displays a Plotly 3D trajectory or velocity plot for a single plane.
    Includes uncertainty ellipses if ellipse_flag is True.
    """
    fig = go.Figure()

    if not ellipse_flag:
        if not vel_flag:
            fig.add_trace(go.Scatter3d(x=processed_data['actual_x'], y=processed_data['actual_y'], z=processed_data['actual_z'], mode='lines', name='Actual_Trajectory'))
            # Only plot GPS measurements that are not NaN
            if processed_data['gps_x'].size > 0: # Check if there are valid GPS measurements
                fig.add_trace(go.Scatter3d(x=processed_data['gps_x'], y=processed_data['gps_y'], z=processed_data['gps_z'], mode='lines', name='Measurements'))
            fig.add_trace(go.Scatter3d(x=processed_data['pred_x'], y=processed_data['pred_y'], z=processed_data['pred_z'], mode='lines', name='Predicted_Trajectory'))
            
            if landmarks_array is not None:
                fig.add_trace(go.Scatter3d(x=[loc[0] for loc in landmarks_array], y=[loc[1] for loc in landmarks_array], z=[loc[2] for loc in landmarks_array], mode='markers', marker=dict(size=3, color='black'), name='ATC Locations'))
                fig.add_trace(go.Scatter3d(x=[loc[0] for loc in landmarks_array], y=[loc[1] for loc in landmarks_array], z=[loc[2] for loc in landmarks_array], mode='markers', marker=dict(size=90, color='blue'), name='ATC Influence sphere', opacity=0.1))
        else:
            fig.add_trace(go.Scatter3d(x=processed_data['actual_xd'], y=processed_data['actual_yd'], z=processed_data['actual_zd'], mode='lines', name='True Velocities'))
            fig.add_trace(go.Scatter3d(x=processed_data['pred_xd'], y=processed_data['pred_yd'], z=processed_data['pred_zd'], mode='lines', name='Predicted_Velocity'))
    else:
        # For ellipse plotting, it's usually 2D projection
        fig.add_trace(go.Scatter(x=processed_data['pred_x'], y=processed_data['pred_y'], mode='lines', name='Predicted_Trajectory'))
        for i in range(len(processed_data['predicted_measurement_raw'])): # Use raw list for iteration
            plot_ellipse(fig, processed_data['predicted_measurement_raw'][i][:2,0], sigma_array[i][:2,:2], 'rgba(255,0,0,0.3)', f'Uncertainty Ellipse - Step {i}')

    fig.update_layout(title='3D Trajectory Plot (x, y, z)',
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    fig.show()

def plot_many_planes_results(all_processed_data, all_sigma_arrays, vel_flag, ellipse_flag):
    """
    Generates and displays a Plotly 3D trajectory or velocity plot for multiple planes.
    Includes uncertainty ellipses if ellipse_flag is True.
    """
    fig = go.Figure()

    num_planes = len(all_processed_data)
    colors = ['blue', 'red', 'green', 'purple'] # Define colors for different planes

    if not ellipse_flag:
        for i in range(num_planes):
            processed_data = all_processed_data[i]
            if not vel_flag:
                fig.add_trace(go.Scatter3d(x=processed_data['actual_x'], y=processed_data['actual_y'], z=processed_data['actual_z'], mode='lines', name=f'Actual_Trajectory{i+1}', line=dict(color=colors[i])))
                # Only plot GPS measurements that are not NaN
                if processed_data['gps_x'].size > 0:
                    fig.add_trace(go.Scatter3d(x=processed_data['gps_x'], y=processed_data['gps_y'], z=processed_data['gps_z'], mode='lines', name=f'Measurements{i+1}', line=dict(color=colors[i], dash='dash')))
                fig.add_trace(go.Scatter3d(x=processed_data['pred_x'], y=processed_data['pred_y'], z=processed_data['pred_z'], mode='lines', name=f'Predicted_Trajectory{i+1}', line=dict(color=colors[i], dash='dot')))
            else:
                # Velocity plots for multiple planes (not implemented in original, but good to have)
                fig.add_trace(go.Scatter3d(x=processed_data['actual_xd'], y=processed_data['actual_yd'], z=processed_data['actual_zd'], mode='lines', name=f'True Velocities{i+1}', line=dict(color=colors[i])))
                fig.add_trace(go.Scatter3d(x=processed_data['pred_xd'], y=processed_data['pred_yd'], z=processed_data['pred_zd'], mode='lines', name=f'Predicted_Velocity{i+1}', line=dict(color=colors[i], dash='dot')))
    else:
        # Ellipse plotting for multiple planes
        for i in range(num_planes):
            processed_data = all_processed_data[i]
            sigma_array = all_sigma_arrays[i]
            fig.add_trace(go.Scatter(x=processed_data['pred_x'], y=processed_data['pred_y'], mode='lines', name=f'Predicted_Trajectory{i+1}', line=dict(color=colors[i])))
            for j in range(len(processed_data['predicted_measurement_raw'])):
                # Using a random color for each ellipse for distinction, could be more systematic
                plot_ellipse(fig, processed_data['predicted_measurement_raw'][j][:2,0], sigma_array[j][:2,:2], f'rgba({np.random.randint(0,256)},{np.random.randint(0,256)},{np.random.randint(0,256)},0.3)', f'Uncertainty Ellipse Plane {i+1} - Step {j}')

    fig.update_layout(title='3D Trajectory Plot (x, y, z)',
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    fig.show()
