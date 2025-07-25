
## Advanced State Estimation and Data Association in Autonomous Systems

This project implements and demonstrates various Kalman filtering techniques for state estimation in dynamic systems, with a focus on applications in autonomous systems. It covers the fundamental linear Kalman Filter for single-object tracking, extends to the Extended Kalman Filter (EKF) for non-linear scenarios like landmark localization, and addresses the critical challenge of data association in multi-object tracking environments.

The simulations are designed to showcase the Kalman filter's performance under diverse conditions, including varying noise parameters, intermittent sensor observations, and different control policies. The modular codebase allows for easy experimentation and visualization of key concepts such as uncertainty propagation and the impact of filter tuning.

### Project Structure
The project is organized into several Python files, each responsible for a specific aspect of the simulation or utility:

#### kalman_filter.py:

Defines the KalmanFilter class, which is the core implementation of both the linear Kalman Filter and the Extended Kalman Filter (EKF).

Includes methods for defining system dynamics (state transition, control input, measurement models), generating actual system states with process noise, simulating noisy sensor measurements (GPS, landmark distances), and executing the prediction and update steps of the Kalman filter.

It supports different control policies for the object's actual trajectory via the traj_type parameter.

#### simulation_utils.py:

Contains a collection of helper functions designed to streamline the simulation process across different scenarios.

Provides utilities for initializing data storage lists, appending simulation results (actual states, measurements, predicted states, covariances), processing raw collected data into NumPy arrays suitable for plotting, and generic plotting functions for both single and multiple planes.

#### plotting_utils.py:

(Assumed to contain helper functions for plotting, such as plot_ellipse, which is used to visualize uncertainty ellipses in 2D projections).

#### brute_force.py:

(Assumed to contain the brute_force function, which implements a brute-force approach for data association in multi-object tracking scenarios).

#### single_plane.py:

Implements a simulation scenario focusing on tracking a single plane using the standard linear Kalman Filter.

Demonstrates basic trajectory estimation and uncertainty propagation.

#### missing_gps.py:

Simulates a single plane tracking scenario where GPS measurements are intermittently available.

Highlights the Kalman Filter's ability to maintain an estimate during periods of observation loss and its recovery upon receiving new data.

#### many_planes.py:

Demonstrates multi-object tracking by simulating several planes simultaneously.

Incorporates data association algorithms (like the brute_force method) to correctly match incoming measurements to the respective plane tracks.

#### landmark_data.py:

Implements a simulation using the Extended Kalman Filter (EKF) for a single plane.

Focuses on landmark localization, where the plane uses non-linear distance measurements from fixed "Air Traffic Control (ATC) towers" to improve its position estimate.

#### main.py:

The primary entry point for executing the simulations.

Utilizes argparse to allow users to select which simulation to run and configure various plotting and trajectory options via command-line arguments.

### Environment Setup
To set up the development environment using Conda, follow these steps:

**Create a Conda environment:**bash
conda create -n kalman_env python=3.9

(You can choose a different Python version if preferred, e.g., `python=3.8` or `python=3.10`).

#### Activate the environment:

conda activate kalman_env

#### Install dependencies:

pip install numpy plotly
(Ensure plotting_utils.py and brute_force.py are present in your project directory. If they have additional external dependencies, install them as needed).

#### Usage
The main.py script is designed to be executed from the command line, allowing you to specify which simulation to run and configure various parameters.

#### General Command Structure

python main.py --simulation <name>[--traj_type <0/1>]

#### Arguments

--simulation <name>:

Required. Specifies which simulation script to execute.

Choices: single_plane, missing_gps, many_planes, landmark_data.
Default: single_plane.

--vel_flag <True/False>:
Optional. If set to True, the plots will display velocities instead of positions.
Default: False.

--ellipse_flag <True/False>:
Optional. If set to True, the plots will include uncertainty ellipses (for 2D projections).
Default: False.

--traj_type <0/1>:

Optional. Determines the control policy for the actual trajectory of the simulated object(s).

0: Uses the control policy ut = [0.128 * cos(0.032t), 0.128 * sin(0.032t), 0.01]^T.
1: Uses the control policy ut = [sin(t), cos(t), sin(t)]^T.
Default: 1.

Examples

Run the single_plane simulation with default settings:
    python main.py --simulation single_plane

Run the missing_gps simulation, showing uncertainty ellipses and using trajectory type 0:
    python main.py --simulation missing_gps --ellipse_flag True --traj_type 0

Run the landmark_data simulation, displaying velocity plots:
    python main.py --simulation landmark_data --vel_flag True

Run the many_planes simulation, showing uncertainty ellipses and using trajectory type 0:
    python main.py --simulation many_planes --ellipse_flag True --traj_type 0

Run the single_plane simulation, displaying velocity plots:
    python main.py --simulation single_plane --vel_flag True


Feel free to experiment with different combinations of arguments to explore the various aspects of the Kalman filter and its applications.