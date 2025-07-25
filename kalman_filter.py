# This code implements a Kalman Filter for state estimation of a system.
# It simulates a plane's movement in 3D space, incorporating process noise and measurement noise.
# The filter uses GPS measurements and can optionally incorporate landmark measurements for improved accuracy.

import numpy as np
import plotly.graph_objects as go
from itertools import permutations

class KalmanFilter:
    def __init__(self,state,mu,sigma,r1,r2,q1,s,traj_type=1):
        # A: State transition matrix. Defines how the state evolves from one time step to the next.
        # Here, it represents a constant velocity model in 3D.
        # The first three rows correspond to position (x, y, z) and the last three to velocity (vx, vy, vz).
        # Position at t+1 = Position at t + Velocity at t
        # Velocity at t+1 = Velocity at t
        self.A = np.asarray([[1,0,0,1,0,0],
                             [0,1,0,0,1,0],
                             [0,0,1,0,0,1],
                             [0,0,0,1,0,0],
                             [0,0,0,0,1,0],
                             [0,0,0,0,0,1]])
        # B: Control input matrix. Maps control inputs to changes in the state.
        # Here, control inputs directly affect the velocity components.
        self.B = np.asarray([[0,0,0],
                             [0,0,0],
                             [0,0,0],
                             [1,0,0],
                             [0,1,0],
                             [0,0,1]])
        # C: Measurement matrix. Maps the state space to the measurement space.
        # Here, it indicates that GPS measurements provide direct observations of position (x, y, z).
        self.C = np.asarray([[1,0,0,0,0,0],
                             [0,1,0,0,0,0],
                             [0,0,1,0,0,0]])
        # R: Process noise covariance matrix. Represents the uncertainty in the system's dynamics.
        # r1 is the standard deviation for position noise, r2 for velocity noise.
        self.R = np.asarray([[r1**2,0,0,0,0,0],
                             [0,r1**2,0,0,0,0],
                             [0,0,r1**2,0,0,0],
                             [0,0,0,r2**2,0,0],
                             [0,0,0,0,r2**2,0],
                             [0,0,0,0,0,r2**2]])
        # Q1: Measurement noise covariance matrix for combined GPS and landmark measurements.
        # q1 is the standard deviation for GPS position noise, s for landmark distance noise.
        self.Q1 = np.asarray([[q1**2,0,0,0],
                             [0,q1**2,0,0],
                             [0,0,q1**2,0],
                             [0,0,0,s**2]])
        # Q2: Measurement noise covariance matrix for GPS measurements only.
        # q1 is the standard deviation for GPS position noise.
        self.Q2 = np.asarray([[q1**2,0,0],
                             [0,q1**2,0],
                             [0,0,q1**2]])
        
        # Mean of the process noise (assumed to be zero).
        self.process_mean = np.array([[0],[0],[0],[0],[0],[0]])
        # Mean of the GPS measurement noise (assumed to be zero).
        self.GPSmeasurement_mean = np.array([[0],[0],[0]])
        # Initial true state of the system (position and velocity).
        self.state = state
        # Initial estimated state (mean of the state estimate).
        self.mu = mu
        # Initial state covariance matrix (uncertainty in the initial state estimate).
        self.sigma = sigma
        # Initial GPS measurement (position part of the initial state).
        self.initial_measurement = state[:3]
        # Coordinates of the known landmarks.
        self.landmarks = np.array([[150,0,100],[-150,0,100],[0,150,100],[0,-150,100]])
        # Mean of the landmark distance measurement noise (assumed to be zero).
        self.landmark_mean = np.array([0])
        # Standard deviation of the landmark distance measurement noise.
        self.S = np.array([s])
        # Time values for specific measurement intervals.
        self.t_values = np.concatenate([np.arange(start, end + 1) for start, end in zip(range(1, 501, 100), range(50, 451, 100))])
        
        # Flag to switch between different control input functions.
        self.u_flag = traj_type 
        
    def get_action(self,t):
        """
        Calculates the control input (action) at a given time t.
        Switches between two predefined control input patterns based on self.u_flag.
        """
        if self.u_flag == 1:
            u = np.array([[np.sin(t)],[np.cos(t)],[np.sin(t)]])
        else:
            u = np.array([[-0.128*np.cos(0.032*t)],
                          [-0.128*np.sin(0.032*t)],
                          [0.01]])
        return u
            
    def get_actual_state(self,action):
        """
        Simulates the actual evolution of the system's state.
        Applies the state transition, control input, and adds process noise.
        """
        process_noise = np.random.multivariate_normal(mean = self.process_mean.flatten(), cov = self.R, size=1).reshape(-1,1)
        new_state = np.dot(self.A,self.state) + np.dot(self.B,action) + process_noise
        return new_state
            
    def get_gps_measurement(self,x):
        """
        Generates a GPS measurement based on the true state x.
        Applies the measurement matrix and adds GPS measurement noise.
        """
        measurement_noise = np.random.multivariate_normal(mean = self.GPSmeasurement_mean.flatten(), cov = self.Q2, size=1).reshape(-1,1)
        z = np.dot(self.C,x) + measurement_noise
        return z
    
    def get_gps_with_landmark(self,state,d):
        """
        Generates a combined GPS and landmark distance measurement.
        Combines GPS measurement with a noisy distance measurement to the closest landmark.
        """
        gps_noise = np.random.multivariate_normal(self.GPSmeasurement_mean.flatten(), self.Q2, size=1).reshape(-1,1)
        landmark_noise = np.random.normal(self.landmark_mean, self.S, size=1).reshape(-1,1)
        gps = np.dot(self.C,state) + gps_noise
        d += landmark_noise
        z = np.vstack((gps,d)) # Stack GPS coordinates and landmark distance
        return z
    
    def get_landmark_distance(self,gps):
        """
        Calculates the distance to the closest landmark from a given GPS position.
        Also computes the Jacobian matrix (H) for the landmark measurement.
        """
        distances = []
        x = gps[0,0]
        y = gps[1,0]
        z = gps[2,0]
        for landmark in self.landmarks:
            l_x = landmark[0]
            l_y = landmark[1]
            l_z = landmark[2]
            d = np.sqrt(np.square(x - l_x) + np.square(y - l_y) + np.square(z - l_z))
            distances.append(d)

        cur_landmark = self.landmarks[np.argmin(distances)] # Identify the closest landmark
        d = np.min(distances) # Minimum distance
        lx,ly,lz = cur_landmark[0],cur_landmark[1],cur_landmark[2]
        tx,ty,tz = (x-lx)/d,(y-ly)/d,(z-lz)/d # Derivatives for the Jacobian
        
        # H: Jacobian of the measurement function with respect to the state.
        # The first three rows correspond to GPS measurements (direct observation of position).
        # The fourth row corresponds to the landmark distance measurement,
        # where tx, ty, tz are the partial derivatives of the distance with respect to x, y, z respectively.
        H = np.array([[1,0,0,0,0,0],
                      [0,1,0,0,0,0],
                      [0,0,1,0,0,0],
                      [tx,ty,tz,0,0,0]])
        return np.min(distances),np.argmin(distances),H
    
    def hofMu(self,mu_hat,n):
        """
        Calculates the expected measurement h(mu_hat) for combined GPS and landmark measurements.
        This is the non-linear measurement function linearized for the Extended Kalman Filter.
        """
        temp = mu_hat[:3,0].reshape(3,1) # Expected GPS coordinates
        x,y,z = mu_hat[0,0],mu_hat[1,0],mu_hat[2,0]
        lx,ly,lz = self.landmarks[n,0],self.landmarks[n,1],self.landmarks[n,2]
        d_arr = np.array([(x-lx),(y-ly),(z-lz)])
        d = np.linalg.norm(d_arr) # Expected distance to the nth landmark
        d_arr = np.array([d])
        h = np.vstack((temp,d_arr)) # Stack expected GPS and landmark distance
        return h
    
    def predict(self,action):
        """
        Performs the prediction step of the Kalman Filter.
        Estimates the next state (mu_hat) and its covariance (sigma_hat) based on the current state and control input.
        """
        mu_hat = np.dot(self.A,self.mu) + np.dot(self.B,action) # Predicted state mean
        sigma_hat = np.dot(self.A, np.dot(self.sigma,self.A.T)) + self.R # Predicted state covariance
        return mu_hat,sigma_hat
    
    def update(self,mu_hat,sigma_hat,Z):
        """
        Performs the update step of the Kalman Filter using only GPS measurements.
        Corrects the predicted state and covariance based on the actual measurement Z.
        """
        # Calculate Kalman Gain K
        denom = np.linalg.inv(np.dot(self.C, np.dot(sigma_hat,self.C.T)) + self.Q2)
        K = np.dot(np.dot(sigma_hat,self.C.T),denom)
        
        # Update state estimate
        temp = Z - np.dot(self.C, mu_hat)
        mu = mu_hat + np.dot(K, temp)
        
        # Update state covariance
        temp2 = np.eye(6) - np.dot(K,self.C)
        sigma = np.dot(temp2, sigma_hat)
        return mu,sigma
    
    def update_with_landmark(self,mu_hat,sigma_hat,Z,H,n):
        """
        Performs the update step of the Extended Kalman Filter using combined GPS and landmark measurements.
        Corrects the predicted state and covariance based on the actual measurement Z and Jacobian H.
        """
        # Calculate Kalman Gain K
        denom = np.linalg.inv(np.dot(H, np.dot(sigma_hat, H.T)) + self.Q1)
        K = np.dot(np.dot(sigma_hat,H.T),denom)
        
        # Calculate expected measurement h(mu_hat)
        h = self.hofMu(mu_hat,n)
        
        # Update state estimate
        temp = Z - h
        mu = mu_hat + np.dot(K, temp)
        
        # Update state covariance
        temp2 = np.eye(6) - np.dot(K,H)
        sigma = np.dot(temp2, sigma_hat)
        return mu,sigma
    
    def estimate(self,t):
        """
        Performs a full prediction and update cycle of the Kalman Filter for a given time step t.
        Simulates actual state, gets a GPS measurement, and then updates the filter's estimate.
        """
        action = self.get_action(t) # Get control input
        new_state = self.get_actual_state(action) # Simulate true state
        mu_hat,sigma_hat = self.predict(action) # Predict
        new_measurement = self.get_gps_measurement(new_state) # Get GPS measurement
        new_mu,new_sigma = self.update(mu_hat,sigma_hat,new_measurement) # Update
        
        self.state = new_state # Update true state for next iteration
        self.mu = new_mu # Update filter's estimated mean
        self.sigma = new_sigma # Update filter's estimated covariance
        return new_state,new_measurement,new_mu,new_sigma