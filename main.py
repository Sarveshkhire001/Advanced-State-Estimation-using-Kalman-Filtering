import numpy as np
import argparse # Import the argparse module

# Import your simulation functions
from single_plane import single_plane
from missing_gps import missing_GPS
from many_planes import many_planes
from landmark_data import landmark_data

def main():
    parser = argparse.ArgumentParser(description="Run Kalman Filter simulations with configurable options.")

    # Argument for selecting the simulation to run
    parser.add_argument('--simulation', type=str, default='single_plane',
                        choices=['single_plane', 'missing_gps', 'many_planes', 'landmark_data'],
                        help="Choose which simulation to run (default: single_plane)")

    # Arguments for flags (converted to boolean)
    parser.add_argument('--vel_flag', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="Set velocity plotting flag (True/False, default: False)")
    parser.add_argument('--ellipse_flag', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="Set ellipse plotting flag (True/False, default: False)")
    
    # Argument for trajectory type (0 or 1)
    parser.add_argument('--traj_type', type=int, default=1, choices=[0, 1],
                        help="Choose trajectory type (0 for ut = [0.128 cos(0.032t), 0.128 sin(0.032t), 0.01]T, 1 for ut = [sin(t), cos(t), sin(t)]T; default: 1)")

    args = parser.parse_args()

    # Call the selected simulation function with the provided flags
    if args.simulation == 'single_plane':
        print("Running Single Plane Simulation...")
        single_plane(args.vel_flag, args.ellipse_flag, args.traj_type)
    elif args.simulation == 'missing_gps':
        print("Running Missing GPS Simulation...")
        missing_GPS(args.vel_flag, args.ellipse_flag, args.traj_type)
    elif args.simulation == 'many_planes':
        print("Running Many Planes Simulation...")
        many_planes(args.vel_flag, args.ellipse_flag, args.traj_type)
    elif args.simulation == 'landmark_data':
        print("Running Landmark Data Simulation...")
        landmark_data(args.vel_flag, args.ellipse_flag, args.traj_type)

if __name__ == '__main__':
    np.random.seed(1) # For reproducibility
    main()

