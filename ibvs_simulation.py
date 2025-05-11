import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from ibvs_rnn_verification import IBVSRNN

class Camera:
    """
    Simple camera model for IBVS simulation
    """
    def __init__(self, fx=500, fy=500, cx=320, cy=240, width=640, height=480):
        # Camera intrinsic parameters
        self.fx = fx  # Focal length in x direction (pixels)
        self.fy = fy  # Focal length in y direction (pixels)
        self.cx = cx  # Principal point x-coordinate (pixels)
        self.cy = cy  # Principal point y-coordinate (pixels)
        self.width = width  # Image width (pixels)
        self.height = height  # Image height (pixels)
        
        # Camera extrinsic parameters (position and orientation)
        self.position = np.array([0.0, 0.0, 0.0])  # Camera position in world frame
        self.orientation = np.array([0.0, 0.0, 0.0])  # Euler angles [roll, pitch, yaw]
        self.R_matrix = np.eye(3)  # Rotation matrix from world to camera
        self.update_extrinsics()
    
    def update_extrinsics(self):
        """Update camera extrinsic parameters"""
        r = R.from_euler('xyz', self.orientation, degrees=False)
        self.R_matrix = r.as_matrix()
    
    def world_to_camera(self, points_3d):
        """
        Transform 3D points from world frame to camera frame
        
        Args:
            points_3d: numpy array of shape (n, 3) containing 3D points in world frame
            
        Returns:
            points_cam: numpy array of shape (n, 3) containing 3D points in camera frame
        """
        # Ensure points_3d is a numpy array
        points_3d = np.asarray(points_3d)
        
        # Reshape to (n, 3) if needed
        if len(points_3d.shape) == 1:
            points_3d = points_3d.reshape(1, 3)
        
        # Apply rotation and translation
        points_cam = np.zeros_like(points_3d)
        for i in range(points_3d.shape[0]):
            # Rotate and translate
            points_cam[i] = np.dot(self.R_matrix, points_3d[i] - self.position)
        
        return points_cam
    
    def project(self, points_3d):
        """
        Project 3D points onto the image plane
        
        Args:
            points_3d: numpy array of shape (n, 3) containing 3D points in world frame
            
        Returns:
            points_2d: numpy array of shape (n, 2) containing 2D projection coordinates (u, v)
            depths: numpy array of shape (n,) containing depths (z-coordinate) of the points
        """
        # Transform points to camera frame
        points_cam = self.world_to_camera(points_3d)
        
        # Get depths (z-coordinates in camera frame)
        depths = points_cam[:, 2]
        
        # Project to image plane
        points_2d = np.zeros((points_cam.shape[0], 2))
        for i in range(points_cam.shape[0]):
            # Skip points behind the camera
            if depths[i] <= 0:
                points_2d[i] = [-1, -1]  # Invalid projection
                continue
            
            # Perspective projection
            points_2d[i, 0] = self.fx * points_cam[i, 0] / depths[i] + self.cx
            points_2d[i, 1] = self.fy * points_cam[i, 1] / depths[i] + self.cy
        
        return points_2d, depths
    
    def normalized_coordinates(self, points_2d):
        """
        Convert pixel coordinates to normalized image coordinates
        
        Args:
            points_2d: numpy array of shape (n, 2) containing 2D points in pixel coordinates
            
        Returns:
            norm_points: numpy array of shape (n, 2) containing normalized coordinates
        """
        norm_points = np.zeros_like(points_2d)
        norm_points[:, 0] = (points_2d[:, 0] - self.cx) / self.fx
        norm_points[:, 1] = (points_2d[:, 1] - self.cy) / self.fy
        return norm_points
    
    def move_camera(self, linear_vel, angular_vel, dt):
        """
        Move camera according to velocity commands
        
        Args:
            linear_vel: linear velocity [vx, vy, vz] in camera frame
            angular_vel: angular velocity [wx, wy, wz] in camera frame
            dt: time step
        """
        # Convert velocities to world frame
        linear_vel_world = np.dot(self.R_matrix.T, linear_vel)
        
        # Update position
        self.position += linear_vel_world * dt
        
        # Update orientation (simplified, assumes small angles)
        delta_euler = angular_vel * dt
        self.orientation += delta_euler
        
        # Update rotation matrix
        self.update_extrinsics()


class IBVSSimulation:
    """
    Simulation environment for IBVS control
    """
    def __init__(self):
        # Create camera
        self.camera = Camera()
        
        # Create feature points in 3D space (world frame)
        self.points_3d = np.array([
            [-0.1, -0.1, 2.0],  # Point 1
            [0.1, -0.1, 2.0],   # Point 2
            [0.1, 0.1, 2.0],    # Point 3
            [-0.1, 0.1, 2.0]    # Point 4
        ])
        
        # Define desired camera pose
        self.desired_position = np.array([0.0, 0.0, 0.0])
        self.desired_orientation = np.array([0.0, 0.0, 0.0])
        
        # Initialize camera at some offset from desired pose
        self.camera.position = np.array([0.2, 0.3, 0.5])
        self.camera.orientation = np.array([0.1, -0.1, 0.2])
        self.camera.update_extrinsics()
        
        # Calculate desired feature points
        desired_camera = Camera()
        desired_camera.position = self.desired_position
        desired_camera.orientation = self.desired_orientation
        desired_camera.update_extrinsics()
        
        points_2d_desired, _ = desired_camera.project(self.points_3d)
        self.s_desired = desired_camera.normalized_coordinates(points_2d_desired)
        self.s_desired = self.s_desired.flatten()  # Flatten to 1D array
        
        # Initialize controller parameters with improved values
        P = 5.0 * np.eye(3)  # Increased weight on control effort
        q = np.zeros(3)  # No linear term in cost function
        epsilon = 0.001  # Smaller time constant for faster convergence
        gamma = 20.0  # Increased convergence rate parameter (was 5.0)
        
        # Create IBVS-RNN controller with debug=False to suppress debug output
        self.controller = IBVSRNN(P, q, epsilon, gamma, debug=False)
    
    def simulate(self, duration=10.0, dt=0.05):
        """
        Simulate IBVS control loop
        
        Args:
            duration: Total simulation time (seconds)
            dt: Time step (seconds)
            
        Returns:
            history: Dictionary containing simulation history
        """
        # Number of steps
        n_steps = int(duration / dt)
        
        # Initialize history
        history = {
            'time': np.zeros(n_steps),
            'position': np.zeros((n_steps, 3)),
            'orientation': np.zeros((n_steps, 3)),
            'features': [],
            'desired_features': np.tile(self.s_desired, (n_steps, 1)),
            'feature_error': np.zeros(n_steps),
            'control_input': np.zeros((n_steps, 6)),
            'cost': np.zeros(n_steps)
        }
        
        # Simulation loop
        for i in range(n_steps):
            # Current time
            t = i * dt
            history['time'][i] = t
            
            # Get current camera state
            history['position'][i] = self.camera.position
            history['orientation'][i] = self.camera.orientation
            
            # Get current feature points
            points_2d_current, depths = self.camera.project(self.points_3d)
            s_current = self.camera.normalized_coordinates(points_2d_current)
            s_current = s_current.flatten()  # Flatten to 1D array
            history['features'].append(s_current)
            
            # Calculate feature error
            feature_error = np.linalg.norm(s_current - self.s_desired)
            history['feature_error'][i] = feature_error
            
            # Solve RNN dynamics for small time window
            t_span = (0, 1.0)  # Longer prediction horizon (was 0.5)
            t_eval = np.linspace(0, 1.0, 50)  # More evaluation points (was 20)
            
            # Solve with the current state
            sol = self.controller.solve(s_current, self.s_desired, self.camera.orientation, t_span, t_eval)
            
            # Extract control input (angular velocity) from RNN solution
            control_input = sol.y[:3, -1]  # Use final state as control input
            
            # Apply scaling factor to control input for better stability
            control_input = control_input * 2.0  # Amplify the control signal
            
            # Zero linear velocity for pure rotation control
            linear_vel = np.zeros(3)
            angular_vel = control_input
            
            # Store control input
            history['control_input'][i, :3] = linear_vel
            history['control_input'][i, 3:] = angular_vel
            
            # Calculate cost
            cost = 0.5 * np.dot(angular_vel.T, np.dot(self.controller.P, angular_vel))
            history['cost'][i] = cost
            
            # Apply control input to camera
            self.camera.move_camera(linear_vel, angular_vel, dt)
            
            # Print progress
            if i % 20 == 0 or i == n_steps - 1:
                print(f"Simulation progress: {i}/{n_steps}, Feature error: {feature_error:.6f}")
                
        # Convert features list to numpy array
        history['features'] = np.array(history['features'])
        
        return history
    
    def plot_results(self, history):
        """
        Plot simulation results
        
        Args:
            history: Dictionary containing simulation history
        """
        time = history['time']
        
        # Plot camera trajectory
        fig = plt.figure(figsize=(12, 10))
        
        # 3D trajectory
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(history['position'][:, 0], history['position'][:, 1], history['position'][:, 2])
        ax1.scatter(self.points_3d[:, 0], self.points_3d[:, 1], self.points_3d[:, 2], c='r', marker='o')
        ax1.scatter(self.desired_position[0], self.desired_position[1], self.desired_position[2], c='g', marker='*')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Camera Trajectory')
        
        # Feature error
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(time, history['feature_error'])
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Feature Error')
        ax2.set_title('Feature Error vs Time')
        ax2.grid(True)
        
        # Control inputs
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(time, history['control_input'][:, 3:])
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angular Velocity')
        ax3.set_title('Angular Velocity vs Time')
        ax3.legend(['ωx', 'ωy', 'ωz'])
        ax3.grid(True)
        
        # Cost function
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(time, history['cost'])
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Cost')
        ax4.set_title('Cost Function vs Time')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot feature trajectories in image plane
        plt.figure(figsize=(8, 6))
        
        # Convert features for plotting (reshape to have points as rows)
        n_steps = len(time)
        n_points = len(self.points_3d)
        features = history['features'].reshape(n_steps, n_points, 2)
        desired_features = history['desired_features'][0].reshape(n_points, 2)
        
        # Plot feature trajectories
        for i in range(n_points):
            plt.plot(features[:, i, 0], features[:, i, 1], '-')
            plt.scatter(features[-1, i, 0], features[-1, i, 1], c='b', marker='o')
            plt.scatter(features[0, i, 0], features[0, i, 1], c='r', marker='x')
            plt.scatter(desired_features[i, 0], desired_features[i, 1], c='g', marker='*')
        
        # Add labels and grid
        plt.xlabel('u (normalized)')
        plt.ylabel('v (normalized)')
        plt.title('Feature Trajectories in Image Plane')
        plt.grid(True)
        plt.axis('equal')
        plt.legend(['Trajectory', 'Final Position', 'Initial Position', 'Desired Position'])
        
        # Zoom in on the trajectories
        plt.xlim([-0.2, 0.2])
        plt.ylim([-0.2, 0.2])
        
        plt.show()
        
        # Plot orientation over time
        plt.figure(figsize=(8, 6))
        plt.plot(time, history['orientation'])
        plt.xlabel('Time (s)')
        plt.ylabel('Orientation (rad)')
        plt.title('Camera Orientation vs Time')
        plt.legend(['Roll', 'Pitch', 'Yaw'])
        plt.grid(True)
        plt.show()


def main():
    """Main function to run the IBVS simulation"""
    # Create simulation
    simulation = IBVSSimulation()
    
    # Run simulation
    print("Starting IBVS simulation...")
    history = simulation.simulate(duration=10.0, dt=0.05)
    
    # Plot results
    simulation.plot_results(history)
    
    # Report final error
    final_error = history['feature_error'][-1]
    print(f"Final feature error: {final_error:.6f}")
    
    # Report final position and orientation
    final_position = history['position'][-1]
    final_orientation = history['orientation'][-1]
    print(f"Final position: {final_position}")
    print(f"Final orientation: {final_orientation}")
    print(f"Desired position: {simulation.desired_position}")
    print(f"Desired orientation: {simulation.desired_orientation}")
    print(f"Position error: {np.linalg.norm(final_position - simulation.desired_position):.6f}")
    print(f"Orientation error: {np.linalg.norm(final_orientation - simulation.desired_orientation):.6f}")


if __name__ == "__main__":
    main() 