import numpy as np
import matplotlib.pyplot as plt
from rnn_optimal_control import RNNOptimalControlSolver
import time
from mpl_toolkits.mplot3d import Axes3D
from cvxopt import matrix, solvers

def generate_robot_problem(n_joints=7, n_timesteps=50):
    """
    Generate a robot trajectory optimization problem
    """
    # Total problem dimension (joints × timesteps)
    n_vars = n_joints * n_timesteps
    
    # Generate sparse P matrix for smoothness and dynamics
    P = np.zeros((n_vars, n_vars))
    
    # Weights for optimization
    pos_weight = 0.1    # Position cost
    vel_weight = 30.0   # Velocity smoothness
    acc_weight = 15.0   # Acceleration smoothness
    
    for t in range(n_timesteps):
        idx = t * n_joints
        for i in range(n_joints):
            # Position cost
            P[idx+i, idx+i] += pos_weight
            
            # Velocity cost
            if t < n_timesteps - 1:
                P[idx+i, idx+i] += vel_weight
                P[idx+i, idx+n_joints+i] = -vel_weight
                P[idx+n_joints+i, idx+i] = -vel_weight
                P[idx+n_joints+i, idx+n_joints+i] += vel_weight
            
            # Acceleration cost
            if t < n_timesteps - 2:
                P[idx+i, idx+i] += acc_weight
                P[idx+i, idx+n_joints+i] -= 2.0 * acc_weight
                P[idx+i, idx+2*n_joints+i] += acc_weight
                P[idx+n_joints+i, idx+i] -= 2.0 * acc_weight
                P[idx+n_joints+i, idx+n_joints+i] += 4.0 * acc_weight
                P[idx+n_joints+i, idx+2*n_joints+i] -= 2.0 * acc_weight
                P[idx+2*n_joints+i, idx+i] += acc_weight
                P[idx+2*n_joints+i, idx+n_joints+i] -= 2.0 * acc_weight
                P[idx+2*n_joints+i, idx+2*n_joints+i] += acc_weight
    
    # Add regularization
    P = P + 0.05 * np.eye(n_vars)
    
    # Linear term - reaching goal configuration
    q = np.zeros(n_vars)
    
    # Equality constraints for end-effector path
    n_eq_per_step = 3  # position(x,y,z) constraints
    n_eq_constraints = n_eq_per_step * n_timesteps
    A = np.zeros((n_eq_constraints, n_vars))
    b = np.zeros(n_eq_constraints)
    
    # Generate desired end-effector path (circular trajectory)
    radius = 0.3
    height_range = 0.2
    center = np.array([0.0, 0.0, 0.4])  # Center of the circle
    
    for t in range(n_timesteps):
        idx_start = t * n_eq_per_step
        phase = t / (n_timesteps - 1)
        # Position trajectory (circular motion in XY plane with smooth height change)
        b[idx_start:idx_start+3] = np.array([
            center[0] + radius * np.cos(2 * np.pi * phase),
            center[1] + radius * np.sin(2 * np.pi * phase),
            center[2] + height_range * np.sin(2 * np.pi * phase)
        ])
        
        # Generate Jacobian
        J = generate_jacobian(n_joints, phase)
        A[idx_start:idx_start+3, t*n_joints:(t+1)*n_joints] = J
    
    # Joint limits
    joint_limits = np.array([
        [-2.0, 2.0],   # Base joint
        [-1.5, 1.5],   # Shoulder
        [-2.0, 2.0],   # Elbow
        [-2.0, 2.0],   # Wrist 1
        [-2.0, 2.0],   # Wrist 2
        [-2.5, 2.5],   # Wrist 3
        [-2.5, 2.5]    # Tool joint
    ])
    
    omega_l = np.zeros(n_vars)
    omega_u = np.zeros(n_vars)
    
    for t in range(n_timesteps):
        idx = t * n_joints
        for j in range(n_joints):
            omega_l[idx + j] = joint_limits[j][0]
            omega_u[idx + j] = joint_limits[j][1]
    
    return P, q, A, b, omega_l, omega_u

def generate_jacobian(n_joints, phase):
    """Generate a Jacobian matrix based on configuration"""
    J = np.zeros((3, n_joints))
    
    # Base joint primarily affects x-y motion
    J[0:2, 0] = np.array([np.cos(2*np.pi*phase), np.sin(2*np.pi*phase)]) * 0.4
    
    # Shoulder and elbow joints affect all directions
    for i in range(1, 3):
        J[:, i] = np.array([
            np.cos(2*np.pi*phase + i*np.pi/3),
            np.sin(2*np.pi*phase + i*np.pi/3),
            0.4
        ]) * (0.5 - 0.1*i)
    
    # Wrist joints provide fine control
    for i in range(3, n_joints):
        J[:, i] = np.array([
            np.cos(2*np.pi*phase + i*np.pi/4),
            np.sin(2*np.pi*phase + i*np.pi/4),
            0.2
        ]) * (0.3 - 0.03*i)
    
    return J

def calculate_error(omega, A, b, n_joints, n_timesteps):
    """Calculate constraint violation error"""
    omega_reshaped = omega.reshape(n_timesteps, n_joints)
    error = 0
    for t in range(n_timesteps):
        idx = t * 3
        A_t = A[idx:idx+3, t*n_joints:(t+1)*n_joints]
        b_t = b[idx:idx+3]
        error += np.linalg.norm(A_t @ omega_reshaped[t] - b_t)
    return error

def solve_qp_cvx(P, q, A, b, omega_l, omega_u):
    """Solve the QP problem using CVXOPT"""
    n = P.shape[0]
    
    # Convert to CVXOPT matrix format
    P_cvx = matrix(P)
    q_cvx = matrix(q)
    
    # Add bounds as inequality constraints
    G = np.vstack([np.eye(n), -np.eye(n)])
    h = np.hstack([omega_u, -omega_l])
    G_cvx = matrix(G)
    h_cvx = matrix(h)
    
    # Set CVXOPT solver options
    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-8
    solvers.options['reltol'] = 1e-8
    solvers.options['feastol'] = 1e-8
    
    # Solve QP with equality constraints
    A_cvx = matrix(A)
    b_cvx = matrix(b)
    start_time = time.time()
    solution = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
    solve_time = time.time() - start_time
    
    return np.array(solution['x']).flatten(), solve_time

def forward_kinematics_simple(theta):
    """Simplified forward kinematics for visualization"""
    # This is a very simplified model - in reality would use proper DH parameters
    n_joints = len(theta)
    points = np.zeros((n_joints + 1, 3))
    
    # Assume each link has length 0.2 units
    link_length = 0.2
    current_pos = np.zeros(3)
    current_angle = np.zeros(3)
    
    for i in range(n_joints):
        points[i] = current_pos
        
        # Update angles based on joint type (simplified)
        if i % 3 == 0:  # Base rotation
            current_angle[2] += theta[i]
        elif i % 3 == 1:  # Shoulder/elbow
            current_angle[1] += theta[i]
        else:  # Wrist
            current_angle[0] += theta[i]
            
        # Move to next joint
        current_pos = current_pos + link_length * np.array([
            np.cos(current_angle[2]) * np.cos(current_angle[1]),
            np.sin(current_angle[2]) * np.cos(current_angle[1]),
            np.sin(current_angle[1])
        ])
    
    points[-1] = current_pos
    return points

def visualize_robot_state(ax, theta, target_pos=None, title="Robot Configuration"):
    """Visualize robot configuration in 3D"""
    points = forward_kinematics_simple(theta)
    
    # Plot robot links
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2, label='Robot Links')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', label='Joints')
    
    # Plot target position if provided
    if target_pos is not None:
        ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], 
                  c='g', marker='*', s=100, label='Target')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio using axis limits
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add legend
    ax.legend()

def visualize_results(trajectory, ee_positions, error_history=None, time_points=None, title_prefix=""):
    """Visualize optimization results"""
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Joint angles over time
    ax1 = fig.add_subplot(231)
    n_joints = trajectory.shape[1]
    for i in range(n_joints):
        ax1.plot(trajectory[:, i], label=f'Joint {i+1}')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Joint Angle (rad)')
    ax1.set_title(f'{title_prefix}Joint Trajectories')
    ax1.legend()
    ax1.grid(True)
    
    # 2. End-effector trajectory in 3D
    ax2 = fig.add_subplot(232, projection='3d')
    ax2.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 'g-', label='EE Path')
    ax2.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2], 
               c='b', marker='o', label='Start')
    ax2.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2], 
               c='r', marker='o', label='End')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'{title_prefix}End-Effector Trajectory')
    
    # Set equal aspect ratio for end-effector trajectory
    max_range = np.array([
        ee_positions[:, 0].max() - ee_positions[:, 0].min(),
        ee_positions[:, 1].max() - ee_positions[:, 1].min(),
        ee_positions[:, 2].max() - ee_positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (ee_positions[:, 0].max() + ee_positions[:, 0].min()) * 0.5
    mid_y = (ee_positions[:, 1].max() + ee_positions[:, 1].min()) * 0.5
    mid_z = (ee_positions[:, 2].max() + ee_positions[:, 2].min()) * 0.5
    
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    ax2.legend()
    
    # 3. Robot configuration at start
    ax3 = fig.add_subplot(233, projection='3d')
    visualize_robot_state(ax3, trajectory[0], ee_positions[-1], f"{title_prefix}Initial Configuration")
    
    # 4. Robot configuration at end
    ax4 = fig.add_subplot(234, projection='3d')
    visualize_robot_state(ax4, trajectory[-1], ee_positions[-1], f"{title_prefix}Final Configuration")
    
    # 5. Convergence plot (if available)
    ax5 = fig.add_subplot(235)
    if error_history is not None and time_points is not None:
        ax5.semilogy(time_points, error_history, 'b-')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Error (log scale)')
        ax5.set_title(f'{title_prefix}Convergence')
        ax5.grid(True)
    
    # 6. Velocity profile
    ax6 = fig.add_subplot(236)
    velocities = np.diff(trajectory, axis=0) / 0.02  # assuming dt=0.02s
    for i in range(n_joints):
        ax6.plot(velocities[:, i], label=f'Joint {i+1}')
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Velocity (rad/s)')
    ax6.set_title(f'{title_prefix}Joint Velocities')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.show()

def run_optimization():
    """Run the optimization using both RNN and CVX methods"""
    n_joints = 7
    n_timesteps = 50
    
    # Generate problem
    P, q, A, b, omega_l, omega_u = generate_robot_problem(n_joints, n_timesteps)
    
    # Initial joint configuration (same for both methods)
    omega_init = np.zeros(n_joints * n_timesteps)
    for t in range(n_timesteps):
        omega_init[t*n_joints:(t+1)*n_joints] = np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # 1. Solve using RNN
    print("Solving optimization problem using RNN...")
    epsilon = 0.01
    rnn_solver = RNNOptimalControlSolver(P, q, A, b, omega_l, omega_u, epsilon=epsilon)
    
    start_time = time.time()
    omega_rnn, history = rnn_solver.solve(
        t_span=(0, 0.5),
        dt=0.01,
        omega_init=omega_init
    )
    rnn_time = time.time() - start_time
    print(f"RNN solution found in {rnn_time:.3f} seconds")
    
    # 2. Solve using CVX
    print("\nSolving optimization problem using CVX...")
    omega_cvx, cvx_time = solve_qp_cvx(P, q, A, b, omega_l, omega_u)
    print(f"CVX solution found in {cvx_time:.3f} seconds")
    
    # Prepare results for visualization
    trajectory_rnn = omega_rnn.reshape(-1, n_joints)
    trajectory_cvx = omega_cvx.reshape(n_timesteps, n_joints)
    
    ee_positions_rnn = np.array([forward_kinematics_simple(theta)[-1] for theta in trajectory_rnn])
    ee_positions_cvx = np.array([forward_kinematics_simple(theta)[-1] for theta in trajectory_cvx])
    
    # Calculate error histories
    rnn_error = []
    for omega_t in history['omega']:
        error = calculate_error(omega_t, A, b, n_joints, n_timesteps)
        rnn_error.append(error)
    
    # For CVX, calculate error at each timestep by interpolation
    cvx_error = []
    t_eval = np.linspace(0, 0.5, 50)  # Match RNN time points
    for t in t_eval:
        alpha = t / 0.5
        omega_t = omega_init * (1 - alpha) + omega_cvx * alpha
        error = calculate_error(omega_t, A, b, n_joints, n_timesteps)
        cvx_error.append(error)
    
    # Visualize results
    print("\nVisualizing RNN results...")
    visualize_results(trajectory_rnn, ee_positions_rnn, rnn_error, history['t'], "RNN: ")
    
    print("\nVisualizing CVX results...")
    visualize_results(trajectory_cvx, ee_positions_cvx, cvx_error, t_eval, "CVX: ")

if __name__ == "__main__":
    run_optimization() 