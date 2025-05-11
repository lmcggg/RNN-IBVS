import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def test_rnn_controller():
    """
    Test the validity of the RNN-based approach for IBVS control
    by simulating a simple case with direct control of image features.
    """
    # Simulation parameters
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 100)
    
    # Initial and desired features (simplified 2D point)
    s0 = np.array([0.3, 0.2])  # Initial feature
    sd = np.array([0.0, 0.0])  # Desired feature
    
    # Controller parameters
    epsilon = 0.01  # Small time constant
    gamma = 10.0    # Increased convergence parameter
    P = np.eye(2)   # Control effort weight
    q = np.zeros(2) # No linear term
    
    # Dynamics model parameters (simplified)
    # Image Jacobian maps control to feature velocity: s_dot = J * u
    J = np.array([
        [-1.0, -0.5],
        [0.5, -1.0]
    ])
    
    # RNN state: [u, alpha]
    # u: control input (2D)
    # alpha: dual variable for constraint (2D)
    initial_state = np.zeros(4)
    
    # RNN dynamics
    def rnn_dynamics(t, state, s, sd, J):
        # Extract states
        u = state[:2]      # Control input
        alpha = state[2:]  # Dual variable
        
        # Compute derivatives
        u_dot = (-u - np.dot(P, u) - np.dot(J.T, alpha)) / epsilon
        alpha_dot = (np.dot(J, u) + gamma * (s - sd)) / epsilon
        
        # Return state derivative
        return np.concatenate([u_dot, alpha_dot])
    
    # Solve RNN dynamics
    sol = solve_ivp(
        lambda t, state: rnn_dynamics(t, state, s0, sd, J),
        t_span,
        initial_state,
        t_eval=t_eval,
        method='RK45'
    )
    
    # Extract solution
    t = sol.t
    u = sol.y[:2, :]
    alpha = sol.y[2:, :]
    
    # Simulate feature trajectory using control input
    s_traj = np.zeros((len(t), 2))
    s_traj[0] = s0
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        s_dot = np.dot(J, u[:, i-1])
        s_traj[i] = s_traj[i-1] + s_dot * dt
    
    # Calculate feature error over time
    error = np.zeros(len(t))
    for i in range(len(t)):
        error[i] = np.linalg.norm(s_traj[i] - sd)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot control input
    plt.subplot(2, 2, 1)
    plt.plot(t, u.T)
    plt.title('Control Input')
    plt.xlabel('Time (s)')
    plt.ylabel('u')
    plt.legend(['u1', 'u2'])
    plt.grid(True)
    
    # Plot dual variable
    plt.subplot(2, 2, 2)
    plt.plot(t, alpha.T)
    plt.title('Dual Variable (α)')
    plt.xlabel('Time (s)')
    plt.ylabel('α')
    plt.legend(['α1', 'α2'])
    plt.grid(True)
    
    # Plot feature trajectory
    plt.subplot(2, 2, 3)
    plt.plot(s_traj[:, 0], s_traj[:, 1], 'b-')
    plt.plot(s0[0], s0[1], 'ro', label='Initial')
    plt.plot(sd[0], sd[1], 'g*', label='Desired')
    plt.title('Feature Trajectory')
    plt.xlabel('s1')
    plt.ylabel('s2')
    plt.grid(True)
    plt.legend()
    
    # Plot feature error
    plt.subplot(2, 2, 4)
    plt.plot(t, error)
    plt.title('Feature Error')
    plt.xlabel('Time (s)')
    plt.ylabel('||s - sd||')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final values
    print(f"Initial feature: {s0}")
    print(f"Desired feature: {sd}")
    print(f"Final feature: {s_traj[-1]}")
    print(f"Final error: {error[-1]}")
    print(f"Final control input: {u[:, -1]}")
    
    # Check if the solution converged
    if error[-1] < 0.05:
        print("\n✓ TEST PASSED: RNN controller successfully converged to the desired feature.")
        print("The RNN-based approach is valid for IBVS control.")
    else:
        print("\n× TEST FAILED: RNN controller did not converge to the desired feature.")
        print("The RNN-based approach may not be valid for IBVS control as implemented.")


def test_rnn_controller_with_depth_invariance():
    """
    Test the validity of the RNN-based approach for IBVS control
    with depth invariance transformation.
    """
    # Simulation parameters
    t_span = (0, 20)   # Longer simulation time
    t_eval = np.linspace(0, 20, 200)  # More evaluation points
    
    # Initial and desired features (simplified 2D point)
    s0 = np.array([0.3, 0.2])  # Initial feature
    sd = np.array([0.0, 0.0])  # Desired feature
    
    # Controller parameters
    epsilon = 0.005  # Smaller time constant for faster convergence
    gamma = 20.0     # Increased convergence parameter
    P = 0.1 * np.eye(3)  # Reduced weight on control effort to allow more aggressive control
    q = np.zeros(3)      # No linear term
    
    # Depth-related parameters
    lambda_val = 1.0  # Depth scaling parameter
    pu = 0.5  # Camera intrinsic parameter
    pv = 0.5  # Camera intrinsic parameter
    
    # Define matrices for depth invariance transformation
    # C matrix (6×4 matrix) - simplified for a 2D point
    C = np.zeros((6, 4))
    C[0, 0] = pu / lambda_val
    C[1, 0] = pv / lambda_val
    C[2, 0] = 1
    C[3, 1] = 1
    C[4, 2] = 1
    C[5, 3] = 1
    
    # J0 matrix (2×4 matrix) - for a single point
    J0 = np.zeros((2, 4))
    # First coordinate (u)
    J0[0, 0] = 0
    J0[0, 1] = pu * pv / lambda_val
    J0[0, 2] = -((pu**2 + lambda_val**2) / lambda_val)
    J0[0, 3] = pv
    # Second coordinate (v)
    J0[1, 0] = 0
    J0[1, 1] = (lambda_val**2 + pv**2) / lambda_val
    J0[1, 2] = -pu * pv / lambda_val
    J0[1, 3] = -pu
    
    # Rotational Jacobian (simplified)
    Jr = np.eye(3)
    
    # RNN state: [omega, k, alpha, beta]
    # omega: angular velocity (3D)
    # k: auxiliary variable (4D)
    # alpha: dual variable for depth invariance (2D)
    # beta: dual variable for constraints (6D)
    initial_state = np.zeros(3 + 4 + 2 + 6)
    
    # RNN dynamics
    def rnn_dynamics(t, state, s, sd, Jr, J0, C):
        # Extract states
        omega = state[:3]    # Angular velocity
        k = state[3:7]       # Auxiliary variable
        alpha = state[7:9]   # Dual variable for feature error
        beta = state[9:]     # Dual variable for constraints
        
        # Compute derivatives
        # ω̇ = (-ω + (ω - Pω - q - Jr^T β)) / ε
        # We only consider the first 3 elements of beta for rotation
        beta_rot = beta[:3]
        omega_dot = (-omega - np.dot(P, omega) - np.dot(Jr.T, beta_rot)) / epsilon
        
        # k̇ = (-J0^T α + C^T β) / ε
        k_dot = (-np.dot(J0.T, alpha) + np.dot(C.T, beta)) / epsilon
        
        # α̇ = (J0 k + γ(s - sd)) / ε
        alpha_dot = (np.dot(J0, k) + gamma * (s - sd)) / epsilon
        
        # β̇ = (Jr ω - C k) / ε
        beta_dot = np.zeros(6)
        # First 3 elements correspond to rotation
        beta_dot[:3] = np.dot(Jr, omega)
        # All 6 elements are affected by C*k
        beta_dot -= np.dot(C, k)
        beta_dot /= epsilon
        
        # Return state derivative
        return np.concatenate([omega_dot, k_dot, alpha_dot, beta_dot])
    
    # Solve RNN dynamics
    sol = solve_ivp(
        lambda t, state: rnn_dynamics(t, state, s0, sd, Jr, J0, C),
        t_span,
        initial_state,
        t_eval=t_eval,
        method='RK45'
    )
    
    # Extract solution
    t = sol.t
    omega = sol.y[:3, :]
    k = sol.y[3:7, :]
    alpha = sol.y[7:9, :]
    beta = sol.y[9:, :]
    
    # Simulate feature trajectory using control input
    s_traj = np.zeros((len(t), 2))
    s_traj[0] = s0
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        s_dot = np.dot(J0, k[:, i-1])  # Feature velocity from auxiliary variable
        s_traj[i] = s_traj[i-1] + s_dot * dt
    
    # Calculate feature error over time
    error = np.zeros(len(t))
    for i in range(len(t)):
        error[i] = np.linalg.norm(s_traj[i] - sd)
    
    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Plot angular velocity
    plt.subplot(3, 2, 1)
    plt.plot(t, omega.T)
    plt.title('Angular Velocity (ω)')
    plt.xlabel('Time (s)')
    plt.ylabel('ω')
    plt.legend(['ωx', 'ωy', 'ωz'])
    plt.grid(True)
    
    # Plot auxiliary variable k
    plt.subplot(3, 2, 2)
    plt.plot(t, k.T)
    plt.title('Auxiliary Variable (k)')
    plt.xlabel('Time (s)')
    plt.ylabel('k')
    plt.legend(['k1', 'k2', 'k3', 'k4'])
    plt.grid(True)
    
    # Plot dual variable alpha
    plt.subplot(3, 2, 3)
    plt.plot(t, alpha.T)
    plt.title('Dual Variable (α)')
    plt.xlabel('Time (s)')
    plt.ylabel('α')
    plt.legend(['α1', 'α2'])
    plt.grid(True)
    
    # Plot dual variable beta
    plt.subplot(3, 2, 4)
    plt.plot(t, beta.T)
    plt.title('Dual Variable (β)')
    plt.xlabel('Time (s)')
    plt.ylabel('β')
    plt.legend([f'β{i+1}' for i in range(6)])
    plt.grid(True)
    
    # Plot feature trajectory
    plt.subplot(3, 2, 5)
    plt.plot(s_traj[:, 0], s_traj[:, 1], 'b-')
    plt.plot(s0[0], s0[1], 'ro', label='Initial')
    plt.plot(sd[0], sd[1], 'g*', label='Desired')
    plt.title('Feature Trajectory')
    plt.xlabel('s1')
    plt.ylabel('s2')
    plt.grid(True)
    plt.legend()
    
    # Plot feature error
    plt.subplot(3, 2, 6)
    plt.plot(t, error)
    plt.title('Feature Error')
    plt.xlabel('Time (s)')
    plt.ylabel('||s - sd||')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final values
    print(f"Initial feature: {s0}")
    print(f"Desired feature: {sd}")
    print(f"Final feature: {s_traj[-1]}")
    print(f"Final error: {error[-1]}")
    print(f"Final angular velocity: {omega[:, -1]}")
    print(f"Final auxiliary variable k: {k[:, -1]}")
    
    # Calculate constraint satisfaction - properly handle the shape mismatch
    # The first 3 elements of final_constraint will be Jr*omega - C*k[:3]
    # The last 3 elements will be -C*k[3:6]
    Jr_omega = np.dot(Jr, omega[:, -1])
    C_k = np.dot(C, k[:, -1])
    final_constraint = np.zeros(6)
    final_constraint[:3] = Jr_omega
    final_constraint -= C_k
    
    print(f"Final constraint violation: {final_constraint}")
    print(f"Constraint satisfaction norm: {np.linalg.norm(final_constraint)}")
    
    # Check if the solution converged
    if error[-1] < 0.05:
        print("\n✓ TEST PASSED: RNN controller with depth invariance successfully converged.")
        print("The RNN-based approach with depth invariance is valid for IBVS control.")
    else:
        print("\n× TEST FAILED: RNN controller with depth invariance did not converge.")
        print("The approach may need adjustments or different parameters.")


if __name__ == "__main__":
    print("=== Testing basic RNN controller for IBVS ===")
    test_rnn_controller()
    
    print("\n=== Testing RNN controller with depth invariance ===")
    test_rnn_controller_with_depth_invariance() 