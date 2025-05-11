import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class IBVSRNN:
    """
    Implementation of Recurrent Neural Network (RNN) for Image-Based Visual Servoing (IBVS)
    based on the mathematical formulation provided.
    """
    def __init__(self, P, q, epsilon, gamma, debug=False):
        # System parameters
        self.P = P  # Positive definite weight matrix
        self.q = q  # Linear term in cost function
        self.epsilon = epsilon  # Small positive constant for RNN convergence
        self.gamma = gamma  # Convergence rate parameter
        self.debug = debug  # Debug flag
        
        # Depth-related parameters
        self.lambda_val = 1.0  # Depth scaling parameter
        self.pu = 0.5  # Camera intrinsic parameter (focal length * x-coordinate of principal point)
        self.pv = 0.5  # Camera intrinsic parameter (focal length * y-coordinate of principal point)
        
        # Initialize matrices
        self._initialize_matrices()
    
    def _initialize_matrices(self):
        """Initialize the system matrices based on the given formulation"""
        # Number of feature points (4 points with 2 coordinates each)
        self.num_features = 8
        
        # C matrix (6×4 matrix)
        self.C = np.zeros((6, 4))
        self.C[0, 0] = self.pu / self.lambda_val
        self.C[1, 0] = self.pv / self.lambda_val
        self.C[2, 0] = 1
        self.C[3, 1] = 1
        self.C[4, 2] = 1
        self.C[5, 3] = 1
        
        # J0 matrix (8×4 matrix) - adapted for 4 points
        # This is a simplification, in reality would be constructed based on the specific feature point Jacobians
        self.J0 = np.zeros((self.num_features, 4))
        
        # Fill J0 with repeated blocks for each feature point
        for i in range(4):  # 4 feature points
            base_idx = i * 2  # Each point has 2 coordinates
            # First coordinate (u)
            self.J0[base_idx, 0] = 0
            self.J0[base_idx, 1] = self.pu * self.pv / self.lambda_val
            self.J0[base_idx, 2] = -((self.pu**2 + self.lambda_val**2) / self.lambda_val)
            self.J0[base_idx, 3] = self.pv
            
            # Second coordinate (v)
            self.J0[base_idx+1, 0] = 0
            self.J0[base_idx+1, 1] = (self.lambda_val**2 + self.pv**2) / self.lambda_val
            self.J0[base_idx+1, 2] = -self.pu * self.pv / self.lambda_val
            self.J0[base_idx+1, 3] = -self.pu
        
        # Expanded matrices for QP formulation
        self.P_bar = np.block([
            [self.P, np.zeros((self.P.shape[0], 4))],
            [np.zeros((4, self.P.shape[0])), np.zeros((4, 4))]
        ])
        
        self.J0_bar = np.block([np.zeros((self.num_features, self.P.shape[0])), self.J0])
        
    def Jr_theta(self, theta):
        """
        Calculate the rotational Jacobian Jr(θ) based on current orientation
        
        Args:
            theta: Current orientation [roll, pitch, yaw]
            
        Returns:
            Jr_theta: Rotational Jacobian matrix (3x3)
        """
        # Implementation of a more realistic rotational Jacobian
        # that depends on the current orientation
        roll, pitch, yaw = theta
        
        # Create rotational Jacobian based on orientation
        # Using a simplified model for demonstration
        Jr = np.eye(3)  # Start with identity
        
        # Add effects of current orientation on the Jacobian
        # This makes rotation around one axis affect the others
        c_roll, s_roll = np.cos(roll), np.sin(roll)
        c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
        c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
        
        # Create a rotation effect matrix
        # This is a simplified approximation of the real Jacobian
        rotation_effect = np.array([
            [c_pitch * c_yaw, s_roll * s_pitch * c_yaw - c_roll * s_yaw, c_roll * s_pitch * c_yaw + s_roll * s_yaw],
            [c_pitch * s_yaw, s_roll * s_pitch * s_yaw + c_roll * c_yaw, c_roll * s_pitch * s_yaw - s_roll * c_yaw],
            [-s_pitch, s_roll * c_pitch, c_roll * c_pitch]
        ])
        
        # Combine with the base Jacobian
        Jr = rotation_effect
        
        return Jr
    
    def G_theta(self, theta):
        """
        Calculate G(θ) = [Jr(θ) -C]
        
        Args:
            theta: Current orientation
            
        Returns:
            G_theta: G(θ) matrix
        """
        Jr = self.Jr_theta(theta)
        return np.hstack([Jr, -self.C])
    
    def rnn_dynamics(self, t, state, s, sd, theta):
        """
        RNN dynamics according to the formulation:
        εω̇ = -ω + P_{11}(ω - Pω - q - J_r^T(θ)β)
        εk̇ = -J_0^T α + C^T β
        εα̇ = J_0 k + γ(s - s_d)
        εβ̇ = J_r(θ)ω - Ck
        
        Args:
            t: Time (not used, required by ODE solver)
            state: Current state [ω, k, α, β]
            s: Current image features
            sd: Desired image features
            theta: Current orientation
            
        Returns:
            state_dot: Time derivative of state
        """
        # Extract states
        omega = state[:3]  # Angular velocity (3x1)
        k = state[3:7]     # Auxiliary variable (4x1)
        alpha = state[7:7+self.num_features]  # Dual variable (num_features x 1)
        beta = state[7+self.num_features:]  # Dual variable (6x1)
        
        # Calculate Jr(θ)
        Jr_theta = self.Jr_theta(theta)  # 3x3 matrix
        
        # Calculate time derivatives based on the RNN dynamics
        # εω̇ = -ω + P_{11}(ω - Pω - q - J_r^T(θ)β)
        P11 = np.eye(3)  # Simplified P_{11} matrix for demonstration
        
        # Fix the matrix multiplication issue
        # Jr_theta is 3x3, beta is 6x1
        # We need to use the first 3 elements of beta for the rotation part
        beta_Jr = beta[:3]  # First 3 elements correspond to rotation
        
        omega_dot = (-omega + np.dot(P11, (omega - np.dot(self.P, omega) - 
                     self.q - np.dot(Jr_theta.T, beta_Jr)))) / self.epsilon
        
        # εk̇ = -J_0^T α + C^T β
        k_dot = (-np.dot(self.J0.T, alpha) + np.dot(self.C.T, beta)) / self.epsilon
        
        # εα̇ = J_0 k + γ(s - s_d)
        alpha_dot = (np.dot(self.J0, k) + self.gamma * (s - sd)) / self.epsilon
        
        # εβ̇ = J_r(θ)ω - Ck
        Jr_omega = np.dot(Jr_theta, omega)
        C_k = np.dot(self.C, k)
        
        beta_dot = np.zeros(6)
        beta_dot[:3] = Jr_omega
        beta_dot -= C_k
        beta_dot /= self.epsilon
        
        # Combine into full state derivative
        state_dot = np.concatenate([omega_dot, k_dot, alpha_dot, beta_dot])
        
        return state_dot
    
    def solve(self, s0, sd, theta0, t_span, t_eval=None):
        """
        Solve the RNN dynamics to find optimal control input
        
        Args:
            s0: Current image features (flattened vector for all feature points)
            sd: Desired image features (flattened vector for all feature points)
            theta0: Current orientation
            t_span: Time span for simulation [t_start, t_end]
            t_eval: Optional specific evaluation times
            
        Returns:
            sol: Solution object from ODE solver
        """
        # Store theta0 for later use
        self.theta0 = theta0
        
        # Initial state [ω, k, α, β]
        # Initialize all to zeros
        initial_state = np.zeros(3 + 4 + self.num_features + 6)  # 3 for ω, 4 for k, num_features for α, 6 for β
        
        # Print debugging information if debug flag is set
        if self.debug:
            print(f"State dimension: {len(initial_state)}")
            print(f"Breakdown: ω(3) + k(4) + α({self.num_features}) + β(6) = {3 + 4 + self.num_features + 6}")
            
            # Check function output dimensions
            test_output = self.rnn_dynamics(0, initial_state, s0, sd, theta0)
            print(f"RNN dynamics output dimension: {len(test_output)}")
        
        # Solve ODE
        sol = solve_ivp(
            lambda t, y: self.rnn_dynamics(t, y, s0, sd, theta0),
            t_span,
            initial_state,
            t_eval=t_eval,
            method='RK45'
        )
        
        return sol
    
    def plot_results(self, sol, s0, sd):
        """
        Plot the results of the RNN optimization
        
        Args:
            sol: Solution from the ODE solver
            s0: Initial image features
            sd: Desired image features
        """
        # Get theta0 from self (set during solve)
        theta0 = self.theta0
        
        # Extract data from solution
        t = sol.t
        omega = sol.y[:3, :]
        k = sol.y[3:7, :]
        alpha = sol.y[7:7+self.num_features, :]
        beta = sol.y[7+self.num_features:, :]
        
        # Plot angular velocity (control input)
        plt.figure(figsize=(12, 10))
        plt.subplot(3, 2, 1)
        plt.plot(t, omega.T)
        plt.title('Angular Velocity (ω)')
        plt.xlabel('Time')
        plt.legend(['ωx', 'ωy', 'ωz'])
        
        # Plot auxiliary variable k
        plt.subplot(3, 2, 2)
        plt.plot(t, k.T)
        plt.title('Auxiliary Variable (k)')
        plt.xlabel('Time')
        plt.legend(['k1', 'k2', 'k3', 'k4'])
        
        # Plot dual variable α (just show a few components for clarity)
        plt.subplot(3, 2, 3)
        plt.plot(t, alpha[:4, :].T)  # Show first 4 components
        plt.title('Dual Variable (α) - First 4 Components')
        plt.xlabel('Time')
        plt.legend([f'α{i+1}' for i in range(4)])
        
        # Plot dual variable β
        plt.subplot(3, 2, 4)
        plt.plot(t, beta.T)
        plt.title('Dual Variable (β)')
        plt.xlabel('Time')
        plt.legend(['β1', 'β2', 'β3', 'β4', 'β5', 'β6'])
        
        # Plot feature error
        plt.subplot(3, 2, 5)
        feature_error = np.linalg.norm(s0 - sd)
        plt.axhline(y=feature_error, color='r', linestyle='-', label='Initial Error')
        
        # Calculate feature error over time (simplified)
        error = np.zeros(len(t))
        for i in range(len(t)):
            # Simplified calculation - in a real system this would depend on the current feature position
            k_i = k[:, i]
            error[i] = np.linalg.norm(np.dot(self.J0, k_i) - (sd - s0))
        
        plt.plot(t, error, label='Error Evolution')
        plt.title('Feature Error')
        plt.xlabel('Time')
        plt.ylabel('||s - s_d||')
        plt.legend()
        
        # Calculate cost function over time
        plt.subplot(3, 2, 6)
        cost = np.zeros(len(t))
        for i in range(len(t)):
            omega_i = omega[:, i]
            cost[i] = 0.5 * np.dot(omega_i.T, np.dot(self.P, omega_i)) + np.dot(self.q.T, omega_i)
        
        plt.plot(t, cost)
        plt.title('Cost Function')
        plt.xlabel('Time')
        plt.ylabel('Cost')
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot for constraints satisfaction
        plt.figure(figsize=(10, 6))
        constraints = np.zeros((len(t), 6))
        for i in range(len(t)):
            omega_i = omega[:, i]
            k_i = k[:, i]
            Jr = self.Jr_theta(theta0)
            constraints[i, :3] = np.dot(Jr, omega_i)  # First 3 constraints (rotation)
            constraints[i, :] -= np.dot(self.C, k_i)  # All 6 constraints
        
        plt.plot(t, constraints)
        plt.title('Constraint Satisfaction: J_r(θ)ω - Ck')
        plt.xlabel('Time')
        plt.ylabel('Constraint Value')
        plt.legend([f'Constraint {i+1}' for i in range(6)])
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.grid(True)
        plt.show()


def main():
    # Problem parameters
    P = np.eye(3)  # Positive definite weight matrix (3×3)
    q = np.array([0.1, 0.1, 0.1])  # Linear term in cost function
    epsilon = 0.01  # Small positive constant for RNN convergence
    gamma = 1.0  # Convergence rate parameter
    
    # Create IBVS-RNN instance with debug enabled for standalone execution
    ibvs_rnn = IBVSRNN(P, q, epsilon, gamma, debug=True)
    
    # Initial and desired image features (4 points with 2 coordinates each)
    # Flatten to 1D array for processing
    s0 = np.array([0.2, 0.3, 0.25, 0.35, 0.3, 0.2, 0.35, 0.25])  # Initial features
    sd = np.array([0.0, 0.0, 0.05, 0.05, 0.05, -0.05, -0.05, -0.05])  # Desired features
    
    # Initial orientation
    theta0 = np.zeros(3)
    
    # Solve RNN dynamics
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 200)
    
    try:
        sol = ibvs_rnn.solve(s0, sd, theta0, t_span, t_eval)
        
        # Plot results
        ibvs_rnn.plot_results(sol, s0, sd)
        
        # Print final values
        final_omega = sol.y[:3, -1]
        print(f"Final angular velocity (ω): {final_omega}")
        
        final_k = sol.y[3:7, -1]
        print(f"Final auxiliary variable (k): {final_k}")
        
        final_alpha = sol.y[7:7+ibvs_rnn.num_features, -1]
        print(f"Final dual variable (α): {final_alpha}")
        
        final_beta = sol.y[7+ibvs_rnn.num_features:, -1]
        print(f"Final dual variable (β): {final_beta}")
        
        final_cost = 0.5 * np.dot(final_omega.T, np.dot(P, final_omega)) + np.dot(q.T, final_omega)
        print(f"Final cost: {final_cost}")
        
        # Check constraint satisfaction
        Jr = ibvs_rnn.Jr_theta(theta0)
        constraint_violation = np.zeros(6)
        constraint_violation[:3] = np.dot(Jr, final_omega)  # First 3 constraints (rotation)
        constraint_violation -= np.dot(ibvs_rnn.C, final_k)  # All 6 constraints
        print(f"Final constraint violation: {constraint_violation}")
        print(f"Constraint satisfaction norm: {np.linalg.norm(constraint_violation)}")
    
    except Exception as e:
        print(f"Error during simulation: {e}")


if __name__ == "__main__":
    main() 