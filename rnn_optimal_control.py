import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cvxopt import matrix, solvers
import time

class RNNOptimalControlSolver:
    def __init__(self, P, q, A, b, omega_l, omega_u, epsilon=0.1):
        """
        Initialize the RNN-based optimal control solver
        
        Parameters:
        -----------
        P : ndarray, shape (n, n)
            Quadratic term in objective function (must be positive definite)
        q : ndarray, shape (n,)
            Linear term in objective function
        A : ndarray, shape (m, n)
            Equality constraint matrix
        b : ndarray, shape (m,)
            Equality constraint vector
        omega_l : ndarray, shape (n,)
            Lower bounds for variables
        omega_u : ndarray, shape (n,)
            Upper bounds for variables
        epsilon : float
            Convergence rate parameter
        """
        self.P = P
        self.q = q
        self.A = A if A.size > 0 else np.zeros((0, P.shape[0]))
        self.b = b if b.size > 0 else np.zeros(0)
        self.omega_l = omega_l
        self.omega_u = omega_u
        self.n = P.shape[0]  # dimension of omega
        self.m = A.shape[0] if A.size > 0 else 0  # number of equality constraints
        
        # Verify P is positive definite
        eigvals = np.linalg.eigvals(P)
        if np.any(eigvals <= 0):
            raise ValueError("P matrix must be positive definite")
        
        # Set epsilon based on the maximum eigenvalue of P
        self.L = np.max(eigvals)
        self.epsilon = min(epsilon, 1.0 / self.L)  # Conservative choice
        
        # Initialize state
        self.omega = np.zeros(self.n)
        self.alpha = np.zeros(self.m)

    def projection(self, x):
        """Project x onto the feasible set defined by bounds"""
        return np.clip(x, self.omega_l, self.omega_u)

    def rnn_dynamics(self, t, state):
        """
        RNN dynamics following the standard format:
        ε * dω/dt = -ω + P_Ω(ω - (Pω + q + A^T α))
        ε * dα/dt = Aω - b
        """
        n = self.n
        m = self.m
        
        # Split state into omega and alpha
        omega = state[:n]
        alpha = state[n:] if m > 0 else np.array([])

        # Compute gradients
        if m > 0:
            # Primal update with projection
            gradient = self.P @ omega + self.q + self.A.T @ alpha
            omega_dot = (-omega + self.projection(omega - gradient)) / self.epsilon
            
            # Dual update for equality constraints
            constraint_violation = self.A @ omega - self.b
            alpha_dot = constraint_violation / self.epsilon
            
            return np.concatenate([omega_dot, alpha_dot])
        else:
            # Only primal update when no equality constraints
            gradient = self.P @ omega + self.q
            omega_dot = (-omega + self.projection(omega - gradient)) / self.epsilon
            return omega_dot

    def solve(self, t_span=(0, 5), dt=0.01, omega_init=None, alpha_init=None):
        """
        Solve the optimal control problem using RNN
        
        Parameters:
        -----------
        t_span : tuple
            Time span for integration (t_start, t_end)
        dt : float
            Time step for integration
        omega_init : ndarray, optional
            Initial value for omega (warm start)
        alpha_init : ndarray, optional
            Initial value for alpha (warm start)
        
        Returns:
        --------
        omega : ndarray
            Optimal solution
        history : dict
            Solution history for visualization
        """
        # Initialize from provided values or defaults
        if omega_init is not None:
            self.omega = omega_init
        if alpha_init is not None and self.m > 0:
            self.alpha = alpha_init
        
        # Combine initial state
        if self.m > 0:
            state0 = np.concatenate([self.omega, self.alpha])
        else:
            state0 = self.omega
        
        # Set solver options for numerical stability
        options = {
            'rtol': 1e-6,
            'atol': 1e-6,
            'max_step': dt,
            'first_step': dt * 0.1,
            'method': 'RK45'
        }
        
        # Solve ODE system
        solution = solve_ivp(
            fun=self.rnn_dynamics,
            t_span=t_span,
            y0=state0,
            t_eval=np.arange(t_span[0], t_span[1], dt),
            **options
        )
        
        if not solution.success:
            print(f"Warning: Integration failed: {solution.message}")
        
        # Extract solution
        if self.m > 0:
            omega_history = solution.y[:self.n, :]
            alpha_history = solution.y[self.n:, :]
        else:
            omega_history = solution.y
            alpha_history = np.array([])
        
        # Get final solution
        omega_final = omega_history[:, -1]
        
        # Store history for visualization
        history = {
            't': solution.t,
            'omega': omega_history.T,
            'alpha': alpha_history.T if alpha_history.size > 0 else alpha_history
        }
        
        return omega_final, history

def solve_qp_traditional(P, q, A, b, omega_l, omega_u):
    """
    Solve the QP problem using traditional solver (CVXOPT)
    """
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
    
    if A.size > 0:
        A_cvx = matrix(A)
        b_cvx = matrix(b)
        # Solve QP with equality constraints
        start_time = time.time()
        solution = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
    else:
        # Solve QP without equality constraints
        start_time = time.time()
        solution = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    
    solve_time = time.time() - start_time
    return np.array(solution['x']).flatten(), solve_time

def plot_results(history, omega_traditional=None, title="Solution Convergence"):
    """Plot the convergence of the solution"""
    plt.figure(figsize=(12, 6))
    
    # Plot RNN solution
    t = history['t']
    omega = history['omega']
    if len(omega.shape) == 1:
        plt.plot(t, omega, label='ω (RNN)')
    else:
        for i in range(omega.shape[1]):
            plt.plot(t, omega[:, i], label=f'ω{i+1} (RNN)')
    
    # Plot traditional solution if provided
    if omega_traditional is not None:
        if len(omega_traditional.shape) == 1:
            plt.axhline(y=omega_traditional, color='C0', linestyle='--',
                       label='ω (Traditional)')
        else:
            for i in range(len(omega_traditional)):
                plt.axhline(y=omega_traditional[i], color=f'C{i}', linestyle='--',
                           label=f'ω{i+1} (Traditional)')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show() 