# Robot Trajectory Optimization with RNN

This project implements a robot trajectory optimization system using Recurrent Neural Networks (RNN) and traditional optimization methods. It provides both simulation and visualization capabilities for robot motion planning and control.
The paper is “Image-Based_Visual_Servoing_of_Manipulators_With_Unknown_Depth_A_Recurrent_Neural_Network_Approach” from TNNLS ,I read it and write this code.

## Features

- Robot trajectory optimization using RNN-based approach
- Traditional optimization using CVX
- Visual servoing control implementation
- 3D visualization of robot configurations
- Comprehensive error analysis and convergence plots
- Support for multi-joint robot systems

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- SciPy
- CVXOPT

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/robot-trajectory-optimization.git
cd robot-trajectory-optimization
```

2. Install required packages:
```bash
pip install numpy matplotlib scipy cvxopt
```

## Usage

### Basic Trajectory Optimization

Run the main example:
```bash
python example_robot_trajectory.py
```

This will:
- Generate a robot trajectory optimization problem
- Solve it using both RNN and CVX methods
- Visualize the results including:
  - Joint trajectories
  - End-effector path
  - Initial and final configurations
  - Convergence analysis
  - Velocity profiles

### Visual Servoing Control

To run the visual servoing simulation:
```bash
python ibvs_simulation.py
```

## Project Structure

```
.
├── example_robot_trajectory.py    # Main example script
├── rnn_optimal_control.py        # RNN optimization controller
├── ibvs_rnn_verification.py      # Visual servoing verification
├── verify_ibvs_rnn.py           # RNN controller tests
├── rnn_ibvs.py                  # RNN visual servoing implementation
├── ibvs_simulation.py           # Visual servoing simulation
└── README.md                    # This file
```

## Key Components

### RNN Optimal Control
The `RNNOptimalControlSolver` class implements the RNN-based optimization approach, which provides:
- Real-time optimization capabilities
- Constraint handling
- Convergence guarantees

### Visual Servoing
The visual servoing implementation includes:
- Image-based visual servoing (IBVS)
- RNN-based control law
- Depth-invariant control

## Results

The optimization results include:
1. Joint angle trajectories
2. End-effector path visualization
3. Robot configuration snapshots
4. Convergence analysis
5. Velocity profiles

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


For questions and feedback, please open an issue in the GitHub repository.
