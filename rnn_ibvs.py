import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

class DHRobot:
    """使用DH参数法的机械臂实现"""
    def __init__(self):
        # PUMA560的DH参数 [theta, d, a, alpha]
        self.dh_params = np.array([
            [0,     0.6604, 0,      np.pi/2],  # Joint 1
            [0,     0,      0.4318, 0],        # Joint 2
            [0,     0,      0.0203, -np.pi/2], # Joint 3
            [0,     0.4331, 0,      np.pi/2],  # Joint 4
            [0,     0,      0,      -np.pi/2], # Joint 5
            [0,     0.0563, 0,      0]         # Joint 6
        ])
        self.n_joints = len(self.dh_params)
    
    def transform_matrix(self, theta, d, a, alpha):
        """计算单个DH变换矩阵"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct, -st*ca, st*sa,  a*ct],
            [st, ct*ca,  -ct*sa, a*st],
            [0,  sa,     ca,     d],
            [0,  0,      0,      1]
        ])
    
    def forward_kinematics(self, joint_angles):
        """计算正向运动学"""
        T = np.eye(4)
        transforms = []
        
        for i in range(self.n_joints):
            theta = joint_angles[i] + self.dh_params[i, 0]
            d = self.dh_params[i, 1]
            a = self.dh_params[i, 2]
            alpha = self.dh_params[i, 3]
            
            Ti = self.transform_matrix(theta, d, a, alpha)
            T = T @ Ti
            transforms.append(T.copy())
        
        return T, transforms
    
    def jacobian(self, joint_angles):
        """计算机械臂雅可比矩阵"""
        T, transforms = self.forward_kinematics(joint_angles)
        J = np.zeros((6, self.n_joints))
        
        # 末端执行器位置
        end_pos = T[:3, 3]
        
        for i in range(self.n_joints):
            if i == 0:
                z_axis = np.array([0, 0, 1])
                pos = np.zeros(3)
            else:
                z_axis = transforms[i-1][:3, 2]
                pos = transforms[i-1][:3, 3]
            
            # 计算线速度雅可比
            J[:3, i] = np.cross(z_axis, end_pos - pos)
            # 计算角速度雅可比
            J[3:, i] = z_axis
        
        return J

class CameraParams:
    def __init__(self):
        self.lambda_cam = 8e-3  # 焦距
        self.u0 = 256  # 图像中心u坐标
        self.v0 = 256  # 图像中心v坐标
        self.ax = 8e4  # x方向像素比例
        self.ay = 8e4  # y方向像素比例

class RNNIBVSController:
    def __init__(self, n_joints=6, n_features=4):
        self.n = n_joints
        self.n_features = n_features
        self.camera = CameraParams()
        
        # 大幅增加控制增益以加快收敛
        self.gamma = 5.0  # 增大控制增益
        self.epsilon = 0.05  # 减小以加快响应
        self.eps_k = 0.01  # 减小以加快深度估计
        self.v = 30.0  # 增大速度约束
        
        # 进一步放宽约束
        self.omega_lower = np.array([-0.5] * self.n)
        self.omega_upper = np.array([0.5] * self.n)
        self.theta_lower = np.array([-np.pi] * self.n)
        self.theta_upper = np.array([np.pi] * self.n)
        
        # 减小阻尼
        self.P = np.eye(self.n) * 0.1
        self.q = np.zeros(self.n)
        
        # 状态初始化
        self.reset_state()
    
    def reset_state(self):
        """重置控制器状态"""
        self.omega = np.zeros(self.n)
        # 初始化k向量，确保深度估计为正
        k_init = np.zeros(4 * self.n_features)
        for i in range(self.n_features):
            k_init[i*4] = 1.0/0.6  # 初始深度估计
            k_init[i*4+1:i*4+4] = 0.0  # 其他参数初始化为0
        self.k = k_init
        self.alpha = np.zeros(2 * self.n_features)
        self.beta = np.zeros(6 * self.n_features)
    
    def compute_C_J0(self, s):
        """计算C矩阵和J0矩阵"""
        features = s.reshape(-1, 2)
        C_list = []
        J0_list = []
        
        for feature in features:
            u, v = feature
            # 归一化图像坐标
            pu = (u - self.camera.u0) / self.camera.ax
            pv = (v - self.camera.v0) / self.camera.ay
            
            # 构建C矩阵
            C = np.zeros((6, 4))
            C[0, 0] = pu / self.camera.lambda_cam
            C[1, 0] = pv / self.camera.lambda_cam
            C[2, 0] = 1
            C[3, 1] = 1
            C[4, 2] = 1
            C[5, 3] = 1
            
            # 构建J0矩阵
            J0 = np.zeros((2, 4))
            J0[0] = [0, pu*pv/self.camera.lambda_cam, 
                     -(pu**2 + self.camera.lambda_cam**2)/self.camera.lambda_cam, pv]
            J0[1] = [0, (self.camera.lambda_cam**2 + pv**2)/self.camera.lambda_cam,
                     -pu*pv/self.camera.lambda_cam, -pu]
            
            # 应用相机内参
            J0 *= np.array([[self.camera.ax, self.camera.ax, self.camera.ax, self.camera.ax],
                           [self.camera.ay, self.camera.ay, self.camera.ay, self.camera.ay]])
            
            # 限制数值范围
            J0 = np.clip(J0, -1e3, 1e3)
            
            C_list.append(C)
            J0_list.append(J0)
        
        # 构建分块矩阵
        C_full = np.zeros((6 * self.n_features, 4 * self.n_features))
        J0_full = np.zeros((2 * self.n_features, 4 * self.n_features))
        
        for i in range(self.n_features):
            C_full[i*6:(i+1)*6, i*4:(i+1)*4] = C_list[i]
            J0_full[i*2:(i+1)*2, i*4:(i+1)*4] = J0_list[i]
        
        return C_full, J0_full
    
    def project_omega(self, omega, theta):
        """投影操作确保关节速度和位置约束"""
        omega_l = np.maximum(self.omega_lower, -self.v * (theta - self.theta_lower))
        omega_u = np.minimum(self.omega_upper, -self.v * (theta - self.theta_upper))
        return np.clip(omega, omega_l, omega_u)
    
    def rnn_dynamics(self, state, t, Jr, s, sd):
        """RNN动力学方程"""
        omega = state[:self.n]
        k = state[self.n:self.n+4*self.n_features]
        alpha = state[self.n+4*self.n_features:self.n+4*self.n_features+2*self.n_features]
        beta = state[-6*self.n_features:]
        
        # 计算当前特征点的C和J0矩阵
        C, J0 = self.compute_C_J0(s)
        
        # 构建分块对角Jr矩阵
        Jr_full = np.zeros((6 * self.n_features, self.n))
        for i in range(self.n_features):
            Jr_full[i*6:(i+1)*6, :] = Jr
        
        # 添加数值稳定性处理
        error = s - sd
        error = np.clip(error, -100, 100)  # 限制误差范围
        
        # 计算各状态的导数
        d_omega = (-omega + self.project_omega(
            omega - self.P @ omega - self.q - Jr_full.T @ beta,
            state[:self.n])) / self.epsilon
        
        d_k = (-J0.T @ alpha + C.T @ beta) / self.eps_k
        d_alpha = (J0 @ k + self.gamma * error) / self.epsilon
        d_beta = (Jr_full @ omega - C @ k) / self.epsilon
        
        # 限制状态导数的范围
        d_omega = np.clip(d_omega, -1, 1)
        d_k = np.clip(d_k, -1, 1)
        d_alpha = np.clip(d_alpha, -1, 1)
        d_beta = np.clip(d_beta, -1, 1)
        
        return np.concatenate([d_omega, d_k, d_alpha, d_beta])
    
    def compute_control(self, theta, Jr, s, sd, dt=0.0005):
        """计算控制输入"""
        state = np.concatenate([self.omega, self.k, self.alpha, self.beta])
        new_state = odeint(self.rnn_dynamics, state, [0, dt], args=(Jr, s, sd))[1]
        
        self.omega = new_state[:self.n]
        self.k = new_state[self.n:self.n+4*self.n_features]
        self.alpha = new_state[self.n+4*self.n_features:self.n+4*self.n_features+2*self.n_features]
        self.beta = new_state[-6*self.n_features:]
        
        return self.omega

class RobotSimulator:
    def __init__(self):
        self.robot = DHRobot()
        self.n_features = 4  # 添加特征点数量属性
        self.controller = RNNIBVSController(n_joints=6, n_features=self.n_features)
        self.camera = CameraParams()
        
        # 调整目标点布局
        self.target_points_3d = np.array([
            [0.2, 0.2, 0.6],    # 增大特征点间距
            [-0.2, 0.2, 0.6],
            [-0.2, -0.2, 0.6],
            [0.2, -0.2, 0.6]
        ])
        
        self.feature_trajectories = []
        self.depth_estimates = []
        self.true_depths = []
    
    def project_points(self, points_3d):
        """将3D点投影到图像平面"""
        points_2d = np.zeros((len(points_3d), 2))
        for i, p in enumerate(points_3d):
            x, y, z = p
            u = self.camera.u0 + self.camera.ax * self.camera.lambda_cam * x / z
            v = self.camera.v0 + self.camera.ay * self.camera.lambda_cam * y / z
            points_2d[i] = [u, v]
        return points_2d
    
    def get_current_features(self, theta):
        """获取当前图像特征"""
        # 计算末端执行器位姿
        T, _ = self.robot.forward_kinematics(theta)
        
        # 将目标点转换到相机坐标系
        points_cam = np.array([np.linalg.inv(T) @ np.append(p, 1) for p in self.target_points_3d])
        points_cam = points_cam[:, :3]
        
        return self.project_points(points_cam)
    
    def simulate(self, theta_init, max_iter=1000, dt=0.001):
        """运行仿真"""
        theta = theta_init.copy()
        trajectory = []
        feature_errors = []
        
        # 计算期望特征
        desired_features = self.project_points(self.target_points_3d)
        desired_features = desired_features.flatten()
        
        # 获取初始特征和深度
        current_features = self.get_current_features(theta)
        self.feature_trajectories.append(current_features.copy())
        
        T, _ = self.robot.forward_kinematics(theta)
        points_cam = np.array([np.linalg.inv(T) @ np.append(p, 1) for p in self.target_points_3d])
        initial_depths = points_cam[:, 2]
        self.true_depths.append(initial_depths)
        
        # 使用真实深度初始化深度估计
        k = self.controller.k.reshape(-1, 4)
        for i in range(self.n_features):
            k[i, 0] = 1.0 / max(0.1, initial_depths[i])  # 确保深度估计为正
        self.controller.k = k.flatten()
        self.depth_estimates.append(initial_depths)
        
        last_error = float('inf')
        no_improvement_count = 0
        min_error = float('inf')
        
        for i in range(max_iter):
            current_features = self.get_current_features(theta)
            current_features_flat = current_features.flatten()
            
            # 记录轨迹
            self.feature_trajectories.append(current_features.copy())
            
            # 计算控制输入
            Jr = self.robot.jacobian(theta)
            omega = self.controller.compute_control(
                theta, Jr, current_features_flat, desired_features, dt)
            
            # 限制控制输入变化
            if len(trajectory) >= 2:
                last_omega = (trajectory[-1] - trajectory[-2]) / dt
                max_change = 0.2  # 增大变化率限制
                omega = np.clip(omega, last_omega - max_change, last_omega + max_change)
            
            # 更新状态
            theta += omega * dt
            trajectory.append(theta.copy())
            
            # 计算误差
            current_error = np.linalg.norm(current_features_flat - desired_features)
            feature_errors.append(current_error)
            min_error = min(min_error, current_error)
            
            # 更新深度估计
            T, _ = self.robot.forward_kinematics(theta)
            points_cam = np.array([np.linalg.inv(T) @ np.append(p, 1) for p in self.target_points_3d])
            true_depths = points_cam[:, 2]
            self.true_depths.append(true_depths)
            
            k = self.controller.k.reshape(-1, 4)
            current_depth_est = np.zeros(self.n_features)
            for j in range(self.n_features):
                if abs(k[j, 0]) > 1e-6:
                    current_depth_est[j] = 1.0 / k[j, 0]
                else:
                    current_depth_est[j] = self.depth_estimates[-1][j]
                # 确保深度估计为正
                current_depth_est[j] = max(0.1, current_depth_est[j])
            
            # 使用自适应步长的滑动平均
            alpha = 0.95 - 0.5 * min(1.0, current_error / feature_errors[0])
            smoothed_depth = alpha * np.array(self.depth_estimates[-1]) + (1 - alpha) * current_depth_est
            self.depth_estimates.append(smoothed_depth)
            
            # 检查收敛
            if current_error < 5.0:  # 使用更实际的收敛条件
                print(f"Converged after {i} iterations")
                break
            
            # 检查局部最小值
            if abs(current_error - last_error) < 1e-6:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            
            if no_improvement_count > 50:
                if current_error < 1.5 * min_error:
                    print(f"Acceptable solution found after {i} iterations")
                    break
                else:
                    print(f"Stuck in local minimum after {i} iterations")
                    break
            
            last_error = current_error
            
            if i % 100 == 0:
                print(f"Iteration {i}, Error: {current_error:.6f}")
        
        return np.array(trajectory), np.array(feature_errors)
    
    def plot_robot(self, ax, theta):
        """在3D图中绘制机械臂"""
        _, transforms = self.robot.forward_kinematics(theta)
        points = np.zeros((len(transforms) + 1, 3))
        points[0] = [0, 0, 0]
        
        for i, T in enumerate(transforms):
            points[i+1] = T[:3, 3]
        
        # 绘制机械臂连杆
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
        
        # 绘制坐标系
        for T in transforms:
            self.plot_frame(ax, T)
    
    def plot_frame(self, ax, T, scale=0.1):
        """绘制坐标系"""
        origin = T[:3, 3]
        x_axis = origin + scale * T[:3, 0]
        y_axis = origin + scale * T[:3, 1]
        z_axis = origin + scale * T[:3, 2]
        
        ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 'r-')
        ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 'g-')
        ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 'b-')
    
    def visualize_results(self, trajectory, feature_errors):
        """增强的可视化结果"""
        fig = plt.figure(figsize=(20, 10))
        
        # 1. 关节角度轨迹
        ax1 = fig.add_subplot(231)
        for i in range(6):
            ax1.plot(trajectory[:, i], label=f'Joint {i+1}')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Joint Angle (rad)')
        ax1.set_title('Joint Trajectories')
        ax1.legend()
        
        # 2. 特征误差收敛
        ax2 = fig.add_subplot(232)
        ax2.plot(feature_errors)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Feature Error')
        ax2.set_title('Feature Error Convergence')
        ax2.set_yscale('log')
        
        # 3. 3D机械臂构型
        ax3 = fig.add_subplot(233, projection='3d')
        self.plot_robot(ax3, trajectory[-1])
        ax3.set_title('Final Robot Configuration')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_box_aspect([1,1,1])
        
        # 4. 图像平面特征点轨迹
        ax4 = fig.add_subplot(234)
        feature_trajectories = np.array(self.feature_trajectories)
        for i in range(4):
            ax4.plot(feature_trajectories[:, i, 0], feature_trajectories[:, i, 1], 
                    'o-', markersize=2, label=f'Point {i+1}')
            # 标记起点和终点
            ax4.plot(feature_trajectories[0, i, 0], feature_trajectories[0, i, 1], 
                    'go', markersize=8, label=f'Start {i+1}')
            ax4.plot(feature_trajectories[-1, i, 0], feature_trajectories[-1, i, 1], 
                    'ro', markersize=8, label=f'End {i+1}')
        ax4.set_xlabel('u (pixels)')
        ax4.set_ylabel('v (pixels)')
        ax4.set_title('Feature Points Trajectory in Image Plane')
        ax4.grid(True)
        
        # 5. 深度估计对比
        ax5 = fig.add_subplot(235)
        true_depths = np.array(self.true_depths)
        depth_estimates = np.array(self.depth_estimates)
        for i in range(4):
            ax5.plot(true_depths[:, i], label=f'True Depth {i+1}')
            ax5.plot(depth_estimates[:, i], '--', label=f'Estimated Depth {i+1}')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Depth (m)')
        ax5.set_title('Depth Estimation vs Ground Truth')
        ax5.legend()
        
        # 6. 3D特征点轨迹
        ax6 = fig.add_subplot(236, projection='3d')
        # 计算3D点轨迹
        feature_3d_trajectories = []
        for theta in trajectory:
            T, _ = self.robot.forward_kinematics(theta)
            points_cam = np.array([np.linalg.inv(T) @ np.append(p, 1) for p in self.target_points_3d])
            feature_3d_trajectories.append(points_cam[:, :3])
        feature_3d_trajectories = np.array(feature_3d_trajectories)
        
        # 绘制3D轨迹
        for i in range(4):
            ax6.plot3D(feature_3d_trajectories[:, i, 0], 
                      feature_3d_trajectories[:, i, 1],
                      feature_3d_trajectories[:, i, 2],
                      label=f'Point {i+1}')
            # 标记起点和终点
            ax6.scatter(feature_3d_trajectories[0, i, 0],
                       feature_3d_trajectories[0, i, 1],
                       feature_3d_trajectories[0, i, 2], c='g', marker='o')
            ax6.scatter(feature_3d_trajectories[-1, i, 0],
                       feature_3d_trajectories[-1, i, 1],
                       feature_3d_trajectories[-1, i, 2], c='r', marker='o')
        ax6.set_xlabel('X (m)')
        ax6.set_ylabel('Y (m)')
        ax6.set_zlabel('Z (m)')
        ax6.set_title('3D Feature Points Trajectory')
        
        plt.tight_layout()
        plt.show()

def main():
    simulator = RobotSimulator()
    
    # 使用更好的初始位置
    theta_init = np.array([0.0, -0.4, 0.0, 0.0, -0.4, 0.0])
    
    trajectory, feature_errors = simulator.simulate(theta_init)
    simulator.visualize_results(trajectory, feature_errors)

if __name__ == "__main__":
    main()
