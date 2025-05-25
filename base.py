
import numpy as np
from scipy.spatial.transform import Rotation as R

def cal_dis(P1, P2):
    P1 = np.array(P1)
    P2 = np.array(P2)
    if P1.shape != P2.shape:
        raise ValueError("P1 and P2 must have the same shape")
    return np.linalg.norm(P2 - P1)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def cal_2_norm(v):
    return np.linalg.norm(v, ord=2)

def rot2quat(Rmat):
    r = R.from_matrix(Rmat)
    # [x, y, z, w] = r.as_quat()
    return r.as_quat()

def quat2rot(q):
    if len(q) != 4:
        raise ValueError("Quaternion must be of length 4")
    q = np.array(q)
    r = R.from_quat(q)
    return r.as_matrix()

def quat2RPY(q):
    if len(q) != 4:
        raise ValueError("Quaternion must be of length 4")
    q = np.array(q)
    r = R.from_quat(q)
    return r.as_euler('xyz', degrees=True)

def RPY2quat(rpy):
    if len(rpy) != 3:
        raise ValueError("RPY must be of length 3")
    rpy = np.array(rpy)
    r = R.from_euler('xyz', rpy, degrees=True)
    return r.as_quat()

# q1: 起始四元数 [x, y, z, w]
# q2: 结束四元数 [x, y, z, w]
# q: 四元数 [x, y, z, w] 
def new_quat(q1, q2):
    if len(q1) != 4 or len(q2) != 4:
        raise ValueError("Both q1 and q2 must be of length 4")
    q1 = np.array(q1)
    q2 = np.array(q2)
    q = np.zeros(4)
    q[0] = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1]
    q[1] = q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[2] - q1[0] * q2[2] 
    q[2] = q1[3] * q2[2] + q1[0] * q2[1] + q1[2] * q2[3] - q1[1] * q2[0]
    q[3] = q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]
    return normalize(q)

# 计算两个四元数之间的夹角和旋转轴(弧度制)
# 输入的四元数必须是单位四元数 q:[x, y, z, w]
def quat_angle_diff_and_axi(q1, q2):
    # 这里求出的theta是四元数空间的夹角，要转到三维空间时需要乘以2
    if len(q1) != 4 or len(q2) != 4:
        raise ValueError("Both q1 and q2 must be of length 4")
    q1 = np.array(q1)
    q2 = np.array(q2)
    dot_product = np.dot(q1, q2)
    # 确保从q1->q2的旋转角度是最短的
    if dot_product < 0:
        q1 = -q1
        dot_product = -dot_product
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure the value is within the valid range for arccos
    radian = np.arccos(dot_product)
    if np.abs(radian) < 1e-6:
        return 0, np.array([0, 0, 1])
    
    # 计算旋转轴v q = q2 * q1^(-1)
    # q1^(-1) = [-x, -y, -z, w]
    # 这里的q1是单位四元数，所以q1^(-1) = q1的共轭
    q1_conjugate = np.array([-q1[0], -q1[1], -q1[2], q1[3]])
    q = new_quat(q2, q1_conjugate)
    v = normalize(q[:3])
    return radian, v

# radian: 四元数空间的旋转弧度
# t_toal: 总时间
# q1: 起始四元数
# q2: 结束四元数
# t: 当前时间 归一化[0,1]
# 返回插补时刻的四元数
def Slerp_orientation(radian, q1, q2, t):
    # 接近0°, slerp退化为线性插值
    if radian - 0 < 1e-2:
        q = (1 - t) * q1 + t * q2
        q_norm = np.linalg.norm(q)
        q = q / q_norm
    else:
        q = (np.sin((1 - t) * radian) * q1 + np.sin(t * radian) * q2) / np.sin(radian)
    return q

def cal_angle_vel(q1, q2, dt):
    # 计算四元数之间的角速度
    if len(q1) != 4 or len(q2) != 4:
        raise ValueError("Both q1 and q2 must be of length 4")
    q1 = np.array(q1)
    q2 = np.array(q2)
    angle_diff, v = quat_angle_diff_and_axi(q1, q2)
    return angle_diff / dt
