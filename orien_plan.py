import base
import numpy as np

def test_orien_plan(ori_rpy1, ori_rpy2, ori_rpy_3):
    quat1 = base.RPY2quat(ori_rpy1)
    quat2 = base.RPY2quat(ori_rpy2)
    quat3 = base.RPY2quat(ori_rpy_3)
    print("quat1:", quat1)
    print("quat2:", quat2)
    print("quat3:", quat3)

    # 计算四元数之间的角度差
    angle_diff1, v1 = base.quat_angle_diff_and_axi(quat1, quat2)
    angle_diff2, v2 = base.quat_angle_diff_and_axi(quat2, quat3)
    print("angle_diff1:", angle_diff1)
    print("angle_diff2:", angle_diff2)
    
    dt = 0.004
    t_total_1 = 2.0
    t_total_2 = 3.0

    steps_1 = int(t_total_1 / dt)
    steps_2 = int(t_total_2 / dt)

    angle1 = np.zeros((steps_1, 1))
    angle2 = np.zeros((steps_2, 1))

    angle_vel1 = angle1
    angle_vel2 = angle2
    # 旋转轴
    v1 = np.zeros((steps_1, 3))
    v2 = np.zeros((steps_2, 3))

    # q1->q2插补的四元数
    q1_2 = np.zeros((steps_1, 4))
    for i in range(steps_1):
        t = i / steps_1 * dt
        q1_2[i] = base.Slerp_orientation(angle_diff1, quat1, quat2, t)
        angle1[i], v1[i] = base.quat_angle_diff_and_axi(quat1, q1_2[i])

    angle_vel1[0] = 0.0    
    for i in range(1, steps_1 - 1):
        angle_vel1[i] = (angle1[i + 1] - angle1[i - 1]) / (2 * dt)
    angle_vel1[steps_1 - 1] = (angle1[steps_1 - 1] - angle1[steps_1 - 2]) / dt

    q2_3 = np.zeros((steps_2, 4))
    for i in range(steps_2):
        t = i / steps_2 * dt
        q2_3[i] = base.Slerp_orientation(angle_diff2, quat2, quat3, t)
        angle2[i], v2[i] = base.quat_angle_diff_and_axi(quat2, q2_3[i])
    angle_vel2[0] = 0.0
    for i in range(1, steps_2 - 1):
        angle_vel2[i] = (angle2[i + 1] - angle2[i - 1]) / (2 * dt)
    angle_vel2[steps_2 - 1] = (angle2[steps_2 - 1] - angle2[steps_2 - 2]) / dt