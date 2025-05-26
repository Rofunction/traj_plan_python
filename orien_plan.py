import base
import numpy as np
import matplotlib.pyplot as plt

def test_orien_plan(ori_rpy1, ori_rpy2, ori_rpy3):
    # quat1 = base.RPY2quat(ori_rpy1)
    # quat2 = base.RPY2quat(ori_rpy2)
    # quat3 = base.RPY2quat(ori_rpy_3)
    quat1 = ori_rpy1
    quat2 = ori_rpy2
    quat3 = ori_rpy3
    print("quat1:", quat1)
    print("quat2:", quat2)
    print("quat3:", quat3)

    # 计算四元数之间的角度差
    quat1_new, angle_diff1, v01 = base.quat_angle_diff_and_axi_first(quat1, quat2)
    quat2_new, angle_diff2, v02 = base.quat_angle_diff_and_axi_first(quat2, quat3)
    print("angle_diff1:", angle_diff1)
    print("angle_diff2:", angle_diff2)
    
    dt = 0.004
    t_total_1 = 2.0
    t_total_2 = 2.0
    # 时间归一化系数
    coef1 = 1.0 / t_total_1
    coef2 = 1.0 / t_total_2

    steps_1 = int(t_total_1 / dt)
    steps_2 = int(t_total_2 / dt)

    angle1 = np.zeros((steps_1 + 1, 1))
    angle2 = np.zeros((steps_2 + 1, 1))

    angle_vel1 = np.zeros((steps_1 + 1, 1))
    angle_vel2 = np.zeros((steps_2 + 1, 1))
    # 旋转轴
    v1 = np.zeros((steps_1 + 1, 3))
    v1[0] = v01
    v2 = np.zeros((steps_2 + 1, 3))
    v2[0] = v02

    # q1->q2插补的四元数
    q1_2 = np.zeros((steps_1 + 1, 4))
    q1_2[0] = quat1
    for i in range(1, steps_1 + 1):
        t = i * dt * coef1
        q1_2[i] = base.Slerp_orientation(angle_diff1, quat1_new, quat2, t)
        angle1[i], v1[i] = base.quat_angle_diff_and_axi(quat1, q1_2[i], v01) # 理论上初始轴只是用于第一次

    # 计算角速度
    for i in range(1, steps_1):
        angle_vel1[i] = (angle1[i + 1] - angle1[i - 1]) / (2 * dt)
    angle_vel1[steps_1] = (angle1[steps_1] - angle1[steps_1 - 1]) / dt

    # q2->q3插补的四元数
    q2_3 = np.zeros((steps_2 + 1, 4))
    for i in range(1, steps_2 + 1):
        t = i * dt * coef2
        q2_3[i] = base.Slerp_orientation(angle_diff2, quat2_new, quat3, t)
        angle2[i], v2[i] = base.quat_angle_diff_and_axi(quat2, q2_3[i], v02) # 理论上初始轴只是用于第一次

    # 计算角速度
    for i in range(1, steps_2):
        angle_vel2[i] = (angle2[i + 1] - angle2[i - 1]) / (2 * dt)
    angle_vel2[steps_2] = (angle2[steps_2] - angle2[steps_2 - 1]) / dt


    # plot 轴
    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')
    axs.plot(v1[:, 0], v1[:, 1], v1[:, 2], label='v1')
    axs.plot(v2[:, 0], v2[:, 1], v2[:, 2], label='v2')
    axs.scatter(v1[0, 0], v1[0, 1], v1[0, 2], c='r', marker='o', label='v1_start')
    axs.scatter(v1[steps_1, 0], v1[steps_1, 1], v1[steps_1, 2], c='g', marker='o', label='v1_end')
    axs.scatter(v2[0, 0], v2[0, 1], v2[0, 2], c='r', marker='o', label='v2_start')
    axs.scatter(v2[steps_2, 0], v2[steps_2, 1], v2[steps_2, 2], c='g', marker='o', label='v2_end')
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.set_zlabel('Z')
    axs.set_title('Rotation Axis')
    axs.legend()

    # plot 角度
    fig = plt.figure()
    axs = fig.add_subplot(111)
    axs.plot(np.arange(steps_1 + 1) * dt, angle1, marker='.', label='angle1')
    axs.plot(np.arange(steps_2 + 1) * dt, angle2, marker='.', label='angle2')
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Angle (rad)')
    axs.set_title('Angle over Time')
    axs.legend()

    # plot 角速度
    fig = plt.figure()
    axs = fig.add_subplot(111)
    axs.plot(np.arange(steps_1 + 1) * dt, angle_vel1, marker='.', label='angle_vel1')
    axs.plot(np.arange(steps_2 + 1) * dt, angle_vel2, marker='.' ,label='angle_vel2')
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Angular Velocity (rad/s)')
    axs.set_title('Angular Velocity over Time')
    axs.legend()
    plt.show()