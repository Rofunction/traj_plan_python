import ruckig
from ruckig import Ruckig, InputParameter, OutputParameter, Result, Trajectory
import numpy as np
import matplotlib.pyplot as plt
import base
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as Rot

def pos_interpolation():
    rr = 0.2 # m
    input = InputParameter(1)
    
    input.max_velocity = [2.5]
    input.max_acceleration = [12.5]
    input.max_jerk = [25]

    P1 = np.array([0.2, 0.3, 0.5])
    P2 = np.array([1.0, 0.1, 0.8])
    P3 = np.array([0.6, 0.0, 0.4])
    
    dis_P2_P1 = base.cal_dis(P1, P2)
    dis_P3_P2 = base.cal_dis(P3, P2)
    P2_1_norm = base.normalize(P2 - P1)
    P2_3_norm = base.normalize(P3 - P2)

    input.current_position = [0]
    input.current_velocity = [0.0]
    input.current_acceleration = [0.0]

    input.target_position = [dis_P2_P1]
    input.target_velocity = [0.0]
    input.target_acceleration = [0.0]

    ruckig = Ruckig(1)
    traj_1 = Trajectory(1)

    res = ruckig.calculate(input, traj_1)
    if res == Result.ErrorInvalidInput:
        raise Exception("Invalid input parameters")
    

    input.target_position = [dis_P3_P2]
    input.target_velocity = [0.0]
    input.target_acceleration = [0.0]

    traj_2 = Trajectory(1)
    res = ruckig.calculate(input, traj_2)
    if res == Result.ErrorInvalidInput:
        raise Exception("Invalid input parameters")
    
    print(f'Trajectory1 duration: {traj_1.duration: 0.4f} [s]')
    print(f'Trajectory2 duration: {traj_2.duration: 0.4f} [s]')

    traj_time_1 = traj_1.duration
    traj_nums_1 = int(traj_time_1 / 0.004)
    pos_1 = np.zeros((traj_nums_1, 3), dtype=float)
    vel_1 = np.zeros((traj_nums_1, 3), dtype=float)
    acc_1 = np.zeros((traj_nums_1, 3), dtype=float)


    traj_time_2 = traj_2.duration
    traj_nums_2 = int(traj_time_2 / 0.004)
    pos_2 = np.zeros((traj_nums_2, 3), dtype=float)
    vel_2 = np.zeros((traj_nums_2, 3), dtype=float)
    acc_2 = np.zeros((traj_nums_2, 3), dtype=float)

    # 混成后的轨迹
    pos_3 = np.zeros((traj_nums_1 + traj_nums_2, 3), float)
    vel_3 = np.zeros((traj_nums_1 + traj_nums_2, 3), float)
    acc_3 = np.zeros((traj_nums_1 + traj_nums_2, 3), float)

    pos_temp = 0.0
    vel_temp = 0.0
    acc_temp = 0.0
    for i in range(traj_nums_1):
        pos_temp, vel_temp, acc_temp = traj_1.at_time(i * 0.004)
        pos_1[i, :] = P1 + P2_1_norm * pos_temp
        vel_1[i, :] = P2_1_norm * vel_temp
        acc_1[i, :] = P2_1_norm * acc_temp
    for i in range(traj_nums_2):
        pos_temp, vel_temp, acc_temp = traj_2.at_time(i * 0.004)
        pos_2[i, :] = P2 + P2_3_norm * pos_temp
        vel_2[i, :] = P2_3_norm * vel_temp
        acc_2[i, :] = P2_3_norm * acc_temp

    # 速度混成
    for i in range(traj_nums_1):
        distance = base.cal_dis(pos_1[i, :], P2)
        pos_3[i, :] = pos_1[i, :]
        vel_3[i, :] = vel_1[i, :]
        acc_3[i, :] = acc_1[i, :]
        if distance - rr <= 0:
            idx_1 = i
            break
    no1 = traj_nums_1 - idx_1 # 剩余的点数(包含idx_1)

    # 实际混成的idx
    idx_rl = 0
    # 第二段轨迹实际开始参与插补的点
    idx_rl2 = 0
    for i in range(traj_nums_2):
        distance = base.cal_dis(P2, pos_2[i, :])
        if distance - rr >= 0:
            no2 = i
            break
    idx_2 = traj_nums_2 - no2 # 剩余的点数(包含idx_2)
    # 混成段
    if no1 > no2:
        idx_rl = idx_1 + no1 - no2
        for i in range(0, no1 - no2):
            pos_3[i + idx_1, :] = pos_1[i + idx_1, :]
            vel_3[i + idx_1, :] = vel_1[i + idx_1, :]
            acc_3[i + idx_1, :] = acc_1[i + idx_1, :]
        for i in range(0, no2):
            pos_3[i + idx_1 + no1 - no2, :] = pos_2[i, :] - P2 + pos_1[i + idx_1 + no1 - no2, :]
            vel_3[i + idx_1 + no1 - no2, :] = vel_2[i, :] + vel_1[i + idx_1 + no1 - no2, :]
            acc_3[i + idx_1 + no1 - no2, :] = acc_2[i, :] + acc_1[i + idx_1 + no1 - no2, :]
    else:
        idx_rl = idx_1
        idx_rl2 = no2 - no1
        for i in range(0, no1):
            pos_3[i + idx_1, :] = pos_1[i + idx_1, :] + pos_2[i + no2 - no1, :] - P2
            vel_3[i + idx_1, :] = vel_1[i + idx_1, :] + vel_2[i + no2 - no1, :]
            acc_3[i + idx_1, :] = acc_1[i + idx_1, :] + acc_2[i + no2 - no1, :]
    
    for i in range(0, idx_2):
        pos_3[i + idx_1 + no1, :] = pos_2[no2 + i, :]
        vel_3[i + idx_1 + no1, :] = vel_2[no2 + i, :]
        acc_3[i + idx_1 + no1, :] = acc_2[no2 + i, :]


    # 笛卡尔图
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    t1 = np.arange(traj_nums_1) * 0.004
    t2 = np.arange(traj_nums_2) * 0.004 + t1[idx_rl - idx_rl2]
    t3 = np.arange(traj_nums_1 + idx_2) * 0.004
    len3 = len(t3)
    axs[0].plot(t1, pos_1[:, 0], label='traj_1_x')
    axs[0].plot(t2, pos_2[:, 0] , label='traj_2_x')
    axs[0].plot(t3, pos_3[0:len3, 0], label='traj_3_x')
    axs[0].scatter(idx_rl * 0.004, pos_1[idx_rl, 0], c='r', marker='o', label='pos')
    axs[0].set_ylabel('pos')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(t1, vel_1[:, 0], label='vel1_x')
    axs[1].plot(t2, vel_2[:, 0], label='vel2_x')
    axs[1].plot(t3, vel_3[0:len3, 0], label='vel3_x')
    axs[1].scatter(idx_rl * 0.004, vel_1[idx_rl, 0], c='r', marker='o', label='pos')
    axs[1].set_ylabel('vel')
    axs[1].grid()
    axs[1].legend()

    axs[2].plot(t1, acc_1[:, 0], label='acc1')
    axs[2].plot(t2, acc_2[:, 0], label='acc2')
    axs[2].plot(t3, acc_3[0:len3, 0], label='acc3')
    axs[2].scatter(idx_rl * 0.004, acc_1[idx_rl, 0], c='r', marker='o', label='pos')
    axs[2].set_ylabel('acc')
    axs[2].grid()
    axs[2].legend()

    plt.xlabel('time [s]')
    plt.suptitle('X_label')
    # y 方向对比
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axs[0].plot(t1, pos_1[:, 1], label='traj_1_y')
    axs[0].plot(t2, pos_2[:, 1] , label='traj_2_y')
    axs[0].plot(t3, pos_3[0:len3, 1], label='traj_3_y')
    axs[0].scatter(idx_rl * 0.004, pos_1[idx_rl, 1], c='r', marker='o', label='pos')
    axs[0].set_ylabel('pos')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(t1, vel_1[:, 1], label='vel1_y')
    axs[1].plot(t2, vel_2[:, 1], label='vel2_y')
    axs[1].plot(t3, vel_3[0:len3, 1], label='vel3_y')
    axs[1].scatter(idx_rl * 0.004, vel_1[idx_rl, 1], c='r', marker='o', label='pos')
    axs[1].set_ylabel('vel')
    axs[1].grid()
    axs[1].legend()

    axs[2].plot(t1, acc_1[:, 1], label='acc1')
    axs[2].plot(t2, acc_2[:, 1], label='acc2')
    axs[2].plot(t3, acc_3[0:len3, 1], label='acc3')
    axs[2].scatter(idx_rl * 0.004, acc_1[idx_rl, 1], c='r', marker='o', label='pos')
    axs[2].set_ylabel('acc')
    axs[2].grid()
    axs[2].legend()

    plt.xlabel('time_1 [s]')
    plt.suptitle('Y_label')
    # Z方向对比
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(t1, pos_1[:, 2], label='traj_1_z')
    axs[0].plot(t2, pos_2[:, 2] , label='traj_2_z')
    axs[0].plot(t3, pos_3[0:len3, 2], label='traj_3_z')
    axs[0].scatter(idx_rl * 0.004, pos_1[idx_rl, 2], c='r', marker='o', label='pos')
    axs[0].set_ylabel('pos')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(t1, vel_1[:, 2], label='vel1_z')
    axs[1].plot(t2, vel_2[:, 2], label='vel2_z')
    axs[1].plot(t3, vel_3[0:len3, 2], label='vel3_z')
    axs[1].scatter(idx_rl * 0.004, vel_1[idx_rl, 2], c='r', marker='o', label='pos')
    axs[1].set_ylabel('vel')
    axs[1].grid()
    axs[1].legend()

    axs[2].plot(t1, acc_1[:, 2], label='acc1')
    axs[2].plot(t2, acc_2[:, 2], label='acc2')
    axs[2].plot(t3, acc_3[0:len3, 2], label='acc3')
    axs[2].scatter(idx_rl * 0.004, acc_1[idx_rl, 2], c='r', marker='o', label='pos')
    axs[2].set_ylabel('acc')
    axs[2].grid()
    axs[2].legend()

    plt.xlabel('time [s]')
    plt.suptitle('Z_label')
    '''
    # # 轨迹1
    # time_1 = np.arange(traj_nums_1) * 0.004
    # fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    # axs[0].plot(time_1, pos_1[:, 0], label='x_pos')
    # axs[0].plot(time_1, vel_1[:, 0], label='x_vel')
    # axs[0].plot(time_1, acc_1[:, 0], label='x_acc')
    # axs[0].set_ylabel('x')
    # axs[0].grid()
    # axs[0].legend()

    # axs[1].plot(time_1, pos_1[:, 1], label='y_pos')
    # axs[1].plot(time_1, vel_1[:, 1], label='y_vel')
    # axs[1].plot(time_1, acc_1[:, 1], label='y_acc')
    # axs[1].set_ylabel('y')
    # axs[1].grid()
    # axs[1].legend()

    # axs[2].plot(time_1, pos_1[:, 2], label='z_pos')
    # axs[2].plot(time_1, vel_1[:, 2], label='z_vel')
    # axs[2].plot(time_1, acc_1[:, 2], label='z_acc')

    # axs[2].set_ylabel('z')
    # axs[2].grid()
    # axs[2].legend()
    # plt.xlabel('time_1 [s]')
    # plt.suptitle('Trajectory2')

    # # 轨迹2
    # time_2 = np.arange(traj_nums_2) * 0.004
    # fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    # axs[0].plot(time_2, pos_2[:, 0], label='x_pos')
    # axs[0].plot(time_2, vel_2[:, 0], label='x_vel')
    # axs[0].plot(time_2, acc_2[:, 0], label='x_acc')
    # axs[0].set_ylabel('x')
    # axs[0].grid()
    # axs[0].legend()

    # axs[1].plot(time_2, pos_2[:, 1], label='y_pos')
    # axs[1].plot(time_2, vel_2[:, 1], label='y_vel')
    # axs[1].plot(time_2, acc_2[:, 1], label='y_acc')
    # axs[1].set_ylabel('y')
    # axs[1].grid()
    # axs[1].legend()

    # axs[2].plot(time_2, pos_2[:, 2], label='z_pos')
    # axs[2].plot(time_2, vel_2[:, 2], label='z_vel')
    # axs[2].plot(time_2, acc_2[:, 2], label='z_acc')

    # axs[2].set_ylabel('z')
    # axs[2].grid()
    # axs[2].legend()
    # plt.xlabel('time_2 [s]')
    # plt.suptitle('Trajectory2')
    '''
    # 3D轨迹
    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')
    axs.plot(pos_3[0:traj_nums_1 + idx_2:, 0], 
             pos_3[0:traj_nums_1 + idx_2:, 1], 
             pos_3[0:traj_nums_1 + idx_2:, 2], c='r', label='Trajectory3')
    axs.plot(pos_1[0:traj_nums_1:, 0],
             pos_1[0:traj_nums_1:, 1],
             pos_1[0:traj_nums_1:, 2], c='g', label='Trajectory1')
    axs.plot(pos_2[0:traj_nums_2:, 0],
             pos_2[0:traj_nums_2:, 1],
             pos_2[0:traj_nums_2:, 2], c='b', label='Trajectory2')
    axs.scatter(P1[0], P1[1], P1[2], c='r', marker='o', label='P1')
    axs.scatter(P2[0], P2[1], P2[2], c='g', marker='o', label='P2')
    axs.scatter(P3[0], P3[1], P3[2], c='b', marker='o', label='P3')
    # points = np.array([P1, P2, P3])
    # axs.plot(points[:, 0], points[:, 1], points[:, 2], c='k', label='Line')
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.set_zlabel('Z')
    axs.set_title('3D Trajectory')
    axs.legend()
    plt.show()

# orien: RPY
def pos_orien_syn_no_blending(p0, p1, p2, q0, q1, q2, zone):
    pos_blend = zone[0]
    orien_blend = np.deg2rad(zone[1]) # deg->rad
    if len(p0) != 3 or len(p1) != 3:
        raise ValueError("Position must be a 3-element vector")
    # if len(start_orien) != 3 or len(end_orien) != 3:
    #     raise ValueError("Orientation must be a 3-element vector (RPY in degrees)")
    
    dis1 = base.cal_dis(p0, p1)
    dis2 = base.cal_dis(p1, p2)
    if dis1 <= 0 or dis2 <= 0:
        raise ValueError("Start and end positions must be different")
    # Convert RPY to quaternion
    # q1 = base.RPY2quat(start_orien)
    # q2 = base.RPY2quat(end_orien)
    # Calculate the distance between the two quaternions: q2 * q1^-1(in world frame)
    init_axi = np.array([0.0, 0.0, 1.0]) # initial axis
    q0, angle_diff1, n1 = base.quat_angle_diff_and_axi(q0, q1, init_axi)
    q1, angle_diff2, n2 = base.quat_angle_diff_and_axi(q1, q2, init_axi)
    angle_diff1 *= 2.0 # convert to 3D space angle
    angle_diff2 *= 2.0 # convert to 3D space angle
    p0_1_norm = base.normalize(p1 - p0)
    p1_2_norm = base.normalize(p2 - p1)

    ruckig = Ruckig(2)
    input = InputParameter(2)
    input.max_velocity = [1.0, 3.14]
    input.max_acceleration = [3.0, 6.28]
    input.max_jerk = [30.0, 62.8]
    input.current_position = [0, 0]
    input.current_velocity = [0.0, 0.0]
    input.current_acceleration = [0.0, 0.0]
    input.target_position = [dis1, angle_diff1]
    input.target_velocity = [0.0, 0.0]
    input.target_acceleration = [0.0, 0.0]
    traj1 = Trajectory(2)
    res = ruckig.calculate(input, traj1)
    if res == Result.ErrorInvalidInput:
        raise Exception("Invalid input parameters")

    input.target_position = [dis2, angle_diff2]
    traj2 = Trajectory(2)
    res = ruckig.calculate(input, traj2)
    if res == Result.ErrorInvalidInput:
        raise Exception("Invalid input parameters")
    
    print(f'Trajectory duration: {traj1.duration: 0.4f} [s]')
    print(f'Trajectory duration: {traj2.duration: 0.4f} [s]')

    traj_time1 = traj1.duration
    traj_num1 = int(traj_time1 / 0.004) + 1 # +1 to include the end point
    traj_time2 = traj1.duration
    traj_num2 = int(traj_time2 / 0.004) + 1 # +1 to include the end point

    pos1 = np.zeros((traj_num1, 3), dtype=float)
    vel1 = np.zeros((traj_num1, 3), dtype=float)
    acc1 = np.zeros((traj_num1, 3), dtype=float)

    q_imp1 = np.zeros((traj_num1, 4), dtype=float)
    angle1 = np.zeros((traj_num1, 1), dtype=float)
    n01 = np.zeros((traj_num1, 3), dtype=float)
    omega1 = np.zeros((traj_num1, 1), dtype=float)
    d_omega1 = np.zeros((traj_num1, 1), dtype=float)
    rpy = np.zeros((traj_num1, 3), dtype=float)

    pos2 = np.zeros((traj_num2, 3), dtype=float)
    vel2 = np.zeros((traj_num2, 3), dtype=float)
    acc2 = np.zeros((traj_num2, 3), dtype=float)

    q_imp2 = np.zeros((traj_num2, 4), dtype=float)
    angle2 = np.zeros((traj_num2, 1), dtype=float)
    n02 = np.zeros((traj_num2, 3), dtype=float)
    omega2 = np.zeros((traj_num2, 1), dtype=float)
    d_omega2 = np.zeros((traj_num2, 1), dtype=float)


    for i in range(traj_num1):
        pos_temp, vel_temp, acc_temp = traj1.at_time(i * 0.004)
        pos1[i, :] = p0 + p0_1_norm * pos_temp[0]
        vel1[i, :] = vel_temp[0] * p0_1_norm
        acc1[i, :] = acc_temp[0] * p0_1_norm
        angle1[i, 0] = pos_temp[1]
        omega1[i, 0] = vel_temp[1]
        d_omega1[i, 0] = acc_temp[1]
        # Calculate the quaternion for the current time step
        q_imp1[i, :] = base.new_quat(q0, base.axis_angle2quat(n1, angle1[i, 0]))
        n01[i, :] = n1
        # rpy[i, :] = base.quat2RPY(q_imp1[i, :])
    
    for i in range(traj_num2):
        pos_temp, vel_temp, acc_temp = traj2.at_time(i * 0.004)
        pos2[i, :] = p1 + p1_2_norm * pos_temp[0]
        vel2[i, :] = vel_temp[0] * p1_2_norm
        acc2[i, :] = acc_temp[0] * p1_2_norm
        angle2[i, 0] = pos_temp[1]
        omega2[i, 0] = vel_temp[1]
        d_omega2[i, 0] = acc_temp[1]
        # Calculate the quaternion for the current time step
        q_imp2[i, :] = base.new_quat(q1, base.axis_angle2quat(n2, angle2[i, 0]))
        n02[i, :] = n2
    # 最终轨迹
    traj_num3 = traj_num1 + traj_num2
    pos3 = np.zeros((traj_num3, 3), dtype=float)
    vel3 = np.zeros((traj_num3, 3), dtype=float)
    acc3 = np.zeros((traj_num3, 3), dtype=float)

    q_imp3 = np.zeros((traj_num3, 4), dtype=float)
    angle3 = np.zeros((traj_num3, 1), dtype=float)
    n03 = np.zeros((traj_num3, 3), dtype=float)
    omega3 = np.zeros((traj_num3, 1), dtype=float)
    d_omega3 = np.zeros((traj_num3, 1), dtype=float)

    for i in range(traj_num1):
        pos3[i, :] = pos1[i, :]
        vel3[i, :] = vel1[i, :]
        acc3[i, :] = acc1[i, :]
        q_imp3[i, :] = q_imp1[i, :]
        angle3[i, 0] = angle1[i, 0]
        omega3[i, 0] = omega1[i, 0]
        d_omega3[i, 0] = d_omega1[i, 0]
        n03[i, :] = n01[i, :]

    for i in range(traj_num2):
        idx = i + traj_num1
        pos3[i + traj_num1, :] = pos2[i, :]
        vel3[i + traj_num1, :] = vel2[i, :]
        acc3[i + traj_num1, :] = acc2[i, :]
        q_imp3[i + traj_num1, :] = q_imp2[i, :]
        q_temp, angle_temp, n03[i + traj_num1, :] = base.quat_angle_diff_and_axi(q_imp3[i + traj_num1 - 1, :], 
                                                                                 q_imp3[i + traj_num1, :],
                                                                                 n03[i + traj_num1 - 1, :])
        q0, angle3[i + traj_num1, 0], n = base.quat_angle_diff_and_axi(q0, q_imp3[i + traj_num1, :], n03[i + traj_num1, :])
        angle3[i + traj_num1, 0] *= 2.0
        omega3[i + traj_num1, 0] = base.cal_quat_vel(q_imp3[idx-1, :], q_imp3[idx, :], 0.004)
        d_omega3[idx, 0] = (omega3[idx, 0] - omega3[idx-1, 0]) / 0.004


    t1 = np.arange(traj_num1) * 0.004
    t2 = np.arange(traj_num2) * 0.004 + t1[-1]
    t3 = np.arange(traj_num3) * 0.004

    # plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axs[0].plot(t1, pos1[:, 0], label='traj1_x')
    axs[0].plot(t2, pos2[:, 0], label='traj2_x')
    axs[0].plot(t3, pos3[:, 0], label='traj3_x')
    axs[0].scatter(t1[-1], pos1[-1, 0], c='r', marker='o')
    axs[0].set_ylabel('Position X')
    axs[0].grid()
    axs[0].legend(loc='best')

    axs[1].plot(t1, vel1[:, 0], label='traj1_x vel')
    axs[1].plot(t2, vel2[:, 0], label='traj1_x vel')
    axs[1].plot(t3, vel3[:, 0], label='traj1_x vel')
    axs[1].scatter(t1[-1], vel1[-1, 0], c='r', marker='o')
    axs[1].set_ylabel('Vel')
    axs[1].grid()
    axs[1].legend()

    axs[2].plot(t1, acc1[:, 0], label='traj1_x acc')
    axs[2].plot(t2, acc2[:, 0], label='traj1_x acc')
    axs[2].plot(t3, acc3[:, 0], label='traj1_x acc')
    axs[2].scatter(t1[-1], acc1[-1, 0], c='r', marker='o')
    axs[2].set_ylabel('Acc')
    axs[2].grid()
    axs[2].legend()
    plt.xlabel('Time [s]')
    plt.suptitle('Position and Orientation Synchronization')
    # plt.show()

    # # 3D 轨迹
    # fig = plt.figure()
    # axs = fig.add_subplot(111, projection='3d')
    # axs.plot(pos3[:, 0], pos3[:, 1], pos3[:, 2], c='r', label='Trajectory')
    # axs.scatter(p0[0], p0[1], p0[2], c='r', marker='o', label='Start Pos')
    # axs.scatter(p2[0], p2[1], p2[2], c='g', marker='o', label='End Pos')
    # axs.set_xlabel('X')
    # axs.set_ylabel('Y')
    # axs.set_zlabel('Z')
    # axs.set_title('3D Trajectory')
    # axs.legend()
    # # plt.show()

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    # 角度
    axs[0].plot(t1, angle1, linestyle='--', label='angle1')
    axs[0].plot(t2, angle2, linestyle='-.', label='angle2')
    axs[0].plot(t3, angle3, label='angle_final')
    axs[0].scatter(t1[-1], angle1[-1], c='r', marker='o')
    axs[0].set_ylabel('Angle (rad)')
    axs[0].set_title('Angle over Time')
    axs[0].legend()
    axs[0].grid()

    # 旋转轴
    axs[1].plot(t3, n03[:, 0], linestyle='-', label='n1_x')
    axs[1].plot(t3, n03[:, 1], linestyle='--', label='n1_y')
    axs[1].plot(t3, n03[:, 2], linestyle='-.', label='n1_z')
    axs[1].scatter(t1[-1], n01[-1, 0], c='r', marker='o', label='n1_x')
    axs[1].scatter(t1[-1], n01[-1, 1], c='g', marker='o', label='n1_y')
    axs[1].scatter(t1[-1], n01[-1, 2], c='b', marker='o', label='n1_z')
    axs[1].set_ylabel('Rotation Axis')
    axs[1].set_title('Rotation Axis over Time')
    axs[1].legend()

    # 角速度
    axs[2].plot(t1, omega1, linestyle='--', label='angle1_vel')
    axs[2].plot(t2, omega2, linestyle='-.', label='angle2_vel')
    axs[2].plot(t3, omega3, marker='.', label='omega_final')
    axs[2].scatter(t1[-1], omega1[-1], c='r', marker='o')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Angular Velocity (rad/s)')
    axs[2].set_title('Angular Velocity over Time')
    axs[2].legend()
    axs[2].grid()
    # 角加速度
    axs[3].plot(t1, d_omega1, linestyle='--', label='angle1_vel')
    axs[3].plot(t2, d_omega2, linestyle='-.', label='angle2_vel')
    axs[3].plot(t3, d_omega3, marker='.', label='omega_final')
    axs[3].scatter(t1[-1], d_omega1[-1], c='r', marker='o')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Angular acc (rad/s^2)')
    axs[3].set_title('Angular acc')
    axs[3].legend()
    axs[3].grid()
    plt.tight_layout()

    pos_final = pos3
    q_final = q_imp3
    # 姿态动画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_final[:, 0], pos_final[:, 1], pos_final[:, 2], c='r', label='Trajectory-noblending') 
    ax.plot(pos1[0:traj_num1:, 0],
            pos1[0:traj_num1:, 1],
            pos1[0:traj_num1:, 2], c='g', label='Trajectory1')
    ax.plot(pos2[0:traj_num2:, 0],
            pos2[0:traj_num2:, 1],
            pos2[0:traj_num2:, 2], c='b', label='Trajectory2')
    ax.scatter(p0[0], p0[1], p0[2], c='r', marker='o', label='p0')
    ax.scatter(p1[0], p1[1], p1[2], c='g', marker='o', label='p1')
    ax.scatter(p2[0], p2[1], p2[2], c='b', marker='o', label='p2')
    # 起点姿态
    rot_start = Rot.from_quat(q_final[0])
    p_start = pos_final[0]
    L = 0.05
    x_s = rot_start.apply([L, 0, 0]) + p_start
    y_s = rot_start.apply([0, L, 0]) + p_start
    z_s = rot_start.apply([0, 0, L]) + p_start
    ax.plot([p_start[0], x_s[0]], [p_start[1], x_s[1]], [p_start[2], x_s[2]], 'r:', lw=2)
    ax.plot([p_start[0], y_s[0]], [p_start[1], y_s[1]], [p_start[2], y_s[2]], 'g:', lw=2)
    ax.plot([p_start[0], z_s[0]], [p_start[1], z_s[1]], [p_start[2], z_s[2]], 'b:', lw=2)

    # 终点姿态
    rot_end = Rot.from_quat(q_final[-1])
    p_end = pos_final[-1]
    x_e = rot_end.apply([L, 0, 0]) + p_end
    y_e = rot_end.apply([0, L, 0]) + p_end
    z_e = rot_end.apply([0, 0, L]) + p_end
    ax.plot([p_end[0], x_e[0]], [p_end[1], x_e[1]], [p_end[2], x_e[2]], 'r--', lw=2)
    ax.plot([p_end[0], y_e[0]], [p_end[1], y_e[1]], [p_end[2], y_e[2]], 'g--', lw=2)
    ax.plot([p_end[0], z_e[0]], [p_end[1], z_e[1]], [p_end[2], z_e[2]], 'b--', lw=2)
    # 坐标轴原点
    # origin = np.zeros((3, 1))
    # 初始化三条轴
    point, = ax.plot([], [], [], 'ro', markersize=8, label='Current Pos')
    x_axis, = ax.plot([], [], [], 'r-', lw=1, label='X')
    y_axis, = ax.plot([], [], [], 'g-', lw=1, label='Y')
    z_axis, = ax.plot([], [], [], 'b-', lw=1, label='Z')
    def init():
        ax.set_xlim([np.min(pos_final[:,0])-0.1, np.max(pos_final[:,0])+0.1])
        ax.set_ylim([np.min(pos_final[:,1])-0.1, np.max(pos_final[:,1])+0.1])
        ax.set_zlim([np.min(pos_final[:,2])-0.1, np.max(pos_final[:,2])+0.1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return point, x_axis, y_axis, z_axis

    def update(i):
        p = pos_final[i]
        quat = q_final[i]
        rot = Rot.from_quat([quat[0], quat[1], quat[2], quat[3]])
        # 单位向量
        L = 0.05
        x = rot.apply([L, 0, 0]) + p
        y = rot.apply([0, L, 0]) + p
        z = rot.apply([0, 0, L]) + p
        # 更新当前点
        point.set_data([p[0]], [p[1]])
        point.set_3d_properties([p[2]])
        # 更新姿态坐标轴
        x_axis.set_data([p[0], x[0]], [p[1], x[1]])
        x_axis.set_3d_properties([p[2], x[2]])
        y_axis.set_data([p[0], y[0]], [p[1], y[1]])
        y_axis.set_3d_properties([p[2], y[2]])
        z_axis.set_data([p[0], z[0]], [p[1], z[1]])
        z_axis.set_3d_properties([p[2], z[2]])
        return point, x_axis, y_axis, z_axis
    # ani = FuncAnimation(fig, update, frames=range(len(q_final)), init_func=init, blit=True, interval=50)
    ani = FuncAnimation(fig, update, frames=len(pos_final), init_func=init, blit=True, interval=50)
    plt.legend()
    plt.tight_layout()
    # ani.save('trajectory.gif', writer='pillow', fps=20)
    plt.show()




# Position and orientation synchronization with blending.
# p0: start position
# p1: middle position
# p2: end position
# q0: start quaternion
# q1: middle quaternion
# q2: end quaternion
def pos_orien_syn_bleding(p0, p1, p2, q0, q1, q2, zone):
    pos_blend = zone[0] # mm
    orien_blend = np.deg2rad(zone[1]) # deg->rad

    if len(p0) != 3 or len(p1) != 3 or len(p2) != 3:
        raise ValueError("Position must be a 3-element vector")
    if len(q0) != 4 or len(q1) != 4 or len(q2) != 4:
        raise ValueError("Quaternion must be a 4-element vector")
    
    dis_01 = base.cal_dis(p0, p1)
    dis_12 = base.cal_dis(p1, p2)
    # 计算四元数的夹角和旋转轴
    init_axi = np.array([0.0, 0.0, 1.0]) # initial rotation axis
    q0, angle_diff1, n1 = base.quat_angle_diff_and_axi(q0, q1, init_axi)
    q1, angle_diff2, n2 = base.quat_angle_diff_and_axi(q1, q2, init_axi)

    angle_diff1 *= 2.0 # convert to 3D space angle
    angle_diff2 *= 2.0 # convert to 3D space angle

    if angle_diff1 < 0 or angle_diff2 < 0:
        raise ValueError("Quaternions must represent valid rotations")
    if dis_01 <= 0 or dis_12 <= 0:
        raise ValueError("Positions must be different")

    p01_norm = base.normalize(p1 - p0)
    p12_norm = base.normalize(p2 - p1)

    ruckig = Ruckig(2)
    input = InputParameter(2)
    input.max_velocity = [1, 3.14]
    input.max_acceleration = [3, 6.28]
    input.max_jerk = [30.0, 62.8]
    input.current_position = [0, 0]
    input.current_velocity = [0.0, 0.0]
    input.current_acceleration = [0.0, 0.0]

    input.target_position = [dis_01, angle_diff1]
    input.target_velocity = [0.0, 0.0]
    input.target_acceleration = [0.0, 0.0]
    # 计算第一段轨迹
    traj1 = Trajectory(2)
    res = ruckig.calculate(input, traj1)
    if res == Result.ErrorInvalidInput:
        raise Exception("Invalid input parameters")
    # 计算第二段轨迹
    input.target_position = [dis_12, angle_diff2]
    traj2 = Trajectory(2)
    res = ruckig.calculate(input, traj2)
    if res == Result.ErrorInvalidInput:
        raise Exception("Invalid input parameters")

    print(f'Trajectory1 duration: {traj1.duration: 0.4f} [s]')
    print(f'Trajectory2 duration: {traj2.duration: 0.4f} [s]')

    traj_nums1 = int(traj1.duration / 0.004) + 1 # +1 to include the end point
    traj_nums2 = int(traj2.duration / 0.004) + 1 # +1 to include the end point
    
    pos1 = np.zeros((traj_nums1, 3), dtype=float)
    vel1 = np.zeros((traj_nums1, 3), dtype=float)
    acc1 = np.zeros((traj_nums1, 3), dtype=float)
    q_imp1 = np.zeros((traj_nums1, 4), dtype=float)
    angle1 = np.zeros((traj_nums1, 1), dtype=float)
    omega1 = np.zeros((traj_nums1, 1), dtype=float)
    d_omega1 = np.zeros((traj_nums1, 1), dtype=float)

    pos2 = np.zeros((traj_nums2, 3), dtype=float)
    vel2 = np.zeros((traj_nums2, 3), dtype=float)
    acc2 = np.zeros((traj_nums2, 3), dtype=float)
    q_imp2 = np.zeros((traj_nums2, 4), dtype=float)
    angle2 = np.zeros((traj_nums2, 1), dtype=float)
    omega2 = np.zeros((traj_nums2, 1), dtype=float)
    d_omega2 = np.zeros((traj_nums2, 1), dtype=float)

    for i in range(traj_nums1):
        pos_temp, vel_temp, acc_temp = traj1.at_time(i * 0.004)
        pos1[i, :] = p0 + p01_norm * pos_temp[0]
        vel1[i, :] = vel_temp[0] * p01_norm
        acc1[i, :] = acc_temp[0] * p01_norm

        angle1[i, 0] = pos_temp[1]
        omega1[i, 0] = vel_temp[1]
        d_omega1[i, 0] = acc_temp[1]
        q_imp1[i, :] = base.new_quat(q0, base.axis_angle2quat(n1, angle1[i, 0]))
        
    for i in range(traj_nums2):
        pos_temp, vel_temp, acc_temp = traj2.at_time(i * 0.004)
        pos2[i, :] = p1 + p12_norm * pos_temp[0]
        vel2[i, :] = vel_temp[0] * p12_norm
        acc2[i, :] = acc_temp[0] * p12_norm

        angle2[i, 0] = pos_temp[1]
        omega2[i, 0] = vel_temp[1]
        d_omega2[i, 0] = acc_temp[1]
        q_imp2[i, :] = base.new_quat(q1, base.axis_angle2quat(n2, angle2[i, 0]))

    # blending
    pos3 = np.zeros((traj_nums1+traj_nums2, 3), dtype=float)
    vel3 = np.zeros((traj_nums1+traj_nums2, 3), dtype=float)
    acc3 = np.zeros((traj_nums1+traj_nums2, 3), dtype=float)

    q_imp3 = np.zeros((traj_nums1+traj_nums2, 4), dtype=float)
    angle3 = np.zeros((traj_nums1+traj_nums2, 1), dtype=float)
    n3 = np.zeros((traj_nums1+traj_nums2, 3), dtype=float) # 旋转轴
    omega3 = np.zeros((traj_nums1+traj_nums2, 1), dtype=float)
    d_omega3 = np.zeros((traj_nums1+traj_nums2, 1), dtype=float)

    idx1 = 0
    idx2 = 0
    idx_rl1 = 0
    idx_rl2 = 0
    no1 = 0 # 第一段需要blending的点数
    no2 = 0 # 第二段需要blending的点数
    for i in range(traj_nums1):
        pos3[i, :] = pos1[i, :]
        vel3[i, :] = vel1[i, :]
        acc3[i, :] = acc1[i, :]
        q_imp3[i, :] = q_imp1[i, :]
        angle3[i, 0] = angle1[i, 0]
        omega3[i, 0] = omega1[i, 0]
        d_omega3[i, 0] = d_omega1[i, 0]
        n3[i, :] = n1
        # 位置和姿态同时满足，才blending
        if base.cal_dis(pos3[i, :], p1) <= pos_blend and (angle_diff1 - angle3[i, 0]) <= orien_blend:
            idx1 = i
            no1 = traj_nums1 - idx1
            break

    for i in range(traj_nums2):
        if base.cal_dis(p1, pos2[i, :]) >= pos_blend and angle2[i, 0] >= orien_blend:
            no2 = i
            idx2 = traj_nums2 - no2
            break

    # blending段
    '''
    if no1 > no2:
        idx_rl1 = idx1 + no1 - no2
        for i in range(no1 - no2):
            pos3[i + idx1, :] = pos1[i + idx1, :]
            vel3[i + idx1, :] = vel1[i + idx1, :]
            acc3[i + idx1, :] = acc1[i + idx1, :]
            q_imp3[i + idx1, :] = q_imp1[i + idx1, :]
            angle3[i + idx1, 0] = angle1[i + idx1, 0]
            omega3[i + idx1, 0] = omega1[i + idx1, 0]
            d_omega3[i + idx1, 0] = d_omega1[i + idx1, 0]
            n3[i + idx1, :] = n1
        for i in range(no2):
            idx = i + idx1 + no1 - no2
            # pos blending
            pos3[idx, :] = pos2[i, :] - p1 + pos1[idx, :]
            vel3[idx, :] = vel2[i, :] + vel1[idx, :]
            acc3[idx, :] = acc2[i, :] + acc1[idx, :]
            # orien blending
            q_imp3[idx, :] = base.new_quat(q_imp1[idx, :], base.axis_angle2quat(n2, angle2[i, 0]))
            # 计算相邻q的旋转轴
            q_temp, angle_tmp, n3[idx, :] = base.quat_angle_diff_and_axi(q_imp3[idx-1, :], q_imp3[idx, :], n3[idx-1, :])
            # 计算q->q0的夹角
            q0, angle3[idx, 0], n = base.quat_angle_diff_and_axi(q0, q_imp3[idx, :], n3[idx, :])
            angle3[idx, 0] *= 2.0 # convert to angle in 3D space
            omega3[idx, 0] = base.cal_quat_vel(q_imp3[idx-1, :], q_imp3[idx, :], 0.004) 
            d_omega3[idx] = (omega3[idx, 0] - omega3[idx-1, 0]) / 0.004 # 计算角加速度
    else:
        # 第二段需要blending的点数大于第一段
        idx_rl1 = idx1
        idx_rl2 = no2 - no1
        for i in range(0, no1):
            pos3[i + idx1, :] = pos1[i + idx1, :] + pos2[i + idx_rl2, :] - p1
            vel3[i + idx1, :] = vel1[i + idx1, :] + vel2[i + idx_rl2, :]
            acc3[i + idx1, :] = acc1[i + idx1, :] + acc2[i + idx_rl2, :]
            # orien blending
            q_imp3[i + idx1, :] = base.new_quat(q_imp1[i + idx1, :], base.axis_angle2quat(n2, angle2[i + idx_rl2, 0]))
            # 计算相邻q的旋转轴
            q_temp, angle_tmp, n3[i + idx1, :] = base.quat_angle_diff_and_axi(q_imp3[i + idx1-1, :], q_imp3[i + idx1, :], n3[i + idx1-1, :])
            # 计算q->q0的夹角
            q0, angle3[i + idx1, 0], n = base.quat_angle_diff_and_axi(q0, q_imp3[i + idx1, :], n3[i + idx1, :])
            angle3[i + idx1, 0] *= 2.0 # convert to angle in 3D space
            omega3[i + idx1, 0] = base.cal_quat_vel(q_imp3[i + idx1-1, :], q_imp3[i + idx1, :], 0.004) 
            d_omega3[i+idx1] = (omega3[i+idx1, 0] - omega3[i+idx1-1, 0]) / 0.004
    
    for i in range(0, idx2):
        pos3[i + idx1 + no1, :] = pos2[no2 + i, :]
        vel3[i + idx1 + no1, :] = vel2[no2 + i, :]
        acc3[i + idx1 + no1, :] = acc2[no2 + i, :]

        q_imp3[i + idx1 + no1, :] = q_imp2[no2 + i, :]
        q0, angle3[i + idx1 + no1, 0], n = base.quat_angle_diff_and_axi(q0, q_imp3[i + idx1 + no1, :], n3[i + idx1 + no1, :])
        angle3[i + idx1 + no1, 0] *= 2.0
        # omega3[i + idx1 + no1, 0] = base.cal_quat_vel(q_imp3[i + idx1 + no1-1, :], q_imp3[i + idx1 + no1, :], 0.004)
        omega3[i + idx1 + no1, 0] = omega2[no2 + i, 0]
        d_omega3[i + idx1 + no1, 0] = (omega3[i + idx1 + no1, 0] - omega3[i + idx1 + no1-1, 0]) / 0.004
        n3[i + idx1 + no1, :] = n2
    '''
    idx2_start = no2
    if no1 > no2:
        idx_rl1 = idx1 + no1 - no2
        for i in range(0, no1 - no2 + 1):
            pos3[i + idx1, :] = pos1[i + idx1, :]
            vel3[i + idx1, :] = vel1[i + idx1, :]
            acc3[i + idx1, :] = acc1[i + idx1, :]
            q_imp3[i + idx1, :] = q_imp1[i + idx1, :]
            angle3[i + idx1, 0] = angle1[i + idx1, 0]
            omega3[i + idx1, 0] = omega1[i + idx1, 0]
            d_omega3[i + idx1, 0] = d_omega1[i + idx1, 0]
            n3[i + idx1, :] = n1
        for i in range(1, no2): # 从第二段第一个点开始
            idx = i + idx1 + no1 - no2
            # pos blending
            pos3[idx, :] = pos2[i, :] - p1 + pos1[idx, :]
            vel3[idx, :] = vel2[i, :] + vel1[idx, :]
            acc3[idx, :] = acc2[i, :] + acc1[idx, :]
            # orien blending
            q_imp3[idx, :] = base.new_quat(q_imp1[idx, :], base.axis_angle2quat(n2, angle2[i, 0]))
            # 计算相邻q的旋转轴
            q_temp, angle_tmp, n3[idx, :] = base.quat_angle_diff_and_axi(q_imp3[idx-1, :], q_imp3[idx, :], n3[idx-1, :])
            # 计算q->q0的夹角
            q0, angle3[idx, 0], n = base.quat_angle_diff_and_axi(q0, q_imp3[idx, :], n3[idx, :])
            angle3[idx, 0] *= 2.0 # convert to angle in 3D space
            omega3[idx, 0] = base.cal_quat_vel(q_imp3[idx-1, :], q_imp3[idx, :], 0.004) 
            d_omega3[idx] = (omega3[idx, 0] - omega3[idx-1, 0]) / 0.004 # 计算角加速度
    else:
        # 第二段需要blending的点数大于第一段
        idx_rl1 = idx1
        idx2_start = no1
        idx2 = idx2 + no2 - no1 # 重置剩余需要插补的点
        for i in range(1, no1):
            pos3[i + idx1, :] = pos1[i + idx1, :] + pos2[i, :] - p1
            vel3[i + idx1, :] = vel1[i + idx1, :] + vel2[i, :]
            acc3[i + idx1, :] = acc1[i + idx1, :] + acc2[i, :]
            # orien blending
            q_imp3[i + idx1, :] = base.new_quat(q_imp1[i + idx1, :], base.axis_angle2quat(n2, angle2[i, 0]))
            # 计算相邻q的旋转轴
            q_temp, angle_tmp, n3[i + idx1, :] = base.quat_angle_diff_and_axi(q_imp3[i + idx1-1, :], q_imp3[i + idx1, :], n3[i + idx1-1, :])
            # 计算q->q0的夹角
            q0, angle3[i + idx1, 0], n = base.quat_angle_diff_and_axi(q0, q_imp3[i + idx1, :], n3[i + idx1, :])
            angle3[i + idx1, 0] *= 2.0 # convert to angle in 3D space
            omega3[i + idx1, 0] = base.cal_quat_vel(q_imp3[i + idx1-1, :], q_imp3[i + idx1, :], 0.004)
            d_omega3[i+idx1] = (omega3[i+idx1, 0] - omega3[i+idx1-1, 0]) / 0.004
    
    for i in range(0, idx2):
        pos3[i + traj_nums1, :] = pos2[idx2_start + i, :]
        vel3[i + traj_nums1, :] = vel2[idx2_start + i, :]
        acc3[i + traj_nums1, :] = acc2[idx2_start + i, :]

        q_imp3[i + traj_nums1, :] = q_imp2[idx2_start + i, :]
        q0, angle3[i + traj_nums1, 0], n = base.quat_angle_diff_and_axi(q0, q_imp3[i + traj_nums1, :], n3[i + traj_nums1, :])
        angle3[i + traj_nums1, 0] *= 2.0
        omega3[i + traj_nums1, 0] = base.cal_quat_vel(q_imp3[i + traj_nums1-1, :], q_imp3[i + traj_nums1, :], 0.004)
        omega3[i + traj_nums1, 0] = omega2[idx2_start + i, 0]
        # d_omega3[i + traj_nums1, 0] = (omega3[i + traj_nums1, 0] - omega3[i + traj_nums1-1, 0]) / 0.004
        d_omega3[i + traj_nums1, 0] = d_omega2[idx2_start + i, 0]
        n3[i + traj_nums1, :] = n2
    # plot
    t1 = np.arange(traj_nums1) * 0.004
    t2 = np.arange(traj_nums2) * 0.004 + t1[idx_rl1 - idx_rl2]
    t3 = np.arange(traj_nums1 + idx2) * 0.004
    pos_final = pos3[0:traj_nums1 + idx2, :]
    vel_final = vel3[0:traj_nums1 + idx2, :]
    acc_final = acc3[0:traj_nums1 + idx2, :]
    q_final = q_imp3[0:traj_nums1 + idx2, :]
    angle_final = angle3[0:traj_nums1 + idx2, :]
    omega_final = omega3[0:traj_nums1 + idx2, :]
    d_omega_final = d_omega3[0:traj_nums1 + idx2, :]
    # 位置对比
    # x
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axs[0].plot(t1, pos1[:, 0], label='traj_1_x')
    axs[0].plot(t2, pos2[:, 0] , label='traj_2_x')
    axs[0].plot(t3, pos3[0:traj_nums1 + idx2, 0], label='traj_3_x')
    axs[0].scatter(t3[idx_rl1], pos3[idx_rl1, 0], c='r', marker='o', label='start')
    axs[0].scatter(t3[traj_nums1-1], pos3[traj_nums1-1, 0], c='g', marker='o', label='end')
    axs[0].set_ylabel('pos')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(t1, vel1[:, 0], label='vel1_x')
    axs[1].plot(t2, vel2[:, 0], label='vel2_x')
    axs[1].plot(t3, vel3[0:traj_nums1 + idx2, 0], label='vel3_x')
    axs[1].scatter(t3[idx_rl1], vel3[idx_rl1, 0], c='r', marker='o', label='start')
    axs[1].scatter(t3[traj_nums1-1], vel3[traj_nums1-1, 0], c='g', marker='o', label='end')
    axs[1].set_ylabel('vel')
    axs[1].grid()
    axs[1].legend()

    axs[2].plot(t1, acc1[:, 0], label='acc1_x')
    axs[2].plot(t2, acc2[:, 0], label='acc2_x')
    axs[2].plot(t3, acc3[0:traj_nums1 + idx2, 0], label='acc3_x')
    axs[2].scatter(t3[idx_rl1], acc3[idx_rl1, 0], c='r', marker='o', label='start')
    axs[2].scatter(t3[traj_nums1-1], acc3[traj_nums1-1, 0], c='g', marker='o', label='end')
    axs[2].set_ylabel('acc_x')
    axs[2].grid()
    axs[2].legend()
    plt.xlabel('time [s]')
    plt.suptitle('X_label')
    # y
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axs[0].plot(t1, pos1[:, 1], label='traj_1_y')
    axs[0].plot(t2, pos2[:, 1] , label='traj_2_y')
    axs[0].plot(t3, pos3[0:traj_nums1 + idx2, 1], label='traj_3_y')
    axs[0].scatter(t3[idx_rl1], pos3[idx_rl1, 1], c='r', marker='o', label='start')
    axs[0].scatter(t3[traj_nums1-1], pos3[traj_nums1-1, 1], c='g', marker='o', label='end')
    axs[0].set_ylabel('pos')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(t1, vel1[:, 1], label='vel1_y')
    axs[1].plot(t2, vel2[:, 1], label='vel2_y')
    axs[1].plot(t3, vel3[0:traj_nums1 + idx2, 1], label='vel3_y')
    axs[1].scatter(t3[idx_rl1], vel3[idx_rl1, 1], c='r', marker='o', label='start')
    axs[1].scatter(t3[traj_nums1-1], vel3[traj_nums1-1, 1], c='g', marker='o', label='end')
    axs[1].set_ylabel('vel')
    axs[1].grid()
    axs[1].legend()

    axs[2].plot(t1, acc1[:, 1], label='acc1_y')
    axs[2].plot(t2, acc2[:, 1], label='acc2_y')
    axs[2].plot(t3, acc3[0:traj_nums1 + idx2, 1], label='acc3_y')
    axs[2].scatter(t3[idx_rl1], acc3[idx_rl1, 1], c='r', marker='o', label='start')
    axs[2].scatter(t3[traj_nums1-1], acc3[traj_nums1-1, 1], c='g', marker='o', label='end')
    axs[2].set_ylabel('acc')
    axs[2].grid()
    axs[2].legend()
    plt.xlabel('time_1 [s]')
    plt.suptitle('Y_label')

    # z
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axs[0].plot(t1, pos1[:, 2], label='traj_1_z')
    axs[0].plot(t2, pos2[:, 2] , label='traj_2_z')
    axs[0].plot(t3, pos3[0:traj_nums1 + idx2, 2], label='traj_3_z')
    axs[0].scatter(t3[idx_rl1], pos3[idx_rl1, 2], c='r', marker='o', label='start')
    axs[0].scatter(t3[traj_nums1-1], pos3[traj_nums1-1, 2], c='g', marker='o', label='end')
    axs[0].set_ylabel('pos')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(t1, vel1[:, 2], label='vel1_z')
    axs[1].plot(t2, vel2[:, 2], label='vel2_z')
    axs[1].plot(t3, vel3[0:traj_nums1 + idx2, 2], label='vel3_z')
    axs[1].scatter(t3[idx_rl1], vel3[idx_rl1, 2], c='r', marker='o', label='start')
    axs[1].scatter(t3[traj_nums1-1], vel3[traj_nums1-1, 2], c='g', marker='o', label='end')
    axs[1].set_ylabel('vel')
    axs[1].grid()
    axs[1].legend()

    axs[2].plot(t1, acc1[:, 2], label='acc1_z')
    axs[2].plot(t2, acc2[:, 2], label='acc2_z')
    axs[2].plot(t3, acc3[0:traj_nums1 + idx2, 2], label='acc3_z')
    axs[2].scatter(t3[idx_rl1], acc3[idx_rl1, 2], c='r', marker='o', label='start')
    axs[2].scatter(t3[traj_nums1-1], acc3[traj_nums1-1, 2], c='g', marker='o', label='end')
    axs[2].set_ylabel('acc')
    axs[2].grid()
    axs[2].legend()
    plt.xlabel('time_1 [s]')
    plt.suptitle('Z_label')

    # 3D轨迹
    # fig = plt.figure()
    # axs = fig.add_subplot(111, projection='3d')
    # axs.plot(pos3[0:traj_nums1 + idx2:, 0], 
    #          pos3[0:traj_nums1 + idx2:, 1], 
    #          pos3[0:traj_nums1 + idx2:, 2], c='r', label='Trajectory3-blending')
    # axs.plot(pos1[0:traj_nums1:, 0],
    #          pos1[0:traj_nums1:, 1],
    #          pos1[0:traj_nums1:, 2], c='g', label='Trajectory1')
    # axs.plot(pos2[0:traj_nums2:, 0],
    #          pos2[0:traj_nums2:, 1],
    #          pos2[0:traj_nums2:, 2], c='b', label='Trajectory2')
    # axs.scatter(p0[0], p0[1], p0[2], c='r', marker='o', label='p0')
    # axs.scatter(p1[0], p1[1], p1[2], c='g', marker='o', label='p1')
    # axs.scatter(p2[0], p2[1], p2[2], c='b', marker='o', label='p2')
    # axs.set_xlabel('X')
    # axs.set_ylabel('Y')
    # axs.set_zlabel('Z')
    # axs.set_title('3D Trajectory')
    # axs.legend()

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    # 角度
    axs[0].plot(t1, angle1, linestyle='--', label='angle1')
    axs[0].plot(t2, angle2, linestyle='-.', label='angle2')
    axs[0].plot(t3, angle3[:traj_nums1+idx2, 0], label='angle_final')
    axs[0].scatter(t3[idx_rl1], angle3[idx_rl1], c='r', marker='o', label='start')
    axs[0].scatter(t3[traj_nums1-1], angle3[traj_nums1-1], c='g', marker='o', label='end')
    axs[0].set_ylabel('Angle (rad)')
    axs[0].set_title('Angle over Time')
    axs[0].legend()
    axs[0].grid()

    # 旋转轴
    axs[1].plot(t3, n3[:traj_nums1+idx2, 0], linestyle='-', label='n1_x')
    axs[1].plot(t3, n3[:traj_nums1+idx2, 1], linestyle='--', label='n1_y')
    axs[1].plot(t3, n3[:traj_nums1+idx2, 2], linestyle='-.', label='n1_z')
    axs[1].scatter(t3[idx_rl1], n3[idx_rl1, 0], c='r', marker='o', label='start')
    axs[1].scatter(t3[traj_nums1-1], n3[traj_nums1-1, 0], c='g', marker='o', label='end')
    axs[1].set_ylabel('Rotation Axis')
    axs[1].set_title('Rotation Axis over Time')
    axs[1].legend()

    # 角速度
    axs[2].plot(t1, omega1, linestyle='--', label='angle1_vel')
    axs[2].plot(t2, omega2, linestyle='-.', label='angle2_vel')
    axs[2].plot(t3, omega3[:traj_nums1+idx2, 0], marker='.', label='omega_final')
    axs[2].scatter(t3[idx_rl1], omega3[idx_rl1], c='r', marker='o', label='start')
    axs[2].scatter(t3[traj_nums1-1], omega3[traj_nums1-1], c='g', marker='o', label='end')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Angular Velocity (rad/s)')
    axs[2].set_title('Angular Velocity over Time')
    axs[2].legend()
    axs[2].grid()
    # 角加速度
    axs[3].plot(t1, d_omega1, linestyle='--', label='angle1_vel')
    axs[3].plot(t2, d_omega2, linestyle='-.', label='angle2_vel')
    axs[3].plot(t3, d_omega3[:traj_nums1+idx2, 0], marker='.', label='omega_final')
    axs[3].scatter(t3[idx_rl1], d_omega3[idx_rl1], c='r', marker='o', label='start')
    axs[3].scatter(t3[traj_nums1-1], d_omega3[traj_nums1-1], c='g', marker='o', label='end')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Angular acc (rad/s^2)')
    axs[3].set_title('Angular acc')
    axs[3].legend()
    axs[3].grid()
    plt.tight_layout()
    # plt.show()

    # 姿态动画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_final[:, 0], pos_final[:, 1], pos_final[:, 2], c='r', label='Trajectory-blending') 
    ax.plot(pos1[0:traj_nums1:, 0],
             pos1[0:traj_nums1:, 1],
             pos1[0:traj_nums1:, 2], c='g', label='Trajectory1')
    ax.plot(pos2[0:traj_nums2:, 0],
             pos2[0:traj_nums2:, 1],
             pos2[0:traj_nums2:, 2], c='b', label='Trajectory2')
    ax.scatter(p0[0], p0[1], p0[2], c='r', marker='o', label='p0')
    ax.scatter(p1[0], p1[1], p1[2], c='g', marker='o', label='p1')
    ax.scatter(p2[0], p2[1], p2[2], c='b', marker='o', label='p2')
    # 起点姿态
    rot_start = Rot.from_quat(q_final[0])
    p_start = pos_final[0]
    L = 0.05
    x_s = rot_start.apply([L, 0, 0]) + p_start
    y_s = rot_start.apply([0, L, 0]) + p_start
    z_s = rot_start.apply([0, 0, L]) + p_start
    ax.plot([p_start[0], x_s[0]], [p_start[1], x_s[1]], [p_start[2], x_s[2]], 'r:', lw=2)
    ax.plot([p_start[0], y_s[0]], [p_start[1], y_s[1]], [p_start[2], y_s[2]], 'g:', lw=2)
    ax.plot([p_start[0], z_s[0]], [p_start[1], z_s[1]], [p_start[2], z_s[2]], 'b:', lw=2)

    # 终点姿态
    rot_end = Rot.from_quat(q_final[-1])
    p_end = pos_final[-1]
    x_e = rot_end.apply([L, 0, 0]) + p_end
    y_e = rot_end.apply([0, L, 0]) + p_end
    z_e = rot_end.apply([0, 0, L]) + p_end
    ax.plot([p_end[0], x_e[0]], [p_end[1], x_e[1]], [p_end[2], x_e[2]], 'r--', lw=2)
    ax.plot([p_end[0], y_e[0]], [p_end[1], y_e[1]], [p_end[2], y_e[2]], 'g--', lw=2)
    ax.plot([p_end[0], z_e[0]], [p_end[1], z_e[1]], [p_end[2], z_e[2]], 'b--', lw=2)
    # 坐标轴原点
    # origin = np.zeros((3, 1))
    # 初始化三条轴
    point, = ax.plot([], [], [], 'ro', markersize=8, label='Current Pos')
    x_axis, = ax.plot([], [], [], 'r-', lw=1, label='X')
    y_axis, = ax.plot([], [], [], 'g-', lw=1, label='Y')
    z_axis, = ax.plot([], [], [], 'b-', lw=1, label='Z')
    def init():
        ax.set_xlim([np.min(pos_final[:,0])-0.1, np.max(pos_final[:,0])+0.1])
        ax.set_ylim([np.min(pos_final[:,1])-0.1, np.max(pos_final[:,1])+0.1])
        ax.set_zlim([np.min(pos_final[:,2])-0.1, np.max(pos_final[:,2])+0.1])
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])
        # ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return point, x_axis, y_axis, z_axis

    def update(i):
        p = pos_final[i]
        quat = q_final[i]
        rot = Rot.from_quat([quat[0], quat[1], quat[2], quat[3]])
        # 单位向量
        L = 0.05
        x = rot.apply([L, 0, 0]) + p
        y = rot.apply([0, L, 0]) + p
        z = rot.apply([0, 0, L]) + p
        # 更新三条轴
        # x_axis.set_data([0, x[0]], [0, x[1]])
        # x_axis.set_3d_properties([0, x[2]])
        # y_axis.set_data([0, y[0]], [0, y[1]])
        # y_axis.set_3d_properties([0, y[2]])
        # z_axis.set_data([0, z[0]], [0, z[1]])
        # z_axis.set_3d_properties([0, z[2]])
        # return x_axis, y_axis, z_axis
        # 更新当前点
        point.set_data([p[0]], [p[1]])
        point.set_3d_properties([p[2]])
        # 更新姿态坐标轴
        x_axis.set_data([p[0], x[0]], [p[1], x[1]])
        x_axis.set_3d_properties([p[2], x[2]])
        y_axis.set_data([p[0], y[0]], [p[1], y[1]])
        y_axis.set_3d_properties([p[2], y[2]])
        z_axis.set_data([p[0], z[0]], [p[1], z[1]])
        z_axis.set_3d_properties([p[2], z[2]])
        return point, x_axis, y_axis, z_axis
    # ani = FuncAnimation(fig, update, frames=range(len(q_final)), init_func=init, blit=True, interval=50)
    ani = FuncAnimation(fig, update, frames=len(pos_final), init_func=init, blit=True, interval=50)
    plt.legend()
    # ani.save('trajectory.gif', writer='pillow', fps=20)
    plt.show()