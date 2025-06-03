import ruckig
from ruckig import Ruckig, InputParameter, OutputParameter, Result, Trajectory
import numpy as np
import matplotlib.pyplot as plt
import base
from mpl_toolkits.mplot3d import Axes3D

def pos_interpolation():
    rr = 0.2 # cm
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
def pos_orien_syn(start_pos, end_pos, q1, q2):
    if len(start_pos) != 3 or len(end_pos) != 3:
        raise ValueError("Position must be a 3-element vector")
    # if len(start_orien) != 3 or len(end_orien) != 3:
    #     raise ValueError("Orientation must be a 3-element vector (RPY in degrees)")
    
    dis = base.cal_dis(start_pos, end_pos)
    if dis <= 0:
        raise ValueError("Start and end positions must be different")
    # Convert RPY to quaternion
    # q1 = base.RPY2quat(start_orien)
    # q2 = base.RPY2quat(end_orien)
    # Calculate the distance between the two quaternions: q2 * q1^-1(in world frame)
    q1, angle_dis, axi = base.quat_angle_diff_and_axi_first(q1, q2)

    p1_2_norm = base.normalize(end_pos - start_pos)

    ruckig = Ruckig(2)
    input = InputParameter(2)
    input.max_velocity = [2.5, 1.0]
    input.max_acceleration = [12.5, 5.0]
    input.max_jerk = [25, 10.0]
    input.current_position = [0, 0]
    input.current_velocity = [0.0, 0.0]
    input.current_acceleration = [0.0, 0.0]
    input.target_position = [dis, 2.0 * angle_dis]
    input.target_velocity = [0.0, 0.0]
    input.target_acceleration = [0.0, 0.0]
    traj = Trajectory(2)
    res = ruckig.calculate(input, traj)
    if res == Result.ErrorInvalidInput:
        raise Exception("Invalid input parameters")

    print(f'Trajectory duration: {traj.duration: 0.4f} [s]')
    traj_time = traj.duration
    traj_nums = int(traj_time / 0.004) + 1 # +1 to include the end point
    pos = np.zeros((traj_nums, 3), dtype=float)
    vel = np.zeros((traj_nums, 3), dtype=float)
    acc = np.zeros((traj_nums, 3), dtype=float)

    q_imp = np.zeros((traj_nums, 4), dtype=float)
    angle = np.zeros((traj_nums, 1), dtype=float)
    omega = np.zeros((traj_nums, 1), dtype=float)
    d_omega = np.zeros((traj_nums, 1), dtype=float)
    rpy = np.zeros((traj_nums, 3), dtype=float)

    for i in range(traj_nums):
        pos_temp, vel_temp, acc_temp = traj.at_time(i * 0.004)
        pos[i, :] = start_pos + p1_2_norm * pos_temp[0]
        vel[i, :] = vel_temp[0] * p1_2_norm
        acc[i, :] = acc_temp[0] * p1_2_norm
        angle[i, 0] = pos_temp[1]
        omega[i, 0] = vel_temp[1]
        d_omega[i, 0] = acc_temp[1]
        # Calculate the quaternion for the current time step
        q_imp[i, :] = base.new_quat(q1, base.axis_angle2quat(axi, angle[i, 0]))
        rpy[i, :] = base.quat2RPY(q_imp[i, :])
    
    print("end quat", q_imp[-1, :])
    
    t = np.arange(traj_nums) * 0.004
    # plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axs[0].plot(t, pos[:, 0], label='X Position')
    axs[0].plot(t, pos[:, 1], label='Y Position')
    axs[0].plot(t, pos[:, 2], label='Z Position')
    axs[0].plot(t, angle[:, 0], label='Angle(axis-angle)')

    axs[0].scatter(0, start_pos[0], c='r', marker='o', label='Start X Pos')
    axs[0].scatter(t[-1], end_pos[0], c='g', marker='o', label='End X Pos')
    axs[0].scatter(0, start_pos[1], c='r', marker='o', label='Start Y Pos')
    axs[0].scatter(t[-1], end_pos[1], c='g', marker='o', label='End Y Pos')
    axs[0].scatter(0, start_pos[2], c='r', marker='o', label='Start Z Pos')
    axs[0].scatter(t[-1], end_pos[2], c='g', marker='o', label='End Z Pos')

    axs[0].set_ylabel('Position/Angle')
    axs[0].grid()
    axs[0].legend(loc='best')

    axs[1].plot(t, vel[:, 0], label='X vel')
    axs[1].plot(t, vel[:, 1], label='Y vel')
    axs[1].plot(t, vel[:, 2], label='Z vel')
    axs[1].plot(t, omega[:, 0], label='Angle_vel(axis-angle)')
    axs[1].set_ylabel('Vel/Angle_vel')
    axs[1].grid()
    axs[1].legend()

    axs[2].plot(t, acc[:, 0], label='X acc')
    axs[2].plot(t, acc[:, 1], label='Y acc')
    axs[2].plot(t, acc[:, 2], label='Z acc')
    axs[2].plot(t, d_omega[:, 0], label='angle_acc(axis-angle)')
    axs[2].set_ylabel('Acc/Angle_acc')
    axs[2].grid()
    axs[2].legend()
    plt.xlabel('Time [s]')
    plt.suptitle('Position and Orientation Synchronization')
    # plt.show()

    # 3D 轨迹
    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')
    axs.plot(pos[:, 0], pos[:, 1], pos[:, 2], c='r', label='Trajectory')
    axs.scatter(start_pos[0], start_pos[1], start_pos[2], c='r', marker='o', label='Start Pos')
    axs.scatter(end_pos[0], end_pos[1], end_pos[2], c='g', marker='o', label='End Pos')
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.set_zlabel('Z')
    axs.set_title('3D Trajectory')
    axs.legend()
    # plt.show()


    fig, axs = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
    axs.plot(t, rpy[:, 0], label='Roll')
    axs.plot(t, rpy[:, 1], label='Pitch')
    axs.plot(t, rpy[:, 2], label='Yaw')
    axs.set_ylabel('RPY (radians)')
    axs.grid()
    axs.legend()
    plt.show()
