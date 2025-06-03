import base
import numpy as np
import matplotlib.pyplot as plt
import ruckig
from ruckig import Ruckig, InputParameter, OutputParameter, Result, Trajectory

def test_orien_plan_slerp(ori_rpy1, ori_rpy2, ori_rpy3):
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
        angle1[i], v1[i] = base.quat_angle_diff_and_axi(quat1_new, q1_2[i], v01) # 理论上初始轴只是用于第一次

    # 计算角速度
    for i in range(1, steps_1):
        angle_vel1[i] = (angle1[i + 1] - angle1[i - 1]) / (2 * dt)
    angle_vel1[steps_1] = (angle1[steps_1] - angle1[steps_1 - 1]) / dt

    # q2->q3插补的四元数
    q2_3 = np.zeros((steps_2 + 1, 4))
    for i in range(1, steps_2 + 1):
        t = i * dt * coef2
        q2_3[i] = base.Slerp_orientation(angle_diff2, quat2_new, quat3, t)
        angle2[i], v2[i] = base.quat_angle_diff_and_axi(quat2_new, q2_3[i], v02) # 理论上初始轴只是用于第一次

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

def test_orien_plan_doubleS(q1, q2, q3):
    # 计算四元数之间的角度差
    quat1_new, angle_diff1, v01 = base.quat_angle_diff_and_axi_first(q1, q2)
    quat2_new, angle_diff2, v02 = base.quat_angle_diff_and_axi_first(q2, q3)
    print("angle_diff1:", angle_diff1)
    print("angle_diff2:", angle_diff2)
    # 轴角的角度
    # angle_diff1 *= 2.0
    # angle_diff2 *= 2.0

    # 设置omega_vel_max, omega_acc_max
    input = InputParameter(1)
    input.max_velocity = [1.0]
    input.max_acceleration = [5.0]
    input.max_jerk = [10.0]

    input.current_position = [0.0]
    input.current_velocity = [0.0]
    input.current_acceleration = [0.0]

    input.target_position = [angle_diff1]
    input.target_velocity = [0.0]
    input.target_acceleration = [0.0]

    ruckig = Ruckig(1)
    traj_1 = Trajectory(1)
    res = ruckig.calculate(input, traj_1)
    if res == Result.ErrorInvalidInput:
        raise Exception("Invalid input for Ruckig calculation")

    # 计算第二段角位移的时间
    input.target_position = [angle_diff2]
    traj_2 = Trajectory(1)
    res = ruckig.calculate(input, traj_2)
    if res == Result.ErrorInvalidInput:
        raise Exception("Invalid input for Ruckig calculation")
    
    print(f'Trajectory 1 duration: {traj_1.duration}, Trajectory 2 duration: {traj_2.duration}')
    t_total_1 = traj_1.duration
    t_total_2 = traj_2.duration
    dt = 0.004

    steps_1 = int(t_total_1 / dt)
    steps_2 = int(t_total_2 / dt)

    angle1 = np.zeros((steps_1 + 1, 1))
    angle2 = np.zeros((steps_2 + 1, 1))

    angle_vel1 = np.zeros((steps_1 + 1, 1))
    angle_vel2 = np.zeros((steps_2 + 1, 1))

    angle1_acc = np.zeros((steps_1 + 1, 1))
    angle2_acc = np.zeros((steps_2 + 1, 1))
    
    angle1_temp = np.zeros((steps_1 + 1, 1))
    angle2_temp = np.zeros((steps_2 + 1, 1))

    # 旋转轴
    v1 = np.zeros((steps_1 + 1, 3))
    v2 = np.zeros((steps_2 + 1, 3))

    # q1->q2插补的四元数
    q1_2 = np.zeros((steps_1 + 1, 4))
    for i in range(0, steps_1 + 1):
        angle1[i], angle_vel1[i], angle1_acc[i] = traj_1.at_time(i * dt)
        delta_quat = np.array([v01[0] * np.sin(angle1[i]),
                              v01[1] * np.sin(angle1[i]),
                              v01[2] * np.sin(angle1[i]),
                              np.cos(angle1[i])])
        q1_2[i] = base.new_quat(quat1_new, delta_quat) # q = q1_new * delta_quat
        angle1_temp[i], v1[i] = base.quat_angle_diff_and_axi(quat1_new, q1_2[i], v01)  # 理论上初始轴只是用于第一次

    print("q1_2:", q1_2[steps_1, :])
    # q2->q3插补的四元数
    q2_3 = np.zeros((steps_2 + 1, 4))
    for i in range(0, steps_2 + 1):
        angle2[i], angle_vel2[i], angle2_acc[i] = traj_2.at_time(i * dt)
        delta_quat = np.array([v02[0] * np.sin(angle2[i]),
                              v02[1] * np.sin(angle2[i]),
                              v02[2] * np.sin(angle2[i]),
                              np.cos(angle2[i])])
        q2_3[i] = base.new_quat(quat2_new, delta_quat) # q = q2_new * delta_quat
        angle2_temp[i], v2[i] = base.quat_angle_diff_and_axi(quat2_new, q2_3[i], v02)  # 理论上初始轴只是用于第一次

    print("q2_3:", q2_3[steps_2, :])
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
    plt.grid()

    # plot 角速度
    fig = plt.figure()
    axs = fig.add_subplot(111)
    axs.plot(np.arange(steps_1 + 1) * dt, angle_vel1, marker='.', label='angle_vel1')
    axs.plot(np.arange(steps_2 + 1) * dt, angle_vel2, marker='.' ,label='angle_vel2')
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Angular Velocity (rad/s)')
    axs.set_title('Angular Velocity over Time')
    axs.legend()
    plt.grid()
    plt.show()

def test_orien_plan_doubleS2(q1, q2, q3, zone_deg):
    zone_rad = np.deg2rad(zone_deg) * 0.5 # 将轴角的角度转换为四元数的弧度
    # 计算四元数之间的角度差
    quat1_new, angle_diff1, v01 = base.quat_angle_diff_and_axi_first(q1, q2)
    quat2_new, angle_diff2, v02 = base.quat_angle_diff_and_axi_first(q2, q3)
    print("angle_diff1:", angle_diff1)
    print("angle_diff2:", angle_diff2)

    # 设置omega_vel_max, omega_acc_max
    input = InputParameter(1)
    input.max_velocity = [1.0]
    input.max_acceleration = [5.0]
    input.max_jerk = [10.0]

    input.current_position = [0.0]
    input.current_velocity = [0.0]
    input.current_acceleration = [0.0]

    input.target_position = [angle_diff1]
    input.target_velocity = [0.0]
    input.target_acceleration = [0.0]

    ruckig = Ruckig(1)
    traj_1 = Trajectory(1)
    res = ruckig.calculate(input, traj_1)
    if res == Result.ErrorInvalidInput:
        raise Exception("Invalid input for Ruckig calculation")

    # 计算第二段角位移的时间
    input.target_position = [angle_diff2]
    traj_2 = Trajectory(1)
    res = ruckig.calculate(input, traj_2)
    if res == Result.ErrorInvalidInput:
        raise Exception("Invalid input for Ruckig calculation")
    
    print(f'Trajectory 1 duration: {traj_1.duration}, Trajectory 2 duration: {traj_2.duration}')
    t_total_1 = traj_1.duration
    t_total_2 = traj_2.duration
    dt = 0.004

    steps_1 = int(t_total_1 / dt) + 1
    steps_2 = int(t_total_2 / dt) + 1

    angle1 = np.zeros((steps_1, 1))
    angle1_vel = np.zeros((steps_1, 1))
    angle1_acc = np.zeros((steps_1, 1))
    q1_imp = np.zeros((steps_1, 4))
    v1 = np.zeros((steps_1, 3))
    
    angle2 = np.zeros((steps_2, 1))
    angle2_vel = np.zeros((steps_2, 1))
    angle2_acc = np.zeros((steps_2, 1))
    q2_imp = np.zeros((steps_2, 4))
    v2 = np.zeros((steps_2, 3))
    idx_1 = 0
    idx_2 = 0
    
    q_temp = np.zeros((4, 1))
    angle_temp = .0
    for i in range(steps_1):
        angle1[i], angle1_vel[i], angle1_acc[i] = traj_1.at_time(i * dt)
        q_temp = np.array([v01[0] * np.sin(angle1[i]),
                            v01[1] * np.sin(angle1[i]),
                            v01[2] * np.sin(angle1[i]),
                            np.cos(angle1[i])])
        q1_imp[i] = base.new_quat(quat1_new, q_temp)
        angle_temp, v1[i] = base.quat_angle_diff_and_axi(quat1_new, q1_imp[i], v01)

    for i in range(steps_2):
        angle2[i], angle2_vel[i], angle2_acc[i] = traj_2.at_time(i * dt)
        q_temp = np.array([v02[0] * np.sin(angle2[i]),
                            v02[1] * np.sin(angle2[i]),
                            v02[2] * np.sin(angle2[i]),
                            np.cos(angle2[i])])
        q2_imp[i] = base.new_quat(quat2_new, q_temp)
        angle_temp, v2[i] = base.quat_angle_diff_and_axi(quat2_new, q2_imp[i], v02)

    # 计算过渡点index
    for i in range(steps_1):
        if angle1[i] - (angle_diff1 - zone_rad) >= 0:
            idx_1 = i
            break
    qa = q1_imp[idx_1]
    angle_a = angle1[idx_1]
    a_vel = angle1_vel[idx_1]
    a_acc = angle1_acc[idx_1]
    # 四元数转换到三维空间
    p1 = np.array([v01[0] * angle_a, v01[1] * angle_a, v01[2] * angle_a])
    p1_vel = np.array([v01[0] * a_vel, v01[1] * a_vel, v01[2] * a_vel])
    p1_acc = np.array([v01[0] * a_acc, v01[1] * a_acc, v01[2] * a_acc])
    
    for i in range(steps_2):
        if angle2[i] - zone_rad >= 0:
            idx_2 = i
            break
    qb = q2_imp[idx_2]
    angle_b = angle2[idx_2]
    b_vel = angle2_vel[idx_2]
    b_acc = angle2_acc[idx_2]
    print("qa: ", qa)
    print("qb: ", qb)
    # 四元数转换到三维空间
    p2 = np.array([v02[0] * angle_b, v02[1] * angle_b, v02[2] * angle_b])
    p2_vel = np.array([v02[0] * b_vel, v02[1] * b_vel, v02[2] * b_vel])
    p2_acc = np.array([v02[0] * b_acc, v02[1] * b_acc, v02[2] * b_acc])

    print(f"idx_1={idx_1}, idx_2={idx_2}")
    print(f"angle_a={angle_a}, angle_b={angle_b}")
    print(f"a_vel={a_vel}, angle_b={b_vel}")
    print(f"a_acc={a_acc}, angle_b={b_acc}")

    print(f"p1={float(p1[0]):.3f}, {float(p1[1]):.3f}, {float(p1[2]):.3f}")
    print(f"p2={float(p2[0]):.3f}, {float(p2[1]):.3f}, {float(p2[2]):.3f}")
    print(f"p1={float(p1_vel[0]):.3f}, {float(p1_vel[1]):.3f}, {float(p1_vel[2]):.3f}")
    print(f"p2={float(p2_vel[0]):.3f}, {float(p2_vel[1]):.3f}, {float(p2_vel[2]):.3f}")
    print(f"p1={float(p1_acc[0]):.3f}, {float(p1_acc[1]):.3f}, {float(p1_acc[2]):.3f}")
    print(f"p2={float(p2_acc[0]):.3f}, {float(p2_acc[1]):.3f}, {float(p2_acc[2]):.3f}")

    # 计算过渡
    input2 = InputParameter(3)
    input2.max_velocity = [1.0, 1.0, 1.0]
    input2.max_acceleration = [5.0, 5.0, 5.0]
    input2.max_jerk = [10.0, 10.0, 10.0]

    input2.current_position = [p1[0], p1[1], p1[2]]
    input2.current_velocity = [p1_vel[0], p1_vel[1], p1_vel[2]]
    input2.current_acceleration = [p1_acc[0], p1_acc[1], p1_acc[2]]

    input2.target_position = [p2[0], p2[1], p2[2]]
    input2.target_velocity = [p2_vel[0], p2_vel[1], p2_vel[2]]
    input2.target_acceleration = [p2_acc[0], p2_acc[1], p2_acc[2]]
    ruckig2 = Ruckig(3)
    traj_3 = Trajectory(3)
    res2 = ruckig2.calculate(input2, traj_3)
    if res2 == Result.ErrorInvalidInput:
        raise Exception("Invalid input for Ruckig calculation")
    print(f'Trajectory 3 duration: {traj_3.duration}')

    t_total_3 = traj_3.duration
    steps_3 = int(t_total_3 / dt) + 1
    Cert_imp = np.zeros((steps_3, 3)) # x,y,z
    Cert_vel = np.zeros((steps_3, 3)) # x,y,z
    Cert_acc = np.zeros((steps_3, 3)) # x,y,z

    va_b = np.zeros((steps_3, 3)) # 过渡段的旋转轴
    anglea_b = np.zeros((steps_3, 1)) # 过渡段的角度
    qa_b = np.zeros((steps_3, 4)) # 四元数

    omega_ab = np.zeros((steps_3, 1))
    omega_v = np.zeros((steps_3, 3)) # 过渡段的角速度对应的旋转轴

    dot_omega_ab = np.zeros((steps_3, 1)) # 过渡段的角加速度
    dot_omega_v = np.zeros((steps_3, 3)) # 过渡段的角加速度对应的旋转轴


    for i in range(steps_3):
        Cert_imp[i], Cert_vel[i], Cert_acc[i] = traj_3.at_time(i * dt)
        anglea_b[i] = np.linalg.norm(Cert_imp[i])
        va_b[i] = Cert_imp[i] / anglea_b[i] if anglea_b[i] != 0 else va_b[i - 1]
        
        qa_b[i] = np.array([va_b[i][0] * np.sin(anglea_b[i][0]),
                            va_b[i][1] * np.sin(anglea_b[i][0]),
                            va_b[i][2] * np.sin(anglea_b[i][0]),
                            np.cos(anglea_b[i][0])])
        
    
    print("q1_3_first:", qa_b[0, :])
    print("q1_3_end:", qa_b[steps_3 - 1, :])

    # 四元数
    q_all = np.zeros((idx_1 + steps_3 + steps_2 - idx_2 - 1, 4))
    q_all[:idx_1, :] = q1_imp[:idx_1, :]
    q_all[idx_1:idx_1 + steps_3, :] = qa_b
    q_all[idx_1 + steps_3:, :] = q2_imp[idx_2 + 1:, :]
    # 四元数的角度
    angle_all = np.zeros((idx_1 + steps_3 + steps_2 - idx_2 - 1, 1))
    angle_all[:idx_1] = angle1[:idx_1]
    angle_all[idx_1:idx_1 + steps_3] = anglea_b
    angle_all[idx_1 + steps_3:] = angle2[idx_2 + 1:]
    # 四元数的角速度
    omega_all = np.zeros((idx_1 + steps_3 + steps_2 - idx_2 - 1, 1))
    omega_all[:idx_1] = angle1_vel[:idx_1]
    omega_all[idx_1:idx_1 + steps_3] = omega_ab
    omega_all[idx_1 + steps_3:] = angle2_vel[idx_2 + 1:]
    # 四元数的角加速度
    dot_omega_ab_all = np.zeros((idx_1 + steps_3 + steps_2 - idx_2 - 1, 1))
    dot_omega_ab_all[:idx_1] = angle1_acc[:idx_1]
    dot_omega_ab_all[idx_1:idx_1 + steps_3] = dot_omega_ab
    dot_omega_ab_all[idx_1 + steps_3:] = angle2_acc[idx_2 + 1:]

    # 旋转轴
    v_all = np.zeros((idx_1 + steps_3 + steps_2 - idx_2 - 1, 3))
    v_all[:idx_1, :] = v1[:idx_1, :]
    v_all[idx_1:idx_1 + steps_3, :] = va_b
    v_all[idx_1 + steps_3:, :] = v2[idx_2 + 1:, :]

    
    # plot
    t = np.arange(idx_1 + steps_3 + steps_2 - idx_2 - 1) * dt
    fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    # 角度
    # axs[0].plot(t, angle_all, marker='.', label='angle')
    axs[0].plot(t, angle_all, label='angle')
    axs[0].scatter(t[idx_1], angle_all[idx_1], c='r', marker='o', label='start')
    axs[0].scatter(t[idx_1 + steps_3 - 1], angle_all[idx_1 + steps_3 - 1], c='g', marker='o', label='end')
    axs[0].set_ylabel('Angle (rad)')
    axs[0].set_title('Angle over Time')
    axs[0].legend()
    axs[0].grid()
    # 角速度
    axs[1].plot(t, omega_all, label='angular velocity')
    axs[1].scatter(t[idx_1], omega_all[idx_1], c='r', marker='o', label='start')
    axs[1].scatter(t[idx_1 + steps_3 - 1], omega_all[idx_1 + steps_3 - 1], c='g', marker='o', label='end')
    axs[1].set_ylabel('Angular Velocity (rad/s)')
    axs[1].set_title('Angular Velocity over Time')
    axs[1].legend()
    axs[1].grid()
    # 角加速度
    axs[2].plot(t, dot_omega_ab_all, label='angular acceleration')
    axs[2].scatter(t[idx_1], dot_omega_ab_all[idx_1], c='r', marker='o', label='start')
    axs[2].scatter(t[idx_1 + steps_3 - 1], dot_omega_ab_all[idx_1 + steps_3 - 1], c='g', marker='o', label='end')
    axs[2].set_ylabel('Angular Acceleration (rad/s^2)')
    axs[2].set_title('Angular Acceleration over Time')
    axs[2].legend()
    axs[2].grid()
    # 旋转轴
    axs[3].plot(t, v_all[:,0], linestyle= '-', label='x axis')
    axs[3].plot(t, v_all[:,1], linestyle= '--', label='y axis')
    axs[3].plot(t, v_all[:,2], linestyle= '-.', label='z axis')
    axs[3].scatter(t[idx_1], v_all[idx_1, 0], c='r', marker='o', label='start')
    axs[3].scatter(t[idx_1 + steps_3 - 1], v_all[idx_1 + steps_3 - 1, 0], c='g', marker='o', label='end')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Rotation Axis')
    axs[3].set_title('Rotation Axis over Time')
    axs[3].legend()
    axs[3].grid()
    plt.tight_layout()
    plt.show()