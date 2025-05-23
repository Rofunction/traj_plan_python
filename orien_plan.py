import base

def test_orien_plan(ori_rpy1, ori_rpy2, ori_rpy_3):
    quat1 = base.RPY2quat(ori_rpy1)
    quat2 = base.RPY2quat(ori_rpy2)
    quat3 = base.RPY2quat(ori_rpy_3)
    print("quat1:", quat1)
    print("quat2:", quat2)
    print("quat3:", quat3)

    # 计算四元数之间的角度差
    angle_diff1 = base.quat_angle_diff(quat1, quat2)
    angle_diff2 = base.quat_angle_diff(quat2, quat3)
    print("angle_diff1:", angle_diff1)
    print("angle_diff2:", angle_diff2)

    t_total_1 = 2.0
    t_total_2 = 3.0
    
    