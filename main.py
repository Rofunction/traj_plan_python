import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import base
import orien_plan
import pos

if __name__ == "__main__":
    # pos.pos_interpolation()
    P1 = np.array([0.2, 0.3, 0.5])
    P2 = np.array([1.0, 0.1, 0.8])
    P3 = np.array([0.6, 0.0, 0.4])
    q1 = np.array([-.495, -.297, -.774, .258])
    q2 = np.array([-.828, -.315, -.177, .429])
    q3 = np.array([.490, -.487, .511, .511])
    # zone = np.array([0.3, 45.0])
    zone = np.array([0.2, 30.0])
    # zone = np.array([0.1, 15.0])
    # orien_plan.test_orien_plan_slerp(q1, q2, q3)
    # orien_plan.test_orien_plan_doubleS(q1, q2, q3)
    # orien_plan.orien_imp_doubleS3(q1, q2, q3, 10.0) # 姿态过渡
    # pos.pos_orien_syn_no_blending(P1, P2, P3, q1, q2, q3, zone) # 位姿同步无过渡
    pos.pos_orien_syn_bleding(P1, P2, P3, q1, q2, q3, zone) # 位姿同步blending
    # pos.pos_interpolation()