import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import base
import orien_plan
import pos

if __name__ == "__main__":
    # pos.pos_interpolation()
    q1 = np.array([-.495, -.297, -.774, .258])
    q2 = np.array([-.828, -.315, -.177, .429])
    q3 = np.array([.490, -.487, .511, .511])
    orien_plan.test_orien_plan(q1, q2, q3)