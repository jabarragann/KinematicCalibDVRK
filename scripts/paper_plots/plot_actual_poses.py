import matplotlib.pyplot as plt
import numpy as np


poses1_cp = np.load("results/attempt_to_reproduce_20241212/poses1_cp_actual-measured.npy")
poses2_cp = np.load("results/attempt_to_reproduce_20241212/poses2_cp_actual-measured.npy")


print(poses1_cp.shape)
print(poses2_cp.shape)

fig, axes = plt.subplots(3,3, sharex=True)

axes[0, 0].plot(poses1_cp[0,3,:], label="poses1_cp_x")
axes[1, 0].plot(poses1_cp[1,3,:], label="poses1_cp_y")
axes[2, 0].plot(poses1_cp[2,3,:], label="poses1_cp_z")

axes[0, 1].plot(poses2_cp[0,3,:], label="poses2_cp_x")
axes[1, 1].plot(poses2_cp[1,3,:], label="poses2_cp_y")
axes[2, 1].plot(poses2_cp[2,3,:], label="poses2_cp_z")

#error plots
axes[0, 2].plot(poses2_cp[0,3,:]-poses1_cp[0,3,:], label="error_x")
axes[1, 2].plot(poses2_cp[1,3,:]-poses1_cp[1,3,:], label="error_y")
axes[2, 2].plot(poses2_cp[2,3,:]-poses1_cp[2,3,:], label="error_z")

plt.show()