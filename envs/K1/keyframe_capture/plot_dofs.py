# plot chosen dofs and imu to get a better understanding of the data and to find good key frames for imitation learning
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

def plot_dofs_and_imu(df):
    # plot dof pos and imu data
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    # we have no time colmn, so we just use the index as time
    plt.plot(df["time"], df["dof_0"], label="dof_0")
    plt.plot(df["time"], df["dof_1"], label="dof_1")
    plt.plot(df["time"], df["dof_2"], label="dof_2")
    plt.plot(df["time"], df["dof_3"], label="dof_3")
    plt.plot(df["time"], df["gyro0"], label="gyro0")
    plt.plot(df["time"], df["rpy0"], label="rpy0")

    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title("DOF Positions and IMU Data Over Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



# check raw data
df = pd.read_csv("/home/max/repos/htwk-gym/dofs_imu.csv")
df["time"] = df.index # correct time is irrelevant here, we just want to see the shape of the data
# plot_dofs_and_imu(df)

# we saw that nothing really happens until around index 4600, so we truncate the dateset
# and save it to a new csv file and plot it again
df_truncated = df[df["time"] >= 4600]
df_truncated.to_csv("/home/max/repos/htwk-gym/dofs_imu_truncated.csv", index=False)
plot_dofs_and_imu(df_truncated)

# now we see stand up 1 plays from around 5000 to 5300
# and stand up 2 from around 5800 to 6000, so we can use these two segments as key frames for imitation learning
df_stand_up_1 = df_truncated[(df_truncated["time"] >= 5000) & (df_truncated["time"] < 5300)]
df_stand_up_2 = df_truncated[(df_truncated["time"] >= 5800) & (df_truncated["time"] < 6000)]
plot_dofs_and_imu(df_stand_up_1)
plot_dofs_and_imu(df_stand_up_2)

# now that we know our key frames for the stand up motions
# we can extract import key positions and use them in our
# reward functions

# XXX Booster docs open source K1 Manual DOFs Reihenfolge 0 bis 21