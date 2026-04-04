# plot chosen dofs and imu to get a better understanding of the data and to find good key frames for imitation learning
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

def plot_dofs_and_imu(df, lbl:str|None=None):
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
    plt.title(f"DOF Positions and IMU Data Over Time {lbl}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



# check raw data
df = pd.read_csv("/home/max/repos/htwk-gym/envs/K1/keyframe_capture/dofs_imu.csv")
df["time"] = df.index # correct time is irrelevant here, we just want to see the shape of the data
# plot_dofs_and_imu(df)

# we saw that nothing really happens until around index 4600, so we truncate the dateset
# and save it to a new csv file and plot it again
df_truncated = df[(df["time"] >= 5000) & (df["time"] < 6100)]
df_truncated.to_csv("/home/max/repos/htwk-gym/envs/K1/keyframe_capture/dofs_imu_truncated.csv", index=False)
plot_dofs_and_imu(df_truncated, lbl="Truncated")

# now we see stand up 1 plays from around 5000 to 5300
# and stand up 2 from around 5800 to 6000, so we can use these two segments as key frames for imitation learning
df_stand_up_1 = df_truncated[(df_truncated["time"] >= 5000) & (df_truncated["time"] < 5300)]
df_stand_up_2 = df_truncated[(df_truncated["time"] >= 5800) & (df_truncated["time"] < 6100)]
plot_dofs_and_imu(df_stand_up_1, lbl="Stand Up 1")
plot_dofs_and_imu(df_stand_up_2, lbl="Stand Up 2")

# now we glue the two segments together to get a dataset for imitation learning
df_stand_up = pd.concat([df_stand_up_1, df_stand_up_2], ignore_index=True)
plot_dofs_and_imu(df_stand_up, lbl="Stand Up Combined")
df_stand_up.to_csv("/home/max/repos/htwk-gym/envs/K1/keyframe_capture/dofs_imu_stand_up.csv", index=False)

# now that we know our key frames for the stand up motions
# we can extract import key positions and use them in our
# reward functions

# XXX Booster docs open source K1 Manual DOFs Reihenfolge 0 bis 21
# Index | Joint Name                     | Max (°) | Min (°) | Kommentar",
# --------------------------------------------------------------------------------",
# 0     | Head Yaw Joint                 | 59      | -59     | Kopf Yaw",
# 1     | Head Pitch Joint               | 43      | -17     | Kopf Pitch",
# 2     | Left Shoulder Pitch Joint      | 69      | -169    | linke Schulter Pitch",
# 3     | Left Shoulder Roll Joint       | 89      | -99     | linke Schulter Roll",
# 4     | Left Shoulder Yaw Joint        | 109     | -109    | linke Schulter Yaw",
# 5     | Left Elbow Joint               | 39      | -129    | linker Ellbogen",
# 6     | Right Shoulder Pitch Joint     | 69      | -169    | rechte Schulter Pitch",
# 7     | Right Shoulder Roll Joint      | 89      | -99     | rechte Schulter Roll",
# 8     | Right Shoulder Yaw Joint       | 109     | -109    | rechte Schulter Yaw",
# 9     | Right Elbow Joint              | 129     | -39     | rechter Ellbogen",
# 10    | Left Hip Pitch Joint           | 126     | -171    | linkes Hüft Pitch",
# 11    | Left Hip Roll Joint            | 89      | -22     | linkes Hüft Roll",
# 12    | Left Hip Yaw Joint             | 59      | -59     | linkes Hüft Yaw",
# 13    | Left Knee Joint                | 127     | 0       | linkes Knie",
# 14    | Left Ankle Up Joint            | 38      | -17     | linkes Sprunggelenk Up",
# 15    | Left Ankle Down Joint          | 41      | -16     | linkes Sprunggelenk Down",
# 16    | Right Hip Pitch Joint          | 126     | -171    | rechtes Hüft Pitch",
# 17    | Right Hip Roll Joint           | 22      | -89     | rechtes Hüft Roll",
# 18    | Right Hip Yaw Joint            | 59      | -59     | rechtes Hüft Yaw",
# 19    | Right Knee Joint               | 127     | 0       | rechtes Knie",
# 20    | Right Ankle Up Joint           | 38      | -17     | rechtes Sprunggelenk Up",
# 21    | Right Ankle Down Joint         | 41      | -16     | rechtes Sprunggelenk Down",