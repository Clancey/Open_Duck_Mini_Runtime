from pypot.feetech import FeetechSTS3215IO
import time
import pickle
import numpy as np

LOG_TIME = 3
io = FeetechSTS3215IO("/dev/ttyACM0")
io.set_mode({1: 0})
# io.set_lock({1: 0})
# maximum_acceleration = 0 # doesn't seem to have an effect

accelerations = [0, 50, 100, 200, 250, 255]
kps = [4, 8, 16, 32]
kds = [0, 4, 8, 16, 32]


for acceleration in accelerations:
    for kp in kps:
        for kd in kds:

            # acceleration = 50
            # kp = 32
            # kd = 0
            # io.set_maximum_acceleration({1: maximum_acceleration})
            io.set_acceleration({1: acceleration})

            io.set_P_coefficient({1: kp})
            io.set_D_coefficient({1: kd})
            print("acceleration", io.get_acceleration([1]))
            print("p", io.get_P_coefficient([1]))
            print("d", io.get_D_coefficient([1]))


            time.sleep(0.2)
            present_positions = []
            speeds = []
            loads = []
            times = []
            goal_positions = []
            currents = []

            goal_position = 90


            def convert_load(raw_load):
                if raw_load > 1023:
                    return (raw_load - 1024) * 0.001
                return -raw_load * 0.001


            io.set_goal_position({1: 0})
            time.sleep(2.0)
            io.set_goal_position({1: goal_position})
            s = time.time()
            while True:
                present_position = np.deg2rad(io.get_present_position([1]))[0]
                raw_load = io.get_present_load([1])[0]
                load = convert_load(raw_load)
                speed = np.deg2rad(io.get_present_speed([1]))[0]
                present_positions.append(present_position)
                present_current = io.get_present_current([1])[0]

                currents.append(present_current)
                speeds.append(speed)
                loads.append(load)
                times.append(time.time() - s)
                goal_positions.append(np.deg2rad(goal_position))

                time.sleep(0.01)
                if time.time() - s > LOG_TIME:
                    break

            data = {
                "acceleration": acceleration,
                "kp": kp,
                "kd": kd,
                # "maximum_acceleration": maximum_acceleration,
                "positions": present_positions,
                "goal_positions": goal_positions,
                "speeds": speeds,
                "loads": loads,
                "currents": currents,
                "times": times,
            }

            pickle.dump(data, open(f"data_acceleration_{acceleration}_kp_{kp}_kd_{kd}.pkl", "wb"))

            time.sleep(0.1)
