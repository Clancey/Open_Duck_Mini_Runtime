from ischedule import schedule, run_loop
import time
import numpy as np
import random
from oclock import Timer
from threading import Thread


def destructor():
    while True:
        time.sleep(0.0001)

t = Thread(target=destructor)
t.start()


DT = 1.0 / 50

times_schedule = []
times_time = []  # time.time
times_oclock = []

@schedule(interval=DT)
def loop():
    times_schedule.append(time.time())
    time.sleep(random.random() * DT/2)


run_loop(return_after=3)

s = time.time()
while True:
    loop_s = time.time()
    times_time.append(time.time())
    if time.time() - s > 3:
        break

    time.sleep(random.random() * DT/2)

    took = time.time() - loop_s

    time.sleep(DT - took)


timer = Timer(interval=DT)
s = time.time()
while True:
    loop_s = time.time()
    times_oclock.append(time.time())
    if time.time() - s > 3:
        break

    time.sleep(random.random() * DT/2)

    timer.checkpt()





import matplotlib.pyplot as plt

diffs_schedule = np.diff(times_schedule)
diffs_time = np.diff(times_time)
diffs_oclock = np.diff(times_oclock)

plt.plot(diffs_schedule, label='schedule')
plt.plot(diffs_time, label='time')
plt.plot(diffs_oclock, label='oclock')
plt.legend()
plt.show()

t.join()