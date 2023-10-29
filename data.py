import json
import random
import numpy as np
import math 

# Number of possible bus plans
M = 4

# Number of trips
T = 8

# Number of shifts
B = 2

# Sample costs
c = {1: 100, 2: 150, 3: 120, 4: 200}

# Sample number of times each trip needs to be covered
S = {1: 2, 2: 1, 3: 3, 4: 2, 5: 1, 6: 2, 7: 1, 8: 2}

# Sample number of buses in each shift
Tb = {1: 2, 2: 2}

# Sample values for UAT
UAT_values = {t: 10 for t in range(1, T + 1)}

# Bus capacities
cobus_capacity = 70
peakbus_capacity = 50

# Initialize the presence matrix h 
h = {
    1: {1: 1, 2: 0, 3: 1, 4: 0},
    2: {1: 0, 2: 1, 3: 1, 4: 0},
    3: {1: 1, 2: 1, 3: 1, 4: 0},
    4: {1: 0, 2: 0, 3: 0, 4: 1},
    5: {1: 1, 2: 0, 3: 0, 4: 0},
    6: {1: 0, 2: 1, 3: 0, 4: 0},
    7: {1: 0, 2: 0, 3: 1, 4: 0},
    8: {1: 0, 2: 0, 3: 0, 4: 1}
}

# Sample values for starting time of trips
Tstart = {t: np.random.uniform(0, 60) for t in range(1, T + 1)}
# Sample values for ending time of trips
Tend = {t: Tstart[t] + np.random.uniform(30, 120) for t in range(1, T + 1)}
# Sample values for the driving time between two successive trips
driving_time = {(t0, t1): np.random.uniform(15, 60) for t0 in range(1, T + 1) for t1 in range(1, T + 1) if t0 != t1}

# Update the costs based on idle time
for j in range(1, M + 1):
    idle_time_sum = 0
    if j in h:
        for t0 in range(1, T + 1):
            if t0 in h[j]:
                for t1 in range(1, T + 1):
                    if t1 in h[j]:
                        if t0 != t1 and h[j][t0] == 1 and h[j][t1] == 1:
                            idle_time_sum += (Tstart[t1] - Tend[t0] + driving_time[(t0, t1)])
    c[j] = c[j] + 1000 * math.atan(-0.21 * idle_time_sum + 1.57)

    
data = {
    "M": M,
    "T": T,
    "B": B,
    "c": c,
    "S": S,
    "Tb": Tb,
    "UAT_values": UAT_values,
    "cobus_capacity": cobus_capacity,
    "peakbus_capacity": peakbus_capacity,
    "h": h,
    
}

with open("data.json", "w") as data_file:
    json.dump(data, data_file)
