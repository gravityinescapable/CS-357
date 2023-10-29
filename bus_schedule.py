import gurobipy as grb
import numpy as np
import matplotlib.pyplot as plt
import math
from data_loader import load_data
import json
import random

# Load sample data from data.json file
data = load_data("data.json")

# Extract data from the file
M = data["M"]
T = data["T"]
B = data["B"]
c = {int(j): data["c"][j] for j in data["c"]}
S = {int(t): data["S"][t] for t in data["S"]}
Tb = {int(b): data["Tb"][b] for b in data["Tb"]}
UAT_values = {int(t): data["UAT_values"][t] for t in data["UAT_values"]}
cobus_capacity = int(data["cobus_capacity"])
peakbus_capacity = int(data["peakbus_capacity"])
h = {int(j): {int(t): data["h"][j][t] for t in data["h"][j]} for j in data["h"]}


# Initialize Gurobi model for the master problem
master_model = grb.Model()
master_model.setParam('Presolve', 0)
master_model.setParam('MIPFocus', 2)
# Disable the use of user-provided MIP start solution
master_model.setParam('StartAlg', 0)  
master_model.setParam('NodeLimit',10)
# Increasing default precision settings
master_model.setParam('MIPGap', 1e-6)


# Define binary decision variables yj for each bus plan j
y = {}
for j in range(1, M + 1):
    y[j] = master_model.addVar(vtype=grb.GRB.BINARY, name=f'y_{j}')

# Define binary decision variables for bus type (Cobus or Peak Bus)
cobus = {}
peakbus = {}
for j in range(1, M + 1):
    cobus[j] = master_model.addVar(vtype=grb.GRB.BINARY, name=f'cobus_{j}')
    
# Define continuous variables UAT for unassigned trips
UAT = {}
for t in range(1, T + 1):
    UAT[t] = master_model.addVar(vtype=grb.GRB.CONTINUOUS, name=f'UAT_{t}')

# Define binary variables for driver breaks
driver_breaks = {}
for t in range(1, T + 1):
    driver_breaks[t] = master_model.addVar(vtype=grb.GRB.BINARY, name=f'driver_break_{t}')

# Update the master model to include the variables
master_model.update()

# Set the objective function to minimize the total cost using the provided cost function
obj_expr = grb.LinExpr()

for j in range(1, M + 1):
    # Cost of selecting bus plans
    obj_expr += c[j] * y[j]  
    # Cost of selecting a Cobus 
    obj_expr += 2 * cobus[j]  
    # Cost of selecting a Peak Bus
    obj_expr += 5 * (1-cobus[j]) 

# Add the unassigned trip cost in the objective function
for t in range(1, T + 1):
    # Use UAT_values dictionary to set the coefficients
    obj_expr += UAT[t] * UAT_values[t]  

# Add the cost of driver breaks
# Cost associated with a break
break_cost = 10  
for t in range(1, T + 1):
    obj_expr += driver_breaks[t] * break_cost

master_model.setObjective(obj_expr, grb.GRB.MINIMIZE)


# Dictionary to store aggregated trip assignment constraints
trip_assignment_constraints = {}  

# Dictionary to store aggregated shift assignment constraints
shift_assignment_constraints = {}  

# Add trip assignment and UAT constraints
for t in range(1, T + 1):
    trip_assignment_constraints[t] = master_model.addConstr(
        grb.quicksum(h[t][j] * y[j] for j in range(1, M + 1)) + UAT[t] == S[t], name=f'TripAssignment_{t}')

# Add shift assignment constraints
for b in range(1, B + 1):
    shift_assignment_constraints[b] = master_model.addConstr(
        grb.quicksum(y[j] for j in range(1, M + 1)) == Tb[b], name=f'ShiftAssignment_{b}')

# Capacity constraints for Cobus and Peak Bus
for j in range(1, M + 1):
    expr_capacity = cobus[j] * cobus_capacity + peakbus_capacity * (1 - cobus[j])
    master_model.addConstr(expr_capacity >= 1, name=f'CapacityConstraint_{j}')

# Heuristic Initialization: Create an initial set of bus plans
# Initialize half of the buses as Cobuses
cobus_count = M // 2
initial_bus_plans = [1] * cobus_count + [0] * (M - cobus_count)
# Shuffle the initial assignment
np.random.shuffle(initial_bus_plans)  

# Apply the initial solution to the model
for j in range(1, M + 1):
    y[j].start = initial_bus_plans[j - 1]

# Generate random dual multiplier values for trips (t)
dual_multipliers_t = {t: random.uniform(100, 200) for t in range(1, T + 1)}

# Generate random dual multiplier values for shifts (b)
dual_multipliers_b = {b: random.uniform(1000, 2000) for b in range(1, B + 1)}

# Integrate the column generation algorithm
reduced_cost = {j: c[j] for j in range(1, M + 1)}
# Maximum number of column generation iterations
max_iterations = 50
# Counter to keep track of the number of iterations
iteration = 0

while True:
    # Solve the LP-relaxation of the master problem
    master_model.setParam('Heuristics', 0)
    master_model.optimize()

    # Function to solve the pricing problem for a given shift b
    def solve_pricing_problem(b):
        pricing_model = grb.Model()

        # Define binary decision variables y_new, cobus_new, and driver_break_new
        y_new = pricing_model.addVar(vtype=grb.GRB.BINARY, name=f'y_new_{b}')
        cobus_new = pricing_model.addVar(vtype=grb.GRB.BINARY, name=f'cobus_new_{b}')
        driver_break_new = pricing_model.addVar(vtype=grb.GRB.BINARY, name=f'driver_break_new_{b}')
        UAT_new = pricing_model.addVar(vtype=grb.GRB.CONTINUOUS, name=f'UAT_new_{b}')  # Add UAT variable

        # Update the pricing model to include the variables
        pricing_model.update()

        # Get the dual multipliers for this shift b
        dual_multiplier_t = dual_multipliers_t[b]
        dual_multiplier_b = dual_multipliers_b[b]

        # Calculate the reduced cost
        reduced_cost = c[b] - dual_multiplier_t - dual_multiplier_b

        # Define the objective function and constraints for the pricing problem based on the shift b
        idle_cost = 1000 * y_new
        pricing_model.setObjective(reduced_cost * y_new + idle_cost + 2 * cobus_new + 5 * (1 - cobus_new) + driver_break_new * break_cost + UAT_new * UAT_values[b], grb.GRB.MINIMIZE)  # Include UAT cost

        # Add constraints
        pricing_model.addConstr(y_new <= 1, name=f'CapacityConstraint_{b}')
        pricing_model.addConstr(cobus_new + (1 - cobus_new) == 1, name=f'TypeConstraint_{b}')
        pricing_model.addConstr(driver_break_new + (1 - driver_break_new) == 1, name=f'DriverBreakConstraint_{b}')
        pricing_model.addConstr(UAT_new >= 0, name=f'UATConstraint_{b}')  # Ensure UAT is non-negative

        # Solve the pricing problem
        pricing_model.optimize()
        if pricing_model.status == grb.GRB.OPTIMAL:
            new_cost = pricing_model.objVal
            return new_cost, driver_break_new.x, UAT_new.x
        else:
            return None, None, None

    # Flag to track if new columns are added in this iteration
    new_columns_added = False

    # Solve the pricing problem for each shift b
    for b in range(1, B + 1):
        new_cost, new_break, new_UAT = solve_pricing_problem(b)
        # Check for negative reduced cost
        if new_cost is not None and new_cost < 0:
            new_columns_added = True
            # Add the new column (bus plan) to the master problem
            y_new = master_model.addVar(vtype=grb.GRB.BINARY, name=f'y_new_{b}')
            cobus_new = master_model.addVar(vtype=grb.GRB.BINARY, name=f'cobus_new_{b}')
            driver_break_new = master_model.addVar(vtype=grb.GRB.BINARY, name=f'driver_break_new_{b}')
            UAT_new = master_model.addVar(vtype=grb.GRB.CONTINUOUS, name=f'UAT_new_{b}')

            # Add the new variables to the master model
            master_model.update()

            # Update the objective function and add constraints for the new column
            idle_cost = 1000 * y_new
            master_model.setObjectiveN(reduced_cost * y_new + idle_cost + 2 * cobus_new + 5 * (1 - cobus_new) + driver_break_new * break_cost + UAT_new * UAT_values[b], 1, weight=1)  # Adjust the cost function as needed
            master_model.addConstr(y_new <= 1, name=f'CapacityConstraint_{b}')
            master_model.addConstr(cobus_new + (1 - cobus_new) == 1, name=f'TypeConstraint_{b}')
            master_model.addConstr(driver_break_new + (1 - driver_break_new) == 1, name=f'DriverBreakConstraint_{b}')
            master_model.addConstr(UAT_new >= 0, name=f'UATConstraint_{b}')

    if not new_columns_added:
        # Terminate if no more negative reduced cost columns can be found
        break
    iteration += 1
   
    
# Solve the final master problem
master_model.optimize()

# Print results
if master_model.status == grb.GRB.OPTIMAL:
    selected_bus_plans = [j for j in range(1, M + 1) if y[j].x > 0.5]
    print(f'Selected Bus Plans: {selected_bus_plans}')
    print(f'Objective Value: {master_model.objVal}')
else:
    print('Optimal solution not found.')

# Plot the selected bus plans and their types
if master_model.status == grb.GRB.OPTIMAL:
    selected_types = []
    for j in selected_bus_plans:
        if cobus[j].x > 0.5:
            selected_types.append("Cobus")
        else:
            selected_types.append("Peak Bus")

    plt.bar(selected_bus_plans, [cobus[j].x for j in selected_bus_plans], color='blue', label='Cobus')
    plt.bar(selected_bus_plans, [1 - cobus[j].x for j in selected_bus_plans], color='red', label='Peak Bus')
    plt.xlabel('Bus Plan')
    plt.ylabel('Bus Type')
    plt.legend(loc='best')
    plt.title('Selected Bus Plans and Their Types')
    plt.show(block=False)

    plt.figure()
    plt.bar(UAT.keys(), [UAT[t].x for t in UAT], color='green')
    plt.xlabel('Trip')
    plt.ylabel('Unassigned Trip Costs')
    plt.title('Unassigned Trip Costs')
    plt.tight_layout()
    plt.show(block=False)

    plt.show()
else:
    print('Optimal solution not found.')
# Dispose the Gurobi model
master_model.dispose()
