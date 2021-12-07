from typing import Tuple
import cvxpy as cp, numpy as np

DORMS = 41
RESTAURANTS = 7
MINUTES_PER_HOUR = 60

# population of dorm d: p[d]
populations = np.array([
    124,
    314,
    397,
    183,
    711,
    569,
    246,
    527,
    281,
    690,
    18,
    8,
    160,
    14,
    147,
    224,
    777,
    489,
    492,
    2763,
    262,
    102,
    3992,
    404,
    477,
    162,
    709,
    1060,
    249,
    28,
    13,
    683,
    48,
    48,
    596,
    280,
    1050,
    644,
    1532,
    797,
    260,
])

# distace (rounded to the next minute) from dorm d to restaurant r: distances[r][d]
distances = np.ceil(np.array([
    [13.42,2.37,5.38,17.37,12.72,14.05,14.05],
    [4.1,12.15,10,8.35,6.12,1.27,1.27],
    [3.05,12.02,10.22,7.3,5.07,1.87,1.87],
    [13.2,4.67,1.45,18.02,13.37,10.48,10.48],
    [12.18,2.48,2.47,16.77,12.12,10.95,10.95],
    [16.27,6.92,4.28,20.5,15.85,12.45,12.45],
    [0.47,13.08,11.28,5.82,2.47,4.73,4.73],
    [14.98,6.73,2.98,19.22,14.57,11.17,11.17],
    [4.1,15.05,13.98,3.47,3.68,6.35,6.35],
    [13.6,2.55,5.22,17.55,12.9,13.97,13.97],
    [4.75,15.38,13.57,3.88,4.1,5.93,5.93],
    [4.13,13.87,12.07,5.82,6.33,4.42,4.42],
    [12.02,4,0.27,16.82,12.17,9.3,9.3],
    [6.02,16.75,14.95,2.22,5.6,7.32,7.32],
    [12.43,2.73,0.93,17.03,12.38,10.05,10.05],
    [16.32,5.53,3.87,20.92,16.27,12.9,12.9],
    [15.77,4.43,4.4,20.37,15.72,13.43,13.43],
    [14.98,3.5,4.55,19.58,14.93,13.77,13.77],
    [19.5,9.08,11.67,22.48,17.83,20.43,20.43],
    [8.72,15.05,16.17,11.65,7.33,13.12,13.12],
    [2.33,13.28,13.65,2.9,1.92,7.52,7.52],
    [11.92,0.38,3.07,15.87,11.22,11.45,11.45],
    [11.97,13.87,16.33,14.9,10.58,16.03,16.03],
    [4.32,14.95,13.15,4.47,4.68,5.52,5.52],
    [5.35,16.32,16.68,0,4.93,9.45,9.45],
    [12.48,0.17,3.62,16.42,11.77,12,12],
    [12.08,2.38,1.25,16.68,12.03,10.85,10.85],
    [4.6,14.33,12.53,6.43,6.62,4.88,4.88],
    [1.47,12.45,10.65,6.83,3.48,3.65,3.65],
    [9.57,20.53,20.8,4.22,9.15,13.17,13.17],
    [5.87,16.9,15.08,2.07,5.45,7.45,7.45],
    [15.5,6.28,3.75,20.3,15.65,12.78,12.78],
    [5.88,16.52,14.72,2.45,5.83,7.08,7.08],
    [5.68,16.65,16.05,1.88,5.27,8.42,8.42],
    [12.32,0,3.45,16.25,11.62,11.83,11.83],
    [4.15,14.78,12.98,4.9,5.12,5.35,5.35],
    [5.98,15.05,12.52,7.82,8,3.8,3.8],
    [15.8,5.92,4.07,21.3,16.65,13.1,13.1],
    [5.67,16.63,15.27,1.87,5.25,7.63,7.63],
    [14.43,3.38,5.83,18.38,13.73,14.6,14.6],
    [5.5,16.13,14.32,2.85,6.23,6.68,6.68],
])).T

def run_sim(log=False, optimize=True, robots=50, proportional_robots=True, hours=12, demand=.05, relative_demand=np.array([1,1,1,1,1,1,1])):
    ROBOTS = robots
    HOURS_PER_DAY = hours
    ORDERS_PER_PERSON_PER_DAY = demand
    RELATIVE_RESTAURANT_DEMAND = relative_demand

    TOTAL_MINUTES = HOURS_PER_DAY * MINUTES_PER_HOUR
    ORDERS_PER_PERSON_PER_MINUTE = ORDERS_PER_PERSON_PER_DAY / TOTAL_MINUTES
    NORMALIZED_RESTAURANT_DEMAND = RELATIVE_RESTAURANT_DEMAND / sum(RELATIVE_RESTAURANT_DEMAND)
    ORDERS_PER_PERSON_PER_MINUTE_PER_RESTAURANT = NORMALIZED_RESTAURANT_DEMAND * ORDERS_PER_PERSON_PER_MINUTE

    # expected number of orders per minute from dorm d to restaurant r: lambdas[r][d]
    # outer product ⊗
    lambdas = np.outer(ORDERS_PER_PERSON_PER_MINUTE_PER_RESTAURANT, populations)

    # robots at each restaurant: robots[r]
    if proportional_robots:
        r = cp.Variable(RESTAURANTS, integer=True)
        constraints = [r[i] >= 1 for i in range(RESTAURANTS)] # no negative robots
        constraints.append(cp.sum(r) == ROBOTS) # only one return restaurant
        obj_func = sum(cp.abs(NORMALIZED_RESTAURANT_DEMAND * ROBOTS - r))
        problem = cp.Problem(cp.Minimize(obj_func), constraints)
        problem.solve(solver=cp.GUROBI)
        robots = [int(r) for r in r.value]
    else:
        robots = [0 for _ in range(RESTAURANTS)]
        for i in range(ROBOTS):
            robots[i % RESTAURANTS] += 1

    t = 0
    minutes_waited = 0

    # orders for minute t from dorm d to restaurant r: orders[t][r][d]
    orders = np.random.poisson(lambdas, size=(TOTAL_MINUTES, RESTAURANTS, DORMS))

    # minute-orders generated at minute t from dorm d to restaurant r: minute_orders[t][r][d]
    # Hadamard product ∘
    minute_orders = orders * distances

    # queue of minute-orders pending at restaurant r: pending_minute_orders[r]
    pending_minute_orders = [[] for _ in range(RESTAURANTS)]

    # queue of minute-orders-legs being processed by restaurant r: processing_minute_orders[r]
    processing_minute_orders_legs = [[] for _ in range(RESTAURANTS)]

    def waiting_time(minute_order_legs, robots) -> float:
        time = 0
        for i in range(robots): # for each robot
            for j in range(i, len(minute_order_legs), robots): # for each order for robot i
                for k in range(i, j, robots): # for each order robot i has already done
                    time += minute_order_legs[k][0] + minute_order_legs[k][0]
                time += minute_order_legs[j][0] # order the robot is doing
        return time

    def waiting_times(r) -> Tuple[float, float]:
        ''' returns the total time in minutes customers
            would wait if restaurant `r` finished all its
            processing and pending orders with n and n+1 robots
        '''
        processing_legs = sorted(processing_minute_orders_legs[r], key=lambda legs: legs[0]+legs[1])

        all_legs = processing_legs + [[minute, minute] for minute in pending_minute_orders[r]]

        return waiting_time(all_legs, robots[r]), waiting_time(all_legs, robots[r]+1)

    def run_minute():
        # insert pending orders
        for r in range(RESTAURANTS):
            pending_minute_orders[r] += list(minute_orders[t][r][minute_orders[t][r] != 0])

        # update processing orders
        returning_robots = [[] for _ in range(RESTAURANTS)]
        for r in range(RESTAURANTS):
            for i, processing_leg in enumerate(processing_minute_orders_legs[r]):
                if processing_leg[0] > 0:
                    processing_leg[0] -= 1
                else:
                    processing_leg[1] -= 1
                    if processing_leg[1] <= 0:
                        returning_robots[r].append(i)
        for r in range(RESTAURANTS):
            filtered_processing_minute_orders_legs = []
            for i in range(len(processing_minute_orders_legs[r])):
                if i not in returning_robots[r]:
                    filtered_processing_minute_orders_legs.append(processing_minute_orders_legs[r][i])
            processing_minute_orders_legs[r] = filtered_processing_minute_orders_legs

        # decide return restaurants
        if optimize:
            for r in range(RESTAURANTS):
                for rr in returning_robots[r]:
                    robots[r] -= 1

                    diffs = []
                    for j in range(RESTAURANTS):
                        time, reduced_time = waiting_times(j)
                        diff = time - reduced_time
                        if diff < 0:
                            diff = 0
                        diffs.append(diff)

                    # x = cp.Variable(RESTAURANTS+1, boolean=True, name=f'(t={t},r={r},i={rr})')
                    # constraints = [(robots[i] + x[i]) >= 1 for i in range(RESTAURANTS)] # all positive robots
                    # constraints.append(cp.sum(x) == 1) # robots 
                    # obj_func = sum([ x[i] * diffs[i] for i in range(RESTAURANTS) ]) + x[RESTAURANTS]/1e3
                    # problem = cp.Problem(cp.Maximize(obj_func), constraints)
                    # problem.solve(solver=cp.GUROBI)

                    # return_restaurant = np.flatnonzero(x.value)[0]
                    # if return_restaurant == RESTAURANTS:
                    #     return_restaurant = r

                    if diffs == [0 for _ in range(RESTAURANTS)] or robots[r] == 0:
                        return_restaurant = r
                    else:
                        return_restaurant = diffs.index(max(diffs))

                    robots[return_restaurant] += 1

        # transfer pending orders to processing orders
        for r in range(RESTAURANTS):
            available_robots = robots[r] - len(processing_minute_orders_legs[r])
            transfering_orders = pending_minute_orders[r][:available_robots]
            pending_minute_orders[r] = pending_minute_orders[r][available_robots:]
            processing_minute_orders_legs[r] += [ [order, order] for order in transfering_orders ]

    for _ in range(TOTAL_MINUTES):
        run_minute()
        minutes_waited += sum([len(i) for i in pending_minute_orders]) + sum([len(list(filter(lambda legs: legs[0] > 0, i))) for i in processing_minute_orders_legs])
        orders = [ len(pending_minute_orders[i]) + len(processing_minute_orders_legs[i]) for i in range(RESTAURANTS) ]
        if log:
            print(f'{t}/{TOTAL_MINUTES}  :  {robots}  :  {orders}  :  {minutes_waited}')
        t += 1

    total_orders = len(minute_orders.flatten()[minute_orders.flatten() != 0])

    return round(minutes_waited / total_orders, 2)

def average_saved(log=False, robots=50, proportional_robots=True, demand=.05, hours=12, relative_demand=np.array([3,1,1.5,1,1,1,3])):
    averages = []
    for _ in range(10):
        averages.append(run_sim(optimize=False, log=log, robots=robots, proportional_robots=proportional_robots, hours=hours, demand=demand, relative_demand=relative_demand))
    unoptomized = sum(averages)/len(averages)

    averages = []
    for _ in range(10):
        averages.append(run_sim(optimize=True, log=log, robots=robots, proportional_robots=proportional_robots, hours=hours, demand=demand, relative_demand=relative_demand))
    optimized = sum(averages)/len(averages)

    return round(unoptomized), round(optimized)


### RUN SIMULATIONS HERE
print('robots,hours,demand,relative_demand,unoptimized,optimized')

relative_demand = np.array([3,1,1.5,1,1,1,3])
demand = .05

normalized_demand = relative_demand / sum(relative_demand)
for robots in range(30,100,5):
    for hours in range(8,13):
        minutes = average_saved(robots=robots, hours=hours, demand=demand, relative_demand=relative_demand)
        
        print(f'{robots},{hours},{demand*100}%,{[round(i,2) for i in normalized_demand]},{minutes[0]},{minutes[1]}')