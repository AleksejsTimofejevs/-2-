import os
import sys
import traci
import numpy as np
from sumolib import checkBinary


# Проверка на наличие переменной окружения SUMO_HOME и добавление пути к инструментам SUMO
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# Параметры
EPISODES = 20  # Number of episodes
MAX_CARS = 10  # Cap the number of cars at 10 for state representation

# Загрузка Q-Table
q_table = np.load("q_table.npy")

# Lists to store results for plotting
average_waiting_times = []

def get_state(junctionID):
    lanes = traci.trafficlight.getControlledLanes(junctionID)
    state = []
    for lane in lanes[:4]:
        num_cars = min(traci.lane.getLastStepHaltingNumber(lane), MAX_CARS)
        state.append(num_cars)
    state_index = np.ravel_multi_index(state, (MAX_CARS + 1,) * 4)
    return state_index

def choose_action(state):
    return np.argmax(q_table[state])

def switch_to_next_phase(junctionID):
    current_phase = traci.trafficlight.getPhase(junctionID)
    program = traci.trafficlight.getAllProgramLogics(junctionID)[0]
    total_phases = len(program.phases)
    next_phase = (current_phase + 1) % total_phases
    traci.trafficlight.setPhase(junctionID, next_phase)
    return next_phase

def run_simulation():
    for episode in range(EPISODES):
        traci.start([checkBinary("sumo-gui"), "-c", "SUMO-files/Crossroad.sumo.cfg", "--tripinfo-output", "tripinfo.xml"])

        step = 0
        total_waiting_time = 0
        total_vehicles = 0

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            if step % 5 == 0:
                for junctionID in traci.trafficlight.getIDList():
                    current_state = get_state(junctionID)
                    action = choose_action(current_state)

                    if action == 1:
                        next_phase = switch_to_next_phase(junctionID)
                    else:
                        next_phase = traci.trafficlight.getPhase(junctionID)

                    traci.simulationStep()

                    waiting_time = sum(
                        traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(junctionID))
                    total_waiting_time += waiting_time
                    total_vehicles += sum(traci.lane.getLastStepVehicleNumber(lane) for lane in
                                          traci.trafficlight.getControlledLanes(junctionID))

            step += 1

        average_waiting_time = total_waiting_time / (total_vehicles + 1e-6)  # To avoid division by zero
        average_waiting_times.append(average_waiting_time)
        traci.close()
        sys.stdout.flush()


if __name__ == "__main__":
    run_simulation()
