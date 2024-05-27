import os
import sys
import traci
import numpy as np
import matplotlib.pyplot as plt
from sumolib import checkBinary

# Проверка наличия переменной окружения SUMO_HOME и установка пути для инструментов
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# Параметры Q-Learning
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.0001
EPISODES = 20  # Уменьшено для быстрой работы, увеличьте по необходимости

# Определение пространства состояний и действий
NUM_ACTIONS = 2  # Два действия: 0 (остаться в текущей фазе), 1 (переключиться на следующую фазу)
MAX_CARS = 10  # Ограничение числа машин до 10 для представления состояния

# Инициализация Q-таблицы с размером пространства состояний (11*11*11*11, NUM_ACTIONS)
state_space_size = (MAX_CARS + 1) ** 4
q_table = np.random.uniform(low=-2, high=0, size=(state_space_size, NUM_ACTIONS))

# Список для хранения результатов для построения графиков
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
    if np.random.random() < EPSILON:
        return np.random.randint(0, NUM_ACTIONS)
    return np.argmax(q_table[state])


def update_q_table(state, action, reward, new_state):
    max_future_q = np.max(q_table[new_state])
    current_q = q_table[state, action]
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    q_table[state, action] = new_q


def switch_to_next_phase(junctionID):
    current_phase = traci.trafficlight.getPhase(junctionID)
    program = traci.trafficlight.getAllProgramLogics(junctionID)[0]
    total_phases = len(program.phases)
    next_phase = (current_phase + 1) % total_phases
    traci.trafficlight.setPhase(junctionID, next_phase)
    return next_phase


def run(train=True):
    global EPSILON
    for episode in range(EPISODES):
        if train:
            traci.start(
                [checkBinary("sumo"), "-c", "SUMO-files/Crossroad.sumo.cfg", "--tripinfo-output", "tripinfo.xml"])
        else:
            traci.start(
                [checkBinary("sumo-gui"), "-c", "SUMO-files/Crossroad.sumo.cfg", "--tripinfo-output", "tripinfo.xml"])

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
                    new_state = get_state(junctionID)

                    waiting_time = sum(
                        traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(junctionID))
                    reward = -waiting_time
                    update_q_table(current_state, action, reward, new_state)

                    total_waiting_time += waiting_time
                    total_vehicles += sum(traci.lane.getLastStepVehicleNumber(lane) for lane in
                                          traci.trafficlight.getControlledLanes(junctionID))

                    if EPSILON > MIN_EPSILON:
                        EPSILON *= EPSILON_DECAY

            step += 1

        average_waiting_time = total_waiting_time / (total_vehicles + 1e-6)  # Деление на 0
        average_waiting_times.append(average_waiting_time)
        traci.close()
        sys.stdout.flush()

    # Сохранение Q-таблицы в конце
    np.save("q_table.npy", q_table)

    # Построение графика и сохранение среднего времени ожидания за эпизод
    plt.plot(range(EPISODES), average_waiting_times)
    plt.xlabel("Episode")
    plt.ylabel("Average Waiting Time per Vehicle (s)")
    plt.title("Average Waiting Time per Vehicle over Episodes")
    plt.savefig("Q-Table Performance.jpg")
    plt.show()


if __name__ == "__main__":
    train_mode = True
    sumoBinary = checkBinary('sumo-gui' if not train_mode else 'sumo')
    run(train=train_mode)
