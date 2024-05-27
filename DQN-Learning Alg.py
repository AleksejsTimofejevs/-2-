import os
import sys
import traci
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from sumolib import checkBinary
import matplotlib.pyplot as plt

# Проверка на наличие переменной окружения SUMO_HOME и добавление пути к инструментам SUMO
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# Параметры симуляции
EPISODES = 500  # Количество эпизодов
MAX_MEMORY_SIZE = 10000  # Максимальный размер памяти для буфера воспроизведения
BATCH_SIZE = 32  # Размер батча для обучения
GAMMA = 0.95  # Коэффициент дисконтирования
ALPHA = 0.001  # Скорость обучения для DQN
EPSILON = 1.0  # Начальное значение ε для ε-жадной политики
EPSILON_DECAY = 0.995  # Коэффициент уменьшения ε
MIN_EPSILON = 0.0001  # Минимальное значение ε
MAX_LANES = 20  # Максимальное количество полос
MAX_CARS = 100  # Ограничение на количество машин для представления состояния
TARGET_UPDATE_FREQ = 5  # Частота обновления целевой модели
SAVE_PATH = "dqn_model.pth"

# Списки для хранения результатов
average_waiting_times = []

# Конфигурация устройства (использование GPU, если доступен)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Определение модели DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Инициализация моделей
model = DQN(MAX_LANES, 2).to(device)
target_model = DQN(MAX_LANES, 2).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

# Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=ALPHA)
criterion = nn.MSELoss()

# Память для воспроизведения опыта
memory = deque(maxlen=MAX_MEMORY_SIZE)


# Функция для выбора действия
def choose_action(state):
    if np.random.rand() < EPSILON:
        return np.random.randint(2)
    state = torch.FloatTensor(np.array(state)).to(device)
    with torch.no_grad():
        q_values = model(state)
    return np.argmax(q_values.cpu().numpy())


# Функция для обновления целевой модели
def update_target_model():
    target_model.load_state_dict(model.state_dict())


# Функция для получения текущего состояния
def get_state(junctionID):
    lanes = traci.trafficlight.getControlledLanes(junctionID)
    state = [min(traci.lane.getLastStepHaltingNumber(lane), MAX_CARS) for lane in lanes[:MAX_LANES]]
    state += [0] * (MAX_LANES - len(state))  # Заполнение состояния нулями до MAX_LANES
    return np.array(state[:MAX_LANES]) / MAX_CARS  # Нормализация состояния


# Обучение модели
def replay():
    if len(memory) < BATCH_SIZE:
        return None  # Возвращаем None, если недостаточно образцов для формирования батча
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(np.array(actions)).to(device)
    rewards = torch.FloatTensor(np.array(rewards)).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(np.array(dones)).to(device)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_model(next_states).max(1)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = criterion(q_values, expected_q_values.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# Функция для переключения на следующую фазу
def switch_to_next_phase(junctionID):
    current_phase = traci.trafficlight.getPhase(junctionID)
    program = traci.trafficlight.getAllProgramLogics(junctionID)[0]
    total_phases = len(program.phases)
    next_phase = (current_phase + 1) % total_phases
    traci.trafficlight.setPhase(junctionID, next_phase)
    return next_phase


# Функция для сохранения модели
def save_model(path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'target_model_state_dict': target_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': EPSILON
    }, path)
    print(f"Model saved to {path}")


# Функция для загрузки модели
def load_model(path):
    global EPSILON
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        target_model.load_state_dict(checkpoint['target_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        EPSILON = checkpoint['epsilon']
        print(f"Model loaded from {path}")
    else:
        print(f"No model found at {path}")


# Запуск симуляции
def run_simulation(train=True):
    global EPSILON
    for episode in range(EPISODES):
        if train:
            traci.start([checkBinary("sumo"), "-c", "SUMO-files/Crossroad.sumo.cfg", "--tripinfo-output", "tripinfo.xml"])
        else:
            traci.start([checkBinary("sumo-gui"), "-c", "SUMO-files/Crossroad.sumo.cfg", "--tripinfo-output", "tripinfo.xml"])
        step = 0
        total_waiting_time = 0
        total_vehicles = 0
        episode_loss = []

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            if step % 5 == 0:
                for junctionID in traci.trafficlight.getIDList():
                    current_state = get_state(junctionID)
                    action = choose_action(current_state)
                    switch_to_next_phase(junctionID) if action == 1 else traci.trafficlight.getPhase(junctionID)
                    traci.simulationStep()
                    next_state = get_state(junctionID)
                    waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(junctionID))
                    reward = -waiting_time / 100.0  # Normalizing the reward
                    done = traci.simulation.getMinExpectedNumber() == 0
                    memory.append((current_state, action, reward, next_state, done))

                    total_waiting_time += waiting_time
                    total_vehicles += sum(traci.lane.getLastStepVehicleNumber(lane) for lane in traci.trafficlight.getControlledLanes(junctionID))

                    loss = replay()
                    if loss is not None:
                        episode_loss.append(loss)

                if step % TARGET_UPDATE_FREQ == 0:
                    update_target_model()

            step += 1

        average_waiting_time = total_waiting_time / (total_vehicles + 1e-6)  # Const, чтобы не делить на 0
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        print(f"Episode {episode + 1}/{EPISODES}, Average Waiting Time: {average_waiting_time:.2f}, Loss: {avg_loss:.4f}")
        average_waiting_times.append(average_waiting_time)
        traci.close()
        sys.stdout.flush()

        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY

        if (episode + 1) % 50 == 0:
            save_model(SAVE_PATH)

    # График среднего времени ожидания по симуляциям (эпизодам)
    plt.plot(range(EPISODES), average_waiting_times)
    plt.xlabel("Episode")
    plt.ylabel("Average Waiting Time per Vehicle (s)")
    plt.title("Average Waiting Time per Vehicle over Episodes")
    plt.savefig("DQN-Table Performance.jpg")
    plt.show()


if __name__ == "__main__":
    load_model(SAVE_PATH)
    run_simulation(train=True)
