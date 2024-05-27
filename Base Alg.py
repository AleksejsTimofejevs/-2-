import os
import sys
import traci
import matplotlib.pyplot as plt
from sumolib import checkBinary

# Check for SUMO_HOME and set the path for tools
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# Simulation parameters
EPISODES = 20  # Number of episodes
PHASE_DURATION = 15  # Duration for each phase in seconds

# Lists to store results for plotting
average_waiting_times = []

def switch_to_next_phase(junctionID):
    current_phase = traci.trafficlight.getPhase(junctionID)
    program = traci.trafficlight.getAllProgramLogics(junctionID)[0]
    total_phases = len(program.phases)
    next_phase = (current_phase + 1) % total_phases
    traci.trafficlight.setPhase(junctionID, next_phase)
    return next_phase

def run_simulation():
    for episode in range(EPISODES):
        traci.start([checkBinary("sumo"), "-c", "SUMO-files/Crossroad.sumo.cfg", "--tripinfo-output", "tripinfo.xml"])

        step = 0
        total_waiting_time = 0
        total_vehicles = 0

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            if step % PHASE_DURATION == 0:
                for junctionID in traci.trafficlight.getIDList():
                    switch_to_next_phase(junctionID)

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

    # Plot and save average waiting time per episode
    plt.plot(range(EPISODES), average_waiting_times)
    plt.xlabel("Episode")
    plt.ylabel("Average Waiting Time per Vehicle (s)")
    plt.title("Average Waiting Time per Vehicle over Episodes")
    plt.savefig("Performance without RL.jpg")
    plt.show()

if __name__ == "__main__":
    run_simulation()
