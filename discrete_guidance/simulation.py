import os
import argparse
import subprocess
import xml.etree.ElementTree as ET

import random
import numpy as np

import traci
from sumolib import checkBinary
from sumolib.miscutils import getFreeSocketPort

from data_collection import get_default_network, get_default_routes, construct_network_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Network parameters
    parser.add_argument("--grid_number", type=int, default=12)
    parser.add_argument("--grid_length", type=float, default=50.0)
    
    # Route parameters
    parser.add_argument("--route_number", type=int, default=20)
    parser.add_argument("--min_num_vehicles", type=int, default=100)
    parser.add_argument("--max_num_vehicles", type=int, default=200)
    
    # Simulation parameters
    parser.add_argument("--simulation_time", type=int, default=1800)
    
    # seed
    parser.add_argument("--seed", type=int, default=42)
    
    # visualize
    parser.add_argument("--visualize", action="store_true")
    
    args = parser.parse_args()
    
    # Set seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    
    folder_name = f"sumo/N{args.grid_number}_L{args.grid_length}/R{args.route_number}_MIN{args.min_num_vehicles}_MAX{args.max_num_vehicles}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    
    # Get Default Network
    network_file = f"{folder_name}/default.net.xml"
    possible_edges, possible_edges_pair, start_edges, end_edges, orig_remove_edges = get_default_network(args.grid_number, args.grid_length, network_file)
    
    # Get Default Routes
    route_file = f"{folder_name}/default.rou.xml"
    
    # Load best layout from the data
    data = np.load("data/random.npz")
    idx = np.argmin(data["y"][:, 0])
    x = data["X"][idx]
    
    remove_edges = [possible_edges[i] for i in range(len(possible_edges)) if x[i] == 1]
    remove_edges += [possible_edges_pair[i] for i in range(len(possible_edges)) if x[i] == 1]
    
    # Construct network file
    new_network_file = f"{folder_name}/test.net.xml"
    construct_network_file(args.grid_number, args.grid_length, new_network_file, remove_edges, orig_remove_edges)

    # Run simulation
    sumocfg_file = f"{folder_name}/test.sumocfg"
    tripinfo_file = f"{folder_name}/test.tripinfo.xml"
    with open(sumocfg_file, "w") as f:
        f.write(f'<configuration>\n')
        f.write(f'    <input>\n')
        f.write(f'        <net-file value="test.net.xml"/>\n')
        f.write(f'        <route-files value="default.rou.xml"/>\n')
        f.write(f'    </input>\n')
        f.write(f'    <time>\n')
        f.write(f'        <begin value="0"/>\n')
        f.write(f'        <end value="{args.simulation_time}"/>\n')
        f.write(f'    </time>\n')
        f.write(f'</configuration>\n')
        
    if args.visualize:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    sumo_cmd = [sumoBinary, "-c", sumocfg_file, "--no-warnings", "--no-step-log", "--tripinfo-output", tripinfo_file]
    port = getFreeSocketPort()
    
    traci.start(sumo_cmd, port=port)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    traci.close()
    
    # Post-process
    waiting_time_list = []
    traveling_time_list = []
    last_vehicle_arrival_time = 0
    tree = ET.parse(tripinfo_file)
    root = tree.getroot()
    for child in root:
        waiting_time_list.append(float(child.attrib['waitingTime']))
        traveling_time_list.append(float(child.attrib['duration']))
        last_vehicle_arrival_time = max(last_vehicle_arrival_time, float(child.attrib['arrival']))
    y = np.mean(waiting_time_list)
    print(f"Average waiting time: {np.mean(waiting_time_list):.2f}", end="\t")
    print(f"Average traveling time: {np.mean(traveling_time_list):.2f}", end="\t")
    print(f"Last vehicle arrival time: {last_vehicle_arrival_time:.2f}")
        