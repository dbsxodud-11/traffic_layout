import os
import argparse
import subprocess
import xml.etree.ElementTree as ET

import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

import traci
from sumolib import checkBinary
from sumolib.miscutils import getFreeSocketPort

import multiprocessing as mp

def get_default_network(grid_number, grid_length, network_file):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    remove_edges = []
    for i in range(grid_number):
        for j in range(grid_number):
            if i == 0 and j < grid_number - 1:
                remove_edges.append(f"{alphabet[i]}{j}{alphabet[i]}{j+1}")
                remove_edges.append(f"{alphabet[i]}{j+1}{alphabet[i]}{j}")
            if j == 0 and i < grid_number - 1:
                remove_edges.append(f"{alphabet[i]}{j}{alphabet[i+1]}{j}")
                remove_edges.append(f"{alphabet[i+1]}{j}{alphabet[i]}{j}")
            if i == grid_number - 1 and j < grid_number - 1:
                remove_edges.append(f"{alphabet[i]}{j}{alphabet[i]}{j+1}")
                remove_edges.append(f"{alphabet[i]}{j+1}{alphabet[i]}{j}")
            if j == grid_number - 1 and i < grid_number - 1:
                remove_edges.append(f"{alphabet[i]}{j}{alphabet[i+1]}{j}")
                remove_edges.append(f"{alphabet[i+1]}{j}{alphabet[i]}{j}")
    remove_edges = ', '.join(remove_edges)
    
    if os.path.exists(network_file):
        print(f"File {network_file} already exists. Use existing file.")
    else:
        # Using netgenerate to create network file
        subprocess.call(f'netgenerate --grid --grid.number {grid_number} --grid.length {grid_length} --output-file sumo/{network_file} --remove-edges.explicit "{remove_edges}"', shell=True)

    start_edges_up, start_edges_down, start_edges_left, start_edges_right = [], [], [], []
    end_edges_up, end_edges_down, end_edges_left, end_edges_right = [], [], [], []
    for i in range(grid_number):
        if i == 0:
            for j in range(grid_number):
                if j != 0 and j != grid_number - 1:
                    start_edges_left.append(f"{alphabet[i]}{j}{alphabet[i+1]}{j}")
                    end_edges_left.append(f"{alphabet[i+1]}{j}{alphabet[i]}{j}")
        elif i == grid_number - 1:
            for j in range(grid_number):
                if j != 0 and j != grid_number - 1:
                    start_edges_right.append(f"{alphabet[i]}{j}{alphabet[i-1]}{j}")
                    end_edges_right.append(f"{alphabet[i-1]}{j}{alphabet[i]}{j}")
        else:
            start_edges_up.append(f"{alphabet[i]}0{alphabet[i]}1")
            end_edges_up.append(f"{alphabet[i]}1{alphabet[i]}0")
            start_edges_down.append(f"{alphabet[i]}{grid_number-1}{alphabet[i]}{grid_number-2}")
            end_edges_down.append(f"{alphabet[i]}{grid_number-2}{alphabet[i]}{grid_number-1}")

    possible_edges = []
    possible_edges_pair = []
    for i in range(1, grid_number-2):
        for j in range(2, grid_number-2):
            possible_edges.append(f"{alphabet[i]}{j}{alphabet[i+1]}{j}")
            possible_edges_pair.append(f"{alphabet[i+1]}{j}{alphabet[i]}{j}")
            
    for i in range(2, grid_number-2):
        for j in range(1, grid_number-2):
            possible_edges.append(f"{alphabet[i]}{j}{alphabet[i]}{j+1}")
            possible_edges_pair.append(f"{alphabet[i]}{j+1}{alphabet[i]}{j}")
            
    start_edges = [start_edges_up, start_edges_down, start_edges_left, start_edges_right]
    end_edges = [end_edges_up, end_edges_down, end_edges_left, end_edges_right]
        
    return possible_edges, possible_edges_pair, start_edges, end_edges, remove_edges

def get_default_routes(route_number, route_type, min_num_vehicles, max_num_vehicles, simulation_time, 
                       start_edges, end_edges, route_file):
    routes = []
    for _ in range(route_number):
        if route_type == "random":
            i, j = np.random.choice(4, 2, replace=False)
            start_edge = np.random.choice(start_edges[i])
            end_edge = np.random.choice(end_edges[j])
        elif route_type == "straight":
            i = np.random.choice(4)
            if i < 2:
                j = (i + 1) % 2
            else:
                j = (i + 1) % 2 + 2
            idx = np.random.choice(len(start_edges[i]))
            start_edge = start_edges[i][idx]
            end_edge = end_edges[j][idx]
        num_vehicles = np.random.randint(min_num_vehicles, max_num_vehicles)
        routes.append((start_edge, end_edge, num_vehicles))
    
    if os.path.exists(route_file):
        print(f"File {route_file} already exists. Use existing file.")
    else:
        with open(f"{route_file}", 'w') as f:
            f.write(f'<routes>\n')
            for i, (start_edge, end_edge, num_vehicles) in enumerate(routes):
                f.write(f'    <flow id="{i}" begin="0" end="{simulation_time}" from="{start_edge}" to="{end_edge}" number="{num_vehicles}" />\n')
            f.write('</routes>\n')

def construct_network_file(grid_number, grid_length, network_file, remove_edges, orig_remove_edges):
    remove_edges = ', '.join(remove_edges)
    remove_edges += ", " + orig_remove_edges
    subprocess.call(f'netgenerate --grid --grid.number {grid_number} --grid.length {grid_length} --output-file {network_file} --remove-edges.explicit "{remove_edges}" -j traffic_light', shell=True)

def simulation_check(idx, args, possible_edges, possible_edges_pair, orig_remove_edges, folder_name, seed, xs):
    np.random.seed(seed+idx)
    remove_ratio = np.random.uniform(0.1, 0.7)
    # xs = np.random.choice(2, size=len(possible_edges), p=[1-remove_ratio, remove_ratio])
    remove_edges = [possible_edges[i] for i in range(len(possible_edges)) if xs[i] == 1]
    remove_edges += [possible_edges_pair[i] for i in range(len(possible_edges)) if xs[i] == 1]
    
    # Preprocessing: Make sure the graph is connected
    G = nx.Graph()
    for x, edge in zip(xs, possible_edges):
        if x == 0:
            nodes = ["" for _ in range(2)]
            i = 0
            for j in range(len(edge)):
                if edge[j] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    if j > 0 and edge[j-1] in "0123456789":
                        i += 1
                    nodes[i] += edge[j]
                else:
                    nodes[i] += edge[j]
            G.add_edge(nodes[0], nodes[1])

    for _ in range(10):
        add_edges = []
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if len(neighbors) <= 1:
                possible_candidates, possible_candidates_reverse = [], []
                alphabet, number = node[0], int(node[1:])
                if number != 0:
                    possible_candidates.append(f"{alphabet}{number-1}{alphabet}{number}")
                    possible_candidates_reverse.append(f"{alphabet}{number}{alphabet}{number-1}")
                if number != args.grid_number - 1:
                    possible_candidates.append(f"{alphabet}{number+1}{alphabet}{number}")
                    possible_candidates_reverse.append(f"{alphabet}{number}{alphabet}{number+1}")
                if alphabet != "A":
                    possible_candidates.append(f"{'ABCDEFGHIJKLMNOPQRSTUVWXYZ'['ABCDEFGHIJKLMNOPQRSTUVWXYZ'.index(alphabet)-1]}{number}{alphabet}{number}")
                    possible_candidates_reverse.append(f"{alphabet}{number}{'ABCDEFGHIJKLMNOPQRSTUVWXYZ'['ABCDEFGHIJKLMNOPQRSTUVWXYZ'.index(alphabet)-1]}{number}")
                if alphabet != "Z":
                    possible_candidates.append(f"{'ABCDEFGHIJKLMNOPQRSTUVWXYZ'['ABCDEFGHIJKLMNOPQRSTUVWXYZ'.index(alphabet)+1]}{number}{alphabet}{number}")
                    possible_candidates_reverse.append(f"{alphabet}{number}{'ABCDEFGHIJKLMNOPQRSTUVWXYZ'['ABCDEFGHIJKLMNOPQRSTUVWXYZ'.index(alphabet)+1]}{number}")
                
                candidate_idx = np.random.choice(len(possible_candidates), 1)[0]
                possible_edge = possible_candidates[candidate_idx]
                possible_edge_reverse = possible_candidates_reverse[candidate_idx]
                add_edges.append(possible_edge)
                if possible_edge in remove_edges:
                    remove_edges.remove(possible_edge)
                    remove_edges.remove(possible_edge_reverse)
                    
        for edge in add_edges:
            nodes = ["" for _ in range(2)]
            i = 0
            for j in range(len(edge)):
                if edge[j] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    if j > 0 and edge[j-1] in "0123456789":
                        i += 1
                    nodes[i] += edge[j]
                else:
                    nodes[i] += edge[j]
            G.add_edge(nodes[0], nodes[1])
    
    # Construct network file
    new_network_file = f"{folder_name}/sample_{idx}_{seed}.net.xml"
    construct_network_file(args.grid_number, args.grid_length, new_network_file, remove_edges, orig_remove_edges)
    
    # Run simulation
    sumocfg_file = f"{folder_name}/sample_{idx}_{seed}.sumocfg"
    tripinfo_file = f"{folder_name}/sample_{idx}_{seed}.tripinfo.xml"
    with open(sumocfg_file, "w") as f:
        f.write(f'<configuration>\n')
        f.write(f'    <input>\n')
        f.write(f'        <net-file value="sample_{idx}_{seed}.net.xml"/>\n')
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
    sumo_cmd = [sumoBinary, "-c", sumocfg_file, "--no-warnings", "--no-step-log", "--seed", str(seed), "--tripinfo-output", tripinfo_file]
    port = getFreeSocketPort()
    
    traci.start(sumo_cmd, port=port)
    # while traci.simulation.getMinExpectedNumber() > 0:
    for _ in range(args.simulation_time):
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
    y = np.array([np.mean(waiting_time_list), np.mean(traveling_time_list), last_vehicle_arrival_time])
    print(f"Average waiting time: {np.mean(waiting_time_list):.2f}", end="\t")
    print(f"Average traveling time: {np.mean(traveling_time_list):.2f}", end="\t")
    print(f"Last vehicle arrival time: {last_vehicle_arrival_time:.2f}")
    return xs, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Network parameters
    parser.add_argument("--grid_number", type=int, default=12)
    parser.add_argument("--grid_length", type=float, default=50.0)
    
    # Route parameters
    parser.add_argument("--route_number", type=int, default=20)
    parser.add_argument("--route_type", type=str, default="random")
    parser.add_argument("--min_num_vehicles", type=int, default=100)
    parser.add_argument("--max_num_vehicles", type=int, default=200)
    
    # Simulation parameters
    parser.add_argument("--simulation_time", type=int, default=1800)
    
    # seed
    parser.add_argument("--seed", type=int, default=42)
    
    # visualize
    parser.add_argument("--visualize", action="store_true")
    
    # data collection
    # parser.add_argument("--num_data_points", type=int, default=10)
    
    # Sampling configuration
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--cond_value", type=int, default=50)
    
    parser.add_argument("--iter", type=int, default=1)
    
    args = parser.parse_args()
    
    # Set seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    
    folder_name = f"sumo/N{args.grid_number}_L{args.grid_length}/R{args.route_number}_T{args.route_type}_MIN{args.min_num_vehicles}_MAX{args.max_num_vehicles}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    
    # Get Default Network
    network_file = f"{folder_name}/default.net.xml"
    possible_edges, possible_edges_pair, start_edges, end_edges, orig_remove_edges = get_default_network(args.grid_number, args.grid_length, network_file)
    
    # Get Default Routes
    route_file = f"{folder_name}/default.rou.xml"
    get_default_routes(args.route_number, args.route_type, args.min_num_vehicles, args.max_num_vehicles, args.simulation_time, 
                       start_edges, end_edges, route_file)
    
    # sample_file = f"data/samples_table_t0.25_100.csv"
    # sample_file = f"data/samples_table_t0.1_w100_n1000.csv"
    save_path = f"results/N{args.grid_number}_L{args.grid_length}/R{args.route_number}_T{args.route_type}_MIN{args.min_num_vehicles}_MAX{args.max_num_vehicles}/generated/"
    save_path += f"c={args.cond_value}|n={args.num_samples}|t={args.temperature}|seed={seed}"
    sample_df = pd.read_csv(f"{save_path}/samples_table_iter{args.iter}_c{args.cond_value}_n{args.num_samples}_t{args.temperature}_seed{seed}.csv")
    sample_xs = sample_df["0"].values
    # string to numpy
    sample_xs = np.stack([np.array(list(map(int, list(x)))) for x in sample_xs])
    num_samples = len(sample_xs)
    
    inputs = [(i, args, possible_edges, possible_edges_pair, orig_remove_edges, folder_name, seed, sample_xs[i]) for i in range(num_samples)]
    with mp.Pool(4) as pool:
        data = pool.starmap(simulation_check, tqdm(inputs, total=num_samples))
        
    # Save data
    X = np.stack([d[0] for d in data])
    y = np.stack([d[1] for d in data])
    # np.savez_compressed("data/random.npz", X=X, y=y)
    npz_save_path = os.path.join(save_path, f"samples_table_iter{args.iter}_c{args.cond_value}_n{args.num_samples}_t{args.temperature}_seed{seed}.npz")
    np.savez_compressed(npz_save_path, X=X, y=y)
    
    if args.iter == 1:
        preprocessed_data_path = f"results/N{args.grid_number}_L{args.grid_length}/R{args.route_number}_T{args.route_type}_MIN{args.min_num_vehicles}_MAX{args.max_num_vehicles}/data/preprocessed/sumo_preprocessed_dataset_iter{args.iter}.csv"
        preprocessed_data = pd.read_csv(preprocessed_data_path)
    else:
        preprocessed_data_path = f"results/N{args.grid_number}_L{args.grid_length}/R{args.route_number}_T{args.route_type}_MIN{args.min_num_vehicles}_MAX{args.max_num_vehicles}/data/preprocessed/sumo_preprocessed_dataset_iter{args.iter}_c{args.cond_value}_n{args.num_samples}_t{args.temperature}_seed{seed}.csv"
        preprocessed_data = pd.read_csv(preprocessed_data_path)
    
    columns = preprocessed_data.columns
    new_data = pd.DataFrame(columns=columns)
    
    for i in range(X.shape[0]):
        new_data.loc[i,'layout'] = ''.join(map(str,X[i]))
        new_data.loc[i,'waiting_time'] = y[i][0]

    combined_df = pd.concat([preprocessed_data, new_data], ignore_index=True)
    combined_data_path = f"results/N{args.grid_number}_L{args.grid_length}/R{args.route_number}_T{args.route_type}_MIN{args.min_num_vehicles}_MAX{args.max_num_vehicles}/data/preprocessed/sumo_preprocessed_dataset_iter{args.iter+1}_c{args.cond_value}_n{args.num_samples}_t{args.temperature}_seed{seed}.csv"
    combined_df.to_csv(combined_data_path, index=False)
    print(f"Combined data saved to {combined_data_path}")