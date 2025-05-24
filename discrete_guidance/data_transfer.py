import os
import numpy as np
import csv
import argparse


def collect_and_save_csv(data_file, output_folder):
    csv_data = []
    data = np.load(data_file)
    
    data_x = data["X"]  # (10, 144)
    data_y = data["y"]  # (10, 3)
    
    for x, y in zip(data_x, data_y):
        layout = "".join(map(str, x))
        waiting_time = y[0]  # y의 첫 번째 값
        csv_data.append([layout, waiting_time])
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_csv = os.path.join(output_folder, f"data_{len(csv_data)}.csv")
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["layout", "waiting_time"])
        writer.writerows(csv_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and save CSV data.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data folder.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output TSV file.")
    args = parser.parse_args()
    collect_and_save_csv(args.data_file, args.output_folder)

# 나중에 argparse로 받을 수 있도록 수정
# data_folder = "/home/son9ih/gs-project/data" 
# output_tsv = "/home/son9ih/gs-project/discrete_guidance/applications/molecules/data/preprocessed/sumo_preprocessed_dataset.tsv"
# collect_and_save_tsv(data_folder, output_tsv)
