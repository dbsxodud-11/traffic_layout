export SUMO_HOME=/usr/share/sumo
export PYTHONPATH=~/traffic-layout/discrete_guidance

device=$1

# data collection configuration
grid_number=$2
grid_length=50.0
route_number=20
route_type=straight
min_num_vehicles=200
max_num_vehicles=400
num_data_points=5000
simulation_time=3600

# training and sampling configuration
cond_value=$3
temperature=$4
num_generated_samples=$5
iteration=$6
cpu_start=$7
cpu_end=$8
seed=$9

taskset -c $7-$8 python data_collection.py --grid_number $grid_number --grid_length $grid_length \
    --route_number $route_number --route_type $route_type --min_num_vehicles $min_num_vehicles --max_num_vehicles $max_num_vehicles \
    --num_data_points $num_data_points --simulation_time $simulation_time
wait

for iter in $(seq 1 $iteration); do
    echo "Starting iteration $iter of $iteration"

    CUDA_VISIBLE_DEVICES=$device python discrete_guidance/applications/molecules/scripts/train.py \
        -c discrete_guidance/applications/molecules/config_files/training_defaults_sumo.yaml -m "all" \
        -o "base_dir=results/N${grid_number}_L${grid_length}/R${route_number}_T${route_type}_MIN${min_num_vehicles}_MAX${max_num_vehicles}|iteration=$iter|cond_value=$cond_value|temperature=$temperature|num_samples=$num_generated_samples|seed=$seed"
    
    CUDA_VISIBLE_DEVICES=$device python discrete_guidance/applications/molecules/scripts/generate.py \
        -c discrete_guidance/applications/molecules/config_files/generation_defaults.yaml \
        -n $num_generated_samples -p "waiting_time=$cond_value" \
        -o "trained_run_folder_dir=results/N${grid_number}_L${grid_length}/R${route_number}_T${route_type}_MIN${min_num_vehicles}_MAX${max_num_vehicles}/trained/c${cond_value}_n${num_generated_samples}_t${temperature}_iter${iter}_seed${seed}|base_dir=results/N${grid_number}_L${grid_length}/R${route_number}_T${route_type}_MIN${min_num_vehicles}_MAX${max_num_vehicles}|iteration=$iter|cond_value=$cond_value|temperature=$temperature|num_samples=$num_generated_samples|seed=$seed"
    taskset -c $7-$8 python sample_check.py --grid_number $grid_number --grid_length $grid_length \
        --route_number $route_number --route_type $route_type --min_num_vehicles $min_num_vehicles --max_num_vehicles $max_num_vehicles \
        --num_samples $num_generated_samples --temperature $temperature --cond_value $cond_value --simulation_time $simulation_time --iter $iter --seed $seed
done
