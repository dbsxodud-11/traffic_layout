
### Installation
```
conda create -n discrete_guidance python=3.9 -y
conda activate discrete_guidance

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.2 matplotlib==3.8.2 pandas==2.1.3 scipy==1.11.4 tqdm jupyter seaborn
pip install tensorboard ml-collections scikit-learn torchmetrics
pip install rdkit chembl_structure_pipeline

echo -e "from setuptools import setup, find_packages\n\nsetup(name='${ENV_NAME}', version='1.0', packages=find_packages())" > setup.py
pip install -e .

pip install sumolib traci
```

### Step1. Data Collection
```
export SUMO_HOME=/usr/share/sumo
python data_collection.py --grid_number $grid_number --grid_length $grid_length \
 --route_number $route_number --route_type $route_type --min_num_vehicles $min_num_vehicles --max_num_vehicles $max_num_vehicles \
 --num_data_points $num_data_points
```

### Step2. Training Diffusion Models and Predictor
```
export PYTHONPATH=~/gs-project/discrete_guidance
CUDA_VISIBLE_DEVICES=$device python discrete_guidance/applications/molecules/scripts/train.py \
 -c discrete_guidance/applications/molecules/config_files/training_defaults_sumo.yaml -m "all" \
 -o "base_dir=results/N${grid_number}_L${grid_length}/R${route_number}_T${route_type}_MIN${min_num_vehicles}_MAX${max_num_vehicles}"
```

### Step3. Generate Layouts
```
CUDA_VISIBLE_DEVICES=$device python discrete_guidance/applications/molecules/scripts/generate.py \
 -c discrete_guidance/applications/molecules/config_files/generation_defaults.yaml \
 -n $num_generated_samples -p "waiting_time=$cond_value" \
 -o "trained_run_folder_dir=results/N${grid_number}_L${grid_length}/R${route_number}_T${route_type}_MIN${min_num_vehicles}_MAX${max_num_vehicles}/trained|base_dir=results/N${grid_number}_L${grid_length}/R${route_number}_T${route_type}_MIN${min_num_vehicles}_MAX${max_num_vehicles}|sampler.guide_temp=${temperature}"
```

### Step4. Check the performance of Generated Layouts
```
python sample_check.py --grid_number $grid_number --grid_length $grid_length \
 --route_number $route_number --route_type $route_type --min_num_vehicles $min_num_vehicles --max_num_vehicles $max_num_vehicles \
 --num_samples $num_generated_samples --temperature $temperature --cond_value $cond_value 
```

You can also check the example in `run.sh` file.

