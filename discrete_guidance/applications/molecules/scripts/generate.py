# Import public modules
import argparse
import collections
import copy
import time
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Import custom modules
from applications.molecules.src import bookkeeping
from applications.molecules.src import cheminf
from applications.molecules.src import config_handling
from applications.molecules.src import factory
from applications.molecules.src import logging
from applications.molecules.src import managers
from applications.molecules.src import plotting
from applications.molecules.src import utils

# Only run as main
if __name__=='__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('-c', '--config',                     type=str, required=True, help='[Required] Path to (generation) config file.')
    parser.add_argument('-n', '--num_valid_molecule_samples', type=int, required=True, help='[Required] Number of valid molecules to be sampled.')
    parser.add_argument('-p', '--property_name_value',        type=str, default='',    help='[Optional] Which target property value the molecule generation should be guided to in the form "<property-name>=<property-value>" (for example "num_rings=0"). If this argument is not passed, use unconditional generation.')
    parser.add_argument('-o', '--overrides',                  type=str, default='',    help='[Optional] Which configs (in the config file) to override (pass configuration names and override-values in the format "<config-name-1>=<config-value-1>|<config-name-2>=<config-value-2>"). If this argument is not passed, no configurations will be overriden.')
    args = parser.parse_args()
    
    # Strip potenial '"' at beginning and end of args.overrides
    args.overrides = args.overrides.strip('"')

    # Load the configs from the passed path to the config file
    generation_cfg = config_handling.load_cfg_from_yaml_file(args.config)

    # Deepcopy the original cfg
    original_generation_cfg = copy.deepcopy(generation_cfg)

    # Parse the overrides
    overrides = config_handling.parse_overrides(args.overrides)

    # Update the configs with the overrides
    generation_cfg.update(overrides)
    generation_cfg.sampler.guide_temp = generation_cfg.temperature
    generation_cfg.sampler.seed = generation_cfg.seed

    # Parse the property name and target property value
    args.property_name_value = args.property_name_value.strip('"')
    if args.property_name_value=='': # Default value if no property value was passed
        property_name                   = 'None'
        target_property_value           = None
        run_folder_name_property_prefix = 'unconditional'
    else: # Target property passed
        # property_name = num_rings, target_property_value = 1
        property_name, target_property_value = args.property_name_value.split('=')
        if target_property_value=='None':
            target_property_value = None
        else:
            target_property_value = float(target_property_value)
            if int(target_property_value)==target_property_value:
                target_property_value = int(target_property_value)

        run_folder_name_property_prefix = args.property_name_value

    # Extract the requested number of unique valid nswcs (uvnswcs) from the arguments
    num_uvnswcs_requested = int(args.num_valid_molecule_samples)
    # num_uvnswcs_requested = int(100*generation_cfg.sampler.max_iterations)

    # Create a folder for the current generation run
    # save_location = ./generated
    save_location = str(Path(generation_cfg.base_dir, 'generated'))
    # run_folder_name = num_rings1|n=1000
    # run_folder_name = f"{run_folder_name_property_prefix}|n={num_uvnswcs_requested}"
    # if args.overrides!='':
    #     run_folder_name += f"|{args.overrides["trained_run_folder_dir"]}"
    run_folder_name = f"c={target_property_value}|n={num_uvnswcs_requested}|t={generation_cfg.sampler.guide_temp}|seed={generation_cfg.seed}"

    outputs_dir = bookkeeping.create_run_folder(save_location, run_folder_name, include_time=False)

    # Define a logger
    log_file_path = str(Path(outputs_dir, 'logs'))
    logger = logging.define_logger(log_file_path, file_logging_level='INFO', stream_logging_level='DEBUG')

    # Set the logging level of matplotlib to 'info' (to avoid a plethora of irrelevant matplotlib DEBUG logs)
    plt.set_loglevel('info')

    # Log initial information
    if target_property_value is None:
        logger.info(f"Unconditional generation.")
    else:
        logger.info(f"Guided generation towards target: {property_name}={target_property_value}")
    logger.info(f"Generate until the sampled number of unique valid nswcs is: {num_uvnswcs_requested}")

    # Construct an orchestrator from a (trained) 'run folder' containing the saved model weights 
    # and other meta-information required to construct the models and generate from them.
    # Set some overrised for the train_cfg
    trained_overrides = {
        'make_figs': True,
        'save_figs': False,
    }
    config_handling.update_dirs_in_cfg(trained_overrides, outputs_dir=generation_cfg.trained_run_folder_dir)

    # Define an orchestrator from the run folder and overrides
    orchestrator = factory.Orchestrator.from_run_folder(run_folder_dir=generation_cfg.trained_run_folder_dir, overrides=trained_overrides, load_data=True, logger=logger)

    # Load all models
    # 여기에 훈련된 diffusion과 predictor 모델이 로드됨
    # manager.models_dict = {
    #     'denoising_model': denoising_model,'waiting_model': waiting_model, 'num_rings_predictor_model': num_rings_predictor_model, 'num_tokens_predictor_model': num_tokens_predictor_model, 'logp_predictor_model': logp_predictor_model, 'num_heavy_atoms_predictor_model': num_heavy_atoms_predictor_model
    orchestrator.manager.load_all_models()

    # Update the generation configurations without overwriting entries
    config_handling.update_without_overwrite(generation_cfg, orchestrator.cfg)

    # Update the directoris in the generation config file
    config_handling.update_dirs_in_cfg(generation_cfg, str(outputs_dir))

    # Update the 'save_figs' flag
    generation_cfg.save_figs = True

    # Define manager for evaluation with trained model
    eval_manager = managers.DFMManager(generation_cfg, 
                                       denoising_model=orchestrator.manager.denoising_model,
                                       predictor_models_dict=orchestrator.manager.predictor_models_dict)

    # Log the overrides
    logger.info(f"Overrides: {overrides}")

    # Log the configs
    logger.info(f"Overriden config: {generation_cfg}")

    # Save the cfg, original_cfg, and overrides as yaml files in cfg.config_dir
    file_path = str(Path(generation_cfg.configs_dir, 'original_config.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, original_generation_cfg.to_dict())
    file_path = str(Path(generation_cfg.configs_dir, 'overrides.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, overrides)
    file_path = str(Path(generation_cfg.configs_dir, 'config.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, generation_cfg.to_dict())

    # Define a writer used to track training
    tensorboard_writer = bookkeeping.setup_tensorboard(generation_cfg.outputs_dir, rank=0)

    # Set a random seed
    utils.set_random_seed(generation_cfg.sampler.seed)

    # Generate until a certain number of unique valid nswcs (uvnswcs) has been sampled 
    sampled_uvnswcs_list = list() # Keep track of the sampled unique valid nswcs (uvnswcs)
    generated_df_list    = list()
    global_start_time    = time.time()
    logger.info(f"Will generate layout are at least {num_uvnswcs_requested} layouts (i.e. nswcs) have been sampled.")
    for iteration in range(generation_cfg.sampler.max_iterations):
        # If no property is specified, use unconditional sampling
        if target_property_value is None:
            x_generated = eval_manager.generate(num_samples=generation_cfg.sampler.batch_size, 
                                                seed=None, # Only use the external seed
                                                stochasticity=generation_cfg.sampler.noise,
                                                dt=generation_cfg.sampler.dt,
                                                batch_size=generation_cfg.sampler.batch_size)
        else:
            # Construct the predictor model name based on the property name
            predictor_model_name = f"{property_name}_predictor_model"
            logger.info(f"Predictor model name: {predictor_model_name}")

            # Check that the predictor model is valid
            if predictor_model_name not in orchestrator.manager.models_dict:
                err_msg = f"There is no predictor model with name '{predictor_model_name}'. Allowed predictor models are: {list(orchestrator.manager.models_dict.keys())}"
                raise ValueError(err_msg)

            # Here we use generation cfg
            x_generated = eval_manager.generate(num_samples=generation_cfg.sampler.batch_size, #500, 
                                                seed=None, # Only use the external seed
                                                stochasticity=generation_cfg.sampler.noise,
                                                dt=generation_cfg.sampler.dt,
                                                predictor_y_dict={predictor_model_name: target_property_value},
                                                guide_temp=generation_cfg.sampler.guide_temp,
                                                grad_approx=generation_cfg.sampler.grad_approx,
                                                batch_size=generation_cfg.sampler.batch_size)


        # Analyze the generated x
        # filtering here?
        
        # generated sample들을 decoding? 그럴 필요 없는데
        # 원소의 기호를 index숫자로 바꿔서 diffusion으로 넣어준 것이고, 이게 잘 됬다면 그냥 포함시켜도 상관없을듯
        generated_smiles_list = [orchestrator.molecules_data_handler.smiles_encoder.decode(utils.to_numpy(smiles_encoded)) for smiles_encoded in x_generated]
        generated_df_list.extend(generated_smiles_list)
        
        if len(generated_df_list) > num_uvnswcs_requested:
            generated_df_list = generated_df_list[:num_uvnswcs_requested]
            break
            
        # filtering duplicated or invalid smiles
    #     analysis_dict = utils.analyze_generated_smiles(generated_smiles_list, 
    #                                                    orchestrator.molecules_data_handler.subset_df_dict['train']['nswcs'],
    #                                                    pad_token=orchestrator.molecules_data_handler.pad_token,
    #                                                    logger=logger)

    #     # Construct a table with the generated molecules
    #     iter_dict = collections.defaultdict(list)
    #     for x, smiles in zip(x_generated, generated_smiles_list):    
    #         # Get the validity
    #         valid = (smiles in analysis_dict['unique_valid_gen_smiles_list'])

    #         # If the smiles is not valid, continue to next smiles
    #         if valid==False:
    #             continue

    #         # Determine the nswcs
    #         # Remark: This is only possible for valid molecules (hence do this after the validation filter above)!
    #         nswcs = analysis_dict['smiles_to_nswcs_map'][smiles]

    #         # Append to corresponding lists
    #         iter_dict['smiles'].append(smiles)
    #         iter_dict['valid'].append(valid)
    #         iter_dict['nswcs'].append(nswcs)

    #         # If a property has been defined, determine the property value(s) of the generated molecule 
    #         if property_name!='None':
    #             # 우린 이걸 시뮬레이션 통하지 않고는 죽어도 모르니까 이걸 그냥 저장하면 되겠네
    #             # invalid샘플 제외하고
    #             # Determine the ground truth (using RDKit) property value
    #             ground_truth_property_value = cheminf.get_property_value(smiles, property_name=property_name)

    #             # Construct the predictor model name based on the property name
    #             predictor_model_name = f"{property_name}_predictor_model"
    #             logger.info(f"Predictor model name: {predictor_model_name}")

    #             # Predict the property value using the corresponding predictor model
    #             predicted_property_value = orchestrator.manager.predict_property(predictor_model_name, x=x, t=1, return_probs=False)

    #             # Append to corresponding lists
    #             if target_property_value is None:
    #                 iter_dict[f"target_{property_name}"].append('None')
    #             else:
    #                 iter_dict[f"target_{property_name}"].append(target_property_value)
    #             iter_dict[f"predicted_{property_name}"].append(predicted_property_value)
    #             iter_dict[f"ground_truth_{property_name}"].append(ground_truth_property_value)

    #     # Transform the dictionary of lists to a pandas.DataFrame
    #     iter_df = pd.DataFrame(iter_dict)
        
    #     # Make the iter_dict a pandas DataFrame and append it to the corresponding list
    #     generated_df_list.append(pd.DataFrame(iter_dict))

    #     # Update the list of unique valid nswcs (uvnswcs)
    #     if 0<len(iter_df): # If there were no molecules in this iteration, we cannot update
    #         filtered_df = iter_df[iter_df['valid']==True]
    #         sampled_uvnswcs_list += list(set(filtered_df['nswcs']))
    #         sampled_uvnswcs_list = list(set(sampled_uvnswcs_list))

    #     # Determine the number of sampled unique valid nswcs (uvnswcs)
    #     num_sampled_uvnswcs = len(sampled_uvnswcs_list)

    #     if logger is None:
    #         print(f"[{iteration}] Number of already sampled unique valid nswcs: {num_sampled_uvnswcs} (Duration since start: {(time.time()-global_start_time)/60:.2f}min)")
    #         print('-'*100)
    #     else:
    #         logger.info(f"[{iteration}] Number of already sampled unique valid nswcs: {num_sampled_uvnswcs} (Duration since start: {(time.time()-global_start_time)/60:.2f}min)")
    #         logger.info('-'*100)

    #     tensorboard_writer.add_scalar(f"{property_name}/Num-sampled-molecules", num_sampled_uvnswcs, iteration)

    #     # If the number of unique valid generated nswcs exceeds the requestes number, 
    #     # halt generation
        
    #     if num_uvnswcs_requested<=num_sampled_uvnswcs:
    #         break

    # logger.info(f"Generated at least {num_uvnswcs_requested} valid molecules. Duration: {(time.time()-global_start_time)/60:.2f}min")

    # # Stack the DataFrames in the list 'generated_df_list' to obtain one big DataFrame
    # generated_df = pd.concat(generated_df_list)

    # # Only keep the first 'args.num_valid_molecule_samples' samples
    # generated_df = generated_df[:args.num_valid_molecule_samples]

    # Save this DataFrame
    file_name = f'samples_table_iter{generation_cfg.iteration}_c{target_property_value}_n{num_uvnswcs_requested}_t{generation_cfg.sampler.guide_temp}_seed{generation_cfg.seed}.csv'
    file_path = str(Path(generation_cfg.outputs_dir, file_name))
    generated_df_list = pd.Series(generated_df_list)
    generated_df_list.to_csv(file_path, index=False, sep='\t')
    # generated_df.to_csv(file_path, index=False, sep='\t')
    logger.info(f"Stored the samples as table in: {file_path}")
    logger.info(f"Number of unique valid nswcs: {len(generated_df_list)}")

    # # Get all unique valid smiles
    # filtered_df = generated_df[generated_df['valid']==True]
    # unique_valid_gen_smiles_list = list(set(filtered_df['smiles']))

    # # Only make a plot if there are any unique valid generated smiles
    # if len(unique_valid_gen_smiles_list)>0 and generation_cfg.make_figs:
    #     property_names = ['num_tokens', 'logp', 'num_rings', 'num_heavy_atoms']
    #     fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    #     if target_property_value is None:
    #         guide_temp_label = None
    #     else:
    #         guide_temp_label = generation_cfg.sampler.guide_temp
    #     plt.suptitle(f"Target {property_name}: {target_property_value} | Stochasticity: {generation_cfg.sampler.noise} | T: {guide_temp_label}")
    #     for index, property_name in enumerate(property_names):
    #         index1 = index%2
    #         index2 = (index-index1)//2
    #         ax = axs[index1, index2]
    #         plotting.plot_gen_vs_train_distribution(property_name, 
    #                                                 orchestrator.molecules_data_handler.subset_df_dict['train'], 
    #                                                 unique_valid_gen_smiles_list,
    #                                                 ax=ax)
            
    #     # Save the figure
    #     if generation_cfg.save_figs and generation_cfg.figs_save_dir is not None:
    #         file_path = str(Path(generation_cfg.figs_save_dir, f"Visualization_samples.png"))
    #         fig.savefig(file_path)
                        
