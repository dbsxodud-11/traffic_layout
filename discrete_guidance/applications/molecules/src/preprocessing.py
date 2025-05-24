# Import public modules
import collections
import os
import tqdm
import pandas as pd
from pathlib import Path
from typing import List, Optional

# Import custom modules
from . import cheminf

# Define all relative file paths (relative to 'base_dir' passed as argument to the functions defined below)
# for each of the datasets
# todo 02/14/2021: Add the relative file paths for the other datasets
# _REL_FILE_PATH_DICT = {
#     # For QMugs dataset:
#     'qmugs': {
#         'raw_dataset':          'data/raw/qmugs/summary.csv',
#         'unique_nswcs_df':      'data/preprocessed/qmugs_unique_nswcs_df.tsv',
#         'preprocessed_dataset': 'data/preprocessed/qmugs_preprocessed_dataset.tsv',
#     }
# }
# 이거 3개 다 쓰나? ㄴㄴ 결국에 여기서 preprocessed_dataset만 사용함, 이를 보면 preprocess.py 에선 raw_dataset에서
# preprocessed_dataset.tsv를 만드는 게 핵심역할인 것 같음
_REL_FILE_PATH_DICT = {
    # For sumo dataset:
    'sumo': {
        'raw_dataset':          'data/raw/qmugs/summary.csv',
        'unique_nswcs_df':      'data/preprocessed/qmugs_unique_nswcs_df.tsv',
        # 'preprocessed_dataset': 'data/preprocessed/qmugs_preprocessed_dataset.tsv',
        'preprocessed_dataset': 'data/preprocessed/sumo_preprocessed_dataset.csv'
    }
}


def load_preprocessed_dataset(which_dataset:str, 
                              base_dir:str, 
                              iteration:int=0,
                              cond_value:int=0,
                              num_samples:int=100,
                              temperature:int=0.1,  
                              seed:int=0,
                              logger:Optional[object]=None) -> pd.DataFrame:
    """
    Load a preprocessed dataset specified in 'which_dataset' and return it.

    Args:
        which_dataset (str): Which dataset (e.g., 'qmugs').
        base_dir (str): Base directory that should contain 'data/' as sub-folder.
        logger (None or object): Optional logger object.
            (Default: None)

    Return:
        (pandas.DataFrame): Loaded preprocessed dataset as pandas.DataFrame.

    """
    # Determine the (absolute) file path
    file_path = get_file_path(which_file='preprocessed_dataset',
                              which_dataset=which_dataset,  
                              base_dir=base_dir)
    if iteration == 1:
        # If the iteration is 1, then we do not need to add any suffix to the file name
        file_path = file_path.replace('.csv', f'_iter{iteration}.csv')
    else:
        file_path = file_path.replace('.csv', f'_iter{iteration}_c{cond_value}_n{num_samples}_t{temperature}_seed{seed}.csv')
    # Either load or create (and save) the preprocessed dataset
    if os.path.isfile(file_path):
        # Preprocessed dataset does exist, thus load it
        preprocessed_dataset_df = pd.read_csv(file_path, sep=',', dtype={"layout": str})
        # Remove rows where waiting_time is -1.0 (failed simulations)
        preprocessed_dataset_df = preprocessed_dataset_df[preprocessed_dataset_df['waiting_time'] != -1.0]
    
        print('I read the preprocessed dataset')
        display_msg(logger, f"Loaded the '{which_dataset}' preprocessed dataset from: {file_path}")
    else:
        # Preprocessed dataset does not exist, thus create and save it
        preprocessed_dataset_df = create_preprocessed_dataset(which_dataset=which_dataset, 
                                                              base_dir=base_dir, 
                                                              logger=logger)

    return preprocessed_dataset_df

def create_preprocessed_dataset(which_dataset:str, 
                                base_dir:str, 
                                logger:Optional[object]=None) -> pd.DataFrame:
    """
    Create preprocessed dataset specified in 'which_dataset', save it, and return it.

    Args:
        which_dataset (str): Which dataset (e.g., 'qmugs').
        base_dir (str): Base directory that should contain 'data/' as sub-folder.
        logger (None or object): Optional logger object.
            (Default: None)

    Return:
        (pandas.DataFrame): Created preprocessed dataset as pandas.DataFrame.

    """
    # Determine the (absolute) file path
    file_path = get_file_path(which_file='preprocessed_dataset',
                              which_dataset=which_dataset,  
                              base_dir=base_dir)
    
    # Load the list of unique non-stereochemical washed canonical SMILES (nswcs)
    # for the specified dataset
    unique_nswcs_list = load_unique_nswcs_list(which_dataset=which_dataset, base_dir=base_dir)
    unique_nswcs_list.sort() # For reproducibility

    # Loop over the nswcs and create a DataFrame containing nswcs, number of rings and log-p
    preprocessed_dataset_dict = collections.defaultdict(list)
    for nswcs in tqdm.tqdm(unique_nswcs_list):
        num_tokens      = len(nswcs)
        num_rings       = cheminf.get_num_rings(nswcs)
        logp            = cheminf.get_logp(nswcs, addHs=True)
        mol_weight      = cheminf.get_molecular_weight(nswcs)
        num_heavy_atoms = cheminf.get_num_heavy_atoms(nswcs)
        preprocessed_dataset_dict['nswcs'].append(nswcs)
        preprocessed_dataset_dict['num_tokens'].append(num_tokens)
        preprocessed_dataset_dict['num_rings'].append(num_rings)
        preprocessed_dataset_dict['num_heavy_atoms'].append(num_heavy_atoms)
        preprocessed_dataset_dict['logp'].append(logp)
        preprocessed_dataset_dict['mol_weight'].append(mol_weight)

    preprocessed_dataset_df = pd.DataFrame(preprocessed_dataset_dict)
    preprocessed_dataset_df.to_csv(file_path, index=False, sep='\t')
    display_msg(logger, f"Created and saved the '{which_dataset}' preprocessed dataset to: {file_path}")

    return preprocessed_dataset_df
    
def load_unique_nswcs_list(which_dataset:str, 
                           base_dir:str, 
                           logger:Optional[object]=None) -> List[str]:
    """
    Load the list of unique non-stereochemical washed canonical SMILES (nswcs) strings
    specified in 'which_dataset' and return it.

    Args:
        which_dataset (str): Which dataset (e.g., 'qmugs').
        base_dir (str): Base directory that should contain 'data/' as sub-folder.
        logger (None or object): Optional logger object.
            (Default: None)

    Return:
        (list of str): Loaded list of unique nswcs.

    """
    # Determine the (absolute) file path
    file_path = get_file_path(which_file='unique_nswcs_df',
                              which_dataset=which_dataset,  
                              base_dir=base_dir)

    # Either load or create (and save) the table
    if os.path.isfile(file_path):
        # Table does exist, thus load it
        unique_nswcs_df = pd.read_csv(file_path, sep='\t')
        display_msg(logger, f"Loaded a table with the unique non-stereochmical washed canonical SMILES (nswcs) strings of the '{which_dataset}' dataset from: {file_path}")
    else:
        # Table does not exist, thus create and save it
        unique_nswcs_df = create_unique_nswcs_df(which_dataset=which_dataset, 
                                                 base_dir=base_dir, 
                                                 logger=logger)
    return list(set(unique_nswcs_df['nswcs']))

def create_unique_nswcs_df(which_dataset:str, 
                           base_dir:str, 
                           logger:Optional[object]=None) -> pd.DataFrame:
    """
    Create list of unique non-stereochemical washed canonical SMILES (nswcs) strings 
    specified in 'which_dataset', save it, and return it.

    Args:
        which_dataset (str): Which dataset (e.g., 'qmugs').
        base_dir (str): Base directory that should contain 'data/' as sub-folder.
        logger (None or object): Optional logger object.
            (Default: None)

    Return:
        (pandas.DataFrame): Created (one-column) table of unique nswcs strings.

    """
    # Determine the (absolute) file path
    file_path = get_file_path(which_file='unique_nswcs_df',
                              which_dataset=which_dataset,  
                              base_dir=base_dir)
    
    # Obtain the list of SMILES strings depending on the dataset
    if which_dataset=='qmugs':
        # Load the raw QMugs dataset as pandas.DataFrame
        raw_dataset = load_QMugs_raw_dataset(base_dir=base_dir, 
                                             logger=logger)

        # Extract the unique SMILES strings from the raw dataset
        smiles_list = list(set(raw_dataset['smiles']))
    else:
        err_msg = f"Loading of raw '{which_dataset}' dataset has not been implemented.\nLoading has for example been implemented for dataset 'qmugs'."
        raise ValueError(err_msg)

    # Create a set of the non-stereochemical washed canonical SMILES (nswcs)
    # string of all SMILES string in the raw dataset
    nswcs_set = set()
    for smiles in tqdm.tqdm(smiles_list):
        nswcs = cheminf.get_washed_canonical_smiles(smiles, remove_stereochemistry=True)
        nswcs_set.add(nswcs)

    # Create a list from the set and sort the elements (for reproducibility)
    unique_nswcs_list = list(nswcs_set)
    unique_nswcs_list.sort()

    # Generate a (one-column) pandas.DataFrame with the unique nswcs strings
    unique_nswcs_df = pd.DataFrame({'nswcs': unique_nswcs_list})

    # Save the pandas.DataFrame as .tsv file
    unique_nswcs_df.to_csv(file_path, index=False, sep='\t')
    display_msg(logger, f"Created and saved the '{which_dataset}' unique non-stereochemical washed canonical SMILES (nswcs) strings (one-column) table to: {file_path}")

    return unique_nswcs_df

def load_QMugs_raw_dataset(base_dir:str, 
                           logger:Optional[object]=None) -> pd.DataFrame:
    """
    Load the raw QMugs dataset and return it.

    Args:
        base_dir (str): Base directory that should contain 'data/' as sub-folder.
        logger (None or object): Optional logger object.
            (Default: None)

    Return:
        (pandas.DataFrame): Loaded raw QMugs dataset as pandas.DataFrame.

    """
    # Determine the (absolute) file path
    file_path = get_file_path(which_file='raw_dataset',
                              which_dataset='qmugs',  
                              base_dir=base_dir)

    # If raw data does not exist yet, inform user how to download and save it
    if os.path.isfile(file_path)==False:
        err_msg = f"The raw 'QMugs' dataset (i.e. its 'summary.csv' file) cannot be found.\nDownload 'summary.csv' from https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM and save it as '{file_path}'."
        raise FileNotFoundError(err_msg)
    
    # Load the raw data
    raw_dataset_df = pd.read_csv(file_path)
    display_msg(logger, f"Loaded QMugs dataset from: {file_path}")
    
    return raw_dataset_df

def display_msg(logger:Optional[object]=None, 
                msg:str='') -> None:
    """
    Display a message using the logger (if defined/not None) or print.

    Args:
        logger (None or object): Optional logger object.
            (Default: None)
        msg (str): Message to be displayed.
            (Default: '')

    """
    if logger is None:
        print(msg)
    else:
        logger.info(msg)

def get_file_path(which_file:str, 
                  which_dataset:str, 
                  base_dir:str) -> str:
    """
    Return the absolute file path for the specified file and dataset.

    Args:
        which_file (str): Which file. Should be a key of the values of _REL_FILE_PATH_DICT.
        which_dataset (str): Which dataset. Should be a key of _REL_FILE_PATH_DICT (e.g., 'qmugs').
        base_dir (str): Base directory that should contain 'data/' as sub-folder.

    Return:
        (str): Absolute file path.

    """
    if which_dataset in _REL_FILE_PATH_DICT:
        if which_file in _REL_FILE_PATH_DICT[which_dataset]:
            file_path = str(Path(base_dir, _REL_FILE_PATH_DICT[which_dataset][which_file]))
        else:
            err_msg = f"'{which_file}' is not an available file, the available files are: {list(_REL_FILE_PATH_DICT[which_dataset].keys())}"
            raise ValueError(err_msg)
    else:
        err_msg = f"'{which_dataset}' is not an available dataset, the available datasets are: {list(_REL_FILE_PATH_DICT.keys())}"
        raise ValueError(err_msg)

    return file_path
