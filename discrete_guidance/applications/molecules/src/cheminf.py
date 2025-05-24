# Import public modules
import rdkit
import numpy as np
import xml.etree.ElementTree as ET
from IPython.display import SVG
from numbers import Number
from rdkit import DataStructs
from rdkit.Chem import Draw, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from typing import List, Optional

# Import module obtained from 'https://github.com/chembl/ChEMBL_Structure_Pipeline'
import chembl_structure_pipeline

# Set logging level for RDKit
lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def draw_molecule_grid(smiles_list:List[str], 
                       property_name:Optional[str]=None, 
                       property_label:str='X', 
                       **kwargs) -> object:
    """
    Draw molecules on a grid.

    Args:
        smiles_list (list of str): List of SMILES strings of
            the molecules that should be drawn on a grid.
        property_name (None or str): Name of the property to be shown next to each molecule. 
            If None, no property is displayed
            (Default: None)
        property_label (str): Label for the property to be displayed.
            (Default: 'X')
        **kwargs: Key-word arguments forwarded to Draw.MolsToGridImage().

    Return:
        (float or int): Requested property value of the molecule
            corresponding to the input SMILES string.
    
    """
    mol_list = list()
    legend = list()
    for smiles in smiles_list:
        # Generate an RDKit molecule object
        mol = rdkit.Chem.MolFromSmiles(smiles)
        mol_list.append(mol)

        # Get the wishes molecular property
        if property_name is not None:
            mol_property = get_property_value(smiles, property_name=property_name)
            if mol_property==int(mol_property):
                mol_property =int(mol_property)
                mol_annotation = str(mol_property)
            else:
                mol_annotation = f"{mol_property:.2f}"
            
            mol_annotation = r'(' + property_label + r'=' + mol_annotation + r')'

            legend.append(mol_annotation)

    img = Draw.MolsToGridImage(mol_list, legends=legend, **kwargs)
    return img

def get_property_value(smiles:str, property_name:str) -> Number:
    """
    Return they property value of the molecule corresponding 
    to the input SMILES string.

    Args:
        smiles (str): SMILES string of the molecule for which
            the requested property should be returned for.
        property_name (str): Property name.

    Return:
        (float or int): Requested property value of the molecule
            corresponding to the input SMILES string.
    
    """
    if property_name=='logp':
        return get_logp(smiles, addHs=True)
    elif property_name=='num_rings':
        return get_num_rings(smiles)
    elif property_name=='num_heavy_atoms':
        return get_num_heavy_atoms(smiles) 
    elif property_name=='mol_weight':
        return get_molecular_weight(smiles)
    elif property_name=='num_tokens':
        return len(smiles)
    else:
        err_msg = f"The passed property_name '{property_name}' does not correspond to an expected property name."
        raise ValueError(err_msg)

def draw_molecule(smiles:str) -> None:
    """
    Draw the molecule corresponding to the passed smiles string.
    
    Args:
        smiles (str): SMILES string of the molecule to be drawn.

    """
    # Suppress constant printouts while standardizing
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    # Generate an RDKit molecule object
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        err_msg = f"The SMILES string '{smiles}' cannot be made into an RDKit molecule object."
        raise ValueError(err_msg)

    # Generate a drawer
    drawer = rdMolDraw2D.MolDraw2DSVG(300,300)
    drawer.drawOptions().addStereoAnnotation = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Display the drawn molecule
    display( SVG(drawer.GetDrawingText()) )

def get_washed_canonical_smiles(smiles:str, 
                                remove_stereochemistry:bool=False) -> str:
    """
    'Wash' the input SMILES string and return its canonical version.
    
    Args:
        smiles (str): (Canonical) SMILES string.
        remove_stereochemistry (bool): Should we also remove stereochemistry?
            (Default: False)

    Return:
        (str): Washed (canonical) SMILES string.
    
    """
    # Suppress constant printouts while standardizing
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    # Generate an RDKit molecule object
    mol = rdkit.Chem.MolFromSmiles(smiles)

    # Standardize (neutralize) the molecular object
    st_mol = chembl_structure_pipeline.standardize_mol(mol)

    # Remove salt and solvent
    st_mol, _ = chembl_structure_pipeline.get_parent_mol(st_mol)

    # If multiple fragments remain, take the one with the most heavy atoms
    st_mol_frags = rdkit.Chem.GetMolFrags(st_mol, asMols=True, sanitizeFrags=False)
    if 1 < len(st_mol_frags):
        st_mol_frags = sorted(
            st_mol_frags, key=lambda x: x.GetNumHeavyAtoms(), reverse=True
        )
        st_mol = st_mol_frags[0]
        
    # If we should remove the stereochemistry from the molecule, remove it
    if remove_stereochemistry:
        rdkit.Chem.RemoveStereochemistry(st_mol) 

    # Get the canonical SMILES string of the 'washed' molecular object and return it
    smiles = rdkit.Chem.MolToSmiles(st_mol, canonical=True)
    return rdkit.Chem.CanonSmiles(smiles)

def get_molecular_weight(smiles:str) -> float:
    """
    Return the molecular weight of the molecule generated by the input SMILES string.
    
    Args:
        smiles (str): SMILES string for which the molecular weight 
            should be returned for.

    Return:
        (float): Molecular weight of the molecule corresponding 
            to the input SMILES string.

    """
    # Suppress constant printouts while standardizing
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    # Generate an RDKit molecule object
    mol = rdkit.Chem.MolFromSmiles(smiles)

    # Determine and return the molecular weight of the RDKit molecule object
    return rdkit.Chem.rdMolDescriptors.CalcExactMolWt(mol)

def get_num_rings(smiles:str) -> int:
    """
    Return the number of rings of a molecule corresponding to the input SMILES string.
    
    Args:
        smiles (str): SMILES string for which the number of
            rings should be returned for.

    Return:
        (int): Number of rings of the molecule corresponding 
            to the input SMILES string.

    """
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        err_msg = f"The SMILES string '{smiles}' cannot be made into an RDKit molecule object."
        raise ValueError(err_msg)
    
    return rdMolDescriptors.CalcNumRings(mol) # number of rings

def get_logp(smiles:str, 
             addHs:bool=True) -> float:
    """
    Return the lipophilicity (LogP) of a molecule corresponding to the input SMILES string.
    
    Args:
        smiles (str): SMILES string for which the lipophilicity 
            should be returned for.
        addHs (bool): Optional boolean flag to specify if H-atoms 
            should be added in the computation or not.
            (Default: True)

    Return:
        (float): Lipophilicity of the molecule corresponding 
            to the input SMILES string.

    """
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        err_msg = f"The SMILES string '{smiles}' cannot be made into an RDKit molecule object."
        raise ValueError(err_msg)
    
    return MolLogP(mol, includeHs=addHs) # logp calculated by Crippen's approach

def get_num_heavy_atoms(smiles:str) -> int:
    """
    Return the number of heavy atoms of a molecule corresponding to the input SMILES string. 
    
    Args:
        smiles (str): SMILES string for which the number of
            heavy atoms should be returned for.

    Return:
        (int): Number of heavy-atoms of the molecule corresponding 
            to the input SMILES string.

    """
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        err_msg = f"The SMILES string '{smiles}' cannot be made into an RDKit molecule object."
        raise ValueError(err_msg)
    
    return mol.GetNumHeavyAtoms() # Number of heavy atoms

def get_morgan_fingerprint(smiles:str, 
                           fp_radius:int=2, 
                           fp_size:int=1024) -> object:
    """
    Return the Morgan fingerprint of molecule corresponding to the input SMILES string. 
    
    Args:
        smiles (str): SMILES string for which the Morgan
            fingerprint should be returned for.
        fp_radius (int): Morgan finger print (mfp) radius.
            (Default: 2)
        fp_size (int): Morgan finger print (mfp) size (i.e. length).
            (Default: 1024)

    Return:
        (object): Morgan fingerprint as RDKit fingerprint object.
    """
    # Get the molecule from the smiles string
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        err_msg = f"The SMILES string '{smiles}' cannot be made into an RDKit molecule object."
        raise ValueError(err_msg)

    # Initialize a fingerprint generator and generate a fingerprint from the molecule object
    # that is returned
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_size)
    return fp_generator.GetFingerprint(mol)

def get_tanimoto_similarities(smiles_list:List[str], 
                              which:str='average', 
                              mfp_radius:int=2, 
                              mfp_size:int=1024) -> np.ndarray:
    """
    (1) Determine the pairwise Tanimoto similarities (using Morgan fingerprints) 
    of the molecules corresponding to the passed list of SMILES string. 
    (2) Per molecule return either the average (which='average') or maximal 
    (which='maximal') Tanimoto similarity to all other molecules.

    Args:
        smiles_list (list): List of SMILES strings.
        which (str): Which Tanimoto similarity to return per molecule;
            - 'average': Return average Tanimoto similarity of one molecule
                to all the other molecules.
            - 'maximal': Return maximal Tanimoto similarity of one molecule
                to all the other molecules (i.e. return the similarity to most
                similar other molecule).
            (Default: 'average')
        mfp_radius (int): Morgan finger print (mfp) radius.
            (Default: 2)
        mfp_size (int): Morgan finger print (mfp) size (i.e. length).
            (Default: 1024)
    
    Return:
        (numpy.array): 1D numpy array of shape (#SMILES,) holding 'average' or 
            'maximal' Tanimoto similarity of each SMILES to all the other SMILES 
            strings.
    
    """
    ### Step (0)
    # Determine fingerprints for each of the SMILES strings
    fp_list = [get_morgan_fingerprint(smiles, fp_radius=mfp_radius, fp_size=mfp_size) for smiles in smiles_list]
    
    ### Step (1)
    # Determine the pairwise Tanimoto similarities (pwts) between all molecules
    # Inspired by:
    # https://stackoverflow.com/questions/51681659/how-to-use-rdkit-to-calculte-molecular-fingerprint-and-similarity-of-a-list-of-s
    # Remark: Exclude the last molecule ('len(fp_list)-1)') because all pairwise
    #         similarities have already been determined for it in the iterations
    #         of the previous molecules.
    pwts_matrix = np.zeros((len(smiles_list), len(smiles_list)))
    for mol_index in range(len(fp_list)-1):
        # Determine similarities between the current molecule and
        # all the 'molecules in the list after this molecule'
        mol_pwts_list = DataStructs.BulkTanimotoSimilarity(fp_list[mol_index], fp_list[mol_index+1:]) # returns as 'list' object

        # Assign the pairwise Tanimoto similarities to their values in
        # the pairwise Tanimoto similairities matrix
        pwts_matrix[mol_index, mol_index+1:] = np.array(mol_pwts_list)

    # The pairwise Tanimoto similairities matrix is upper diagonal matrix (with zero on diagonal), 
    # transform it to a symmetric matrix with np.nan on the diagonal.
    pwts_matrix = pwts_matrix + pwts_matrix.T
    pwts_matrix = pwts_matrix + np.diag(np.ones(pwts_matrix.shape[0])*np.nan)

    ### Step (2)
    if which=='average':
        return np.nanmean(pwts_matrix, axis=0)
    elif which=='maximal':
        return np.nanmax(pwts_matrix, axis=0)
    else:
        err_msg = f"Input 'which' must be either 'average' or 'maximal', got '{which}' instead."
        raise ValueError(err_msg)
