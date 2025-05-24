import numpy as np
import pandas as pd


preprocessed_data = pd.read_csv('/home/son9ih/gs-project/discrete_guidance/applications/molecules/data/preprocessed/sumo_preprocessed_dataset.tsv', sep='\t')
# preprocessed_data = pd.read_csv('/home/son9ih/gs-project/discrete_guidance/applications/molecules/data/preprocessed/qmugs_preprocessed_dataset.tsv', sep='\t')

print('Head of the preprocessed data:')
print(preprocessed_data.head())

print('Shape of the preprocessed data:')
print(preprocessed_data.shape)

print('Columns of the preprocessed data:')
print(preprocessed_data.columns)