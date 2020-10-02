import data_process
import globalConfig
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

resource_file_dir_path = "data/real_raw_data"
labeled_root_dir = "data/real_labeled_data"
real_sampleset_file = "data/real_sampleset.csv"
adversarial_sampleset_random_file = "data/adversarial_sampleset_random.csv"
adversarial_sampleset_mac_disrupt_file = "data/adversarial_sampleset_mac_disrupt.csv"
mixed_dataset_file = "data/mixed_dataset.csv"
train_dataset_file = "data/train_dataset.csv"
test_dataset_file = "data/test_dataset.csv"

real_data_tag = "1"
seq_len = globalConfig.seq_len
ID_min_value = 5
ID_max_value = 25*60+5-1

point_label_map = {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5,
                   '7':6, '8':7, '9':8, '10':9, '11':10, '12':11,
                   '13':12, '14':13, '15':14, '16':15, "17":16, '18':17,
                   '19':18, '20':19, '21':20, '22':21, '23':22, '24': 23,
                   '25': 24, '26': 25, '27': 26, '28': 27, '29': 28, '30': 29,
                   '31': 30, '32': 31, '33': 32, '34': 33}

# data_process.file_to_tag_dir(resource_file_dir_path, labeled_root_dir, point_label_map)
# data_process.create_real_sampleset(labeled_root_dir, real_sampleset_file, real_data_tag)
# data_process.create_adversarial_sampleset_random(seq_len, ID_min_value, ID_max_value, adversarial_sampleset_random_file)
data_process.create_adversarial_sampleset_mac_disrupt(resource_file_dir_path, adversarial_sampleset_mac_disrupt_file)
data_process.get_mixed_dataset(real_sampleset_file, adversarial_sampleset_random_file, mixed_dataset_file)
data_process.get_mixed_dataset(mixed_dataset_file, adversarial_sampleset_mac_disrupt_file, mixed_dataset_file)

df = pd.read_csv(mixed_dataset_file, index_col=0)
x, y = df.values[:, :seq_len], df.values[:, seq_len:]  # x，y这里已经是ndarray
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
train_data = np.concatenate([x_train, y_train], axis=1)
test_data = np.concatenate([x_test, y_test], axis=1)
pd.DataFrame(train_data).to_csv(train_dataset_file)
pd.DataFrame(test_data).to_csv(test_dataset_file)



