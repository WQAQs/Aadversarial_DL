import os
import shutil
from adversarial_dl_test import data_process

import globalConfig


setType = "test"
setType = "train"
point_label_map = {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5,
                   '7':6, '8':7, '9':8, '10':9, '11':10, '12':11,
                   '13':12, '14':13, '15':14, '16':15, "17":16, '18':17,
                   '19':18, '20':19, '21':20, '22':21, '23':22, '24': 23,
                   '25': 24, '26': 25, '27': 26, '28': 27, '29': 28, '30': 29,
                   '31': 30, '32': 31, '33': 32, '34': 33}
labeled_root_dir = "./points_csv/" + setType
resource_root_dir = "./raw_data/" + setType

# dirs = os.listdir(source_root_dir)
# for data_dir in dirs:
#     dir_path = os.path.join(source_root_dir, data_dir)
#     data_files = os.listdir(dir_path)
#     for data_file in data_files:
#         file_path = os.path.join(dir_path, data_file)
#         point_tag = data_file.split(".")[0].split('_')[1]
#         class_tag = point_label_map[point_tag]
#         new_dir = labeled_root_dir + "\\" + str(class_tag)
#         new_file_path = new_dir + "\\" + data_dir + '_' + data_file
#         if not os.path.exists(new_dir):  # 目标目录不存在时创建目录
#             os.makedirs(new_dir)
#         shutil.copyfile(file_path, new_file_path)

data_process.file_to_tag_dir(resource_root_dir, labeled_root_dir, point_label_map)
