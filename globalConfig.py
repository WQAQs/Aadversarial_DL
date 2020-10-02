import os
import pandas as pd
from keras_bert.bert import get_base_dict

# '''设置工作默认目录，使数据保持统一的存放格式，方便管理，若路径不存在将创建相应的目录及其结构'''
# data_name = "remove9i"
# dirs = [ "remove3i", "remove6i", "remove9i", "remove12i"]#, "best_param6pretrain2tune" , "best_param9pretrain2tune"]
# # dirs = ["best_param"]#,"feed_forward_dim16","feed_forward_dim32","feed_forward_dim64","feed_forward_dim128", "feed_forward_dim256"]

'''这里设置公共参数'''
STEP_SIZE = 1
timeInterval = 500
seq_len = 40
head_num = 2
token_num = len(get_base_dict()) + 26 * 60//STEP_SIZE
transformer_num = 8
embed_dim = 32
feed_forward_dim = 32


# def copy_dirs(source, dist):
#     """
#     复制source下所有目录到dist目录中，若dist不存在将创建
#     :param source: string
#         源目录
#     :param dist: string
#         目标目录
#     :return: nothing
#     """
#     if not os.path.exists(dist):
#         os.makedirs(dist)
#         children = os.listdir(source)
#     for child_dir in children:
#         next_source = os.path.join(source, child_dir)
#         if os.path.isdir(next_source):
#             next_dist = os.path.join(dist, child_dir)
#             copy_dirs(next_source, next_dist)
#
#
mac_dict = None
mac_list_file = "data/mac_list.csv"

def get_mac_dict():
    """
    获取mac字典
    :return:
    """
    global mac_dict
    if not mac_dict:
        mac_dict = get_base_dict()
        mac_list = pd.read_csv(mac_list_file).mac
        for mac in mac_list:
            mac_dict[mac] = len(mac_dict)
    return mac_dict


#
# for d in dirs:
#     base_dir = ".\\format_dir\\"
#     full_path = base_dir + d
#     model_path = base_dir + "example"
#     if not os.path.exists(full_path):
#         copy_dirs(model_path, full_path)
    # os.chdir(full_path)  # 改变当前路径到设置的data_name文件夹下
#
#
# base_dir = ".\\format_dir\\"
# full_path = base_dir+data_name
# model_path = base_dir+"example"
# if not os.path.exists(full_path):
#     copy_dirs(model_path, full_path)
# os.chdir(full_path)   # 改变当前路径到设置的data_name文件夹下
# print(os.getcwd())
