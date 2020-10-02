import random
import pandas as pd
import globalConfig
import os
import shutil
import os
import pandas as pd
from keras_bert.bert import get_base_dict

timeInterval = globalConfig.timeInterval
seq_len = globalConfig.seq_len
STEP_SIZE = globalConfig.STEP_SIZE
time_each_file = 10000  # 每个文件生成数据集时取的时长，单位s
LEAST = -100
MOST = -40
BASE_LEN = globalConfig.get_base_dict().__len__()

TIME_IDX = 1
MAC_IDX = 3
RSSI_IDX = 4

# mac过滤列表，将该下标位置对应的数据过滤掉，模拟ibeacon失效
# 这里过滤1805的3、7号两个ibeacon
# del_mac_idx = []
# del_mac_idx = [10, 14,21]
# del_mac_idx = [10, 12,14,11,21,18]
del_mac_idx = []
# del_mac_idx = [10, 12, 14, 11, 21, 18, 8, 13, 23, 9, 22, 24]


def create_real_sampleset(resource_file_dir_path, result_token_file, data_tag):
    if os.path.exists(result_token_file):  # 如果文件存在
        os.remove(result_token_file)
    mac_dict = globalConfig.get_mac_dict()
    dirs = os.listdir(resource_file_dir_path)
    for point_tag in dirs:
        data_path = os.path.join(resource_file_dir_path, point_tag)
        files = os.listdir(data_path)
        for file in files:
            data_file = os.path.join(data_path, file)
            data = pd.read_csv(data_file, header=None)
            tokens = []
            token = [0] * (seq_len + 1)
            begin_time = data.iloc[0][0]
            first_time = begin_time
            i = 0  # 序列号
            for row in data.itertuples():
                if row[TIME_IDX] - begin_time > timeInterval:
                    token[seq_len] = data_tag
                    tokens.append(token)
                    begin_time = row[TIME_IDX]
                    token = [0] * (seq_len + 1)
                    i = 0
                if row[TIME_IDX] - first_time > time_each_file * 1000:
                    break
                idx = mac_dict.get(row[MAC_IDX], -1)
                if idx > -1 and i < seq_len:
                    value = getNumber(idx, row[RSSI_IDX])
                    if value > -1:
                        token[i] = value
                        i += 1
            pd.DataFrame(tokens).to_csv(result_token_file, index=False, encoding='utf-8', mode="a+", header=None)
            print(file + " finish")
    df = pd.read_csv(result_token_file, header=None)
    df.to_csv(result_token_file, columns=[i for i in range(seq_len + 1)])


def create_adversarial_sampleset_random(seq_len, min_value, max_value, file_path, n=10000):
    samples = []
    sample = [0 for i in range(seq_len + 1)]
    for i in range(n):
        num = random.randint(2, seq_len)  # 随机产生一个随机数，作为一个样本中出现的mac-rssi token的数量
        for j in range(num):
            value = random.randint(min_value, max_value)
            sample[j] = value
        samples.append(sample)
        sample[seq_len] = '0'
        sample = [0 for i in range(seq_len + 1)]
    pd.DataFrame(samples).to_csv(file_path)


def create_adversarial_sampleset_constant_noise(resource_file_dir_path, result_token_file, noise):
    if os.path.exists(result_token_file):  # 如果文件存在
        os.remove(result_token_file)
    data_tag = '0'
    mac_dict = globalConfig.get_mac_dict()
    dirs = os.listdir(resource_file_dir_path)
    for point_tag in dirs:
        data_path = os.path.join(resource_file_dir_path, point_tag)
        files = os.listdir(data_path)
        for file in files:
            data_file = os.path.join(data_path, file)
            data = pd.read_csv(data_file, header=None)
            tokens = []
            token = [0] * (seq_len + 1)
            begin_time = data.iloc[0][0]
            first_time = begin_time
            i = 0  # 序列号
            for row in data.itertuples():
                if row[TIME_IDX] - begin_time > timeInterval:
                    token[seq_len] = data_tag
                    tokens.append(token)
                    begin_time = row[TIME_IDX]
                    token = [0] * (seq_len + 1)
                    i = 0
                if row[TIME_IDX] - first_time > time_each_file * 1000:
                    break
                idx = mac_dict.get(row[MAC_IDX], -1)
                if idx > -1 and i < seq_len:
                    value = getNumber(idx, row[RSSI_IDX] + noise)  #加噪声
                    if value > -1:
                        token[i] = value
                        i += 1
            pd.DataFrame(tokens).to_csv(result_token_file, index=False, encoding='utf-8', mode="a+", header=None)
            print(file + " finish")
    df = pd.read_csv(result_token_file, header=None)
    df.to_csv(result_token_file, columns=[i for i in range(seq_len + 1)])


def create_adversarial_sampleset_alternating_noise(resource_file_dir_path, result_token_file, noise):
    if os.path.exists(result_token_file):  # 如果文件存在
        os.remove(result_token_file)
    data_tag = '0'
    noise_flag = 1
    mac_dict = globalConfig.get_mac_dict()
    dirs = os.listdir(resource_file_dir_path)
    for point_tag in dirs:
        data_path = os.path.join(resource_file_dir_path, point_tag)
        files = os.listdir(data_path)
        for file in files:
            data_file = os.path.join(data_path, file)
            data = pd.read_csv(data_file, header=None)
            tokens = []
            token = [0] * (seq_len + 1)
            begin_time = data.iloc[0][0]
            first_time = begin_time
            i = 0  # 序列号
            for row in data.itertuples():
                if row[TIME_IDX] - begin_time > timeInterval:
                    token[seq_len] = data_tag
                    tokens.append(token)
                    begin_time = row[TIME_IDX]
                    token = [0] * (seq_len + 1)
                    i = 0
                if row[TIME_IDX] - first_time > time_each_file * 1000:
                    break
                idx = mac_dict.get(row[MAC_IDX], -1)
                if idx > -1 and i < seq_len:
                    if noise_flag:
                        value = getNumber(idx, row[RSSI_IDX] + noise)  # 加噪声
                        noise_flag = 0
                    else:
                        value = getNumber(idx, row[RSSI_IDX] - noise)  # 加噪声
                        noise_flag = 1
                    if value > -1:
                        token[i] = value
                        i += 1
            pd.DataFrame(tokens).to_csv(result_token_file, index=False, encoding='utf-8', mode="a+", header=None)
            print(file + " finish")
    df = pd.read_csv(result_token_file, header=None)
    df.to_csv(result_token_file, columns=[i for i in range(seq_len + 1)])


def create_adversarial_sampleset_mac_disrupt(resource_file_dir_path, file_path, n=10000, k_min=2, k_max=5):
    if os.path.exists(file_path):  # 如果文件存在
        os.remove(file_path)
    mac_dict = globalConfig.get_mac_dict()
    dirs = os.listdir(resource_file_dir_path)
    j = 0  # 统计生成的对抗样本数量
    k = 0  # 一个对抗样本中有几个mac对应的rssi被打乱，计数变量
    k_radom = random.randint(k_min, k_max)
    for point_tag in dirs:
        data_path = os.path.join(resource_file_dir_path, point_tag)
        files = os.listdir(data_path)
        for file in files:
            data_file = os.path.join(data_path, file)
            data = pd.read_csv(data_file, header=None)
            tokens = []
            token = [0] * (seq_len + 1)
            begin_time = data.iloc[0][0]
            first_time = begin_time
            i = 0  # 序列号
            for row in data.itertuples():
                if j < n:
                    if row[TIME_IDX] - begin_time > timeInterval:
                        token[seq_len] = '0'
                        if k >= k_min:
                            tokens.append(token)
                            j += 1
                        begin_time = row[TIME_IDX]
                        token = [0] * (seq_len + 1)
                        i = 0
                        k = 0
                        k_radom = random.randint(k_min, k_max)
                    if row[TIME_IDX] - first_time > time_each_file * 1000:
                        break
                    idx = mac_dict.get(row[MAC_IDX], -1)
                    if idx > -1 and i < seq_len:
                        if k < k_radom:
                            value = getNumber(idx + random.randint(0, 25), row[RSSI_IDX])  # 打乱mac对应的rssi
                            k += 1
                        else:
                            value = getNumber(idx, row[RSSI_IDX])  # 不打乱
                        if value > -1:
                            token[i] = value
                            i += 1
            if j >= n:
                break
            pd.DataFrame(tokens).to_csv(file_path, index=False, encoding='utf-8', mode="a+", header=None)
            print(file + " finish")
        else:
            continue
        if j >= n:
            pd.DataFrame(tokens).to_csv(file_path, index=False, encoding='utf-8', mode="a+", header=None)
            break
    df = pd.read_csv(file_path, header=None)
    df.to_csv(file_path, columns=[i for i in range(seq_len + 1)])


def create_adversarial_sampleset_2remote_mac_disrupt(resource_file_dir_path, file_path, n=10000, k_min=2, k_max=5):
    if os.path.exists(file_path):  # 如果文件存在
        os.remove(file_path)
    mac_dict = globalConfig.get_mac_dict()
    dirs = os.listdir(resource_file_dir_path)
    j = 0  # 统计生成的对抗样本数量
    for point_tag in dirs:
        data_path = os.path.join(resource_file_dir_path, point_tag)
        files = os.listdir(data_path)
        for file in files:
            data_file = os.path.join(data_path, file)
            data = pd.read_csv(data_file, header=None)
            tokens = []
            token = [0] * (seq_len + 1)
            begin_time = data.iloc[0][0]
            first_time = begin_time
            i = 0  # 序列号
            for row in data.itertuples():
                if j < n:
                    if row[TIME_IDX] - begin_time > timeInterval:
                        token[seq_len] = '0'
                        tokens.append(token)
                        j += 1
                        begin_time = row[TIME_IDX]
                        token = [0] * (seq_len + 1)
                        i = 0
                    if row[TIME_IDX] - first_time > time_each_file * 1000:
                        break
                    idx = mac_dict.get(row[MAC_IDX], -1)
                    if idx == 13 or idx == 25:  # 更换相隔很远的两个ble，ble14和ble26
                        idx = 25 + 13 - idx
                    if idx > -1 and i < seq_len:
                        value = getNumber(idx, row[RSSI_IDX])
                        if value > -1:
                            token[i] = value
                            i += 1
            if j >= n:
                break
            pd.DataFrame(tokens).to_csv(file_path, index=False, encoding='utf-8', mode="a+", header=None)
            print(file + " finish")
        else:
            continue
        if j >= n:
            pd.DataFrame(tokens).to_csv(file_path, index=False, encoding='utf-8', mode="a+", header=None)
            break
    df = pd.read_csv(file_path, header=None)
    df.to_csv(file_path, columns=[i for i in range(seq_len + 1)])

def get_mixed_dataset(file1_path, file2_path, res_file_path):
    # if os.path.exists(res_file_path):  # 如果文件存在
    #     os.remove(res_file_path)
    df1 = pd.read_csv(file1_path, index_col=0)
    df2 = pd.read_csv(file2_path, index_col=0)
    ## 合并 然后 混匀
    df_mixed = pd.concat([df1, df2], axis=0)
    ##  pandas的dataframe有自带的sample功能，当设参数frac = 1 的时候，就相当于对行做shuffle:
    df_mixed_shuffled = df_mixed.sample(frac=1)
    df_mixed_shuffled.to_csv(res_file_path)

def getNumber(macIndex, rssi):
    if del_mac_idx.__contains__(macIndex):
        return -1
    if rssi-LEAST < 0:
        rssi = LEAST
    if MOST - rssi <= 0:  # 超出范围的就略过
        rssi = MOST
    return (MOST - LEAST)//STEP_SIZE*(macIndex - BASE_LEN)+(rssi-LEAST)//STEP_SIZE+BASE_LEN

def file_to_tag_dir(source_root_dir, labeled_root_dir, point_label_map):
    dirs = os.listdir(source_root_dir)
    for data_dir in dirs:
        dir_path = os.path.join(source_root_dir, data_dir)
        data_files = os.listdir(dir_path)
        for data_file in data_files:
            file_path = os.path.join(dir_path, data_file)
            point_tag = data_file.split(".")[0].split('_')[1]
            class_tag = point_label_map[point_tag]
            new_dir = labeled_root_dir + "\\" + str(class_tag)
            new_file_path = new_dir + "\\" + data_dir + '_' + data_file
            if not os.path.exists(new_dir):  # 目标目录不存在时创建目录
                os.makedirs(new_dir)
            shutil.copyfile(file_path, new_file_path)

