import numpy as np
from scipy.io import loadmat
import os

import numpy as np
from sklearn.discriminant_analysis import _cov
import scipy

from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, resample
import random
from tqdm import tqdm

def load_eeg_data(folder_path,save_path,do_preprocessing=False):
    subject_folders = [f for f in os.listdir(folder_path) if f.startswith('subject')]

    for subject_folder in subject_folders:
        subject_number = subject_folder.replace('subject', '').lstrip('0')
        
        eeg_file_path = os.path.join(folder_path, subject_folder, 'EEG', f'subject{subject_number}_eeg.mat')
        label_file_path = os.path.join(folder_path, subject_folder, 'EEG', f'subject{subject_number}_eeg_label.mat')

        eeg_data = loadmat(eeg_file_path)
        label_data = loadmat(label_file_path)

        if 'seg' in eeg_data:
            eeg_array = np.array(eeg_data['seg'])
        elif 'seg1' in eeg_data:
            eeg_array = np.array(eeg_data['seg1']) 
        else:
            raise ValueError(f"Neither 'seg' nor 'seg1' found in EEG data file {subject_folder}")

        label_array = np.array(label_data['label']) 

        # Denoise EEG data
        fs = 500  
        new_fs = 100  
        lowcut = 0.5  
        highcut = 50  
        order = 3  

        if do_preprocessing:
            print(eeg_array.shape)
            eeg_array = denoise_and_resample_eeg_data(eeg_array, fs, lowcut, highcut, order,new_fs)
            print(eeg_array.shape)

        target_subject_folder = os.path.join(save_path,  f'subject{subject_number}')
        if not os.path.exists(target_subject_folder):
            os.makedirs(target_subject_folder)

        eeg_save_path = os.path.join(target_subject_folder, f'subject{subject_number}_eeg.npy')
        label_save_path = os.path.join(target_subject_folder, f'subject{subject_number}_eeg_label.npy')

        np.save(eeg_save_path, eeg_array)
        np.save(label_save_path, label_array)

        print(f"Saved EEG data to {eeg_save_path}")
        print(f"Saved label data to {label_save_path}")

    print("EEG data and labels have been prepared.")


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, padlen=None):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, padlen=padlen)
    return y

def denoise_and_resample_eeg_data(eeg_data, fs=500, lowcut=0.5, highcut=50, order=5, new_fs=250):
    num_samples, num_channels, num_trials = eeg_data.shape
    new_num_samples = int(num_samples * new_fs / fs)
    resampled_data = np.zeros((new_num_samples, num_channels, num_trials))

    for trial in tqdm(range(num_trials), desc="Processing Trials"):
        for channel in range(num_channels):
            denoised_channel = butter_bandpass_filter(eeg_data[:, channel, trial], lowcut, highcut, fs, order)
            resampled_data[:, channel, trial] = resample(denoised_channel, new_num_samples)
    
    return resampled_data

def load_data_from_file(args,folder_path,save_path, preprocessing_data_path):
    data_list = []
    label_list = []
    
    if not os.path.exists(preprocessing_data_path):
        print(f"目录 {preprocessing_data_path} 不存在，将加载原始数据并进行预处理。")
        load_eeg_data(folder_path, save_path, do_preprocessing=False)
        preprocessing_data_path = save_path  
    else:
        print(f"从目录 {preprocessing_data_path} 加载预处理数据。")
    
    for i in range(1, args.preprocessed_max_subject + 1):
        eeg_path = os.path.join(preprocessing_data_path,f'subject{i}' ,f"subject{i}_eeg.npy")
        label_path = os.path.join(preprocessing_data_path,f'subject{i}' ,f"subject{i}_eeg_label.npy")
        if os.path.exists(eeg_path) and os.path.exists(label_path):
            eeg_data = np.load(eeg_path) 
            label_data = np.load(label_path)
            data_list.append(eeg_data)
            label_list.append(label_data)
        else:
            print(f"文件 {preprocessing_data_path} 不存在，将加载原始数据并进行预处理。")
            load_eeg_data(folder_path, save_path, do_preprocessing=False)
            if os.path.exists(eeg_path) and os.path.exists(label_path):
                eeg_data = np.load(eeg_path)  
                label_data = np.load(label_path)
                data_list.append(eeg_data)
                label_list.append(label_data)
            else:
                print(f"文件 {file_path} 仍然不存在，跳过该主体。")
    
    return data_list,label_list

class Dataset_prepare(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        label = self.labels[idx]
        return input_data, label

import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(data, labels, test_size=0.2, random_seed=None):
    if data.shape[0] != labels.shape[0]:
        raise ValueError("数据集和标签的样本数量不匹配！")
    
    if random_seed is not None:
        print(f"使用的随机数种子为: {random_seed}")
        np.random.seed(random_seed)
    else:
        print("未设置随机数种子，切分结果可能每次不同。")
    
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_seed, stratify=labels)
    
    unique_labels = np.unique(labels)
    print("\n训练集和测试集中每种标签的占比：")
    for label in unique_labels:
        train_label_ratio = np.mean(train_labels == label)
        test_label_ratio = np.mean(test_labels == label)
        print(f"标签 {label}: 训练集占比={train_label_ratio:.2f}, 测试集占比={test_label_ratio:.2f}")
    
    return train_data, train_labels, test_data, test_labels

def cut_and_extend_data(data, labels, l):
    extended_data = []
    extended_labels = []

    for sample, label in zip(data, labels):
        sample_length = sample.shape[0]  
        num_segments = sample_length // l  

        for i in range(num_segments):
            start = i * l
            end = start + l
            segment = sample[start:end, :]  
            extended_data.append(segment)
            extended_labels.append(label)  

    extended_data = np.array(extended_data)
    extended_labels = np.array(extended_labels)

    return extended_data, extended_labels

def cut_and_extend_data_new(data, labels, l, shift_steps):
    combined_data = []
    combined_labels = []
    seen_segments = set()  

    for sample, label in zip(data, labels):
        sample_length = sample.shape[0] 
        num_segments = sample_length // l  

        for i in range(num_segments):
            start = i * l
            end = start + l
            segment = sample[start:end, :] 

            segment_str = str(segment.tobytes())

            if segment_str not in seen_segments:
                combined_data.append(segment)
                combined_labels.append(label)
                seen_segments.add(segment_str)

            shift_start = start + shift_steps
            shift_end = shift_start + l

            if 0 <= shift_start < sample_length - l + 1:
                shifted_segment = sample[shift_start:shift_end, :]

                shifted_segment_str = str(shifted_segment.tobytes())

                if shifted_segment_str not in seen_segments:
                    combined_data.append(shifted_segment)
                    combined_labels.append(label)
                    seen_segments.add(shifted_segment_str)

    # 转换为 NumPy 数组
    combined_data = np.array(combined_data)
    combined_labels = np.array(combined_labels)

    return combined_data, combined_labels

def expand_and_shuffle(data, labels, x):
    if len(data) != len(labels):
        raise ValueError("数据和标签的长度不一致！")

    new_data = np.tile(data, (x, 1, 1))  
    new_labels = np.tile(labels, (x, 1))   

    indices = np.arange(new_data.shape[0])
    np.random.shuffle(indices)
    new_data = new_data[indices]
    new_labels = new_labels[indices]

    return new_data, new_labels


