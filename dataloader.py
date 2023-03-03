import torch
import numpy as np
import mne
import os
from tqdm.auto import tqdm
import math
from random import shuffle
from torch import nn


class CreateIDs():
    def __init__(self, dataset_path, task_mode, sample_length=336, split_mode='recordings', split_ratio=(1., 0., 0.), n_samples_per_recording=29):

        assert task_mode in ['rest_unrest', 'left_right', 'upper_lower',
                             'all_tasks'], "Mode must be set to one of the following: 'rest_unrest', 'left_right', 'upper_lower', 'all_tasks'"

        assert sum(
            split_ratio) <= 1, 'Sum of ratios must be equal or smaller than one'

        self.dataset_path = dataset_path
        self.sample_length = sample_length
        self.task_mode = task_mode
        self.split_mode = split_mode
        self.split_ratio = split_ratio
        self.n_samples_per_recording = n_samples_per_recording
        self.left_right_runs = [3, 4, 7, 8, 11, 12]
        self.upper_lower_runs = [5, 6, 9, 10, 13, 14]

    def check_events(self, sample_name):
        """
        Checks if at least two events are present in the recording
        """

        recording_name = sample_name.split('-')[0] + '.edf'
        participant_name = sample_name[:4]
        recording_path = os.path.join(
            self.dataset_path, participant_name, recording_name)
        recording = mne.io.read_raw_edf(recording_path, verbose=0)
        events = mne.events_from_annotations(recording, verbose=0)[0]

        if len(events) < 2:
            return False
        else:
            return True

    def create_samples_patients(self, patients_list):
        """
        Creates list of ids of samples with train-val-test split according to patients
        """

        ids_list = []
        for patient_folder in patients_list:
            recordings = [x for x in os.listdir(
                patient_folder) if x.endswith('edf')]

            if self.task_mode == 'rest_unrest' or self.task_mode == 'all_tasks':
                recordings = recordings
            elif self.task_mode == 'left_right':
                recordings = [x for x in recordings if int(
                    x[5:-4]) in self.left_right_runs]
            elif self.task_mode == 'upper_lower':
                recordings = [x for x in recordings if int(
                    x[5:-4]) in self.upper_lower_runs]

            for r in recordings:
                for i in range(self.n_samples_per_recording):  # here self
                    ids_list.append(r[:-4] + '-' + str(i) + '.edf')
        return ids_list

    def create_samples_recordings(self, patient_folder, task_mode, split_ratio):
        train_ids = []
        val_ids = []
        test_ids = []

        recordings = [x for x in os.listdir(
            patient_folder) if x.endswith('edf')]
        shuffle(recordings)  # inplace shuffling

        if self.task_mode == 'rest_unrest' or task_mode == 'all_tasks':
            recordings = recordings
        elif self.task_mode == 'left_right':
            recordings = [x for x in recordings if int(
                x[5:-4]) in self.left_right_runs]
        elif self.task_mode == 'upper_lower':
            recordings = [x for x in recordings if int(
                x[5:-4]) in self.upper_lower_runs]

        n_train = math.floor(len(recordings)*split_ratio[0])
        n_val = math.ceil(len(recordings)*split_ratio[1])  # warning if len 0
        n_test = math.ceil(len(recordings)*split_ratio[2])

        if len(recordings) - n_train - n_val > n_test:
            n_train = len(recordings) - n_val - n_test

        for r in recordings[:n_train]:
            for i in range(self.n_samples_per_recording):
                train_ids.append(r[:-4] + '-' + str(i) + '.edf')

        for r in recordings[n_train:n_train+n_val]:
            for i in range(self.n_samples_per_recording):
                val_ids.append(r[:-4] + '-' + str(i) + '.edf')

        for r in recordings[len(recordings)-n_test:]:
            for i in range(self.n_samples_per_recording):
                test_ids.append(r[:-4] + '-' + str(i) + '.edf')

        return {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}

    def create(self):

        patients_folders = [os.path.join(self.dataset_path, x) for x in os.listdir(
            self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, x))]
        shuffle(patients_folders)  # in place shuffling
        print("Found {} patient's folders".format(len(patients_folders)))
        print('Data will be split with ratio {}% train, {}% val, {}% test'.format(
            self.split_ratio[0]*100, self.split_ratio[1]*100, self.split_ratio[2]*100))

        if self.split_mode == 'patients':
            print('Splitting data into train, val and test set according to patients')
            # later print how many recordings in each set and from how many patients
            n_train = int(len(patients_folders)*self.split_ratio[0])
            n_val = int(len(patients_folders)*self.split_ratio[1])
            n_test = int(len(patients_folders)*self.split_ratio[1])

            train_patients_folders = patients_folders[:n_train]
            val_patients_folders = patients_folders[n_train:n_train+n_val]
            test_patients_folders = patients_folders[-n_test:]

            train_ids = self.create_samples_patients(
                train_patients_folders, self.task_mode)
            val_ids = self.create_samples_patients(
                val_patients_folders, self.task_mode)
            test_ids = self.create_samples_patients(
                test_patients_folders, self.task_mode)

        elif self.split_mode == 'recordings':
            print('Splitting data into train, val and test set according to recordings, val and test will use ceil int')
            train_ids, val_ids, test_ids = [], [], []

            for patient_folder in patients_folders:
                ids_dict = self.create_samples_recordings(
                    patient_folder, self.task_mode, self.split_ratio)
                train_ids += ids_dict['train_ids']
                val_ids += ids_dict['val_ids']
                test_ids += ids_dict['test_ids']

        # cleaning ids

        print('Checking train IDs')
        clean_train_ids = [x for x in tqdm(train_ids) if self.check_events(x)]
        print('Checking validation IDs')
        clean_val_ids = [x for x in tqdm(val_ids) if self.check_events(x)]
        print('Checking test IDs')
        clean_test_ids = [x for x in tqdm(test_ids) if self.check_events(x)]
        print('Created {} train, {} validation, and {} test IDs'.format(
            len(clean_train_ids), len(clean_val_ids), len(clean_test_ids)))

        return clean_train_ids, clean_val_ids, clean_test_ids


class DataSet(torch.utils.data.Dataset):

    def __init__(self, list_IDs, dataset_path, sample_length, task_mode, downsampling_factor=2, verbose=0):

        assert task_mode in ['rest_unrest', 'left_right', 'upper_lower',
                             'all_tasks'], "Mode must be set to one of the following: 'rest_unrest', 'left_right', 'upper_lower', 'all_tasks'"

        self.list_IDs = list_IDs
        self.dataset_path = dataset_path
        self.sample_length = sample_length
        self.downsampling_factor = downsampling_factor
        self.task_mode = task_mode
        self.verbose = verbose

        self.left_right_runs = [3, 4, 7, 8, 11, 12]
        self.upper_lower_runs = [5, 6, 9, 10, 13, 14]

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        IDs are given in a format 'SXXBRYY-NN.edf' where NN is the number of sample
        eg. 'S038R14-0.edf'
        """

        ID = self.list_IDs[index]
        recording_name = ID.split('-')[0] + '.edf'
        run_index = int(ID.split('-')[0][-2:])
        participant_name = ID[:4]
        sample_number = int(ID.split('-')[1][:-4])

        whole_recording_path = os.path.join(
            self.dataset_path, participant_name, recording_name)
        whole_recording = mne.io.read_raw_edf(whole_recording_path, verbose=0)
        whole_recording = whole_recording.resample(
            sfreq=(whole_recording.info['sfreq']/self.downsampling_factor))
        events = mne.events_from_annotations(whole_recording, verbose=0)[
            0]  # removing description of values
        sample_start = events[sample_number][0]
        sample_end = sample_start + self.sample_length

        whole_recording = whole_recording.to_data_frame().drop(
            ['time'], axis=1).to_numpy()
        whole_recording = whole_recording.reshape(
            np.shape(whole_recording)[1], -1)

        X = np.expand_dims(whole_recording[:, sample_start:sample_end], axis=0)
        y = [x[2] for x in events if x[0] >= sample_start][0] - 1

        """
        Task modes:
        first y = y-1 to make T0 = 0, t1 = 1 from t0 = 1, t1 = 2 etc
            
        rest_unrest changes labels into 0 for rest and 1 for unrest
            if y!= 0:
                y = 1
        
        left_right and upper_lower - one hot encoding with num classes 3

        all_tasks - one hot encoding with num classes 5
            if y = 0 class 0
            if y = 1 and run in left_right runs y = 1
            if y = 2 and run in left_right runs y = 2
            if y = 1 and run in upper_lower_runs y = 3
            if y = 2 and run in upper_lower_runs y = 4
        """

        if self.task_mode == 'rest_unrest':
            if y != 0:  # taking rest as 1 and other actions as 0
                y = 1
            # this should be cleaned
            y = torch.from_numpy(np.array([y])).float()
        elif self.task_mode == 'left_right' or self.task_mode == 'upper_lower':
            y = nn.functional.one_hot(torch.from_numpy(np.array(y)), 3)
        elif self.task_mode == 'all_tasks':
            if run_index in self.upper_lower_runs and y == 1:
                y = nn.functional.one_hot(
                    torch.from_numpy(np.array(3)), 5).float()
            elif run_index in self.upper_lower_runs and y == 2:
                y = nn.functional.one_hot(
                    torch.from_numpy(np.array(4)), 5).float()
            else:
                y = nn.functional.one_hot(
                    torch.from_numpy(np.array(y)), 5).float()

        X = torch.from_numpy(X).float()
        mean = X.mean()
        std = X.std()
        X = (X-mean)/std

        return X, y