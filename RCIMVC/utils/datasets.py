import os
import random
import sys
import h5py
import numpy as np
import scipy.io as sio
from scipy import sparse
from utils import util
import math


def load_data(config, train_dir=False):
    data_name = config['dataset']
    X_list = []
    Y_list = []
    main_dir = sys.path[0]
    if train_dir:
        main_dir = os.path.join(main_dir, '')

    if data_name in ['Scene-15']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'Scene-15.mat'))
        X = mat['X'][0]
        X_list.append(X[2].astype('float32'))  # 40
        X_list.append(X[1].astype('float32'))  # 59
        Y_list.append(np.squeeze(mat['Y']))
        Y_list.append(np.squeeze(mat['Y']))
    elif data_name in ['Scene-15-3']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'Scene-15.mat'))
        X = mat['X'][0]
        X_list.append(X[2].astype('float32'))  # 40
        X_list.append(X[1].astype('float32'))  # 59
        X_list.append(X[0].astype('float32'))  # 20
        Y_list.append(np.squeeze(mat['Y']))
        Y_list.append(np.squeeze(mat['Y']))
        Y_list.append(np.squeeze(mat['Y']))


    elif data_name in ['MSRC_v1']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        for view in ['msr2', 'msr3']:
            X_list.append(mat[view].astype('float32'))
            Y_list.append(np.squeeze(mat['truth']))
    
    
    elif data_name in ['CCV']:
        data2 = np.load('/home/yanwenbiao/SCMRL/SCMRL/data/SIFT.npy')
        data3 = np.load('/home/yanwenbiao/SCMRL/SCMRL/data/STIP.npy')
        # data3 = np.load('/home/yanwenbiao/SCMRL/SCMRL/data/MFCC.npy')
        labels = np.load('/home/yanwenbiao/SCMRL/SCMRL/data/label.npy')
        
        x1 = data2.astype(np.float32)
        x2 = data3.astype(np.float32)
        y = labels

        from sklearn.preprocessing import normalize
        x1 = normalize(x1, axis=1, norm='max')
        x2 = normalize(x2, axis=1, norm='max')
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        x1 = min_max_scaler.fit_transform(x1)
        x2 = min_max_scaler.fit_transform(x2)

        X_list.append(x1.astype('float32').reshape(6773, 5000))
        X_list.append(x2.astype('float32').reshape(6773, 5000))
        Y_list.append(np.squeeze(labels.astype(np.int32).reshape(6773,)))
    
    
    elif data_name in ['MFeat']:
        mat = sio.loadmat('/home/yanwenbiao/CODE/SCMRL/SCMRL/data/Mfeat.mat')
        x1 = np.copy(mat['X'][0][0]).astype('float32')
        x2 = np.copy(mat['X'][0][4]).astype('float32')
        y = np.copy(mat['Y'].T)

        from sklearn.preprocessing import normalize
        x1 = normalize(x1, axis=1, norm='max')
        x2 = normalize(x2, axis=1, norm='max')
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        x1 = min_max_scaler.fit_transform(x1)
        x2 = min_max_scaler.fit_transform(x2)

        X_list.append(x1.astype('float32').reshape(2000, 216))
        X_list.append(x2.astype('float32').reshape(2000, 240))
        Y_list.append(np.squeeze(y.astype(np.int32).reshape(2000,)))
    
    
    elif data_name in ['BBC']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'BBC.mat'))
        x1 = mat['data1'].toarray().astype('float32')
        x1 = x1.T
        x2 = mat['data2'].toarray().astype('float32')
        x2 = x2.T
        y = mat['truth'].astype('float32').flatten() 
        from sklearn.preprocessing import normalize
        x1 = normalize(x1, axis=1, norm='max')
        x2 = normalize(x2, axis=1, norm='max')
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        x1 = min_max_scaler.fit_transform(x1)
        x2 = min_max_scaler.fit_transform(x2)

        X_list.append(x1.astype('float32').reshape(685, 4659))
        X_list.append(x2.astype('float32').reshape(685, 4633))
        Y_list.append(np.squeeze(y.astype(np.int32).reshape(685,)))

    elif data_name in ['ORL']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'ORL.mat'))
        x1 = mat['X'][0][0].astype('float32')
        x2 = mat['X'][0][2].astype('float32')
        y = mat['Y'].astype('float32').flatten()  # 使用 flatten() 将其转为一维数组

        from sklearn.preprocessing import normalize
        x1 = normalize(x1, axis=1, norm='max')
        x2 = normalize(x2, axis=1, norm='max')
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        x1 = min_max_scaler.fit_transform(x1)
        x2 = min_max_scaler.fit_transform(x2)

        X_list.append(x1.astype('float32').reshape(400, 512))
        X_list.append(x2.astype('float32').reshape(400, 864))
        Y_list.append(np.squeeze(y.astype(np.int32).reshape(400,)))

    elif data_name in ['cub_googlenet']:
        mat = sio.loadmat('/home/yanwenbiao/CODE/2023-IJCAI-ProImp/data/cub_googlenet_doc2vec_c10.mat')
        x1 = np.copy(mat['X'][0][0]).astype('float32')
        x2 = np.copy(mat['X'][0][1]).astype('float32')
        y = np.copy(mat['gt'])

        from sklearn.preprocessing import normalize
        x1 = normalize(x1, axis=1, norm='max')
        x2 = normalize(x2, axis=1, norm='max')
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        x1 = min_max_scaler.fit_transform(x1)
        x2 = min_max_scaler.fit_transform(x2)

        X_list.append(x1.astype('float32').reshape(600, 1024))
        X_list.append(x2.astype('float32').reshape(600, 300))
        Y_list.append(np.squeeze(y.astype(np.int32).reshape(600,)))
        
    elif data_name in ['CCV-3']:
        data1 = np.load('/home/yanwenbiao/SCMRL/SCMRL/data/SIFT.npy')
        data2 = np.load('/home/yanwenbiao/SCMRL/SCMRL/data/STIP.npy')
        data3 = np.load('/home/yanwenbiao/SCMRL/SCMRL/data/MFCC.npy')
        labels = np.load('/home/yanwenbiao/SCMRL/SCMRL/data/label.npy')
        
        x1 = data1.astype(np.float32)
        x2 = data2.astype(np.float32)
        x3 = data3.astype(np.float32)
        y = labels

        from sklearn.preprocessing import normalize
        x1 = normalize(x1, axis=1, norm='max')
        x2 = normalize(x2, axis=1, norm='max')
        x3 = normalize(x3, axis=1, norm='max')
    
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        x1 = min_max_scaler.fit_transform(x1)
        x2 = min_max_scaler.fit_transform(x2)
        x3 = min_max_scaler.fit_transform(x3)
    

        X_list.append(x1.astype('float32').reshape(6773, 5000))
        X_list.append(x2.astype('float32').reshape(6773, 5000))
        X_list.append(x3.astype('float32').reshape(6773, 4000))
    
        Y_list.append(np.squeeze(labels.astype(np.int32).reshape(6773,)))


    elif data_name in ['ORL-4']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'ORL.mat'))
        x1 = mat['X'][0][0].astype('float32')
        x2 = mat['X'][0][2].astype('float32')
        x3 = mat['X'][0][3].astype('float32')
        x4 = mat['X'][0][1].astype('float32')
        y = mat['Y'].astype('float32').flatten()  # 使用 flatten() 将其转为一维数组

        from sklearn.preprocessing import normalize
        x1 = normalize(x1, axis=1, norm='max')
        x2 = normalize(x2, axis=1, norm='max')
        x3 = normalize(x3, axis=1, norm='max')
        x4 = normalize(x4, axis=1, norm='max')

        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        x1 = min_max_scaler.fit_transform(x1)
        x2 = min_max_scaler.fit_transform(x2)
        x3 = min_max_scaler.fit_transform(x3)
        x4 = min_max_scaler.fit_transform(x4)

        X_list.append(x1.astype('float32').reshape(400, 512))
        X_list.append(x2.astype('float32').reshape(400, 864))
        X_list.append(x3.astype('float32').reshape(400, 254))
        X_list.append(x4.astype('float32').reshape(400, 59))

        Y_list.append(np.squeeze(y.astype(np.int32).reshape(400,)))

    else:
        raise Exception('Undefined data_name')
    return X_list, Y_list


def next_batch(X1, X2, batch_size):
    # generate next batch, just two views
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size) - 1  # fix the last batch
    if tot % batch_size == 0:
        total += 1

    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        yield batch_x1, batch_x2, (i + 1)


def next_batch_gt(X1, X2, gt, batch_size):
    # generate next batch with label
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size) - 1  # fix the last batch
    for i in range(int(total) - 1):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        gt_now = gt[start_idx: end_idx, ...]
        yield batch_x1, batch_x2, gt_now, (i + 1)


def next_batch_list(X, batch_size):
    # generate next batch list and X is a list
    tot = X[0].shape[0]
    total = math.ceil(tot / batch_size) - 1  # fix the last batch
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x_list = []
        for k in X.shape[0]:
            batch_x = X[k][start_idx: end_idx, ...]
            batch_x_list.append(batch_x)
        yield batch_x_list, (i + 1)


class DataSet_NoisyMNIST(object):

    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False, dtype=np.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into `[0, 1]`.
        """
        t = 2
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                    'images1.shape: %s labels.shape: %s' % (images1.shape, labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                    'images2.shape: %s labels.shape: %s' % (images2.shape, labels.shape))
            self._num_examples = images1.shape[0] // t

            if dtype == np.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                # print("type conversion view 1")
                images1 = images1.astype(np.float32)

            if dtype == np.float32 and images2.dtype != np.float32:
                # print("type conversion view 2")
                images2 = images2.astype(np.float32)

        self._images1 = images1[::t]
        self._images2 = images2[::t]
        self._labels = labels[::t]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_image for _ in range(batch_size)], [fake_label for _
                                                                                                      in range(
                    batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]
    
    
