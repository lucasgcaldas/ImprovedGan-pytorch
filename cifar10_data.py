import pickle as pk
import os
import sys
import tarfile
from six.moves import urllib
import numpy as np

def maybe_download_and_extract(data_dir, url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    """
    Download and extract the CIFAR-10 dataset if it doesn't exist in the specified directory.

    Args:
        data_dir (str): Directory path where the CIFAR-10 dataset should be stored.
        url (str, optional): URL to download the CIFAR-10 dataset. Defaults to 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'.
    """
    if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(data_dir)

def unpickle(file):
    """
    Unpickle a CIFAR-10 batch file.

    Args:
        file (str): Path to the CIFAR-10 batch file.

    Returns:
        dict: Dictionary containing the unpickled CIFAR-10 data with keys 'x' for images and 'y' for labels.
    """
    fo = open(file, 'rb')
    d = pk.load(fo, encoding='iso-8859-1')
    fo.close()
    return {'x': np.cast[np.float32]((-127.5 + d['data'].reshape((10000, 3, 32, 32))) / 128.), 'y': np.array(d['labels']).astype(np.uint8)}

def load(data_dir, subset='train', download=True):
    """
    Load the CIFAR-10 dataset.

    Args:
        data_dir (str): Directory path where the CIFAR-10 dataset is stored or will be downloaded.
        subset (str, optional): Dataset subset to load. Either 'train' or 'test'. Defaults to 'train'.
        download (bool, optional): Whether to download the CIFAR-10 dataset if it doesn't exist. Defaults to True.

    Returns:
        tuple: Tuple containing the loaded images and labels as numpy arrays.
    """
    if download:
        maybe_download_and_extract(data_dir)
    if subset == 'train':
        train_data = [unpickle(os.path.join(data_dir, 'cifar-10-batches-py/data_batch_' + str(i))) for i in range(1, 6)]
        trainx = np.concatenate([d['x'] for d in train_data], axis=0)
        trainy = np.concatenate([d['y'] for d in train_data], axis=0)
        return trainx, trainy
    elif subset == 'test':
        test_data = unpickle(os.path.join(data_dir, 'cifar-10-batches-py/test_batch'))
        testx = test_data['x']
        testy = test_data['y']
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')
