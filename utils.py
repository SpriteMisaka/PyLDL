import scipy.io as sio

def load_dataset(name):
    data = sio.loadmat('dataset/' + name)
    return data['features'], data['labels']
