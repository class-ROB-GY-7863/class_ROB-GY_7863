
import numpy as np
import h5py


def save_h5py(fn, np_dict, param=None):
    with h5py.File(fn, "w") as h5file:
        for key, data in np_dict.items():
            h5file.create_dataset(key, data=data)
        if param is not None:
            for key, value in param.items():
                h5file.attrs[key] = value


def load_h5py(fn):
    np_dict = {}
    with h5py.File(fn, "r") as h5file:
        for key in h5file.keys():
            np_dict[key] = h5file[key][:]
        param = {}
        for key, value in h5file.attrs.items():
            param[key] = value
    return np_dict, param


class DummyContext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return None