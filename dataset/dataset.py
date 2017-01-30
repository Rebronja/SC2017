import numpy as np
import h5py as h5


def test(file):
    f = h5.File(file + ".hdf5", "w")
    for i in range(10):
        dset = f.create_dataset("dset" + str(i), (50,), dtype='f')
        dset.attrs['label'] = str(i)
        dset.attrs['kana'] = 'kana' + str(i)

    return f


class HiraSet:
    def __init__(self, file_path, size):
        self._file_path = file_path
        self._file = h5.File(file_path + ".hdf5")
        self._entries = None
        self.__size = size

    def file_path(self, file_path=None):
        if file_path is None:
            return self._file_path
        else:
            self._file_path = file_path

    def entries(self):
        return self.__entries

    def close(self):
        for name in self._file:
            dset = self._file[name]
            self.__entries.add(HiraEntry(dset.attrs['label'], dset.attrs['kana'], dset[:, :]))

        self._file.close()
        self._file = None

    def add(self, entry):
        if self._file is None:
            file = h5.File(self.file_path() + ".hdf5")

        dset = file.create_dataset('dset_' + entry.label(), (self.__size, ), dtype='f')
        dset.attrs['label'] = entry.label()
        dset.attrs['kana'] = entry.kana()

    def process(self, picture):
        print(picture)
        # napraviti odgovarajuce HiraEntry objekte za prosledjenu sliku (onu veliku, valja seci poprilicno)


class HiraEntry:
    def __init__(self, label, kana, data=None):
        self._label = label
        self._kana = kana
        self._data = data

    def label(self, label=None):
        if label is None:
            return self._label
        else:
            self._label = label

    def kana(self, kana=None):
        if kana is None:
            return self._kana
        else:
            self._kana = kana

    def data(self, data=None):
        if data is None:
            return self._data
        else:
            self._data = data

    def add(self, row):
        if self._data is None:
            self.data(row)
        else:
            self._data = np.vstack((self._data, row))
