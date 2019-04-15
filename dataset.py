import numpy as np
import h5py as h5
from skimage.color import rgb2gray
from skimage.io import imsave
from skimage.morphology import opening, closing, square, diamond, disk, erosion, dilation
import matplotlib.pyplot as plt
import os as os
from skimage import img_as_uint


class HiraSet:
    def __init__(self, file_path, size):
        self.__file_path = file_path
        self.__file = h5.File(file_path + ".hdf5")
        self.__entries = []
        self.__size = size

    def entries(self):
        return self.__entries

    def pull(self):
        entries = []
        for name in self.__file:
            dset = self.__file[name]
            entries.append(HiraEntry(dset.attrs['label'], dset.attrs['kana'], dset[:, :]))

        self.__entries = entries
        return entries

    def close(self):
        for name in self.__file:
            dset = self.__file[name]
            self.__entries.append(HiraEntry(dset.attrs['label'], dset.attrs['kana'], dset[:, :]))

            self.__file.flush()
        self.__file.close()
        self.__file = None

    def add(self, entry):
        file = self.__file
        if file is None:
            file = h5.File(self.__file_path + ".hdf5")

        length = entry.data().shape[0]
        dset = file.create_dataset(entry.label(), (length, self.__size), dtype='f', maxshape=(None, self.__size))
        dset.attrs['label'] = entry.label()
        dset.attrs['kana'] = entry.kana()
        dset[:, :] = entry.data()
        file.flush()

    def require(self):
        data = []
        labels = []
        for index, entry in enumerate(self.__entries):
            length = len(entry.data())
            zeros = np.zeros(len(characters()))
            zeros[index] = 1

            for ind in range(length):
                data.append(entry.data()[ind])
                labels.append(zeros)

        return np.array(data), labels

    def require_new(self, train_num, test_num, shouldIndex = False):
        train = []
        test = []
        tr_labels = []
        te_labels = []

        for index, entry in enumerate(self.__entries):
            length = len(entry.data())
            zeros = np.zeros(len(characters()))
            zeros[index] = 1

            tr_inds = np.random.choice(length, train_num, replace=False)
            for ind in tr_inds:
                train.append(entry.data()[ind])
                if shouldIndex:
                    tr_labels.append(index)
                else:
                    tr_labels.append(zeros)

            te_inds = []
            for num in range(length):
                if not num in tr_inds:
                    te_inds.append(num)

            te_inds = np.random.choice(te_inds, test_num, replace=False)
            for ind in te_inds:
                test.append(entry.data()[ind])
                if shouldIndex:
                    te_labels.append(index)
                else:
                    te_labels.append(zeros)

        return train, test, tr_labels, te_labels


class HiraEntry:
    def __init__(self, label, kana, data=None):
        self.__label = label
        self.__kana = kana
        self.__data = data

    def label(self, label=None):
        if label is None:
            return self.__label
        else:
            self.__label = label

    def kana(self, kana=None):
        if kana is None:
            return self.__kana
        else:
            self.__kana = kana

    def data(self, data=None):
        if data is None:
            return self.__data
        else:
            self.__data = data

    def add(self, row):
        if self.__data is None:
            self.data(row)
        else:
            self.__data = np.vstack((self.__data, row))


def crop(img, coords, offset):
    images = list()
    images.append(img[coords[0]:coords[1], coords[2]:coords[3]])
    images.append(img[coords[0]+offset:coords[1]+offset, coords[2]:coords[3]])
    images.append(img[coords[0]-offset:coords[1]-offset, coords[2]:coords[3]])
    images.append(img[coords[0]:coords[1], coords[2]+offset:coords[3]+offset])
    images.append(img[coords[0]:coords[1], coords[2]-offset:coords[3]-offset])

    bin_images = list()
    for image in images:
        tmp = rgb2gray(image)
        bin_images.append(tmp)

    return bin_images


def save_fig(imgs, label):
    path = 'images/' + label
    cnt = 0

    if not os.path.exists(path):
        os.makedirs(path)

    for img in imgs:
        cnt += 1
        fig = plt.figure()
        fig.set_size_inches(5, 5)
        plt.imshow(img, 'gray')
        plt.axis('off')

        plt.savefig(path + '/' + label + '_' + str(cnt), dpi='figure')
        plt.close(fig)


def save(imgs, label, page):
    path = 'images/' + label
    cnt = (page-1)*5

    if not os.path.exists(path):
        os.makedirs(path)

    for img in imgs:
        cnt += 1
        imsave(path + '/' + label + '_' + str(cnt) + '.png', img_as_uint(img))


def characters():
    kana = {'a': 'あ', 'i': 'い', 'u': 'う', 'e': 'え', 'o': 'お',
            'ka': 'か', 'ki': 'き', 'ku': 'く', 'ke': 'け', 'ko': 'こ',
            'sa': 'さ', 'shi': 'し', 'su': 'す', 'se': 'せ', 'so': 'そ',
            'ta': 'た', 'chi': 'ち', 'tsu': 'つ', 'te': 'て', 'to': 'と',
            'na': 'な', 'ni': 'に', 'nu': 'ぬ', 'ne': 'ね', 'no': 'の',
            'ha': 'は', 'hi': 'ひ', 'fu': 'ふ', 'he': 'へ', 'ho': 'ほ',
            'ma': 'ま', 'mi': 'み', 'mu': 'む', 'me': 'め', 'mo': 'も',
            'ra': 'ら', 'ri': 'り', 'ru': 'る', 're': 'れ', 'ro': 'ろ',
            'ya': 'や', 'yu': 'ゆ', 'yo': 'よ', 'wa': 'わ', 'wo': 'を',
            'lowerCaseYa': 'ゃ', 'lowerCaseYu': 'ゅ', 'lowerCaseYo': 'ょ', 'n': 'ん',
            'ga': 'が', 'gi': 'ぎ', 'gu': 'ぐ', 'ge': 'げ', 'go': 'ご',
            'za': 'ざ', 'ji': 'じ', 'zu': 'ず', 'ze': 'ぜ', 'zo': 'ぞ',
            'da': 'だ', 'di': 'ぢ', 'du': 'づ', 'de': 'で', 'do': 'ど',
            'ba': 'ば', 'bi': 'び', 'bu': 'ぶ', 'be': 'べ', 'bo': 'ぼ',
            'pa': 'ぱ', 'pi': 'ぴ', 'pu': 'ぷ', 'pe': 'ぺ', 'po': 'ぽ'}

    return kana
