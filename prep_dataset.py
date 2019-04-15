import os
import dataset
import skimage.io
import numpy as np

#UCITAVA SLICICE U ONAJ FAJL
def prepare_dataset():

    rel_path = 'images'
    size = 22500 # 150x150 slicica

    folders = os.listdir(rel_path)

    kana_dict = dataset.characters()
    hiragana_dataset = dataset.HiraSet('dset', size)

    for folder in folders:
        entry = dataset.HiraEntry(folder, kana_dict[folder])

        files = os.listdir(rel_path + '/' + folder)
        for file in files:
            # img = Image.open(rel_path + '/' + folder + '/' + file)

            # img = img.resize((50, 50), PIL.Image.ANTIALIAS)
            # img.save('images50x50' + '/' + folder + '/' + file)

            img = skimage.io.imread(rel_path + '/' + folder + '/' + file)
            re_img = np.reshape(img, size)
            flt_img = re_img / 65535.0

            entry.add(flt_img)

        hiragana_dataset.add(entry)


if __name__ == '__main__':
    prepare_dataset()
