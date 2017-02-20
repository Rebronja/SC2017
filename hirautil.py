import numpy as np


def morph(img, threshold):
    size = 1
    block_center = [1, 1]
    ret_img = np.full((img.shape[0], img.shape[1]), False, dtype=bool)

    while not block_center[0] == img.shape[0]:

        # block = crop(size, block_center, img)
        block = img[block_center[0] - size:block_center[0] + size+1, block_center[1] - size:block_center[1] + size + 1]

        cnt = 0
        for row in range(block.shape[0]):
            for col in range(block.shape[1]):
                if block[row, col] == True:
                    cnt += 1

        if cnt > threshold:
            ret_img[block_center[0], block_center[1]] = True

        else:
            ret_img[block_center[0], block_center[1]] = img[block_center[0], block_center[1]]

        block_center = calculate(block_center, img)

    return ret_img


def calculate(center, img):
    x = center[1]
    y = center[0]
    height = img.shape[0]
    width = img.shape[1]

    if x + 1 <= width - 2:
        x += 1
    else:
        x = 1
        y += 1

    block_center = [y, x]
    return block_center


class Region:
    bbox = []
    used = False
    regions = []
    x = 0
    cands = []

    def __init__(self, reg=None):
        if reg is not None:
            self.bbox = []
            self.regions = []
            self.used = False
            self.cands = []
            self.bbox.append(reg.bbox[0])
            self.bbox.append(reg.bbox[1])
            self.bbox.append(reg.bbox[2])
            self.bbox.append(reg.bbox[3])
            self.x = reg.bbox[1]


def fill(bbox, size):
    new = bbox
    w = bbox[3] - bbox[1]
    h = bbox[2] - bbox[0]

    fill_w = round((size - w) / 2)
    new[1] = bbox[1] - fill_w
    new[3] = bbox[3] + fill_w

    fill_h = round((size - h) / 2)
    new[0] = bbox[0] - fill_h
    new[2] = bbox[2] + fill_h

    w = new[3] - new[1]
    h = new[2] - new[0]

    if w > size:
        new[3] -= size - w
    if h > size:
        new[2] -= size - h

    return new


def reg_result(ind, origin):
    lab_han = [25, 25, 25, 9]
    lab_sent = [16, 20]
    lab = [49, 49, 49]
    real_han = [10, 5, 6, 25, 10]
    real = [49, 10, 8, 5, 49, 9, 10]

    if origin == 0:
        return lab_han[ind]
    if origin == 1:
        return lab_sent[ind]
    if origin == 2:
        return lab[ind]
    if origin == 3:
        return real_han[ind]
    if origin == 4:
        return real[ind]


def nn_result_lab(ind, origin):
    lab_han = list()
    lab_han.append(['ga', 'gi', 'gu', 'ge', 'go', 'za', 'ji', 'zu', 'ze', 'zo', 'da', 'di', 'du', 'de', 'do',
                    'ba', 'bi', 'bu', 'be', 'bo', 'pa', 'pi', 'pu', 'pe', 'po'])
    lab_han.append(['ga', 'gi', 'gu', 'ge', 'go', 'za', 'ji', 'zu', 'ze', 'zo', 'da', 'di', 'du', 'de', 'do',
                    'ba', 'bi', 'bu', 'be', 'bo', 'pa', 'pi', 'pu', 'pe', 'po'])
    lab_han.append(['ga', 'gi', 'gu', 'ge', 'go', 'za', 'ji', 'zu', 'ze', 'zo', 'da', 'di', 'du', 'de', 'do',
                    'ba', 'bi', 'bu', 'be', 'bo', 'pa', 'pi', 'pu', 'pe', 'po'])
    lab_han.append(['ga', 'gi', 'gu', 'da', 'di', 'du', 'pa', 'pi', 'pu'])

    lab_sent = list()
    lab_sent.append(['a', 'na', 'ta', 'no', 'ke', 'ta', 'i', 'de', 'n', 'wa', 'wa', 'a', 'ka', 'i', 'de', 'su'])
    lab_sent.append(['bo', 'ku', 'wa', 'be', 'n', 'ki', 'lowerCaseYo', 'o', 'su', 'ru', 'ga', 'su', 'ki'
                     'de', 'wa', 'a', 'ri', 'ma', 'se', 'n'])

    lab = list()
    lab.append(['a', 'i', 'u', 'e', 'o', 'ka', 'ki', 'ku', 'ke', 'ko', 'sa', 'shi', 'su', 'se', 'so',
                'ta', 'chi', 'tsu', 'te', 'to', 'na', 'ni', 'nu', 'ne', 'no', 'ha', 'hi', 'fu', 'he', 'ho',
                'ma', 'mi', 'mu', 'me', 'mo', 'ra', 'ri', 'ru', 're', 'ro', 'ya', 'yu', 'yo', 'wa', 'wo', 'n',
                'lowerCaseYa', 'lowerCaseYu', 'lowerCaseYo'])
    lab.append(['a', 'i', 'u', 'e', 'o', 'ka', 'ki', 'ku', 'ke', 'ko', 'sa', 'shi', 'su', 'se', 'so',
                'ta', 'chi', 'tsu', 'te', 'to', 'na', 'ni', 'nu', 'ne', 'no', 'ha', 'hi', 'fu', 'he', 'ho',
                'ma', 'mi', 'mu', 'me', 'mo', 'ra', 'ri', 'ru', 're', 'ro', 'ya', 'yu', 'yo', 'wa', 'wo', 'n',
                'lowerCaseYa', 'lowerCaseYu', 'lowerCaseYo'])
    lab.append(['a', 'i', 'u', 'e', 'o', 'ka', 'ki', 'ku', 'ke', 'ko', 'sa', 'shi', 'su', 'se', 'so',
                'ta', 'chi', 'tsu', 'te', 'to', 'na', 'ni', 'nu', 'ne', 'no', 'ha', 'hi', 'fu', 'he', 'ho',
                'ma', 'mi', 'mu', 'me', 'mo', 'ra', 'ri', 'ru', 're', 'ro', 'ya', 'yu', 'yo', 'wa', 'wo', 'n',
                'lowerCaseYa', 'lowerCaseYu', 'lowerCaseYo'])

    if origin == 0:
        return lab_han[ind]
    if origin == 1:
        return lab_sent[ind]
    if origin == 3:
        return lab[ind]


def nn_result_real(ind, origin):
    real_han = list()
    real_han.append(['da', 'di', 'du', 'de', 'do', 'ba', 'bi', 'bu', 'be', 'bo'])
    real_han.append(['pa', 'pi', 'pu', 'pe', 'po'])
    real_han.append(['gu', 'ge', 'du', 'de', 'pu', 'pe'])
    real_han.append(['ga', 'gi', 'gu', 'ge', 'go', 'za', 'ji', 'zu', 'ze', 'zo', 'da', 'di', 'du', 'de', 'do',
                    'ba', 'bi', 'bu', 'be', 'bo', 'pa', 'pi', 'pu', 'pe', 'po'])
    real_han.append(['za', 'ji', 'zu', 'ze', 'zo', 'ba', 'bi', 'bu', 'be', 'bo'])

    real = list()
    real.append(['a', 'i', 'u', 'e', 'o', 'ka', 'ki', 'ku', 'ke', 'ko', 'sa', 'shi', 'su', 'se', 'so',
                'ta', 'chi', 'tsu', 'te', 'to', 'na', 'ni', 'nu', 'ne', 'no', 'ha', 'hi', 'fu', 'he', 'ho',
                'ma', 'mi', 'mu', 'me', 'mo', 'ra', 'ri', 'ru', 're', 'ro', 'ya', 'yu', 'yo', 'wa', 'wo', 'n',
                'lowerCaseYa', 'lowerCaseYu', 'lowerCaseYo'])
    real.append(['a', 'i', 'u', 'e', 'o', 'ka', 'ki', 'ku', 'ke', 'ko'])
    real.append(['se', 'so', 'ne', 'no', 'me', 'mo', 'wa', 'wo'])
    real.append(['i', 'shi', 'ni', 'mi', 'yu'])
    real.append(['a', 'i', 'u', 'e', 'o', 'ka', 'ki', 'ku', 'ke', 'ko', 'sa', 'shi', 'su', 'se', 'so',
                'ta', 'chi', 'tsu', 'te', 'to', 'na', 'ni', 'nu', 'ne', 'no', 'ha', 'hi', 'fu', 'he', 'ho',
                'ma', 'mi', 'mu', 'me', 'mo', 'ra', 'ri', 'ru', 're', 'ro', 'ya', 'yu', 'yo', 'wa', 'wo', 'n',
                'lowerCaseYa', 'lowerCaseYu', 'lowerCaseYo'])
    real.append(['na', 'ni', 'nu', 'ma', 'mi', 'mu', 'ya', 'yu', 'yo'])
    real.append(['na', 'ni', 'nu', 'ne', 'no', 'ha', 'hi', 'fu', 'he', 'ho'])

    if origin == 0:
        return real_han[ind]
    if origin == 1:
        return real[ind]
