from skimage.io import imread, imsave
from skimage import img_as_uint

img = imread('dataset_raw/page1_1.jpg')

imsave('test/real/test1.png', img_as_uint(img[800:1760, 100:2220]))
imsave('test/real/test2.png', img_as_uint(img[800:1000, 100:2220]))
imsave('test/real/test3.png', img_as_uint(img[1030:1760, 760:1130]))
imsave('test/real/test4.png', img_as_uint(img[840:1750, 320:480]))
imsave('test/real/test_han1.png', img_as_uint(img[2160:2310, 110:2210]))
imsave('test/real/test_han2.png', img_as_uint(img[2350:2500, 120:1130]))
imsave('test/real/test_han3.png', img_as_uint(img[1980:2500, 560:920]))

img = imread('dataset_raw/page2_1.jpg')

imsave('test/real/test5.png', img_as_uint(img[900:1810, 150:2245]))
imsave('test/real/test6.png', img_as_uint(img[1275:1800, 150:735]))
imsave('test/real/test7.png', img_as_uint(img[1275:1425, 150:2245]))
imsave('test/real/test_han4.png', img_as_uint(img[2020:2545, 150:2240]))
imsave('test/real/test_han5.png', img_as_uint(img[2020:2360, 1230:2235]))