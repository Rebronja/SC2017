import skimage.io
import skimage

img = skimage.io.imread('dataset_raw/page1_1.jpg')

skimage.io.imsave('test/real/test1.png', skimage.img_as_uint(img[800:1760, 100:2220]))
skimage.io.imsave('test/real/test2.png', skimage.img_as_uint(img[800:1000, 100:2220]))
skimage.io.imsave('test/real/test3.png', skimage.img_as_uint(img[1030:1760, 760:1130]))
skimage.io.imsave('test/real/test4.png', skimage.img_as_uint(img[840:1750, 320:480]))
skimage.io.imsave('test/real/test_han1.png', skimage.img_as_uint(img[2160:2310, 110:2210]))
skimage.io.imsave('test/real/test_han2.png', skimage.img_as_uint(img[2350:2500, 120:1130]))
skimage.io.imsave('test/real/test_han3.png', skimage.img_as_uint(img[1980:2500, 560:920]))

img = skimage.io.imread('dataset_raw/page2_1.jpg')

skimage.io.imsave('test/real/test5.png', skimage.img_as_uint(img[900:1810, 150:2245]))
skimage.io.imsave('test/real/test6.png', skimage.img_as_uint(img[1275:1800, 150:735]))
skimage.io.imsave('test/real/test7.png', skimage.img_as_uint(img[1275:1425, 150:2245]))
skimage.io.imsave('test/real/test_han4.png', skimage.img_as_uint(img[2020:2545, 150:2240]))
skimage.io.imsave('test/real/test_han5.png', skimage.img_as_uint(img[2020:2360, 1230:2235]))
