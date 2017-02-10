
# coding: utf-8

# In[673]:

import matplotlib.pyplot as plt


import numpy as np
from skimage.io import imread
img = imread('images/page1_1.jpg')
plt.imshow(img)


# In[674]:

from skimage.color import rgb2gray
from skimage.morphology import opening, closing, square, diamond, disk, erosion, dilation

from skimage.measure import label
from skimage.measure import regionprops

img_gray = rgb2gray(img)
img_tr = img_gray > 0.2
img_tr_er = erosion(img_tr,selem=disk(5))
img_tr_er = dilation(img_tr,selem=disk(0.5))

labeled_img = label(img_tr_er)
regions = regionprops(labeled_img)

plt.imshow(img_tr_er, 'gray')


# In[562]:

from skimage.io import imsave
img_crop = img_tr_er[850:1000, 100:250] 

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop,'gray')
plt.axis('off')

plt.savefig("images/train/a/a_1.png", dpi='figure')
plt.close(fig)

# In[678]:

img_crop2 = img_tr_er[860:1010, 100:250] 


plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/a/a_2.png", dpi='figure')
plt.close(fig)


# In[679]:

img_crop2 = img_tr_er[860:970, 100:220] 


plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/a/a_3.png", dpi='figure')
plt.close(fig)

# In[677]:

img_crop2 = img_tr_er[850:1000, 120:250] 


plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/a/a_4.png", dpi='figure')
plt.close(fig)

# In[670]:

fig = plt.figure()
fig.set_size_inches(26,26)
plt.imshow(img_tr_er,'gray')
plt.axis('off')

plt.savefig("images/prikaz.png", dpi='figure')
plt.close(fig)

# In[680]:

img_crop2 = img_tr_er[850:1000, 100:230] 


plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/a/a_5.png", dpi='figure')
plt.close(fig)

# In[898]:

img_crop2 = img_tr_er[860:970, 320:450]
plt.imshow(img_crop2,'gray')


# In[899]:


fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/i/i_1.png", dpi='figure')
plt.close(fig)

# In[901]:

img_crop2 = img_tr_er[860:970, 340:500]
plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/i/i_2.png", dpi='figure')
plt.close(fig)

# In[902]:

img_crop2 = img_tr_er[860:970, 300:450]
plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/i/i_3.png", dpi='figure')
plt.close(fig)

# In[905]:

img_crop2 = img_tr_er[830:960, 330:450]
plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/i/i_4.png", dpi='figure')
plt.close(fig)

# In[907]:

img_crop2 = img_tr_er[880:1000, 330:450]
plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/i/i_5.png", dpi='figure')
plt.close(fig)

# In[682]:

img_crop2 = img_tr_er[860:980, 560:640] 
plt.imshow(img_crop2,'gray')


# In[683]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/u/u_1.png", dpi='figure')
plt.close(fig)

# In[911]:

img_crop2 = img_tr_er[870:1020, 560:640] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/u/u_2.png", dpi='figure')
plt.close(fig)

# In[914]:

img_crop2 = img_tr_er[820:970, 560:640] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/u/u_3.png", dpi='figure')
plt.close(fig)

# In[919]:

img_crop2 = img_tr_er[850:980, 570:680] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/u/u_4.png", dpi='figure')
plt.close(fig)

# In[921]:

img_crop2 = img_tr_er[860:980, 530:640] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/u/u_5.png", dpi='figure')
plt.close(fig)

# In[685]:

img_crop2 = img_tr_er[850:980, 760:880] 
plt.imshow(img_crop2,'gray')


# In[686]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/e/e_1.png", dpi='figure')
plt.close(fig)

# In[929]:

img_crop2 = img_tr_er[850:980, 780:920]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/e/e_2.png", dpi='figure')
plt.close(fig)

# In[924]:

img_crop2 = img_tr_er[850:980, 730:880]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/e/e_3.png", dpi='figure')
plt.close(fig)

# In[926]:

img_crop2 = img_tr_er[820:970, 760:880] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/e/e_4.png", dpi='figure')
plt.close(fig)

# In[928]:

img_crop2 = img_tr_er[860:1020, 760:880] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/e/e_5.png", dpi='figure')
plt.close(fig)

# In[687]:

img_crop2 = img_tr_er[860:980, 980:1100]
plt.imshow(img_crop2,'gray')


# In[688]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/o/o_1.png", dpi='figure')
plt.close(fig)

# In[930]:

img_crop2 = img_tr_er[860:980, 990:1150]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/o/o_2.png", dpi='figure')
plt.close(fig)

# In[931]:

img_crop2 = img_tr_er[860:980, 940:1100]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/o/o_3.png", dpi='figure')
plt.close(fig)

# In[934]:

img_crop2 = img_tr_er[870:1020, 980:1100]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/o/o_4.png", dpi='figure')
plt.close(fig)

# In[935]:

img_crop2 = img_tr_er[830:980, 980:1100]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/o/o_5.png", dpi='figure')
plt.close(fig)

# In[689]:

img_crop2 = img_tr_er[860:980, 1190:1320] 
plt.imshow(img_crop2,'gray')


# In[690]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ka/ka_1.png", dpi='figure')
plt.close(fig)

# In[936]:

img_crop2 = img_tr_er[860:980, 1200:1360] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ka/ka_2.png", dpi='figure')
plt.close(fig)

# In[938]:

img_crop2 = img_tr_er[860:980, 1150:1320]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ka/ka_3.png", dpi='figure')
plt.close(fig)

# In[941]:

img_crop2 = img_tr_er[870:1030, 1190:1320] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ka/ka_4.png", dpi='figure')
plt.close(fig)

# In[942]:

img_crop2 = img_tr_er[820:970, 1190:1320] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ka/ka_5.png", dpi='figure')
plt.close(fig)

# In[691]:

img_crop2 = img_tr_er[860:980, 1420:1520] 
plt.imshow(img_crop2,'gray')


# In[692]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ki/ki_1.png", dpi='figure')
plt.close(fig)

# In[948]:

img_crop2 = img_tr_er[860:980, 1430:1570] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ki/ki_2.png", dpi='figure')
plt.close(fig)

# In[945]:

img_crop2 = img_tr_er[860:980, 1380:1520] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ki/ki_3.png", dpi='figure')
plt.close(fig)

# In[946]:

img_crop2 = img_tr_er[820:970, 1420:1520] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ki/ki_4.png", dpi='figure')
plt.close(fig)

# In[949]:

img_crop2 = img_tr_er[870:1030, 1420:1520] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ki/ki_5.png", dpi='figure')
plt.close(fig)

# In[693]:

img_crop2 = img_tr_er[860:970, 1640:1740] 
plt.imshow(img_crop2,'gray')


# In[694]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ku/ku_1.png", dpi='figure')
plt.close(fig)

# In[950]:

img_crop2 = img_tr_er[860:970, 1650:1790] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ku/ku_2.png", dpi='figure')
plt.close(fig)

# In[953]:

img_crop2 = img_tr_er[860:970, 1590:1730]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ku/ku_3.png", dpi='figure')
plt.close(fig)

# In[955]:

img_crop2 = img_tr_er[810:970, 1640:1730] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ku/ku_4.png", dpi='figure')
plt.close(fig)

# In[956]:

img_crop2 = img_tr_er[870:1020, 1640:1730] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ku/ku_5.png", dpi='figure')
plt.close(fig)

# In[696]:

img_crop2 = img_tr_er[860:970, 1850:1970] 
plt.imshow(img_crop2,'gray')


# In[697]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ke/ke_1.png", dpi='figure')
plt.close(fig)

# In[957]:

img_crop2 = img_tr_er[860:970, 1860:2020] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ke/ke_2.png", dpi='figure')
plt.close(fig)

# In[959]:

img_crop2 = img_tr_er[860:970, 1800:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ke/ke_3.png", dpi='figure')
plt.close(fig)

# In[960]:

img_crop2 = img_tr_er[810:960, 1850:1970] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ke/ke_4.png", dpi='figure')
plt.close(fig)

# In[961]:

img_crop2 = img_tr_er[870:1020, 1850:1970] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ke/ke_5.png", dpi='figure')
plt.close(fig)

# In[698]:

img_crop2 = img_tr_er[860:970, 2080:2180] 
plt.imshow(img_crop2,'gray')


# In[699]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ko/ko_1.png", dpi='figure')
plt.close(fig)

# In[962]:

img_crop2 = img_tr_er[860:970, 2090:2230] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ko/ko_2.png", dpi='figure')
plt.close(fig)

# In[965]:

img_crop2 = img_tr_er[860:970, 2030:2180]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ko/ko_3.png", dpi='figure')
plt.close(fig)

# In[966]:

img_crop2 = img_tr_er[810:960, 2080:2180] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ko/ko_4.png", dpi='figure')
plt.close(fig)

# In[967]:

img_crop2 = img_tr_er[870:1020, 2080:2180] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ko/ko_5.png", dpi='figure')
plt.close(fig)

# In[701]:

img_crop2 = img_tr_er[1050:1170, 340:450]
plt.imshow(img_crop2,'gray')


# In[702]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/shi/shi_1.png", dpi='figure')
plt.close(fig)

# In[968]:

img_crop2 = img_tr_er[1050:1170, 350:500]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/shi/shi_2.png", dpi='figure')
plt.close(fig)

# In[969]:

img_crop2 = img_tr_er[1050:1170, 290:450]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/shi/shi_3.png", dpi='figure')
plt.close(fig)

# In[970]:

img_crop2 = img_tr_er[1010:1160, 340:450]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/shi/shi_4.png", dpi='figure')
plt.close(fig)

# In[971]:

img_crop2 = img_tr_er[1060:1220, 340:450]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/shi/shi_5.png", dpi='figure')
plt.close(fig)

# In[703]:

img_crop2 = img_tr_er[1050:1170, 120:220] 
plt.imshow(img_crop2,'gray')


# In[704]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/sa/sa_1.png", dpi='figure')
plt.close(fig)

# In[972]:

img_crop2 = img_tr_er[1050:1170, 130:270] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/sa/sa_2.png", dpi='figure')
plt.close(fig)

# In[973]:

img_crop2 = img_tr_er[1050:1170, 70:210]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/sa/sa_3.png", dpi='figure')
plt.close(fig)

# In[974]:

img_crop2 = img_tr_er[1000:1160, 120:220] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/sa/sa_4.png", dpi='figure')
plt.close(fig)

# In[975]:

img_crop2 = img_tr_er[1060:1220, 120:220] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/sa/sa_5.png", dpi='figure')
plt.close(fig)

# In[707]:

img_crop2 = img_tr_er[1050:1170, 540:670] 
plt.imshow(img_crop2,'gray')


# In[708]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/su/su_1.png", dpi='figure')
plt.close(fig)

# In[976]:

img_crop2 = img_tr_er[1050:1170, 550:720] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/su/su_2.png", dpi='figure')
plt.close(fig)

# In[977]:

img_crop2 = img_tr_er[1050:1170, 490:670]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/su/su_3.png", dpi='figure')
plt.close(fig)

# In[978]:

img_crop2 = img_tr_er[1060:1220, 540:670] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/su/su_4.png", dpi='figure')
plt.close(fig)

# In[980]:

img_crop2 = img_tr_er[1000:1160, 540:670] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/su/su_5.png", dpi='figure')
plt.close(fig)

# In[709]:

img_crop2 = img_tr_er[1050:1170, 760:890]
plt.imshow(img_crop2,'gray')


# In[710]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/se/se_1.png", dpi='figure')
plt.close(fig)

# In[981]:

img_crop2 = img_tr_er[1050:1170, 770:940]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/se/se_2.png", dpi='figure')
plt.close(fig)

# In[982]:

img_crop2 = img_tr_er[1050:1170, 710:890]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/se/se_3.png", dpi='figure')
plt.close(fig)

# In[983]:

img_crop2 = img_tr_er[1000:1160, 760:890]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/se/se_4.png", dpi='figure')
plt.close(fig)

# In[984]:

img_crop2 = img_tr_er[1060:1220, 760:890]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/se/se_5.png", dpi='figure')
plt.close(fig)

# In[711]:

img_crop2 = img_tr_er[1050:1170, 980:1100]
plt.imshow(img_crop2,'gray')


# In[712]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/so/so_1.png", dpi='figure')
plt.close(fig)

# In[985]:

img_crop2 = img_tr_er[1050:1170, 990:1150]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/so/so_2.png", dpi='figure')
plt.close(fig)

# In[986]:

img_crop2 = img_tr_er[1050:1170, 930:1100]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/so/so_3.png", dpi='figure')
plt.close(fig)

# In[987]:

img_crop2 = img_tr_er[1000:1160, 980:1100]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/so/so_4.png", dpi='figure')
plt.close(fig)

# In[988]:

img_crop2 = img_tr_er[1060:1220, 980:1100]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/so/so_5.png", dpi='figure')
plt.close(fig)

# In[713]:

img_crop2 = img_tr_er[1050:1170, 1190:1320]
plt.imshow(img_crop2,'gray')


# In[714]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ta/ta_1.png", dpi='figure')
plt.close(fig)

# In[989]:

img_crop2 = img_tr_er[1050:1170, 1200:1370]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ta/ta_2.png", dpi='figure')
plt.close(fig)

# In[992]:

img_crop2 = img_tr_er[1050:1170, 1140:1320]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ta/ta_3.png", dpi='figure')
plt.close(fig)

# In[993]:

img_crop2 = img_tr_er[1000:1160, 1190:1320]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ta/ta_4.png", dpi='figure')
plt.close(fig)

# In[994]:

img_crop2 = img_tr_er[1060:1220, 1190:1320]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ta/ta_5.png", dpi='figure')
plt.close(fig)

# In[715]:

img_crop2 = img_tr_er[1050:1170, 1420:1530] 
plt.imshow(img_crop2,'gray')


# In[716]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/chi/chi_1.png", dpi='figure')
plt.close(fig)

# In[995]:

img_crop2 = img_tr_er[1050:1170, 1410:1580] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/chi/chi_2.png", dpi='figure')
plt.close(fig)

# In[996]:

img_crop2 = img_tr_er[1050:1170, 1370:1530]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/chi/chi_3.png", dpi='figure')
plt.close(fig)

# In[997]:

img_crop2 = img_tr_er[1000:1160, 1420:1530] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/chi/chi_4.png", dpi='figure')
plt.close(fig)

# In[998]:

img_crop2 = img_tr_er[1060:1220, 1420:1530] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/chi/chi_5.png", dpi='figure')
plt.close(fig)

# In[717]:

img_crop2 = img_tr_er[1050:1170, 1630:1750] 
plt.imshow(img_crop2,'gray')


# In[718]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/tsu/tsu_1.png", dpi='figure')
plt.close(fig)

# In[999]:

img_crop2 = img_tr_er[1050:1170, 1640:1800] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/tsu/tsu_2.png", dpi='figure')
plt.close(fig)

# In[1000]:

img_crop2 = img_tr_er[1050:1170, 1580:1750]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/tsu/tsu_3.png", dpi='figure')
plt.close(fig)

# In[1001]:

img_crop2 = img_tr_er[1000:1160, 1630:1750] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/tsu/tsu_4.png", dpi='figure')
plt.close(fig)

# In[1002]:

img_crop2 = img_tr_er[1060:1220, 1630:1750] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/tsu/tsu_5.png", dpi='figure')
plt.close(fig)

# In[719]:

img_crop2 = img_tr_er[1050:1170, 1860:1970]
plt.imshow(img_crop2,'gray')


# In[720]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/te/te_1.png", dpi='figure')
plt.close(fig)

# In[1003]:

img_crop2 = img_tr_er[1050:1170, 1810:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/te/te_2.png", dpi='figure')
plt.close(fig)

# In[1010]:

img_crop2 = img_tr_er[1050:1170, 1860:2020]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/te/te_3.png", dpi='figure')
plt.close(fig)

# In[1011]:

img_crop2 = img_tr_er[1000:1160, 1860:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/te/te_4.png", dpi='figure')
plt.close(fig)

# In[1012]:

img_crop2 = img_tr_er[1060:1220, 1860:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/te/te_5.png", dpi='figure')
plt.close(fig)

# In[721]:

img_crop2 = img_tr_er[1050:1170, 2080:2180]
plt.imshow(img_crop2,'gray')


# In[722]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/to/to_1.png", dpi='figure')
plt.close(fig)

# In[1013]:

img_crop2 = img_tr_er[1050:1170, 2090:2230]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/to/to_2.png", dpi='figure')
plt.close(fig)

# In[1014]:

img_crop2 = img_tr_er[1050:1170, 2030:2180]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/to/to_3.png", dpi='figure')
plt.close(fig)

# In[1015]:

img_crop2 = img_tr_er[1000:1160, 2080:2180]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/to/to_4.png", dpi='figure')
plt.close(fig)

# In[1016]:

img_crop2 = img_tr_er[1060:1220, 2080:2180]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/to/to_5.png", dpi='figure')
plt.close(fig)

# In[723]:

img_crop2 = img_tr_er[1240:1350, 110:230]
plt.imshow(img_crop2,'gray')


# In[724]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/na/na_1.png", dpi='figure')
plt.close(fig)

# In[1017]:

img_crop2 = img_tr_er[1240:1350, 120:280]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/na/na_2.png", dpi='figure')
plt.close(fig)

# In[1019]:

img_crop2 = img_tr_er[1240:1350, 70:230]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/na/na_3.png", dpi='figure')
plt.close(fig)

# In[1020]:

img_crop2 = img_tr_er[1250:1400, 110:230]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/na/na_4.png", dpi='figure')
plt.close(fig)

# In[1022]:

img_crop2 = img_tr_er[1190:1350, 110:230]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/na/na_5.png", dpi='figure')
plt.close(fig)

# In[725]:

img_crop2 = img_tr_er[1240:1350, 330:440] 
plt.imshow(img_crop2,'gray')


# In[726]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ni/ni_1.png", dpi='figure')
plt.close(fig)

# In[1023]:

img_crop2 = img_tr_er[1240:1350, 340:490] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ni/ni_2.png", dpi='figure')
plt.close(fig)

# In[1024]:

img_crop2 = img_tr_er[1240:1350, 280:440]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ni/ni_3.png", dpi='figure')
plt.close(fig)

# In[1026]:

img_crop2 = img_tr_er[1190:1350, 330:440] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ni/ni_4.png", dpi='figure')
plt.close(fig)

# In[1027]:

img_crop2 = img_tr_er[1250:1400, 330:440] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ni/ni_5.png", dpi='figure')
plt.close(fig)

# In[727]:

img_crop2 = img_tr_er[1240:1350, 540:670] 
plt.imshow(img_crop2,'gray')


# In[728]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/nu/nu_1.png", dpi='figure')
plt.close(fig)

# In[1028]:

img_crop2 = img_tr_er[1240:1350, 550:720] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/nu/nu_2.png", dpi='figure')
plt.close(fig)

# In[1029]:

img_crop2 = img_tr_er[1240:1350, 490:670]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/nu/nu_3.png", dpi='figure')
plt.close(fig)

# In[1030]:

img_crop2 = img_tr_er[1190:1340, 540:670] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/nu/nu_4.png", dpi='figure')
plt.close(fig)

# In[1031]:

img_crop2 = img_tr_er[1250:1390, 540:670] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/nu/nu_5.png", dpi='figure')
plt.close(fig)

# In[729]:

img_crop2 = img_tr_er[1240:1350, 760:890] 
plt.imshow(img_crop2,'gray')


# In[730]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ne/ne_1.png", dpi='figure')
plt.close(fig)

# In[1032]:

img_crop2 = img_tr_er[1250:1390, 760:890] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ne/ne_2.png", dpi='figure')
plt.close(fig)

# In[1034]:

img_crop2 = img_tr_er[1190:1350, 760:890] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ne/ne_3.png", dpi='figure')
plt.close(fig)

# In[1036]:

img_crop2 = img_tr_er[1240:1350, 770:940] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ne/ne_4.png", dpi='figure')
plt.close(fig)

# In[1037]:

img_crop2 = img_tr_er[1240:1350, 710:890]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ne/ne_5.png", dpi='figure')
plt.close(fig)

# In[1042]:

img_crop2 = img_tr_er[1260:1350, 980:1100] 
plt.imshow(img_crop2,'gray')


# In[1043]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/no/no_1.png", dpi='figure')
plt.close(fig)

# In[1044]:

img_crop2 = img_tr_er[1210:1340, 980:1100] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/no/no_2.png", dpi='figure')
plt.close(fig)

# In[1040]:

img_crop2 = img_tr_er[1260:1390, 980:1100] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/no/no_3.png", dpi='figure')
plt.close(fig)

# In[1045]:

img_crop2 = img_tr_er[1260:1350, 990:1150] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/no/no_4.png", dpi='figure')
plt.close(fig)

# In[1046]:

img_crop2 = img_tr_er[1260:1350, 930:1100]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/no/no_5.png", dpi='figure')
plt.close(fig)

# In[1049]:

img_crop2 = img_tr_er[1250:1360, 1180:1330] 
plt.imshow(img_crop2,'gray')


# In[1050]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ha/ha_1.png", dpi='figure')
plt.close(fig)

# In[1048]:

img_crop2 = img_tr_er[1250:1360, 1130:1320] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ha/ha_2.png", dpi='figure')
plt.close(fig)

# In[1051]:

img_crop2 = img_tr_er[1250:1360, 1190:1380]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ha/ha_3.png", dpi='figure')
plt.close(fig)

# In[1053]:

img_crop2 = img_tr_er[1200:1350, 1180:1330] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ha/ha_4.png", dpi='figure')
plt.close(fig)

# In[1054]:

img_crop2 = img_tr_er[1260:1410, 1180:1330] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ha/ha_5.png", dpi='figure')
plt.close(fig)

# In[1055]:

img_crop2 = img_tr_er[1250:1360, 1410:1550]
plt.imshow(img_crop2,'gray')


# In[1056]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/hi/hi_1.png", dpi='figure')
plt.close(fig)

# In[1057]:

img_crop2 = img_tr_er[1250:1360, 1420:1590] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/hi/hi_2.png", dpi='figure')
plt.close(fig)

# In[1058]:

img_crop2 = img_tr_er[1250:1360, 1360:1550]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/hi/hi_3.png", dpi='figure')
plt.close(fig)

# In[1059]:

img_crop2 = img_tr_er[1200:1350, 1410:1550]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/hi/hi_4.png", dpi='figure')
plt.close(fig)

# In[1060]:

img_crop2 = img_tr_er[1260:1410, 1410:1550]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/hi/hi_5.png", dpi='figure')
plt.close(fig)

# In[740]:

img_crop2 = img_tr_er[1240:1360, 1630:1750] 
plt.imshow(img_crop2,'gray')


# In[741]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/fu/fu_1.png", dpi='figure')
plt.close(fig)

# In[1061]:

img_crop2 = img_tr_er[1240:1360, 1580:1750]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/fu/fu_2.png", dpi='figure')
plt.close(fig)

# In[1062]:

img_crop2 = img_tr_er[1240:1360, 1640:1800] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/fu/fu_3.png", dpi='figure')
plt.close(fig)

# In[1063]:

img_crop2 = img_tr_er[1190:1350, 1630:1750] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/fu/fu_4.png", dpi='figure')
plt.close(fig)

# In[1064]:

img_crop2 = img_tr_er[1250:1410, 1630:1750] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/fu/fu_5.png", dpi='figure')
plt.close(fig)

# In[1068]:

img_crop2 = img_tr_er[1270:1330, 1860:1970]
plt.imshow(img_crop2,'gray')


# In[1070]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/he/he_1.png", dpi='figure')
plt.close(fig)

# In[1072]:

img_crop2 = img_tr_er[1270:1330, 1810:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/he/he_2.png", dpi='figure')
plt.close(fig)

# In[1074]:

img_crop2 = img_tr_er[1270:1330, 1860:2010] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/he/he_3.png", dpi='figure')
plt.close(fig)

# In[1075]:

img_crop2 = img_tr_er[1220:1320, 1860:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/he/he_4.png", dpi='figure')
plt.close(fig)

# In[1076]:

img_crop2 = img_tr_er[1280:1380, 1860:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/he/he_5.png", dpi='figure')
plt.close(fig)

# In[745]:

img_crop2 = img_tr_er[1240:1360, 2060:2200]
plt.imshow(img_crop2,'gray')


# In[746]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ho/ho_1.png", dpi='figure')
plt.close(fig)

# In[1079]:

img_crop2 = img_tr_er[1250:1350, 2080:2250]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ho/ho_2.png", dpi='figure')
plt.close(fig)

# In[1081]:

img_crop2 = img_tr_er[1250:1350, 2020:2200]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ho/ho_3.png", dpi='figure')
plt.close(fig)

# In[1082]:

img_crop2 = img_tr_er[1200:1340, 2070:2200]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ho/ho_4.png", dpi='figure')
plt.close(fig)

# In[1084]:

img_crop2 = img_tr_er[1250:1400, 2070:2200]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ho/ho_5.png", dpi='figure')
plt.close(fig)

# In[748]:

img_crop2 = img_tr_er[1430:1550, 120:230]
plt.imshow(img_crop2,'gray')


# In[749]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ma/ma_1.png", dpi='figure')
plt.close(fig)

# In[1085]:

img_crop2 = img_tr_er[1430:1550, 70:230]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ma/ma_2.png", dpi='figure')
plt.close(fig)

# In[1087]:

img_crop2 = img_tr_er[1430:1550, 120:280]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ma/ma_3.png", dpi='figure')
plt.close(fig)

# In[1088]:

img_crop2 = img_tr_er[1380:1540, 120:230]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ma/ma_4.png", dpi='figure')
plt.close(fig)

# In[1089]:

img_crop2 = img_tr_er[1440:1600, 120:230]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ma/ma_5.png", dpi='figure')
plt.close(fig)

# In[751]:

img_crop2 = img_tr_er[1430:1550, 330:460]
plt.imshow(img_crop2,'gray')


# In[752]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mi/mi_1.png", dpi='figure')
plt.close(fig)

# In[1090]:

img_crop2 = img_tr_er[1430:1550, 340:510]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mi/mi_2.png", dpi='figure')
plt.close(fig)

# In[1091]:

img_crop2 = img_tr_er[1430:1550, 280:460]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mi/mi_3.png", dpi='figure')
plt.close(fig)

# In[1092]:

img_crop2 = img_tr_er[1440:1600, 330:460]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mi/mi_4.png", dpi='figure')
plt.close(fig)

# In[1093]:

img_crop2 = img_tr_er[1380:1540, 330:460]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mi/mi_5.png", dpi='figure')
plt.close(fig)

# In[754]:

img_crop2 = img_tr_er[1430:1550, 550:670] 
plt.imshow(img_crop2,'gray')


# In[755]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mu/mu_1.png", dpi='figure')
plt.close(fig)

# In[1094]:

img_crop2 = img_tr_er[1430:1550, 500:670]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mu/mu_2.png", dpi='figure')
plt.close(fig)

# In[1095]:

img_crop2 = img_tr_er[1430:1550, 560:720] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mu/mu_3.png", dpi='figure')
plt.close(fig)

# In[1096]:

img_crop2 = img_tr_er[1440:1600, 550:670] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mu/mu_4.png", dpi='figure')
plt.close(fig)

# In[1097]:

img_crop2 = img_tr_er[1380:1540, 550:670] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mu/mu_5.png", dpi='figure')
plt.close(fig)

# In[757]:

img_crop2 = img_tr_er[1430:1550, 770:880] 
plt.imshow(img_crop2,'gray')


# In[758]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/me/me_1.png", dpi='figure')
plt.close(fig)

# In[1098]:

img_crop2 = img_tr_er[1430:1550, 780:930] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/me/me_2.png", dpi='figure')
plt.close(fig)

# In[1099]:

img_crop2 = img_tr_er[1430:1550, 720:880]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/me/me_3.png", dpi='figure')
plt.close(fig)

# In[1100]:

img_crop2 = img_tr_er[1440:1600, 770:880] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/me/me_4.png", dpi='figure')
plt.close(fig)

# In[1101]:

img_crop2 = img_tr_er[1380:1540, 770:880] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/me/me_5.png", dpi='figure')
plt.close(fig)

# In[761]:

img_crop2 = img_tr_er[1440:1550, 1000:1090] 
plt.imshow(img_crop2,'gray')


# In[762]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mo/mo_1.png", dpi='figure')
plt.close(fig)

# In[1102]:

img_crop2 = img_tr_er[1440:1550, 950:1090]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mo/mo_2.png", dpi='figure')
plt.close(fig)

# In[1104]:

img_crop2 = img_tr_er[1440:1550, 1000:1140] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mo/mo_3.png", dpi='figure')
plt.close(fig)

# In[1105]:

img_crop2 = img_tr_er[1450:1600, 1000:1090] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mo/mo_4.png", dpi='figure')
plt.close(fig)

# In[1106]:

img_crop2 = img_tr_er[1390:1540, 1000:1090] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mo/mo_5.png", dpi='figure')
plt.close(fig)

# In[765]:

img_crop2 = img_tr_er[1440:1550, 1220:1310]
plt.imshow(img_crop2,'gray')


# In[766]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ra/ra_1.png", dpi='figure')
plt.close(fig)

# In[1108]:

img_crop2 = img_tr_er[1440:1550, 1170:1310]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ra/ra_2.png", dpi='figure')
plt.close(fig)

# In[1110]:

img_crop2 = img_tr_er[1440:1550, 1220:1360]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ra/ra_3.png", dpi='figure')
plt.close(fig)

# In[1112]:

img_crop2 = img_tr_er[1440:1600, 1220:1310]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ra/ra_4.png", dpi='figure')
plt.close(fig)

# In[1113]:

img_crop2 = img_tr_er[1390:1540, 1220:1310]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ra/ra_5.png", dpi='figure')
plt.close(fig)

# In[1116]:

img_crop2 = img_tr_er[1440:1550, 1430:1510] 
plt.imshow(img_crop2,'gray')


# In[1117]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ri/ri_1.png", dpi='figure')
plt.close(fig)

# In[1115]:

img_crop2 = img_tr_er[1440:1550, 1380:1510]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ri/ri_2.png", dpi='figure')
plt.close(fig)

# In[1118]:

img_crop2 = img_tr_er[1440:1550, 1440:1560] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ri/ri_3.png", dpi='figure')
plt.close(fig)

# In[1120]:

img_crop2 = img_tr_er[1450:1600, 1430:1510] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ri/ri_4.png", dpi='figure')
plt.close(fig)

# In[1121]:

img_crop2 = img_tr_er[1390:1540, 1430:1510] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ri/ri_5.png", dpi='figure')
plt.close(fig)

# In[774]:

img_crop2 = img_tr_er[1430:1550, 1650:1740]
plt.imshow(img_crop2,'gray')


# In[775]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ru/ru_1.png", dpi='figure')
plt.close(fig)

# In[1123]:

img_crop2 = img_tr_er[1430:1550, 1650:1790]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ru/ru_2.png", dpi='figure')
plt.close(fig)

# In[1125]:

img_crop2 = img_tr_er[1430:1550, 1600:1740]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ru/ru_3.png", dpi='figure')
plt.close(fig)

# In[1126]:

img_crop2 = img_tr_er[1440:1600, 1650:1740]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ru/ru_4.png", dpi='figure')
plt.close(fig)

# In[1127]:

img_crop2 = img_tr_er[1380:1540, 1650:1740]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ru/ru_5.png", dpi='figure')
plt.close(fig)

# In[1131]:

img_crop2 = img_tr_er[1430:1550, 1850:1970]
plt.imshow(img_crop2,'gray')


# In[1132]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/re/re_1.png", dpi='figure')
plt.close(fig)

# In[1129]:

img_crop2 = img_tr_er[1430:1550, 1850:2020]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/re/re_2.png", dpi='figure')
plt.close(fig)

# In[1130]:

img_crop2 = img_tr_er[1430:1550, 1800:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/re/re_3.png", dpi='figure')
plt.close(fig)

# In[1133]:

img_crop2 = img_tr_er[1380:1540, 1850:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/re/re_4.png", dpi='figure')
plt.close(fig)

# In[1134]:

img_crop2 = img_tr_er[1440:1600, 1850:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/re/re_5.png", dpi='figure')
plt.close(fig)

# In[780]:

img_crop2 = img_tr_er[1430:1550, 2080:2180] 
plt.imshow(img_crop2,'gray')


# In[781]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ro/ro_1.png", dpi='figure')
plt.close(fig)

# In[1135]:

img_crop2 = img_tr_er[1430:1550, 2030:2180]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ro/ro_2.png", dpi='figure')
plt.close(fig)

# In[1137]:

img_crop2 = img_tr_er[1430:1550, 2080:2230]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ro/ro_3.png", dpi='figure')
plt.close(fig)

# In[1138]:

img_crop2 = img_tr_er[1380:1540, 2080:2180] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ro/ro_4.png", dpi='figure')
plt.close(fig)

# In[1139]:

img_crop2 = img_tr_er[1440:1590, 2080:2180] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ro/ro_5.png", dpi='figure')
plt.close(fig)

# In[784]:

img_crop2 = img_tr_er[1620:1740, 120:240] 
plt.imshow(img_crop2,'gray')


# In[785]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ya/ya_1.png", dpi='figure')
plt.close(fig)

# In[1140]:

img_crop2 = img_tr_er[1620:1740, 70:240]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ya/ya_2.png", dpi='figure')
plt.close(fig)

# In[1142]:

img_crop2 = img_tr_er[1620:1740, 120:290] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ya/ya_3.png", dpi='figure')
plt.close(fig)

# In[1143]:

img_crop2 = img_tr_er[1630:1790, 120:240] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ya/ya_4.png", dpi='figure')
plt.close(fig)

# In[1144]:

img_crop2 = img_tr_er[1570:1730, 120:240] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ya/ya_5.png", dpi='figure')
plt.close(fig)

# In[787]:

img_crop2 = img_tr_er[1620:1740, 340:450] 
plt.imshow(img_crop2,'gray')


# In[788]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yu/yu_1.png", dpi='figure')
plt.close(fig)

# In[1145]:

img_crop2 = img_tr_er[1620:1740, 290:450]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yu/yu_2.png", dpi='figure')
plt.close(fig)

# In[1146]:

img_crop2 = img_tr_er[1620:1740, 350:500] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yu/yu_3.png", dpi='figure')
plt.close(fig)

# In[1147]:

img_crop2 = img_tr_er[1570:1730, 340:450] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yu/yu_4.png", dpi='figure')
plt.close(fig)

# In[1148]:

img_crop2 = img_tr_er[1630:1790, 340:450] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yu/yu_5.png", dpi='figure')
plt.close(fig)

# In[790]:

img_crop2 = img_tr_er[1620:1740, 570:660] 
plt.imshow(img_crop2,'gray')


# In[791]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yo/yo_1.png", dpi='figure')
plt.close(fig)

# In[1150]:

img_crop2 = img_tr_er[1620:1740, 570:710] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yo/yo_2.png", dpi='figure')
plt.close(fig)

# In[1151]:

img_crop2 = img_tr_er[1620:1740, 520:660]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yo/yo_3.png", dpi='figure')
plt.close(fig)

# In[1152]:

img_crop2 = img_tr_er[1630:1790, 570:660] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yo/yo_4.png", dpi='figure')
plt.close(fig)

# In[1154]:

img_crop2 = img_tr_er[1570:1730, 570:660] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yo/yo_5.png", dpi='figure')
plt.close(fig)

# In[792]:

img_crop2 = img_tr_er[1620:1740, 770:890]
plt.imshow(img_crop2,'gray')


# In[793]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wa/wa_1.png", dpi='figure')
plt.close(fig)

# In[1156]:

img_crop2 = img_tr_er[1620:1740, 720:890]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wa/wa_2.png", dpi='figure')
plt.close(fig)

# In[1157]:

img_crop2 = img_tr_er[1620:1740, 780:940]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wa/wa_3.png", dpi='figure')
plt.close(fig)

# In[1158]:

img_crop2 = img_tr_er[1570:1730, 770:890]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wa/wa_4.png", dpi='figure')
plt.close(fig)

# In[1159]:

img_crop2 = img_tr_er[1630:1790, 770:890]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wa/wa_5.png", dpi='figure')
plt.close(fig)

# In[795]:

img_crop2 = img_tr_er[1630:1730, 1000:1100]
plt.imshow(img_crop2,'gray')


# In[796]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wo/wo_1.png", dpi='figure')
plt.close(fig)

# In[1161]:

img_crop2 = img_tr_er[1630:1730, 950:1100]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wo/wo_2.png", dpi='figure')
plt.close(fig)

# In[1163]:

img_crop2 = img_tr_er[1630:1730, 1000:1150]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wo/wo_3.png", dpi='figure')
plt.close(fig)

# In[1164]:

img_crop2 = img_tr_er[1580:1730, 1000:1100]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wo/wo_4.png", dpi='figure')
plt.close(fig)

# In[1165]:

img_crop2 = img_tr_er[1630:1780, 1000:1100]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wo/wo_5.png", dpi='figure')
plt.close(fig)

# In[797]:

img_crop2 = img_tr_er[1620:1740, 1210:1320]
plt.imshow(img_crop2,'gray')


# In[798]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/n/n_1.png", dpi='figure')
plt.close(fig)

# In[1167]:

img_crop2 = img_tr_er[1620:1740, 1160:1320]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/n/n_2.png", dpi='figure')
plt.close(fig)

# In[1169]:

img_crop2 = img_tr_er[1620:1740, 1210:1370]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/n/n_3.png", dpi='figure')
plt.close(fig)

# In[1170]:

img_crop2 = img_tr_er[1570:1730, 1210:1320]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/n/n_4.png", dpi='figure')
plt.close(fig)

# In[1171]:

img_crop2 = img_tr_er[1630:1790, 1210:1320]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/n/n_5.png", dpi='figure')
plt.close(fig)

# In[1172]:

img_crop2 = img_tr_er[1640:1740, 1420:1530]
plt.imshow(img_crop2,'gray')


# In[1173]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYa/lowerCaseYa_1.png", dpi='figure')
plt.close(fig)

# In[1175]:

img_crop2 = img_tr_er[1640:1740, 1370:1530]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYa/lowerCaseYa_2.png", dpi='figure')
plt.close(fig)

# In[1176]:

img_crop2 = img_tr_er[1640:1740, 1430:1570] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYa/lowerCaseYa_3.png", dpi='figure')
plt.close(fig)

# In[1177]:

img_crop2 = img_tr_er[1590:1730, 1420:1530]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYa/lowerCaseYa_4.png", dpi='figure')
plt.close(fig)

# In[1179]:

img_crop2 = img_tr_er[1650:1790, 1420:1530]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYa/lowerCaseYa_5.png", dpi='figure')
plt.close(fig)

# In[1185]:

img_crop2 = img_tr_er[1640:1740, 1640:1740] 
plt.imshow(img_crop2,'gray')


# In[1186]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYu/lowerCaseYu_1.png", dpi='figure')
plt.close(fig)

# In[1181]:

img_crop2 = img_tr_er[1640:1740, 1650:1790] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYu/lowerCaseYu_2.png", dpi='figure')
plt.close(fig)

# In[1182]:

img_crop2 = img_tr_er[1640:1740, 1590:1740]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYu/lowerCaseYu_3.png", dpi='figure')
plt.close(fig)

# In[1183]:

img_crop2 = img_tr_er[1590:1730, 1640:1740] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYu/lowerCaseYu_4.png", dpi='figure')
plt.close(fig)

# In[1184]:

img_crop2 = img_tr_er[1650:1790, 1640:1740] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYu/lowerCaseYu_5.png", dpi='figure')
plt.close(fig)

# In[1187]:

img_crop2 = img_tr_er[1650:1730, 1870:1950]
plt.imshow(img_crop2,'gray')


# In[1188]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYo/lowerCaseYo_1.png", dpi='figure')
plt.close(fig)

# In[1191]:

img_crop2 = img_tr_er[1650:1730, 1820:1950]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYo/lowerCaseYo_2.png", dpi='figure')
plt.close(fig)

# In[1192]:

img_crop2 = img_tr_er[1650:1730, 1870:2000]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYo/lowerCaseYo_3.png", dpi='figure')
plt.close(fig)

# In[1193]:

img_crop2 = img_tr_er[1600:1730, 1870:1950]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYo/lowerCaseYo_4.png", dpi='figure')
plt.close(fig)

# In[1194]:

img_crop2 = img_tr_er[1650:1780, 1870:1950]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYo/lowerCaseYo_5.png", dpi='figure')
plt.close(fig)

# In[818]:

img_crop2 = img_tr_er[1990:2100, 120:240]
plt.imshow(img_crop2,'gray')


# In[819]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ga/ga_1.png", dpi='figure')
plt.close(fig)

# In[1195]:

img_crop2 = img_tr_er[1990:2100, 70:240]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ga/ga_2.png", dpi='figure')
plt.close(fig)

# In[1196]:

img_crop2 = img_tr_er[1990:2100, 130:280] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ga/ga_3.png", dpi='figure')
plt.close(fig)

# In[1197]:

img_crop2 = img_tr_er[2000:2150, 120:240]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ga/ga_4.png", dpi='figure')
plt.close(fig)

# In[1198]:

img_crop2 = img_tr_er[1940:2100, 120:240]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ga/ga_5.png", dpi='figure')
plt.close(fig)

# In[820]:

img_crop2 = img_tr_er[1990:2110, 340:460]
plt.imshow(img_crop2,'gray')


# In[821]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gi/gi_1.png", dpi='figure')
plt.close(fig)

# In[1200]:

img_crop2 = img_tr_er[1990:2110, 290:460]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gi/gi_2.png", dpi='figure')
plt.close(fig)

# In[1201]:

img_crop2 = img_tr_er[1990:2110, 350:500] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gi/gi_3.png", dpi='figure')
plt.close(fig)

# In[1202]:

img_crop2 = img_tr_er[2000:2160, 340:460]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gi/gi_4.png", dpi='figure')
plt.close(fig)

# In[1203]:

img_crop2 = img_tr_er[1940:2100, 340:450] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gi/gi_5.png", dpi='figure')
plt.close(fig)

# In[822]:

img_crop2 = img_tr_er[1990:2110, 560:670] 
plt.imshow(img_crop2,'gray')


# In[823]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gu/gu_1.png", dpi='figure')
plt.close(fig)

# In[1204]:

img_crop2 = img_tr_er[1990:2110, 510:670]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gu/gu_2.png", dpi='figure')
plt.close(fig)

# In[1205]:

img_crop2 = img_tr_er[1990:2110, 570:720] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gu/gu_3.png", dpi='figure')
plt.close(fig)

# In[1206]:

img_crop2 = img_tr_er[1940:2100, 560:670] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gu/gu_4.png", dpi='figure')
plt.close(fig)

# In[1207]:

img_crop2 = img_tr_er[2000:2160, 560:670] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gu/gu_5.png", dpi='figure')
plt.close(fig)

# In[825]:

img_crop2 = img_tr_er[1990:2110, 770:900]
plt.imshow(img_crop2,'gray')


# In[826]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ge/ge_1.png", dpi='figure')
plt.close(fig)

# In[1208]:

img_crop2 = img_tr_er[1990:2110, 720:900]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ge/ge_2.png", dpi='figure')
plt.close(fig)

# In[1209]:

img_crop2 = img_tr_er[1990:2110, 780:940]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ge/ge_3.png", dpi='figure')
plt.close(fig)

# In[1210]:

img_crop2 = img_tr_er[2000:2160, 770:900]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ge/ge_4.png", dpi='figure')
plt.close(fig)

# In[1211]:

img_crop2 = img_tr_er[1940:2100, 770:900]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ge/ge_5.png", dpi='figure')
plt.close(fig)

# In[829]:

img_crop2 = img_tr_er[1990:2110, 990:1110] 
plt.imshow(img_crop2,'gray')


# In[830]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/go/go_1.png", dpi='figure')
plt.close(fig)

# In[1212]:

img_crop2 = img_tr_er[1990:2110, 1000:1160] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/go/go_2.png", dpi='figure')
plt.close(fig)

# In[1213]:

img_crop2 = img_tr_er[1990:2110, 940:1110]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/go/go_3.png", dpi='figure')
plt.close(fig)

# In[1214]:

img_crop2 = img_tr_er[2000:2160, 990:1110] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/go/go_4.png", dpi='figure')
plt.close(fig)

# In[1215]:

img_crop2 = img_tr_er[1940:2100, 990:1110] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/go/go_5.png", dpi='figure')
plt.close(fig)

# In[833]:

img_crop2 = img_tr_er[2000:2100, 1220:1320]
plt.imshow(img_crop2,'gray')


# In[834]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/za/za_1.png", dpi='figure')
plt.close(fig)

# In[1217]:

img_crop2 = img_tr_er[2000:2100, 1220:1370]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/za/za_2.png", dpi='figure')
plt.close(fig)

# In[1219]:

img_crop2 = img_tr_er[2000:2100, 1170:1320]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/za/za_3.png", dpi='figure')
plt.close(fig)

# In[1220]:

img_crop2 = img_tr_er[2000:2150, 1220:1320]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/za/za_4.png", dpi='figure')
plt.close(fig)

# In[1222]:

img_crop2 = img_tr_er[1950:2100, 1220:1320]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/za/za_5.png", dpi='figure')
plt.close(fig)

# In[1223]:

img_crop2 = img_tr_er[2000:2100, 1440:1530] 
plt.imshow(img_crop2,'gray')


# In[1224]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji1_1.png", dpi='figure')
plt.close(fig)

# In[1225]:

img_crop2 = img_tr_er[2000:2100, 1390:1530]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji1_2.png", dpi='figure')
plt.close(fig)

# In[1226]:

img_crop2 = img_tr_er[2000:2100, 1450:1580] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji1_3.png", dpi='figure')
plt.close(fig)

# In[1227]:

img_crop2 = img_tr_er[1950:2100, 1440:1530] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji1_4.png", dpi='figure')
plt.close(fig)

# In[1228]:

img_crop2 = img_tr_er[2000:2150, 1440:1530] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji1_5.png", dpi='figure')
plt.close(fig)

# In[1229]:

img_crop2 = img_tr_er[2000:2100, 1630:1750] 
plt.imshow(img_crop2,'gray')


# In[1230]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu1_1.png", dpi='figure')
plt.close(fig)

# In[1231]:

img_crop2 = img_tr_er[2000:2100, 1640:1800] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu1_2.png", dpi='figure')
plt.close(fig)

# In[1232]:

img_crop2 = img_tr_er[2000:2100, 1580:1750]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu1_3.png", dpi='figure')
plt.close(fig)

# In[1233]:

img_crop2 = img_tr_er[1950:2100, 1630:1750] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu1_4.png", dpi='figure')
plt.close(fig)

# In[1234]:

img_crop2 = img_tr_er[2000:2150, 1630:1750] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu1_5.png", dpi='figure')
plt.close(fig)

# In[842]:

img_crop2 = img_tr_er[1990:2110, 1850:1970] 
plt.imshow(img_crop2,'gray')


# In[843]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ze/ze_1.png", dpi='figure')
plt.close(fig)

# In[1236]:

img_crop2 = img_tr_er[1990:2110, 1800:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ze/ze_2.png", dpi='figure')
plt.close(fig)

# In[1238]:

img_crop2 = img_tr_er[1990:2110, 1850:2020] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ze/ze_3.png", dpi='figure')
plt.close(fig)

# In[1239]:

img_crop2 = img_tr_er[1940:2100, 1850:1970] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ze/ze_4.png", dpi='figure')
plt.close(fig)

# In[1240]:

img_crop2 = img_tr_er[2000:2160, 1850:1970] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ze/ze_5.png", dpi='figure')
plt.close(fig)

# In[845]:

img_crop2 = img_tr_er[1990:2110, 2070:2190] 
plt.imshow(img_crop2,'gray')


# In[846]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zo/zo_1.png", dpi='figure')
plt.close(fig)

# In[1241]:

img_crop2 = img_tr_er[1990:2110, 2020:2190]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zo/zo_2.png", dpi='figure')
plt.close(fig)

# In[1243]:

img_crop2 = img_tr_er[1990:2110, 2070:2240] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zo/zo_3.png", dpi='figure')
plt.close(fig)

# In[1244]:

img_crop2 = img_tr_er[1940:2100, 2070:2190] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zo/zo_4.png", dpi='figure')
plt.close(fig)

# In[1245]:

img_crop2 = img_tr_er[2000:2160, 2070:2190] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zo/zo_5.png", dpi='figure')
plt.close(fig)

# In[848]:

img_crop2 = img_tr_er[2180:2300, 110:250] 
plt.imshow(img_crop2,'gray')


# In[849]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/da/da_1.png", dpi='figure')
plt.close(fig)

# In[1246]:

img_crop2 = img_tr_er[2180:2300, 60:240] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/da/da_2.png", dpi='figure')
plt.close(fig)

# In[1247]:

img_crop2 = img_tr_er[2180:2300, 120:300] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/da/da_3.png", dpi='figure')
plt.close(fig)

# In[1248]:

img_crop2 = img_tr_er[2130:2290, 110:250] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/da/da_4.png", dpi='figure')
plt.close(fig)

# In[1249]:

img_crop2 = img_tr_er[2190:2350, 110:250] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/da/da_5.png", dpi='figure')
plt.close(fig)

# In[1250]:

img_crop2 = img_tr_er[2180:2300, 340:460] 
plt.imshow(img_crop2,'gray')


# In[1251]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji2_1.png", dpi='figure')
plt.close(fig)

# In[1252]:

img_crop2 = img_tr_er[2180:2300, 290:450] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji2_2.png", dpi='figure')
plt.close(fig)

# In[1253]:

img_crop2 = img_tr_er[2180:2300, 350:510] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji2_3.png", dpi='figure')
plt.close(fig)

# In[1254]:

img_crop2 = img_tr_er[2190:2350, 340:460] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji2_4.png", dpi='figure')
plt.close(fig)

# In[1255]:

img_crop2 = img_tr_er[2130:2290, 340:460] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji2_5.png", dpi='figure')
plt.close(fig)

# In[1256]:

img_crop2 = img_tr_er[2180:2300, 560:680]
plt.imshow(img_crop2,'gray')


# In[1257]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu2_1.png", dpi='figure')
plt.close(fig)

# In[1258]:

img_crop2 = img_tr_er[2180:2300, 510:680]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu2_2.png", dpi='figure')
plt.close(fig)

# In[1260]:

img_crop2 = img_tr_er[2180:2300, 560:730]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu2_3.png", dpi='figure')
plt.close(fig)

# In[1261]:

img_crop2 = img_tr_er[2190:2350, 560:680]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu2_4.png", dpi='figure')
plt.close(fig)

# In[1262]:

img_crop2 = img_tr_er[2130:2290, 560:680]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu2_5.png", dpi='figure')
plt.close(fig)

# In[869]:

img_crop2 = img_tr_er[2180:2300, 770:890] 
plt.imshow(img_crop2,'gray')


# In[870]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/de/de_1.png", dpi='figure')
plt.close(fig)

# In[1263]:

img_crop2 = img_tr_er[2180:2300, 720:890]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/de/de_2.png", dpi='figure')
plt.close(fig)

# In[1264]:

img_crop2 = img_tr_er[2180:2300, 780:940] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/de/de_3.png", dpi='figure')
plt.close(fig)

# In[1265]:

img_crop2 = img_tr_er[2190:2350, 770:890] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/de/de_4.png", dpi='figure')
plt.close(fig)

# In[1266]:

img_crop2 = img_tr_er[2130:2290, 770:890] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/de/de_5.png", dpi='figure')
plt.close(fig)

# In[867]:

img_crop2 = img_tr_er[2180:2300, 1000:1110]
plt.imshow(img_crop2,'gray')


# In[868]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/do/do_1.png", dpi='figure')
plt.close(fig)

# In[1267]:

img_crop2 = img_tr_er[2180:2300, 1010:1150] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/do/do_2.png", dpi='figure')
plt.close(fig)

# In[1268]:

img_crop2 = img_tr_er[2180:2300, 950:1110]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/do/do_3.png", dpi='figure')
plt.close(fig)

# In[1269]:

img_crop2 = img_tr_er[2190:2350, 1000:1110]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/do/do_4.png", dpi='figure')
plt.close(fig)

# In[1270]:

img_crop2 = img_tr_er[2130:2290, 1000:1110]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/do/do_5.png", dpi='figure')
plt.close(fig)

# In[865]:

img_crop2 = img_tr_er[2180:2300, 1200:1330]
plt.imshow(img_crop2,'gray')


# In[866]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ba/ba_1.png", dpi='figure')
plt.close(fig)

# In[1271]:

img_crop2 = img_tr_er[2180:2300, 1210:1370] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ba/ba_2.png", dpi='figure')
plt.close(fig)

# In[1272]:

img_crop2 = img_tr_er[2180:2300, 1150:1330]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ba/ba_3.png", dpi='figure')
plt.close(fig)

# In[1273]:

img_crop2 = img_tr_er[2130:2290, 1200:1330]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ba/ba_4.png", dpi='figure')
plt.close(fig)

# In[1274]:

img_crop2 = img_tr_er[2190:2350, 1200:1330]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ba/ba_5.png", dpi='figure')
plt.close(fig)

# In[896]:

img_crop2 = img_tr_er[2180:2300, 1420:1540]
plt.imshow(img_crop2,'gray')


# In[897]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bi/bi_1.png", dpi='figure')
plt.close(fig)

# In[1277]:

img_crop2 = img_tr_er[2180:2300, 1420:1580] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bi/bi_2.png", dpi='figure')
plt.close(fig)

# In[1278]:

img_crop2 = img_tr_er[2180:2300, 1370:1540]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bi/bi_3.png", dpi='figure')
plt.close(fig)

# In[1279]:

img_crop2 = img_tr_er[2130:2290, 1420:1540]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bi/bi_4.png", dpi='figure')
plt.close(fig)

# In[1280]:

img_crop2 = img_tr_er[2190:2350, 1420:1540]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bi/bi_5.png", dpi='figure')
plt.close(fig)

# In[872]:

img_crop2 = img_tr_er[2180:2300, 1630:1750] 
plt.imshow(img_crop2,'gray')


# In[873]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bu/bu_1.png", dpi='figure')
plt.close(fig)

# In[1281]:

img_crop2 = img_tr_er[2180:2300, 1640:1790] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bu/bu_2.png", dpi='figure')
plt.close(fig)

# In[1282]:

img_crop2 = img_tr_er[2180:2300, 1580:1750]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bu/bu_3.png", dpi='figure')
plt.close(fig)

# In[1283]:

img_crop2 = img_tr_er[2130:2290, 1630:1750] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bu/bu_4.png", dpi='figure')
plt.close(fig)

# In[1284]:

img_crop2 = img_tr_er[2190:2350, 1630:1750]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bu/bu_5.png", dpi='figure')
plt.close(fig)

# In[1285]:

img_crop2 = img_tr_er[2200:2270, 1850:1970]
plt.imshow(img_crop2,'gray')


# In[1286]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/be/be_1.png", dpi='figure')
plt.close(fig)

# In[1287]:

img_crop2 = img_tr_er[2200:2270, 1850:2010] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/be/be_2.png", dpi='figure')
plt.close(fig)

# In[1288]:

img_crop2 = img_tr_er[2200:2270, 1800:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/be/be_3.png", dpi='figure')
plt.close(fig)

# In[1289]:

img_crop2 = img_tr_er[2150:2270, 1850:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/be/be_4.png", dpi='figure')
plt.close(fig)

# In[1290]:

img_crop2 = img_tr_er[2200:2320, 1850:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/be/be_5.png", dpi='figure')
plt.close(fig)

# In[879]:

img_crop2 = img_tr_er[2180:2300, 2050:2200]
plt.imshow(img_crop2,'gray')


# In[880]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bo/bo_1.png", dpi='figure')
plt.close(fig)

# In[1291]:

img_crop2 = img_tr_er[2180:2300, 2000:2200]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bo/bo_2.png", dpi='figure')
plt.close(fig)

# In[1292]:

img_crop2 = img_tr_er[2180:2300, 2060:2240] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bo/bo_3.png", dpi='figure')
plt.close(fig)

# In[1294]:

img_crop2 = img_tr_er[2180:2350, 2050:2200]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bo/bo_4.png", dpi='figure')
plt.close(fig)

# In[1295]:

img_crop2 = img_tr_er[2130:2290, 2050:2200]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bo/bo_5.png", dpi='figure')
plt.close(fig)

# In[881]:

img_crop2 = img_tr_er[2360:2490, 120:250]
plt.imshow(img_crop2,'gray')


# In[882]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pa/pa_1.png", dpi='figure')
plt.close(fig)

# In[1296]:

img_crop2 = img_tr_er[2360:2490, 70:250]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pa/pa_2.png", dpi='figure')
plt.close(fig)

# In[1297]:

img_crop2 = img_tr_er[2360:2490, 130:300]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pa/pa_3.png", dpi='figure')
plt.close(fig)

# In[1298]:

img_crop2 = img_tr_er[2370:2540, 120:250]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pa/pa_4.png", dpi='figure')
plt.close(fig)

# In[1299]:

img_crop2 = img_tr_er[2310:2480, 120:250]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pa/pa_5.png", dpi='figure')
plt.close(fig)

# In[883]:

img_crop2 = img_tr_er[2360:2490, 340:460] 
plt.imshow(img_crop2,'gray')


# In[884]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pi/pi_1.png", dpi='figure')
plt.close(fig)

# In[1300]:

img_crop2 = img_tr_er[2360:2490, 290:460]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pi/pi_2.png", dpi='figure')
plt.close(fig)

# In[1301]:

img_crop2 = img_tr_er[2360:2490, 350:510] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pi/pi_3.png", dpi='figure')
plt.close(fig)

# In[1302]:

img_crop2 = img_tr_er[2370:2540, 340:460] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pi/pi_4.png", dpi='figure')
plt.close(fig)

# In[1303]:

img_crop2 = img_tr_er[2310:2480, 340:460] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pi/pi_5.png", dpi='figure')
plt.close(fig)

# In[885]:

img_crop2 = img_tr_er[2360:2490, 550:680] 
plt.imshow(img_crop2,'gray')


# In[886]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pu/pu_1.png", dpi='figure')
plt.close(fig)

# In[1304]:

img_crop2 = img_tr_er[2360:2490, 500:680]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pu/pu_2.png", dpi='figure')
plt.close(fig)

# In[1305]:

img_crop2 = img_tr_er[2360:2490, 560:730] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pu/pu_3.png", dpi='figure')
plt.close(fig)

# In[1306]:

img_crop2 = img_tr_er[2310:2480, 550:680] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pu/pu_4.png", dpi='figure')
plt.close(fig)

# In[1307]:

img_crop2 = img_tr_er[2370:2540, 550:680] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pu/pu_5.png", dpi='figure')
plt.close(fig)

# In[891]:

img_crop2 = img_tr_er[2370:2470, 770:890] 
plt.imshow(img_crop2,'gray')


# In[892]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pe/pe_1.png", dpi='figure')
plt.close(fig)

# In[1308]:

img_crop2 = img_tr_er[2370:2470, 720:890]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pe/pe_2.png", dpi='figure')
plt.close(fig)

# In[1309]:

img_crop2 = img_tr_er[2370:2470, 780:940] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pe/pe_3.png", dpi='figure')
plt.close(fig)

# In[1310]:

img_crop2 = img_tr_er[2320:2460, 770:890] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pe/pe_4.png", dpi='figure')
plt.close(fig)

# In[1311]:

img_crop2 = img_tr_er[2380:2520, 770:890] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pe/pe_5.png", dpi='figure')
plt.close(fig)

# In[894]:

img_crop2 = img_tr_er[2360:2490, 990:1110] 
plt.imshow(img_crop2,'gray')


# In[895]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/po/po_1.png", dpi='figure')
plt.close(fig)

# In[1312]:

img_crop2 = img_tr_er[2360:2490, 1000:1160] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/po/po_2.png", dpi='figure')
plt.close(fig)

# In[1313]:

img_crop2 = img_tr_er[2360:2490, 940:1110]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/po/po_3.png", dpi='figure')
plt.close(fig)

# In[1314]:

img_crop2 = img_tr_er[2310:2480, 990:1110] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/po/po_4.png", dpi='figure')
plt.close(fig)

# In[1315]:

img_crop2 = img_tr_er[2370:2540, 990:1110] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/po/po_5.png", dpi='figure')
plt.close(fig)

# In[ ]:



