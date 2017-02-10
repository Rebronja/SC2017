
# coding: utf-8

# In[1138]:

import matplotlib.pyplot as plt


import numpy as np
from skimage.io import imread
img = imread('images/page3_2.jpg')
plt.imshow(img)


# In[1139]:

from skimage.color import rgb2gray
from skimage.morphology import opening, closing, square, diamond, disk, erosion, dilation

from skimage.measure import label
from skimage.measure import regionprops

img_gray = rgb2gray(img)
img_tr = img_gray > 0.3
img_tr_er = erosion(img_tr,selem=disk(5))
img_tr_er = dilation(img_tr,selem=disk(0.5))

labeled_img = label(img_tr_er)
regions = regionprops(labeled_img)

plt.imshow(img_tr_er, 'gray')


# In[1140]:

fig = plt.figure()
fig.set_size_inches(26,26)
plt.imshow(img_tr_er,'gray')
plt.axis('off')

plt.savefig("images/prikazSeta2.png", dpi='figure')
plt.close(fig)


# In[1141]:

from skimage.io import imsave
img_crop = img_tr_er[790:930, 70:230] 
plt.imshow(img_crop,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop,'gray')
plt.axis('off')

plt.savefig("images/train/a/a_6.png", dpi='figure')
plt.close(fig)


# In[1136]:

img_crop2 = img_tr_er[790:930, 70:280] 


plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/a/a_7.png", dpi='figure')
plt.close(fig)



# In[636]:

img_crop2 = img_tr_er[790:930, 70:230] 


plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/a/a_8.png", dpi='figure')
plt.close(fig)


# In[637]:

img_crop2 = img_tr_er[740:930, 70:230] 


plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/a/a_9.png", dpi='figure')
plt.close(fig)


# In[630]:

img_crop2 = img_tr_er[790:980, 70:230]  


plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/a/a_10.png", dpi='figure')
plt.close(fig)


# In[639]:

img_crop2 = img_tr_er[810:900, 290:440] 
plt.imshow(img_crop2,'gray')


# In[640]:


fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/i/i_6.png", dpi='figure')
plt.close(fig)


# In[641]:

img_crop2 = img_tr_er[810:900, 290:490]
plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/i/i_7.png", dpi='figure')
plt.close(fig)


# In[642]:

img_crop2 = img_tr_er[810:900, 240:440]
plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/i/i_8.png", dpi='figure')
plt.close(fig)


# In[643]:

img_crop2 = img_tr_er[760:900, 290:440]
plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/i/i_9.png", dpi='figure')
plt.close(fig)


# In[644]:

img_crop2 = img_tr_er[810:950, 290:440]
plt.imshow(img_crop2,'gray')

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/i/i_10.png", dpi='figure')
plt.close(fig)


# In[647]:

img_crop2 = img_tr_er[800:900, 520:660]
plt.imshow(img_crop2,'gray')


# In[648]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/u/u_6.png", dpi='figure')
plt.close(fig)


# In[649]:

img_crop2 = img_tr_er[800:900, 520:700]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/u/u_7.png", dpi='figure')
plt.close(fig)


# In[650]:

img_crop2 = img_tr_er[800:900, 470:660]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/u/u_8.png", dpi='figure')
plt.close(fig)


# In[651]:

img_crop2 = img_tr_er[750:900,520:660]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/u/u_9.png", dpi='figure')
plt.close(fig)


# In[652]:

img_crop2 = img_tr_er[800:950,520:660]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/u/u_10.png", dpi='figure')
plt.close(fig)


# In[654]:

img_crop2 = img_tr_er[800:900, 730:880] 
plt.imshow(img_crop2,'gray')


# In[655]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/e/e_6.png", dpi='figure')
plt.close(fig)


# In[656]:

img_crop2 = img_tr_er[800:900, 730:930] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/e/e_7.png", dpi='figure')
plt.close(fig)


# In[657]:

img_crop2 = img_tr_er[800:900, 680:880] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/e/e_8.png", dpi='figure')
plt.close(fig)


# In[658]:

img_crop2 = img_tr_er[750:900, 730:880] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/e/e_9.png", dpi='figure')
plt.close(fig)


# In[659]:

img_crop2 = img_tr_er[800:950, 730:880] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/e/e_10.png", dpi='figure')
plt.close(fig)


# In[662]:

img_crop2 = img_tr_er[780:900, 950:1100] 
plt.imshow(img_crop2,'gray')


# In[663]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/o/o_6.png", dpi='figure')
plt.close(fig)


# In[664]:

img_crop2 = img_tr_er[780:900, 950:1150] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/o/o_7.png", dpi='figure')
plt.close(fig)


# In[665]:

img_crop2 = img_tr_er[780:900, 900:1100] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/o/o_8.png", dpi='figure')
plt.close(fig)


# In[666]:

img_crop2 = img_tr_er[780:950,950:1100] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/o/o_9.png", dpi='figure')
plt.close(fig)


# In[667]:

img_crop2 = img_tr_er[730:900,950:1100] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/o/o_10.png", dpi='figure')
plt.close(fig)


# In[669]:

img_crop2 = img_tr_er[780:900, 1160:1320]
plt.imshow(img_crop2,'gray')


# In[68]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ka/ka_6.png", dpi='figure')
plt.close(fig)


# In[1129]:

img_crop2 = img_tr_er[780:900, 1160:1370]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ka/ka_7.png", dpi='figure')
plt.close(fig)


# In[1130]:

img_crop2 = img_tr_er[780:900, 1110:1320]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ka/ka_8.png", dpi='figure')
plt.close(fig)


# In[672]:

img_crop2 = img_tr_er[780:950, 1160:1320]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ka/ka_9.png", dpi='figure')
plt.close(fig)


# In[673]:

img_crop2 = img_tr_er[730:900, 1160:1320]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ka/ka_10.png", dpi='figure')
plt.close(fig)


# In[1131]:

img_crop2 = img_tr_er[780:880, 1380:1530]
plt.imshow(img_crop2,'gray')


# In[676]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ki/ki_6.png", dpi='figure')
plt.close(fig)


# In[677]:

img_crop2 = img_tr_er[780:880, 1380:1580]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ki/ki_7.png", dpi='figure')
plt.close(fig)


# In[678]:

img_crop2 = img_tr_er[780:880, 1330:1530]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ki/ki_8.png", dpi='figure')
plt.close(fig)


# In[679]:

img_crop2 = img_tr_er[730:880, 1380:1530]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ki/ki_9.png", dpi='figure')
plt.close(fig)


# In[680]:

img_crop2 = img_tr_er[780:930, 1380:1530]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ki/ki_10.png", dpi='figure')
plt.close(fig)


# In[683]:

img_crop2 = img_tr_er[770:870, 1620:1730] 
plt.imshow(img_crop2,'gray')


# In[684]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ku/ku_6.png", dpi='figure')
plt.close(fig)


# In[685]:

img_crop2 = img_tr_er[770:870, 1620:1780] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ku/ku_7.png", dpi='figure')
plt.close(fig)


# In[686]:

img_crop2 = img_tr_er[770:870, 1570:1730] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ku/ku_8.png", dpi='figure')
plt.close(fig)


# In[687]:

img_crop2 = img_tr_er[770:930, 1620:1730] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ku/ku_9.png", dpi='figure')
plt.close(fig)


# In[688]:

img_crop2 = img_tr_er[720:870,1620:1730] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ku/ku_10.png", dpi='figure')
plt.close(fig)


# In[690]:

img_crop2 = img_tr_er[770:870, 1820:1970] 
plt.imshow(img_crop2,'gray')


# In[691]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ke/ke_6.png", dpi='figure')
plt.close(fig)


# In[692]:

img_crop2 = img_tr_er[770:870, 1820:2020] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ke/ke_7.png", dpi='figure')
plt.close(fig)


# In[693]:

img_crop2 = img_tr_er[770:870,1770:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ke/ke_8.png", dpi='figure')
plt.close(fig)


# In[694]:

img_crop2 = img_tr_er[720:870, 1820:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ke/ke_9.png", dpi='figure')
plt.close(fig)


# In[695]:

img_crop2 = img_tr_er[770:920,1820:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ke/ke_10.png", dpi='figure')
plt.close(fig)


# In[697]:

img_crop2 = img_tr_er[770:870, 2040:2180] 
plt.imshow(img_crop2,'gray')


# In[698]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ko/ko_6.png", dpi='figure')
plt.close(fig)


# In[699]:

img_crop2 = img_tr_er[770:870, 2040:2230] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ko/ko_7.png", dpi='figure')
plt.close(fig)


# In[700]:

img_crop2 = img_tr_er[770:870, 2040:2180] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ko/ko_8.png", dpi='figure')
plt.close(fig)


# In[701]:

img_crop2 = img_tr_er[770:920, 2040:2180] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ko/ko_9.png", dpi='figure')
plt.close(fig)


# In[702]:

img_crop2 = img_tr_er[720:870, 2040:2180] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ko/ko_10.png", dpi='figure')
plt.close(fig)


# In[704]:

img_crop2 = img_tr_er[1070:1180, 80:230]
plt.imshow(img_crop2,'gray')


# In[705]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/sa/sa_6.png", dpi='figure')
plt.close(fig)


# In[706]:

img_crop2 = img_tr_er[1070:1180, 30:230]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/sa/sa_7.png", dpi='figure')
plt.close(fig)


# In[707]:

img_crop2 = img_tr_er[1070:1180,80:280]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/sa/sa_8.png", dpi='figure')
plt.close(fig)


# In[708]:

img_crop2 = img_tr_er[1020:1180, 80:230]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/sa/sa_9.png", dpi='figure')
plt.close(fig)


# In[709]:

img_crop2 = img_tr_er[1070:1230, 80:230]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/sa/sa_10.png", dpi='figure')
plt.close(fig)


# In[712]:

img_crop2 = img_tr_er[1070:1180, 310:450]
plt.imshow(img_crop2,'gray')


# In[713]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/shi/shi_6.png", dpi='figure')
plt.close(fig)


# In[714]:

img_crop2 = img_tr_er[1070:1180, 310:490]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/shi/shi_7.png", dpi='figure')
plt.close(fig)


# In[715]:

img_crop2 = img_tr_er[1070:1180, 260:450]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/shi/shi_8.png", dpi='figure')
plt.close(fig)


# In[716]:

img_crop2 = img_tr_er[1070:1230, 310:450]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/shi/shi_9.png", dpi='figure')
plt.close(fig)


# In[717]:

img_crop2 = img_tr_er[1020:1180, 310:450]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/shi/shi_10.png", dpi='figure')
plt.close(fig)


# In[719]:

img_crop2 = img_tr_er[1070:1170, 520:670]
plt.imshow(img_crop2,'gray')


# In[720]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/su/su_6.png", dpi='figure')
plt.close(fig)


# In[721]:

img_crop2 = img_tr_er[1070:1170, 520:720]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/su/su_7.png", dpi='figure')
plt.close(fig)


# In[722]:

img_crop2 = img_tr_er[1070:1170, 470:670]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/su/su_8.png", dpi='figure')
plt.close(fig)


# In[723]:

img_crop2 = img_tr_er[1020:1170, 520:670]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/su/su_9.png", dpi='figure')
plt.close(fig)


# In[724]:

img_crop2 = img_tr_er[1070:1220, 520:670]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/su/su_10.png", dpi='figure')
plt.close(fig)


# In[727]:

img_crop2 = img_tr_er[1070:1170, 740:890]
plt.imshow(img_crop2,'gray')


# In[728]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/se/se_6.png", dpi='figure')
plt.close(fig)


# In[729]:

img_crop2 = img_tr_er[1070:1170, 740:940]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/se/se_7.png", dpi='figure')
plt.close(fig)


# In[730]:

img_crop2 = img_tr_er[1070:1170, 690:890]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/se/se_8.png", dpi='figure')
plt.close(fig)


# In[731]:

img_crop2 = img_tr_er[1070:1220, 740:890]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/se/se_9.png", dpi='figure')
plt.close(fig)


# In[732]:

img_crop2 = img_tr_er[1020:1170, 740:890]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/se/se_10.png", dpi='figure')
plt.close(fig)


# In[733]:

img_crop2 = img_tr_er[1060:1170, 950:1100] 
plt.imshow(img_crop2,'gray')


# In[734]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/so/so_6.png", dpi='figure')
plt.close(fig)


# In[735]:

img_crop2 = img_tr_er[1060:1170,950:1150] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/so/so_7.png", dpi='figure')
plt.close(fig)


# In[736]:

img_crop2 = img_tr_er[1060:1170, 900:1100] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/so/so_8.png", dpi='figure')
plt.close(fig)


# In[737]:

img_crop2 = img_tr_er[1060:1220,950:1100] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/so/so_9.png", dpi='figure')
plt.close(fig)


# In[738]:

img_crop2 = img_tr_er[1010:1170, 950:1100] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/so/so_10.png", dpi='figure')
plt.close(fig)


# In[739]:

img_crop2 = img_tr_er[1050:1170, 1160:1320] 
plt.imshow(img_crop2,'gray')


# In[740]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ta/ta_6.png", dpi='figure')
plt.close(fig)


# In[742]:

img_crop2 = img_tr_er[1050:1170, 1160:1370] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ta/ta_7.png", dpi='figure')
plt.close(fig)


# In[743]:

img_crop2 = img_tr_er[1050:1170, 1110:1320] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ta/ta_8.png", dpi='figure')
plt.close(fig)


# In[744]:

img_crop2 = img_tr_er[1050:1220, 1160:1320] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ta/ta_9.png", dpi='figure')
plt.close(fig)


# In[745]:

img_crop2 = img_tr_er[1000:1170, 1160:1320] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ta/ta_10.png", dpi='figure')
plt.close(fig)


# In[747]:

img_crop2 = img_tr_er[1050:1150, 1390:1530] 
plt.imshow(img_crop2,'gray')


# In[748]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/chi/chi_6.png", dpi='figure')
plt.close(fig)


# In[749]:

img_crop2 = img_tr_er[1050:1150, 1390:1580] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/chi/chi_7.png", dpi='figure')
plt.close(fig)


# In[750]:

img_crop2 = img_tr_er[1050:1150, 1340:1530] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/chi/chi_8.png", dpi='figure')
plt.close(fig)


# In[751]:

img_crop2 = img_tr_er[1050:1200, 1390:1530] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/chi/chi_9.png", dpi='figure')
plt.close(fig)


# In[752]:

img_crop2 = img_tr_er[1000:1150, 1390:1530] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/chi/chi_10.png", dpi='figure')
plt.close(fig)


# In[755]:

img_crop2 = img_tr_er[1050:1150, 1600:1760] 
plt.imshow(img_crop2,'gray')


# In[756]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/tsu/tsu_6.png", dpi='figure')
plt.close(fig)


# In[757]:

img_crop2 = img_tr_er[1050:1150, 1600:1800] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/tsu/tsu_7.png", dpi='figure')
plt.close(fig)


# In[758]:

img_crop2 = img_tr_er[1050:1150,1550:1760] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/tsu/tsu_8.png", dpi='figure')
plt.close(fig)


# In[759]:

img_crop2 = img_tr_er[1050:1200, 1600:1760] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/tsu/tsu_9.png", dpi='figure')
plt.close(fig)


# In[760]:

img_crop2 = img_tr_er[1000:1150, 1600:1760] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/tsu/tsu_10.png", dpi='figure')
plt.close(fig)


# In[763]:

img_crop2 = img_tr_er[1050:1140, 1820:1970] 
plt.imshow(img_crop2,'gray')


# In[764]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/te/te_6.png", dpi='figure')
plt.close(fig)


# In[765]:

img_crop2 = img_tr_er[1050:1140, 1820:2020] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/te/te_7.png", dpi='figure')
plt.close(fig)


# In[766]:

img_crop2 = img_tr_er[1050:1140, 1770:1970] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/te/te_8.png", dpi='figure')
plt.close(fig)


# In[767]:

img_crop2 = img_tr_er[1000:1140, 1820:1970]  
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/te/te_9.png", dpi='figure')
plt.close(fig)


# In[768]:

img_crop2 = img_tr_er[1050:1190, 1820:1970] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/te/te_10.png", dpi='figure')
plt.close(fig)


# In[770]:

img_crop2 = img_tr_er[1030:1130, 2040:2190] 
plt.imshow(img_crop2,'gray')


# In[771]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/to/to_6.png", dpi='figure')
plt.close(fig)


# In[772]:

img_crop2 = img_tr_er[1030:1130, 2040:2240] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/to/to_7.png", dpi='figure')
plt.close(fig)


# In[773]:

img_crop2 = img_tr_er[1030:1130, 1990:2190] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/to/to_8.png", dpi='figure')
plt.close(fig)


# In[774]:

img_crop2 = img_tr_er[1030:1180, 2040:2190] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/to/to_9.png", dpi='figure')
plt.close(fig)


# In[775]:

img_crop2 = img_tr_er[980:1130, 2040:2190] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/to/to_10.png", dpi='figure')
plt.close(fig)


# In[778]:

img_crop2 = img_tr_er[1340:1450, 80:250] 
plt.imshow(img_crop2,'gray')


# In[779]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/na/na_6.png", dpi='figure')
plt.close(fig)


# In[780]:

img_crop2 = img_tr_er[1340:1450,  80:300] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/na/na_7.png", dpi='figure')
plt.close(fig)


# In[781]:

img_crop2 = img_tr_er[1340:1450,30:250] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/na/na_8.png", dpi='figure')
plt.close(fig)


# In[782]:

img_crop2 = img_tr_er[1340:1500,  80:250] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/na/na_9.png", dpi='figure')
plt.close(fig)


# In[783]:

img_crop2 = img_tr_er[1290:1450,  80:250] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/na/na_10.png", dpi='figure')
plt.close(fig)


# In[785]:

img_crop2 = img_tr_er[1340:1450, 300:460] 
plt.imshow(img_crop2,'gray')


# In[786]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ni/ni_6.png", dpi='figure')
plt.close(fig)


# In[787]:

img_crop2 = img_tr_er[1340:1450, 300:510] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ni/ni_7.png", dpi='figure')
plt.close(fig)


# In[788]:

img_crop2 = img_tr_er[1340:1450, 250:460]  
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ni/ni_8.png", dpi='figure')
plt.close(fig)


# In[789]:

img_crop2 = img_tr_er[1340:1500,300:460] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ni/ni_9.png", dpi='figure')
plt.close(fig)


# In[790]:

img_crop2 = img_tr_er[1290:1450,300:460] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ni/ni_10.png", dpi='figure')
plt.close(fig)


# In[793]:

img_crop2 = img_tr_er[1340:1440, 520:680] 
plt.imshow(img_crop2,'gray')


# In[794]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/nu/nu_6.png", dpi='figure')
plt.close(fig)


# In[795]:

img_crop2 = img_tr_er[1340:1440,520:730] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/nu/nu_7.png", dpi='figure')
plt.close(fig)


# In[796]:

img_crop2 = img_tr_er[1340:1440, 470:680] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/nu/nu_8.png", dpi='figure')
plt.close(fig)


# In[797]:

img_crop2 = img_tr_er[1340:1490, 520:680] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/nu/nu_9.png", dpi='figure')
plt.close(fig)


# In[798]:

img_crop2 = img_tr_er[1290:1440, 520:680] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/nu/nu_10.png", dpi='figure')
plt.close(fig)


# In[801]:

img_crop2 = img_tr_er[1330:1440, 730:900] 
plt.imshow(img_crop2,'gray')


# In[802]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ne/ne_6.png", dpi='figure')
plt.close(fig)


# In[803]:

img_crop2 = img_tr_er[1330:1440,730:950] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ne/ne_7.png", dpi='figure')
plt.close(fig)


# In[804]:

img_crop2 = img_tr_er[1330:1440,680:900] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ne/ne_8.png", dpi='figure')
plt.close(fig)


# In[805]:

img_crop2 = img_tr_er[1330:1490, 730:900] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ne/ne_9.png", dpi='figure')
plt.close(fig)


# In[806]:

img_crop2 = img_tr_er[1280:1440, 730:900]  
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ne/ne_10.png", dpi='figure')
plt.close(fig)


# In[808]:

img_crop2 = img_tr_er[1330:1440, 950:1110] 
plt.imshow(img_crop2,'gray')


# In[809]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/no/no_6.png", dpi='figure')
plt.close(fig)


# In[810]:

img_crop2 = img_tr_er[1330:1440, 950:1160] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/no/no_7.png", dpi='figure')
plt.close(fig)


# In[811]:

img_crop2 = img_tr_er[1330:1440, 900:1110] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/no/no_8.png", dpi='figure')
plt.close(fig)


# In[812]:

img_crop2 = img_tr_er[1330:1490, 950:1110] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/no/no_9.png", dpi='figure')
plt.close(fig)


# In[813]:

img_crop2 = img_tr_er[1280:1440, 950:1110] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/no/no_10.png", dpi='figure')
plt.close(fig)


# In[815]:

img_crop2 = img_tr_er[1320:1430, 1170:1320] 
plt.imshow(img_crop2,'gray')


# In[816]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ha/ha_6.png", dpi='figure')
plt.close(fig)


# In[817]:

img_crop2 = img_tr_er[1320:1430, 1170:1370] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ha/ha_7.png", dpi='figure')
plt.close(fig)


# In[818]:

img_crop2 = img_tr_er[1320:1430, 1120:1320] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ha/ha_8.png", dpi='figure')
plt.close(fig)


# In[819]:

img_crop2 = img_tr_er[1320:1480, 1170:1320] 
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ha/ha_9.png", dpi='figure')
plt.close(fig)


# In[820]:

img_crop2 = img_tr_er[1270:1430, 1170:1320]  
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ha/ha_10.png", dpi='figure')
plt.close(fig)


# In[822]:

img_crop2 = img_tr_er[1320:1420, 1380:1550] 
plt.imshow(img_crop2,'gray')


# In[823]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/hi/hi_6.png", dpi='figure')
plt.close(fig)


# In[824]:

img_crop2 = img_tr_er[1320:1420,1380:1600]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/hi/hi_7.png", dpi='figure')
plt.close(fig)


# In[825]:

img_crop2 = img_tr_er[1320:1420, 1330:1550]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/hi/hi_8.png", dpi='figure')
plt.close(fig)


# In[826]:

img_crop2 = img_tr_er[1320:1470,1380:1550]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/hi/hi_9.png", dpi='figure')
plt.close(fig)


# In[827]:

img_crop2 = img_tr_er[1270:1420, 1380:1550]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/hi/hi_10.png", dpi='figure')
plt.close(fig)


# In[829]:

img_crop2 = img_tr_er[1310:1420, 1600:1760]
plt.imshow(img_crop2,'gray')


# In[830]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/fu/fu_6.png", dpi='figure')
plt.close(fig)


# In[831]:

img_crop2 = img_tr_er[1310:1420, 1600:1810]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/fu/fu_7.png", dpi='figure')
plt.close(fig)


# In[832]:

img_crop2 = img_tr_er[1310:1420, 1550:1760]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/fu/fu_8.png", dpi='figure')
plt.close(fig)


# In[833]:

img_crop2 = img_tr_er[1310:1470, 1600:1760]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/fu/fu_9.png", dpi='figure')
plt.close(fig)


# In[834]:

img_crop2 = img_tr_er[1260:1420, 1600:1760]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/fu/fu_10.png", dpi='figure')
plt.close(fig)


# In[836]:

img_crop2 = img_tr_er[1310:1400, 1820:1970]
plt.imshow(img_crop2,'gray')


# In[837]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/he/he_6.png", dpi='figure')
plt.close(fig)


# In[838]:

img_crop2 = img_tr_er[1310:1400, 1820:2020]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/he/he_7.png", dpi='figure')
plt.close(fig)


# In[839]:

img_crop2 = img_tr_er[1310:1400, 1770:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/he/he_8.png", dpi='figure')
plt.close(fig)


# In[840]:

img_crop2 = img_tr_er[1310:1450, 1820:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/he/he_9.png", dpi='figure')
plt.close(fig)


# In[841]:

img_crop2 = img_tr_er[1260:1400, 1820:1970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/he/he_10.png", dpi='figure')
plt.close(fig)


# In[843]:

img_crop2 = img_tr_er[1300:1400, 2040:2200]
plt.imshow(img_crop2,'gray')


# In[844]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ho/ho_6.png", dpi='figure')
plt.close(fig)


# In[845]:

img_crop2 = img_tr_er[1300:1400, 2040:2250]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ho/ho_7.png", dpi='figure')
plt.close(fig)


# In[846]:

img_crop2 = img_tr_er[1300:1400, 2040:2200]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ho/ho_8.png", dpi='figure')
plt.close(fig)


# In[847]:

img_crop2 = img_tr_er[1250:1400, 2040:2200]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ho/ho_9.png", dpi='figure')
plt.close(fig)


# In[848]:

img_crop2 = img_tr_er[1300:1450, 2040:2200]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ho/ho_10.png", dpi='figure')
plt.close(fig)


# In[851]:

img_crop2 = img_tr_er[1610:1710, 100:250]
plt.imshow(img_crop2,'gray')


# In[852]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ma/ma_6.png", dpi='figure')
plt.close(fig)


# In[853]:

img_crop2 = img_tr_er[1610:1710,100:300]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ma/ma_7.png", dpi='figure')
plt.close(fig)


# In[854]:

img_crop2 = img_tr_er[1610:1710, 50:250]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ma/ma_8.png", dpi='figure')
plt.close(fig)


# In[855]:

img_crop2 = img_tr_er[1560:1710, 100:250]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ma/ma_9.png", dpi='figure')
plt.close(fig)


# In[856]:

img_crop2 = img_tr_er[1610:1760, 100:250]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ma/ma_10.png", dpi='figure')
plt.close(fig)


# In[860]:

img_crop2 = img_tr_er[1610:1710, 310:470]
plt.imshow(img_crop2,'gray')


# In[861]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mi/mi_6.png", dpi='figure')
plt.close(fig)


# In[862]:

img_crop2 = img_tr_er[1610:1710, 310:520]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mi/mi_7.png", dpi='figure')
plt.close(fig)


# In[863]:

img_crop2 = img_tr_er[1610:1710, 260:470]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mi/mi_8.png", dpi='figure')
plt.close(fig)


# In[864]:

img_crop2 = img_tr_er[1610:1760, 310:470]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mi/mi_9.png", dpi='figure')
plt.close(fig)


# In[865]:

img_crop2 = img_tr_er[1560:1710, 310:470]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mi/mi_10.png", dpi='figure')
plt.close(fig)


# In[867]:

img_crop2 = img_tr_er[1600:1700, 530:690]
plt.imshow(img_crop2,'gray')


# In[868]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mu/mu_6.png", dpi='figure')
plt.close(fig)


# In[869]:

img_crop2 = img_tr_er[1600:1700, 530:750]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mu/mu_7.png", dpi='figure')
plt.close(fig)


# In[870]:

img_crop2 = img_tr_er[1600:1700, 480:690]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mu/mu_8.png", dpi='figure')
plt.close(fig)


# In[871]:

img_crop2 = img_tr_er[1600:1750, 530:690]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mu/mu_9.png", dpi='figure')
plt.close(fig)


# In[872]:

img_crop2 = img_tr_er[1550:1700,530:690]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mu/mu_10.png", dpi='figure')
plt.close(fig)


# In[874]:

img_crop2 = img_tr_er[1600:1700, 750:900]
plt.imshow(img_crop2,'gray')


# In[875]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/me/me_6.png", dpi='figure')
plt.close(fig)


# In[876]:

img_crop2 = img_tr_er[1600:1700, 700:900]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/me/me_7.png", dpi='figure')
plt.close(fig)


# In[877]:

img_crop2 = img_tr_er[1600:1700, 750:950]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/me/me_8.png", dpi='figure')
plt.close(fig)


# In[878]:

img_crop2 = img_tr_er[1600:1750,750:900]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/me/me_9.png", dpi='figure')
plt.close(fig)


# In[879]:

img_crop2 = img_tr_er[1550:1700, 750:900]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/me/me_10.png", dpi='figure')
plt.close(fig)


# In[880]:

img_crop2 = img_tr_er[1590:1700, 960:1110]
plt.imshow(img_crop2,'gray')


# In[881]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mo/mo_6.png", dpi='figure')
plt.close(fig)


# In[882]:

img_crop2 = img_tr_er[1590:1700, 960:1160]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mo/mo_7.png", dpi='figure')
plt.close(fig)


# In[883]:

img_crop2 = img_tr_er[1590:1700, 910:1110]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mo/mo_8.png", dpi='figure')
plt.close(fig)


# In[884]:

img_crop2 = img_tr_er[1590:1750, 960:1110]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mo/mo_9.png", dpi='figure')
plt.close(fig)


# In[885]:

img_crop2 = img_tr_er[1540:1700, 960:1110]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/mo/mo_10.png", dpi='figure')
plt.close(fig)


# In[886]:

img_crop2 = img_tr_er[1590:1700, 1180:1330]
plt.imshow(img_crop2,'gray')


# In[887]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ra/ra_6.png", dpi='figure')
plt.close(fig)


# In[888]:

img_crop2 = img_tr_er[1590:1700, 1180:1370]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ra/ra_7.png", dpi='figure')
plt.close(fig)


# In[890]:

img_crop2 = img_tr_er[1590:1700, 1130:1330]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ra/ra_8.png", dpi='figure')
plt.close(fig)


# In[891]:

img_crop2 = img_tr_er[1590:1750, 1180:1330]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ra/ra_9.png", dpi='figure')
plt.close(fig)


# In[892]:

img_crop2 = img_tr_er[1540:1700, 1180:1330]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ra/ra_10.png", dpi='figure')
plt.close(fig)


# In[894]:

img_crop2 = img_tr_er[1580:1690, 1400:1540]
plt.imshow(img_crop2,'gray')


# In[895]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ri/ri_6.png", dpi='figure')
plt.close(fig)


# In[896]:

img_crop2 = img_tr_er[1580:1690, 1400:1580]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ri/ri_7.png", dpi='figure')
plt.close(fig)


# In[897]:

img_crop2 = img_tr_er[1580:1690, 1350:1540]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ri/ri_8.png", dpi='figure')
plt.close(fig)


# In[898]:

img_crop2 = img_tr_er[1580:1740, 1400:1540]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ri/ri_9.png", dpi='figure')
plt.close(fig)


# In[899]:

img_crop2 = img_tr_er[1530:1690, 1400:1540]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ri/ri_10.png", dpi='figure')
plt.close(fig)


# In[900]:

img_crop2 = img_tr_er[1580:1690, 1610:1760]
plt.imshow(img_crop2,'gray')


# In[901]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ru/ru_6.png", dpi='figure')
plt.close(fig)


# In[902]:

img_crop2 = img_tr_er[1580:1690, 1610:1800]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ru/ru_7.png", dpi='figure')
plt.close(fig)


# In[903]:

img_crop2 = img_tr_er[1580:1690, 1560:1760]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ru/ru_8.png", dpi='figure')
plt.close(fig)


# In[904]:

img_crop2 = img_tr_er[1580:1740, 1610:1760]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ru/ru_9.png", dpi='figure')
plt.close(fig)


# In[905]:

img_crop2 = img_tr_er[1530:1690, 1610:1760]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ru/ru_10.png", dpi='figure')
plt.close(fig)


# In[906]:

img_crop2 = img_tr_er[1570:1680, 1810:1980]
plt.imshow(img_crop2,'gray')


# In[907]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/re/re_6.png", dpi='figure')
plt.close(fig)


# In[908]:

img_crop2 = img_tr_er[1570:1680, 1810:2030]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/re/re_7.png", dpi='figure')
plt.close(fig)


# In[909]:

img_crop2 = img_tr_er[1570:1680, 1760:1980]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/re/re_8.png", dpi='figure')
plt.close(fig)


# In[910]:

img_crop2 = img_tr_er[1570:1730, 1810:1980]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/re/re_9.png", dpi='figure')
plt.close(fig)


# In[911]:

img_crop2 = img_tr_er[1520:1680, 1810:1980]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/re/re_10.png", dpi='figure')
plt.close(fig)


# In[912]:

img_crop2 = img_tr_er[1570:1670, 2040:2190]
plt.imshow(img_crop2,'gray')


# In[913]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ro/ro_6.png", dpi='figure')
plt.close(fig)


# In[914]:

img_crop2 = img_tr_er[1570:1670, 2040:2240]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ro/ro_7.png", dpi='figure')
plt.close(fig)


# In[915]:

img_crop2 = img_tr_er[1570:1670, 2040:2190]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ro/ro_8.png", dpi='figure')
plt.close(fig)


# In[916]:

img_crop2 = img_tr_er[1570:1720, 2040:2190]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ro/ro_9.png", dpi='figure')
plt.close(fig)


# In[917]:

img_crop2 = img_tr_er[1520:1670, 2040:2190]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ro/ro_10.png", dpi='figure')
plt.close(fig)


# In[921]:

img_crop2 = img_tr_er[1870:1970, 100:260]
plt.imshow(img_crop2,'gray')


# In[922]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ya/ya_6.png", dpi='figure')
plt.close(fig)


# In[923]:

img_crop2 = img_tr_er[1870:1970,100:310]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ya/ya_7.png", dpi='figure')
plt.close(fig)


# In[924]:

img_crop2 = img_tr_er[1870:1970,  50:260]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ya/ya_8.png", dpi='figure')
plt.close(fig)


# In[925]:

img_crop2 = img_tr_er[1870:2020,  100:260]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ya/ya_9.png", dpi='figure')
plt.close(fig)


# In[926]:

img_crop2 = img_tr_er[1820:1970,  100:260]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ya/ya_10.png", dpi='figure')
plt.close(fig)


# In[927]:

img_crop2 = img_tr_er[1870:1970, 320:480]
plt.imshow(img_crop2,'gray')


# In[928]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yu/yu_6.png", dpi='figure')
plt.close(fig)


# In[929]:

img_crop2 = img_tr_er[1870:1970, 320:520]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yu/yu_7.png", dpi='figure')
plt.close(fig)


# In[930]:

img_crop2 = img_tr_er[1870:1970, 270:480]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yu/yu_8.png", dpi='figure')
plt.close(fig)


# In[931]:

img_crop2 = img_tr_er[1870:2020, 320:480]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yu/yu_9.png", dpi='figure')
plt.close(fig)


# In[932]:

img_crop2 = img_tr_er[1820:1970, 320:480]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yu/yu_10.png", dpi='figure')
plt.close(fig)


# In[934]:

img_crop2 = img_tr_er[1860:1970, 540:690]
plt.imshow(img_crop2,'gray')


# In[935]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yo/yo_6.png", dpi='figure')
plt.close(fig)


# In[936]:

img_crop2 = img_tr_er[1860:1970, 540:740]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yo/yo_7.png", dpi='figure')
plt.close(fig)


# In[937]:

img_crop2 = img_tr_er[1860:1970,490:690]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yo/yo_8.png", dpi='figure')
plt.close(fig)


# In[938]:

img_crop2 = img_tr_er[1860:2020, 540:690]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yo/yo_9.png", dpi='figure')
plt.close(fig)


# In[939]:

img_crop2 = img_tr_er[1810:1970, 540:690]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/yo/yo_10.png", dpi='figure')
plt.close(fig)


# In[940]:

img_crop2 = img_tr_er[1860:1960, 750:910]
plt.imshow(img_crop2,'gray')


# In[941]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wa/wa_6.png", dpi='figure')
plt.close(fig)


# In[942]:

img_crop2 = img_tr_er[1860:1960, 750:950]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wa/wa_7.png", dpi='figure')
plt.close(fig)


# In[943]:

img_crop2 = img_tr_er[1860:1960,700:910]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wa/wa_8.png", dpi='figure')
plt.close(fig)


# In[944]:

img_crop2 = img_tr_er[1860:2010, 750:910]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wa/wa_9.png", dpi='figure')
plt.close(fig)


# In[945]:

img_crop2 = img_tr_er[1810:1960, 750:910]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wa/wa_10.png", dpi='figure')
plt.close(fig)


# In[946]:

img_crop2 = img_tr_er[1850:1960, 970:1110]
plt.imshow(img_crop2,'gray')


# In[947]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wo/wo_6.png", dpi='figure')
plt.close(fig)


# In[948]:

img_crop2 = img_tr_er[1850:1960, 970:1160]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wo/wo_7.png", dpi='figure')
plt.close(fig)


# In[949]:

img_crop2 = img_tr_er[1850:1960, 920:1110]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wo/wo_8.png", dpi='figure')
plt.close(fig)


# In[950]:

img_crop2 = img_tr_er[1850:2010, 970:1110]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wo/wo_9.png", dpi='figure')
plt.close(fig)


# In[951]:

img_crop2 = img_tr_er[1800:1960, 970:1110]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/wo/wo_10.png", dpi='figure')
plt.close(fig)


# In[952]:

img_crop2 = img_tr_er[1850:1960, 1180:1330]
plt.imshow(img_crop2,'gray')


# In[953]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/n/n_6.png", dpi='figure')
plt.close(fig)


# In[954]:

img_crop2 = img_tr_er[1850:1960, 1180:1380]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/n/n_7.png", dpi='figure')
plt.close(fig)


# In[955]:

img_crop2 = img_tr_er[1850:1960, 1130:1330]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/n/n_8.png", dpi='figure')
plt.close(fig)


# In[956]:

img_crop2 = img_tr_er[1850:2010, 1180:1330]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/n/n_9.png", dpi='figure')
plt.close(fig)


# In[957]:

img_crop2 = img_tr_er[1800:1960,1180:1330]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/n/n_10.png", dpi='figure')
plt.close(fig)


# In[958]:

img_crop2 = img_tr_er[1860:1950, 1400:1540]
plt.imshow(img_crop2,'gray')


# In[959]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYa/lowerCaseYa_6.png", dpi='figure')
plt.close(fig)


# In[960]:

img_crop2 = img_tr_er[1860:1950, 1400:1580]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYa/lowerCaseYa_7.png", dpi='figure')
plt.close(fig)


# In[961]:

img_crop2 = img_tr_er[1860:1950, 1350:1540]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYa/lowerCaseYa_8.png", dpi='figure')
plt.close(fig)


# In[962]:

img_crop2 = img_tr_er[1860:2000, 1400:1540]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYa/lowerCaseYa_9.png", dpi='figure')
plt.close(fig)


# In[963]:

img_crop2 = img_tr_er[1810:1950, 1400:1540]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYa/lowerCaseYa_10.png", dpi='figure')
plt.close(fig)


# In[964]:

img_crop2 = img_tr_er[1860:1940, 1620:1750]
plt.imshow(img_crop2,'gray')


# In[965]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYu/lowerCaseYu_6.png", dpi='figure')
plt.close(fig)


# In[966]:

img_crop2 = img_tr_er[1860:1940, 1620:1800]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYu/lowerCaseYu_7.png", dpi='figure')
plt.close(fig)


# In[967]:

img_crop2 = img_tr_er[1860:1940, 1570:1750]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYu/lowerCaseYu_8.png", dpi='figure')
plt.close(fig)


# In[968]:

img_crop2 = img_tr_er[1860:1990,1620:1750]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYu/lowerCaseYu_9.png", dpi='figure')
plt.close(fig)


# In[969]:

img_crop2 = img_tr_er[1810:1940, 1620:1750]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYu/lowerCaseYu_10.png", dpi='figure')
plt.close(fig)


# In[972]:

img_crop2 = img_tr_er[1850:1940, 1830:1960]
plt.imshow(img_crop2,'gray')


# In[973]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYo/lowerCaseYo_6.png", dpi='figure')
plt.close(fig)


# In[974]:

img_crop2 = img_tr_er[1850:1940, 1830:2010]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYo/lowerCaseYo_7.png", dpi='figure')
plt.close(fig)


# In[975]:

img_crop2 = img_tr_er[1850:1940,1780:1960]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYo/lowerCaseYo_8.png", dpi='figure')
plt.close(fig)


# In[976]:

img_crop2 = img_tr_er[1850:1990, 1830:1960]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYo/lowerCaseYo_9.png", dpi='figure')
plt.close(fig)


# In[977]:

img_crop2 = img_tr_er[1800:1940, 1830:1960]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/lowerCaseYo/lowerCaseYo_10.png", dpi='figure')
plt.close(fig)


# In[978]:

img_crop2 = img_tr_er[2390:2490, 110:280]
plt.imshow(img_crop2,'gray')


# In[979]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ga/ga_6.png", dpi='figure')
plt.close(fig)


# In[980]:

img_crop2 = img_tr_er[2390:2490,110:330]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ga/ga_7.png", dpi='figure')
plt.close(fig)


# In[981]:

img_crop2 = img_tr_er[2390:2490,60:280]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ga/ga_8.png", dpi='figure')
plt.close(fig)


# In[982]:

img_crop2 = img_tr_er[2390:2540, 110:280]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ga/ga_9.png", dpi='figure')
plt.close(fig)


# In[983]:

img_crop2 = img_tr_er[2340:2490, 110:280]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ga/ga_10.png", dpi='figure')
plt.close(fig)


# In[984]:

img_crop2 = img_tr_er[2380:2490, 330:490]
plt.imshow(img_crop2,'gray')


# In[985]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gi/gi_6.png", dpi='figure')
plt.close(fig)


# In[986]:

img_crop2 = img_tr_er[2380:2490,330:540]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gi/gi_7.png", dpi='figure')
plt.close(fig)


# In[987]:

img_crop2 = img_tr_er[2380:2490, 280:490]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gi/gi_8.png", dpi='figure')
plt.close(fig)


# In[988]:

img_crop2 = img_tr_er[2380:2540, 330:490]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gi/gi_9.png", dpi='figure')
plt.close(fig)


# In[989]:

img_crop2 = img_tr_er[2330:2490, 330:490]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gi/gi_10.png", dpi='figure')
plt.close(fig)


# In[990]:

img_crop2 = img_tr_er[2380:2490, 550:700]
plt.imshow(img_crop2,'gray')


# In[991]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gu/gu_6.png", dpi='figure')
plt.close(fig)


# In[992]:

img_crop2 = img_tr_er[2380:2490, 550:750]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gu/gu_7.png", dpi='figure')
plt.close(fig)


# In[993]:

img_crop2 = img_tr_er[2380:2490, 500:700]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gu/gu_8.png", dpi='figure')
plt.close(fig)


# In[994]:

img_crop2 = img_tr_er[2380:2540, 550:700]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gu/gu_9.png", dpi='figure')
plt.close(fig)


# In[995]:

img_crop2 = img_tr_er[2330:2490, 550:700]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/gu/gu_10.png", dpi='figure')
plt.close(fig)


# In[996]:

img_crop2 = img_tr_er[2370:2490, 760:920]
plt.imshow(img_crop2,'gray')


# In[997]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ge/ge_6.png", dpi='figure')
plt.close(fig)


# In[998]:

img_crop2 = img_tr_er[2370:2490, 760:970]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ge/ge_7.png", dpi='figure')
plt.close(fig)


# In[999]:

img_crop2 = img_tr_er[2370:2490, 710:920]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ge/ge_8.png", dpi='figure')
plt.close(fig)


# In[1000]:

img_crop2 = img_tr_er[2370:2540, 760:920]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ge/ge_9.png", dpi='figure')
plt.close(fig)


# In[1001]:

img_crop2 = img_tr_er[2320:2490,760:920]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ge/ge_10.png", dpi='figure')
plt.close(fig)


# In[1003]:

img_crop2 = img_tr_er[2370:2480, 980:1130]
plt.imshow(img_crop2,'gray')


# In[1004]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/go/go_6.png", dpi='figure')
plt.close(fig)


# In[1005]:

img_crop2 = img_tr_er[2370:2480, 980:1180]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/go/go_7.png", dpi='figure')
plt.close(fig)


# In[1006]:

img_crop2 = img_tr_er[2370:2480, 930:1130]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/go/go_8.png", dpi='figure')
plt.close(fig)


# In[1007]:

img_crop2 = img_tr_er[2370:2530, 980:1130]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/go/go_9.png", dpi='figure')
plt.close(fig)


# In[1008]:

img_crop2 = img_tr_er[2320:2480, 980:1130]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/go/go_10.png", dpi='figure')
plt.close(fig)


# In[1009]:

img_crop2 = img_tr_er[2360:2480, 1190:1350]
plt.imshow(img_crop2,'gray')


# In[1010]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/za/za_6.png", dpi='figure')
plt.close(fig)


# In[1011]:

img_crop2 = img_tr_er[2360:2480, 1190:1390]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/za/za_7.png", dpi='figure')
plt.close(fig)


# In[1012]:

img_crop2 = img_tr_er[2360:2480, 1140:1350]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/za/za_8.png", dpi='figure')
plt.close(fig)


# In[1013]:

img_crop2 = img_tr_er[2360:2530, 1190:1350]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/za/za_9.png", dpi='figure')
plt.close(fig)


# In[1014]:

img_crop2 = img_tr_er[2310:2480, 1190:1350]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/za/za_10.png", dpi='figure')
plt.close(fig)


# In[1015]:

img_crop2 = img_tr_er[2360:2480, 1410:1550]
plt.imshow(img_crop2,'gray')


# In[1016]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji1_6.png", dpi='figure')
plt.close(fig)


# In[1017]:

img_crop2 = img_tr_er[2360:2480, 1410:1600]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji1_7.png", dpi='figure')
plt.close(fig)


# In[1018]:

img_crop2 = img_tr_er[2360:2480,1360:1550]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji1_8.png", dpi='figure')
plt.close(fig)


# In[1019]:

img_crop2 = img_tr_er[2360:2530, 1410:1550]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji1_9.png", dpi='figure')
plt.close(fig)


# In[1020]:

img_crop2 = img_tr_er[2310:2480, 1410:1550]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji1_10.png", dpi='figure')
plt.close(fig)


# In[1021]:

img_crop2 = img_tr_er[2360:2470, 1620:1780]
plt.imshow(img_crop2,'gray')


# In[478]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu1_6.png", dpi='figure')
plt.close(fig)


# In[1022]:

img_crop2 = img_tr_er[2360:2470, 1620:1820]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu1_7.png", dpi='figure')
plt.close(fig)


# In[1023]:

img_crop2 = img_tr_er[2360:2470, 1570:1780]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu1_8.png", dpi='figure')
plt.close(fig)


# In[1024]:

img_crop2 = img_tr_er[2360:2520, 1620:1780]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu1_9.png", dpi='figure')
plt.close(fig)


# In[1025]:

img_crop2 = img_tr_er[2310:2470, 1620:1780]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu1_10.png", dpi='figure')
plt.close(fig)


# In[1026]:

img_crop2 = img_tr_er[2340:2470, 1830:1990]
plt.imshow(img_crop2,'gray')


# In[1027]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ze/ze_6.png", dpi='figure')
plt.close(fig)


# In[1028]:

img_crop2 = img_tr_er[2340:2470, 1830:2050]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ze/ze_7.png", dpi='figure')
plt.close(fig)


# In[1029]:

img_crop2 = img_tr_er[2340:2470, 1780:1990]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ze/ze_8.png", dpi='figure')
plt.close(fig)


# In[1030]:

img_crop2 = img_tr_er[2340:2520, 1830:1990]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ze/ze_9.png", dpi='figure')
plt.close(fig)


# In[1031]:

img_crop2 = img_tr_er[2290:2470, 1830:1990]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ze/ze_10.png", dpi='figure')
plt.close(fig)


# In[1032]:

img_crop2 = img_tr_er[2340:2470, 2050:2210]
plt.imshow(img_crop2,'gray')


# In[1033]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zo/zo_6.png", dpi='figure')
plt.close(fig)


# In[1034]:

img_crop2 = img_tr_er[2340:2470, 2050:2260]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zo/zo_7.png", dpi='figure')
plt.close(fig)


# In[1035]:

img_crop2 = img_tr_er[2340:2470, 2000:2210]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zo/zo_8.png", dpi='figure')
plt.close(fig)


# In[1036]:

img_crop2 = img_tr_er[2340:2520, 2050:2210]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zo/zo_9.png", dpi='figure')
plt.close(fig)


# In[1037]:

img_crop2 = img_tr_er[2290:2470, 2050:2210]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zo/zo_10.png", dpi='figure')
plt.close(fig)


# In[1038]:

img_crop2 = img_tr_er[2640:2770, 120:280]
plt.imshow(img_crop2,'gray')


# In[1039]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/da/da_6.png", dpi='figure')
plt.close(fig)


# In[1040]:

img_crop2 = img_tr_er[2640:2770, 120:340]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/da/da_7.png", dpi='figure')
plt.close(fig)


# In[1041]:

img_crop2 = img_tr_er[2640:2770, 70:280]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/da/da_8.png", dpi='figure')
plt.close(fig)


# In[1042]:

img_crop2 = img_tr_er[2590:2770, 120:280]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/da/da_9.png", dpi='figure')
plt.close(fig)


# In[1043]:

img_crop2 = img_tr_er[2640:2820, 120:280]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/da/da_10.png", dpi='figure')
plt.close(fig)


# In[1044]:

img_crop2 = img_tr_er[2640:2760, 330:490]
plt.imshow(img_crop2,'gray')


# In[1045]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji2_6.png", dpi='figure')
plt.close(fig)


# In[1046]:

img_crop2 = img_tr_er[2640:2760, 330:540]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji2_7.png", dpi='figure')
plt.close(fig)


# In[1047]:

img_crop2 = img_tr_er[2640:2760,280:490]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji2_8.png", dpi='figure')
plt.close(fig)


# In[1048]:

img_crop2 = img_tr_er[2640:2810, 330:490]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji2_9.png", dpi='figure')
plt.close(fig)


# In[1049]:

img_crop2 = img_tr_er[2590:2760, 330:490]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ji/ji2_10.png", dpi='figure')
plt.close(fig)


# In[1050]:

img_crop2 = img_tr_er[2640:2750, 550:710]
plt.imshow(img_crop2,'gray')


# In[1051]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu2_6.png", dpi='figure')
plt.close(fig)


# In[1052]:

img_crop2 = img_tr_er[2640:2750, 550:760]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu2_7.png", dpi='figure')
plt.close(fig)


# In[1053]:

img_crop2 = img_tr_er[2640:2750, 500:710]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu2_8.png", dpi='figure')
plt.close(fig)


# In[1054]:

img_crop2 = img_tr_er[2640:2800, 550:710]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu2_9.png", dpi='figure')
plt.close(fig)


# In[1055]:

img_crop2 = img_tr_er[2590:2750, 550:710]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/zu/zu2_10.png", dpi='figure')
plt.close(fig)


# In[1056]:

img_crop2 = img_tr_er[2640:2750, 770:920]
plt.imshow(img_crop2,'gray')


# In[1057]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/de/de_6.png", dpi='figure')
plt.close(fig)


# In[1058]:

img_crop2 = img_tr_er[2640:2750, 770:960]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/de/de_7.png", dpi='figure')
plt.close(fig)


# In[1059]:

img_crop2 = img_tr_er[2640:2750, 720:920]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/de/de_8.png", dpi='figure')
plt.close(fig)


# In[1060]:

img_crop2 = img_tr_er[2640:2800, 770:920]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/de/de_9.png", dpi='figure')
plt.close(fig)


# In[1061]:

img_crop2 = img_tr_er[2590:2750, 770:920]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/de/de_10.png", dpi='figure')
plt.close(fig)


# In[1062]:

img_crop2 = img_tr_er[2630:2750, 990:1130]
plt.imshow(img_crop2,'gray')


# In[1063]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/do/do_6.png", dpi='figure')
plt.close(fig)


# In[1064]:

img_crop2 = img_tr_er[2630:2750, 990:1180]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/do/do_7.png", dpi='figure')
plt.close(fig)


# In[1065]:

img_crop2 = img_tr_er[2630:2750,940:1130]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/do/do_8.png", dpi='figure')
plt.close(fig)


# In[1066]:

img_crop2 = img_tr_er[2630:2800, 990:1130]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/do/do_9.png", dpi='figure')
plt.close(fig)


# In[1067]:

img_crop2 = img_tr_er[2580:2750, 990:1130]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/do/do_10.png", dpi='figure')
plt.close(fig)


# In[1069]:

img_crop2 = img_tr_er[2620:2740, 1190:1350]
plt.imshow(img_crop2,'gray')


# In[1070]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ba/ba_6.png", dpi='figure')
plt.close(fig)


# In[1072]:

img_crop2 = img_tr_er[2620:2740, 1190:1400]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ba/ba_7.png", dpi='figure')
plt.close(fig)


# In[1071]:

img_crop2 = img_tr_er[2620:2740, 1140:1350]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ba/ba_8.png", dpi='figure')
plt.close(fig)


# In[1073]:

img_crop2 = img_tr_er[2620:2790, 1190:1350]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ba/ba_9.png", dpi='figure')
plt.close(fig)


# In[1074]:

img_crop2 = img_tr_er[2570:2740, 1190:1350]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/ba/ba_10.png", dpi='figure')
plt.close(fig)


# In[1075]:

img_crop2 = img_tr_er[2620:2740, 1400:1570]
plt.imshow(img_crop2,'gray')


# In[1076]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bi/bi_6.png", dpi='figure')
plt.close(fig)


# In[1077]:

img_crop2 = img_tr_er[2620:2740, 1400:1620]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bi/bi_7.png", dpi='figure')
plt.close(fig)


# In[1078]:

img_crop2 = img_tr_er[2620:2740, 1350:1570]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bi/bi_8.png", dpi='figure')
plt.close(fig)


# In[1079]:

img_crop2 = img_tr_er[2620:2790, 1400:1570]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bi/bi_9.png", dpi='figure')
plt.close(fig)


# In[1080]:

img_crop2 = img_tr_er[2570:2740, 1400:1570]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bi/bi_10.png", dpi='figure')
plt.close(fig)


# In[1081]:

img_crop2 = img_tr_er[2620:2730, 1610:1780]
plt.imshow(img_crop2,'gray')


# In[1082]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bu/bu_6.png", dpi='figure')
plt.close(fig)


# In[1083]:

img_crop2 = img_tr_er[2620:2730, 1610:1830]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bu/bu_7.png", dpi='figure')
plt.close(fig)


# In[1084]:

img_crop2 = img_tr_er[2620:2730,1570:1780]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bu/bu_8.png", dpi='figure')
plt.close(fig)


# In[1085]:

img_crop2 = img_tr_er[2620:2780, 1610:1780]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bu/bu_9.png", dpi='figure')
plt.close(fig)


# In[1086]:

img_crop2 = img_tr_er[2570:2730, 1610:1780]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bu/bu_10.png", dpi='figure')
plt.close(fig)


# In[1087]:

img_crop2 = img_tr_er[2620:2710, 1830:1990]
plt.imshow(img_crop2,'gray')


# In[1088]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/be/be_6.png", dpi='figure')
plt.close(fig)


# In[1089]:

img_crop2 = img_tr_er[2620:2710,1830:2030]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/be/be_7.png", dpi='figure')
plt.close(fig)


# In[1090]:

img_crop2 = img_tr_er[2620:2710,  1780:1990]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/be/be_8.png", dpi='figure')
plt.close(fig)


# In[1091]:

img_crop2 = img_tr_er[2620:2760,  1830:1990]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/be/be_9.png", dpi='figure')
plt.close(fig)


# In[1092]:

img_crop2 = img_tr_er[2570:2710,  1830:1990]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/be/be_10.png", dpi='figure')
plt.close(fig)


# In[1093]:

img_crop2 = img_tr_er[2610:2720, 2050:2210]
plt.imshow(img_crop2,'gray')


# In[1094]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bo/bo_6.png", dpi='figure')
plt.close(fig)


# In[1095]:

img_crop2 = img_tr_er[2610:2720, 2050:2260]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bo/bo_7.png", dpi='figure')
plt.close(fig)


# In[1096]:

img_crop2 = img_tr_er[2610:2720, 2000:2210]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bo/bo_8.png", dpi='figure')
plt.close(fig)


# In[1097]:

img_crop2 = img_tr_er[2610:2770, 2050:2210]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bo/bo_9.png", dpi='figure')
plt.close(fig)


# In[1098]:

img_crop2 = img_tr_er[2560:2720, 2050:2210]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/bo/bo_10.png", dpi='figure')
plt.close(fig)


# In[1099]:

img_crop2 = img_tr_er[2910:3020, 130:290]
plt.imshow(img_crop2,'gray')


# In[1100]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pa/pa_6.png", dpi='figure')
plt.close(fig)


# In[1101]:

img_crop2 = img_tr_er[2910:3020, 130:330]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pa/pa_7.png", dpi='figure')
plt.close(fig)


# In[1102]:

img_crop2 = img_tr_er[2910:3020, 80:290]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pa/pa_8.png", dpi='figure')
plt.close(fig)


# In[1103]:

img_crop2 = img_tr_er[2910:3070, 130:290]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pa/pa_9.png", dpi='figure')
plt.close(fig)


# In[1104]:

img_crop2 = img_tr_er[2860:3020, 130:290]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pa/pa_10.png", dpi='figure')
plt.close(fig)


# In[1105]:

img_crop2 = img_tr_er[2900:3020, 330:500]
plt.imshow(img_crop2,'gray')


# In[1106]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pi/pi_6.png", dpi='figure')
plt.close(fig)


# In[1107]:

img_crop2 = img_tr_er[2900:3020,330:550]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pi/pi_7.png", dpi='figure')
plt.close(fig)


# In[1108]:

img_crop2 = img_tr_er[2900:3020, 290:500]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pi/pi_8.png", dpi='figure')
plt.close(fig)


# In[1109]:

img_crop2 = img_tr_er[2900:3070, 330:500]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pi/pi_9.png", dpi='figure')
plt.close(fig)


# In[1110]:

img_crop2 = img_tr_er[2850:3020, 330:500]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pi/pi_10.png", dpi='figure')
plt.close(fig)


# In[1111]:

img_crop2 = img_tr_er[2900:3020, 560:720]
plt.imshow(img_crop2,'gray')


# In[1112]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pu/pu_6.png", dpi='figure')
plt.close(fig)


# In[1113]:

img_crop2 = img_tr_er[2900:3020, 560:750]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pu/pu_7.png", dpi='figure')
plt.close(fig)


# In[1114]:

img_crop2 = img_tr_er[2900:3020, 510:720]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pu/pu_8.png", dpi='figure')
plt.close(fig)


# In[1115]:

img_crop2 = img_tr_er[2900:3070, 560:720]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pu/pu_9.png", dpi='figure')
plt.close(fig)


# In[1116]:

img_crop2 = img_tr_er[2850:3020, 560:720]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pu/pu_10.png", dpi='figure')
plt.close(fig)


# In[1117]:

img_crop2 = img_tr_er[2900:3000, 770:930]
plt.imshow(img_crop2,'gray')


# In[1118]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pe/pe_6.png", dpi='figure')
plt.close(fig)


# In[1119]:

img_crop2 = img_tr_er[2900:3000, 770:980]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pe/pe_7.png", dpi='figure')
plt.close(fig)


# In[1120]:

img_crop2 = img_tr_er[2900:3000, 720:930]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pe/pe_8.png", dpi='figure')
plt.close(fig)


# In[1121]:

img_crop2 = img_tr_er[2900:3050, 770:930]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pe/pe_9.png", dpi='figure')
plt.close(fig)


# In[1122]:

img_crop2 = img_tr_er[2850:3000, 770:930]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/pe/pe_10.png", dpi='figure')
plt.close(fig)


# In[1123]:

img_crop2 = img_tr_er[2890:3000, 980:1150]
plt.imshow(img_crop2,'gray')


# In[1124]:

fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/po/po_6.png", dpi='figure')
plt.close(fig)


# In[1125]:

img_crop2 = img_tr_er[2890:3000, 980:1190]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/po/po_7.png", dpi='figure')
plt.close(fig)


# In[1126]:

img_crop2 = img_tr_er[2890:3000, 930:1150]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/po/po_8.png", dpi='figure')
plt.close(fig)


# In[1127]:

img_crop2 = img_tr_er[2890:3050, 980:1150]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/po/po_9.png", dpi='figure')
plt.close(fig)


# In[1128]:

img_crop2 = img_tr_er[2840:3000, 980:1150]
plt.imshow(img_crop2,'gray')
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(img_crop2,'gray')
plt.axis('off')

plt.savefig("images/train/po/po_10.png", dpi='figure')
plt.close(fig)


# In[ ]:



