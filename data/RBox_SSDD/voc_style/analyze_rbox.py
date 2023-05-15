# -*- coding: utf-8 -*-
from __future__ import division 
import os
import xml.dom.minidom
import cv2
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

font = {'family' : 'Palatino Linotype',
'weight' : 'normal',
'size'   : 15,
}

ImgPath = 'JPEGImages_test/'
AnnoPath = 'Annotations_test/'

imagelist = os.listdir(AnnoPath)
width_list = []
height_list = []
# bbox_xmin_list = []
# bbox_xmax_list = []
# bbox_ymin_list = []
# bbox_ymax_list = []

rotated_bbox_w_list = []
rotated_bbox_h_list = []

rotated_bbox_cx_list = []
rotated_bbox_cy_list = []

rotated_bbox_theta_list = []
x1_list = []
x2_list = []
x3_list = []
x4_list = []
y1_list = []
y2_list = []
y3_list = []
y4_list = []


for image in imagelist:
    image_pre, ext = os.path.splitext(image)
    imgfile = ImgPath + '/' + image_pre + '.jpg'
    xmlfile = AnnoPath + '/'  + image_pre + '.xml'
    
    im = cv2.imread(imgfile)
    
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement
    
    filename = annotation.getElementsByTagName('filename')[0].childNodes[0].data
    
    size = annotation.getElementsByTagName('size')[0]
    
    # print('size =', size)
    
    width = int(size.getElementsByTagName('width')[0].childNodes[0].data)
    height = int(size.getElementsByTagName('height')[0].childNodes[0].data)
    
    width_list.append(width)
    height_list.append(height)
    
    objectlist = annotation.getElementsByTagName('object')
    
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        objectname = namelist[0].childNodes[0].data
        rotated_bndbox = objects.getElementsByTagName('rotated_bndbox')
        for box in rotated_bndbox:
            rotated_bbox_w = box.getElementsByTagName('rotated_bbox_w')
            rotated_bbox_w = int(rotated_bbox_w[0].childNodes[0].data)
            rotated_bbox_h = box.getElementsByTagName('rotated_bbox_h')
            rotated_bbox_h = int(rotated_bbox_h[0].childNodes[0].data)
            
            rotated_bbox_cx = box.getElementsByTagName('rotated_bbox_cx')
            rotated_bbox_cx = int(rotated_bbox_cx[0].childNodes[0].data)
            rotated_bbox_cy = box.getElementsByTagName('rotated_bbox_cy')
            rotated_bbox_cy = int(rotated_bbox_cy[0].childNodes[0].data)
            
            
            rotated_bbox_theta = box.getElementsByTagName('rotated_bbox_theta')
            rotated_bbox_theta = float(rotated_bbox_theta[0].childNodes[0].data)
            
            x1 = box.getElementsByTagName('x1')
            x1 = int(x1[0].childNodes[0].data)
            y1 = box.getElementsByTagName('y1')
            y1 = int(y1[0].childNodes[0].data)
            
            x2 = box.getElementsByTagName('x2')
            x2 = int(x2[0].childNodes[0].data)
            y2 = box.getElementsByTagName('y2')
            y2 = int(y2[0].childNodes[0].data)
            
            x3 = box.getElementsByTagName('x3')
            x3 = int(x3[0].childNodes[0].data)
            y3 = box.getElementsByTagName('y3')
            y3 = int(y3[0].childNodes[0].data)
            
            x4 = box.getElementsByTagName('x4')
            x4 = int(x4[0].childNodes[0].data)
            y4 = box.getElementsByTagName('y4')
            y4 = int(y4[0].childNodes[0].data)
            
            rotated_bbox_w_list.append(rotated_bbox_w)
            rotated_bbox_h_list.append(rotated_bbox_h)
            rotated_bbox_cx_list.append(rotated_bbox_cx)
            rotated_bbox_cy_list.append(rotated_bbox_cy)
            rotated_bbox_theta_list.append(rotated_bbox_theta)
            
            x1_list.append(x1)
            x2_list.append(x2)
            x3_list.append(x3)
            x4_list.append(x4)
            
            y1_list.append(y1)
            y2_list.append(y2)
            y3_list.append(y3)
            y4_list.append(y4)
            

    
    # print('width, height = ', width, height)

w_np_list = np.array(rotated_bbox_w_list)
h_np_list = np.array(rotated_bbox_h_list)

area_np_list = w_np_list * h_np_list

print('w_np_list = ', w_np_list)

center_x_np_list = np.array(rotated_bbox_cx_list)
center_y_np_list = np.array(rotated_bbox_cy_list)

max_w = int(np.max(w_np_list))
max_h = int(np.max(h_np_list))

min_w = int(np.min(w_np_list))
min_h = int(np.min(h_np_list))

mean_w = int(np.mean(w_np_list))
mean_h = int(np.mean(h_np_list))

median_w = int(np.median(w_np_list))
median_h = int(np.median(h_np_list))

# mode_w = int(np.argmax(np.bincount(w_np_list)))
# mode_h = int(np.argmax(np.bincount(h_np_list)))

# ------------------------------------------------------------------------------------
plt.figure(1)
s1 = plt.scatter(w_np_list, h_np_list, c='limegreen', marker='o')
s2 = plt.scatter(mean_w, mean_h, s=40, c='r', marker='p')
s3 = plt.scatter(min_w, min_h, s=40, c='g', marker='p')
s4 = plt.scatter(max_w, max_h, s=40, c='b', marker='p')
s5 = plt.scatter(median_w, median_h, s=40, c='k', marker='p')
# s6 = plt.scatter(mode_w, mode_h, s=40, c='m', marker='p')

lines_x = [0, 400]
lines_y = [0, 400]

plt.plot(lines_x, lines_y, c='g')


plt.legend((s1, s2, s3, s4, s5), \
('RBox Ship, Weight, Height', \
'RBox Ship, Weight, Height | mean    = {}, {}'.format(mean_w, mean_h),
'RBox Ship, Weight, Height | min       = {}, {}'.format(min_w, min_h),
'RBox Ship, Weight, Height | max      = {}, {}'.format(max_w, max_h),
'RBox Ship, Weight, Height | median = {}, {}'.format(median_w, median_h),
# 'w, h | mode    = {}, {}'.format(mode_w, mode_h),
), \
loc='best')

plt.xlim(0, 400)
plt.ylim(0, 400)

plt.xlabel('RBox Ship Width', font)
plt.ylabel('RBox Ship Height', font)
plt.savefig('RBox_Ship_Width_Height.png', bbox_inches='tight')
plt.show()

ratio_w_h_list = w_np_list / h_np_list

# print(ratio_w_h_list)


max_ratio_width_height = np.max(ratio_w_h_list)
min_ratio_width_height = np.min(ratio_w_h_list)
mean_ratio_width_height = np.mean(ratio_w_h_list)

# print(ratio_w_h_list, ratio_w_h_list)

# ------------------------------------------------------------------------------------
plt.figure(2)

arr=plt.hist(ratio_w_h_list, bins=20, color='deeppink', edgecolor='w')

for i in range(20):
    plt.text(arr[1][i],arr[0][i], str(int(arr[0][i])))
    
plt.xlim(0.1, 10.5)
plt.ylim(0, 900)

plt.xlabel('Ratio between RBox Ship Width and Height', font)
plt.ylabel('Number of Images', font)
plt.savefig('Hist_RBox_Ship_Width_Height.png', bbox_inches='tight')
plt.show()



# ------------------------------------------------------------------------------------
plt.figure(3)
area_np_list = area_np_list / 10000
arr=plt.hist(area_np_list, bins=50, color='chartreuse', edgecolor='w')

for i in range(50):
    plt.text(arr[1][i],arr[0][i], str(int(arr[0][i])), fontsize=5)
    
plt.xlim(0, 30500/10000)
plt.ylim(0, 1700)

plt.xlabel('Area of RBox Ship ' + "(×${10}^{4}$)", font)
plt.ylabel('Number of Images', font)
plt.savefig('Hist_Area_RBox_Ship.png', bbox_inches='tight', dpi=1200)
plt.show()

# ------------------------------------------------------------------------------------
plt.figure(4)
s1 = plt.scatter(center_x_np_list, center_y_np_list, c='darkorange', marker='o')

max_cx = int(np.max(center_x_np_list))
max_cy = int(np.max(center_y_np_list))

min_cx = int(np.min(center_x_np_list))
min_cy = int(np.min(center_y_np_list))

mean_cx = int(np.mean(center_x_np_list))
mean_cy = int(np.mean(center_y_np_list))

median_cx = int(np.median(center_x_np_list))
median_cy = int(np.median(center_y_np_list))

s2 = plt.scatter(mean_cx, mean_cy, s=40, c='r', marker='p')
s3 = plt.scatter(min_cx, min_cy, s=40, c='g', marker='p')
s4 = plt.scatter(max_cx, max_cy, s=40, c='b', marker='p')
s5 = plt.scatter(median_cx, median_cy, s=40, c='k', marker='p')
# s6 = plt.scatter(mode_w, mode_h, s=40, c='m', marker='p')

lines_x = [0, 550]
lines_y = [0, 550]

plt.plot(lines_x, lines_y, c='g')


plt.legend((s1, s2, s3, s4, s5), \
('RBox Ship, Center x, Center y', \
'RBox Ship, Center x, Center y| mean    = {}, {}'.format(mean_cx, mean_cy),
'RBox Ship, Center x, Center y | min       = {}, {}'.format(min_cx, min_cy),
'RBox Ship, Center x, Center y | max      = {}, {}'.format(max_cx, max_cy),
'RBox Ship, Center x, Center y | median = {}, {}'.format(median_cx, median_cy),
# 'w, h | mode    = {}, {}'.format(mode_w, mode_h),
), \
loc='best')

plt.xlim(0, 550)
plt.ylim(0, 550)

plt.xlabel('RBox Ship Center x', font)
plt.ylabel('RBox Ship Center y', font)
plt.savefig('RBox_Ship_Center_x_Center_y.png', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------------
plt.figure(5)

arr=plt.hist(rotated_bbox_theta_list, bins=20, color='green', edgecolor='w')

for i in range(20):
    plt.text(arr[1][i],arr[0][i], str(int(arr[0][i])))
    
plt.xlim(0, 90)
plt.ylim(0, 400)

plt.xlabel('RBox Theta (°)', font)
plt.ylabel('Number of Images', font)
plt.savefig('Hist_RBox_Theta.png', bbox_inches='tight')
plt.show()