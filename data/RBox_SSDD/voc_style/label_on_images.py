# -*- coding: utf-8 -*-
from __future__ import division 
import os
import xml.dom.minidom
import cv2
import numpy as np

def read_xml(ImgPath = 'JPEGImages/', AnnoPath = 'Annotations/', Savepath = 'JPEGImages_RBox_GT_Mask/'):

    if not os.path.isdir(Savepath):
        os.makedirs(Savepath)

    imagelist = os.listdir(AnnoPath)

    for image in imagelist:  
        image_pre, ext = os.path.splitext(image)
        imgfile = ImgPath + '/' + image_pre + '.jpg'
        xmlfile = AnnoPath + '/'  + image_pre + '.xml'
        im = cv2.imread(imgfile)
        DomTree = xml.dom.minidom.parse(xmlfile)
        annotation = DomTree.documentElement
        filenamelist = annotation.getElementsByTagName('filename')
        filename = filenamelist[0].childNodes[0].data
        objectlist = annotation.getElementsByTagName('object')
        for objects in objectlist:
            namelist = objects.getElementsByTagName('name')
            objectname = namelist[0].childNodes[0].data
            rotated_bndbox = objects.getElementsByTagName('rotated_bndbox')
            for box in rotated_bndbox:
                    
                    
                    
                    
                    x1 = box.getElementsByTagName('x1')
                    x1 = int(x1[0].childNodes[0].data)
                    print('x1 = ', x1)
                    
                    y1 = box.getElementsByTagName('y1')
                    y1 = int(y1[0].childNodes[0].data)
                    print('y1 = ', y1)
                    
                    x2 = box.getElementsByTagName('x2')
                    x2 = int(x2[0].childNodes[0].data)
                    print('x2 = ', x2)
                    
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
                    
                    pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], np.int32)
                    

                    # cv2.rectangle(im,(xmin,ymin),(xmax,ymax), (0, 255, 0), 2)
                    
                    
                    
                    
                    cv2.polylines(im,[pts],True,(0,255,0), 2)
                    # cv2.putText(im, 'ship', (xmin,ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)
        path = Savepath + '/' + image_pre + '.jpg'
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(im, objectname, (xmin,ymin - 7), font, 0.5, (0, 0, 255), 1)
                    
        cv2.imwrite(path, im)
read_xml()