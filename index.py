import cv2l
import matplotlib.pyplot as plt
import numpy as np
import win32api

from PIL import Image

from tkinter import Tk
from tkinter.filedialog import askopenfilename


root = Tk()
root.withdraw()


file_path = askopenfilename()


A = cv2.imread('Real.jpg')
#A = cv2.imread('e.png')

P = cv2.imread(file_path)
plt.imshow(P)

a = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
p = cv2.cvtColor(P, cv2.COLOR_BGR2GRAY)
plt.imshow(a)

# plt.show()


a2tr = a
plt.imshow(a2tr)

# plt.show()

b2tr = p
plt.imshow(b2tr)

# plt.show()

print(a.shape)
a2_str = a
plt.imshow(a2_str)

# plt.show()


print(p.shape)
p2_str = p
plt.imshow(p2_str)

# plt.show()

hsvImageReal = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
hsvImageFake = cv2.cvtColor(P, cv2.COLOR_BGR2HSV)

plt.imshow(hsvImageReal)

# plt.show()

plt.imshow(hsvImageFake)
# plt.show()


croppedImageReal = hsvImageReal
plt.imshow(croppedImageReal)
# plt.show()


croppedImageFake = hsvImageFake
plt.imshow(croppedImageFake)

# plt.show()


satThresh = 0.3
valThresh = 0.9
g = croppedImageReal[:,:,1]>satThresh
h = croppedImageReal[:,:,2] < valThresh

g1 = croppedImageFake[:,:,1]>satThresh
h1 = croppedImageFake[:,:,2] < valThresh

BWImageReal = g&h
BWImageFake = g1&h1


def bwareaopen(img, min_size, connectivity=8):
       
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            img, connectivity=connectivity)
        
        
        for i in range(num_labels):
            label_size = stats[i, cv2.CC_STAT_AREA]
            
            
            if label_size < min_size:
                img[labels == i] = 0
                
        return img


binr = cv2.threshold(a2_str, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
 

kernel = np.ones((3, 3), np.uint8)
 

invert = cv2.bitwise_not(binr)
 

BWImageCloseReal = cv2.morphologyEx(invert, cv2.MORPH_GRADIENT, kernel)

binr2 = cv2.threshold(p2_str, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
 

kernel2 = np.ones((3, 3), np.uint8)
 

invert2 = cv2.bitwise_not(binr2)
 

BWImageCloseFake = cv2.morphologyEx(invert2, cv2.MORPH_GRADIENT, kernel2)

areaopenReal = bwareaopen(BWImageCloseReal, 15)

areaopenFake = bwareaopen(BWImageCloseFake, 15)

bw = areaopenReal
 
labels = np.zeros(bw.shape)
countReal = cv2.connectedComponentsWithStats(bw, labels,8)

bw2 = areaopenFake
 
labels2 = np.zeros(bw2.shape)
countFake = cv2.connectedComponentsWithStats(bw2, labels2,8)

def corr2(A, B):
    
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
 
  
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
 
    try:
       result =  np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
    except Exception as e:
        result = "Fake"
    return result


    

print(a2tr)
print(b2tr)   

co=corr2 (a2tr, b2tr)

if co == "Fake":
        win32api.MessageBox(0, 'currency is fake', 'error')
elif (co.any()>=0.5):
    # print ('correlevance of transparent gandhi > 0.5')
    if (countReal[0] == countFake[0] ):
        win32api.MessageBox(0, 'currency is real', 'success')
    else:
        win32api.MessageBox(0, 'green strip is legitimate', 'error', 0x00001000)
else:
    print ('correlevance of transparent gandhi < 0.5')


