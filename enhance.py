import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from IPython.display import clear_output
import os

def show_img(img):
    plt.style.use('dark_background')
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()

def img_processing(img):
    # do something here

    # 先將圖片轉為灰階
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # 將圖片做模糊化，可以降噪
    blur_img = cv2.medianBlur(img,5) 

    # 一般圖二值化(未模糊降噪)
    ret, th1 = cv2.threshold(img,130,255,cv2.THRESH_BINARY)

    # 一般圖二值化(有模糊降噪)
    ret, th4 = cv2.threshold(blur_img,127,255,cv2.THRESH_BINARY)

    # 高斯模糊
    bfilter = cv2.GaussianBlur(img,(3,3),10)
    
    #鋭化
    usm = cv2.addWeighted(img, 1.5, bfilter, -0.5, 0)
    
    # 侵蝕 
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))  
    dilate = cv2.dilate(th1,kernel1)
    
    # 膨脹
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,2))
    eroded = cv2.erode(dilate, kernel)
    
    # opening
    thresh = cv2.threshold(bfilter, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel0 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel0, iterations=1)
    invert = 255 - opening
    
    # 邊緣偵測
    #bfilter = cv2.bilateralFilter(img, 11, 17, 17) #Noise reduction
    Sobel = cv2.Sobel(eroded,cv2.CV_8U,1,0,ksize=1)
    Canny = cv2.Canny(eroded, 30, 200) #Edge detection
    #plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
       
    """titles = ['opening', 'eroded', 'dilate']
    images = [invert, eroded, dilate]
    
    plt.figure(figsize=(15,10)) 
    for i in range(3):        
        plt.subplot(2,3,i+1)
        plt.imshow(images[i],'gray')
        plt.title(titles[i])
    
    plt.show()"""
    
    return invert

def main(filepath = r'./ESRGAN/results/'):
    for filename in os.listdir(filepath):
        try:
            #file_name = "test4.png"
            print(filename)
            origin_img = cv2.imread(filepath+filename)

            #print("origin picture:")
            #show_img(origin_img)

            oup = img_processing(origin_img)

            #oup = cv2.resize(oup,[160,120])

            cv2.imwrite(r'./yolov5/data/traffic_identify/'+filename, oup)
        except:
            continue

if __name__=="__main__":
    main(r'./ESRGAN/results/')