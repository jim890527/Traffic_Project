import cv2
import math
import numpy as np
import os
import matplotlib.pyplot as plt

def Img_Outline(input_dir):
    original_img = cv2.imread(input_dir)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)                     # 高斯模糊去噪（设定卷积核大小影响效果）
    _, RedThresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)  # 设定阈值150（阈值越大越亮）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))          # 定义矩形结构元素
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)       # 闭运算（链接块）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)           # 开运算（去噪点）
    
    closed = 255 - closed
    opened = 255 - opened
    
    # 新增白色邊框 = 255
    print(opened.shape[0],opened.shape[1]) #h,w
    for i in range (opened.shape[0]):
        if i==0 or i==opened.shape[0]-1:
            for j in range (opened.shape[1]):
                #print(i,j)
                if opened[i,j]==0:
                    opened[i,j] = 255
        else:
            #print(i,0)
            if opened[i,0]==0:
                opened[i,0] = 255
            #print(i,opened.shape[1])
            if opened[i,opened.shape[1]-1]==0:
                opened[i,opened.shape[1]-1] = 255  
    
    # 新增三角白框
    for i in range (60,opened.shape[0]):
        slope = opened.shape[1] / (opened.shape[0]-60)
        x = int(((i-60)*slope))
        for j in range (x):
            opened[i,j] = 255        
    
    return original_img, gray_img, RedThresh, closed, opened


def findContours_img(original_img, opened):
    contours = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    c = sorted(contours, key=cv2.contourArea, reverse=True)[1]   # 计算最大轮廓的旋转包围盒
    rect = cv2.minAreaRect(c)                                    # 获取包围盒（中心点，宽高，旋转角度）
    box = np.int0(cv2.boxPoints(rect))                           # box
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)

    print("box[0]:", box[0])
    print("box[1]:", box[1])
    print("box[2]:", box[2])
    print("box[3]:", box[3])
    
    return box,draw_img

def Perspective_transform(box,original_img):
    # 获取画框宽高(x=orignal_W,y=orignal_H)
    orignal_W = math.ceil(np.sqrt((box[3][1] - box[2][1])**2 + (box[3][0] - box[2][0])**2))
    orignal_H= math.ceil(np.sqrt((box[3][1] - box[0][1])**2 + (box[3][0] - box[0][0])**2))

    # 原图中的四个顶点,与变换矩阵
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    pts2 = np.float32([[int(orignal_W+1),int(orignal_H+1)], [0, int(orignal_H+1)], [0, 0], [int(orignal_W+1), 0]])

    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result_img = cv2.warpPerspective(original_img, M, (int(orignal_W+3),int(orignal_H+1)))

    return result_img


def Rotate(path):
    for filename in os.listdir(path):
        try:
            print(filename)
            input_dir = path+filename
            original_img, gray_img, RedThresh, closed, opened = Img_Outline(input_dir)
            box, draw_img = findContours_img(original_img, opened)
            result_img = Perspective_transform(box,original_img)
            #result_img = np.rot90(result_img,3)       
            cv2.imwrite('./rotate/'+filename, result_img)
        except:
            print("ERROR")
            continue
    
def main(path = "./yolov5/runs/license_plate/"):
    Rotate(path)

if __name__=="__main__":
    Rotate(r"./yolov5/runs/license_plate/")