from yolov5 import detect_light as lig
from yolov5 import detect_license as lic
from yolov5 import detect_identify as ident
from ESRGAN import inference_realesrgan as esrgan
from read_csv import Process
import rotate
import enhance

from apscheduler.schedulers.background import BackgroundScheduler
from PIL import Image, ImageTk
import time
import os
import cv2
import threading
import tkinter as tk
import numpy as np
import shutil

wait_lig = 3    # x秒檢查一次紅綠燈狀態
wait_lic = 8    # 檢測到綠燈等待x秒後開始車牌偵測
stop_event = threading.Event()      # be used to exit threading

def Remove(path):
    for filename in os.listdir(path):
        os.remove(path+filename)

def move(t, path):      # move photo to destination
    save_dir = 'C:/camera_sqlite/static/Traffic/'+ t +'/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for filename in os.listdir(path):
        shutil.move(path+filename, save_dir+filename)

def sched_job():
    # run once a day
    t = time.strftime("%Y-%m-%d")
    rotate.main()
    move(t, './yolov5/runs/license_plate/')
    esrgan.main()
    Remove('./rotate/')
    enhance.main()
    Remove('./ESRGAN/results/')
    csv = ident.main(t) 
    Remove('./yolov5/data/traffic_identify/')
    move(t, './identify/'+ t + '/')
    shutil.rmtree('./identify/')
    Process(csv)


class Schedule():
    def __init__(self):
        self.scheduler = BackgroundScheduler()   
        self.scheduler.add_job(sched_job, 'cron', hour=10, misfire_grace_time=60)     # , misfire_grace_time=60
        self.scheduler.start()

    def stop(self):
        self.scheduler.shutdown()


class Threader(threading.Thread):
    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.show1 = kwargs['args'][0]     # ph1
        self.show2 = kwargs['args'][1]     # ph2
        self.button = kwargs['args'][2]    # start button
        self.stop_button = kwargs['args'][3]
        self.start()

    def run(self):
        self.button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        while True:
            if stop_event.is_set():
                stop_event.clear()
                self.button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                break
            
            _, frame = lig.main() # 紅綠燈辨識結果
            if _ == True:         # 判斷紅綠燈，是否需開啟車牌偵測
                tt = time.strftime("%Y-%m-%d-%H-%M-%S")
                show_img = Image.fromarray(cv2.cvtColor((frame).astype(np.uint8),cv2.COLOR_BGRA2RGBA))
                imgtk = ImageTk.PhotoImage(image=show_img.resize((384, 384)))
                time.sleep(0.1)
                self.show1.imgtk=imgtk
                self.show1.config(image=imgtk)
                print('waiting ' + str(wait_lic) + ' seconds')
                time.sleep(wait_lic)     # 等待紅燈前通過的車輛
                arr = lic.main(tt)
                for i in range(0,len(arr)):
                    show_img2 = Image.fromarray(cv2.cvtColor((arr[i]).astype(np.uint8),cv2.COLOR_BGRA2RGBA))
                    imgtk = ImageTk.PhotoImage(image=show_img2.resize((384, 384)))
                    time.sleep(0.1)
                    self.show2.imgtk=imgtk
                    self.show2.config(image=imgtk)
                    time.sleep(0.2)
            else:
                show_img = Image.fromarray(cv2.cvtColor((frame).astype(np.uint8),cv2.COLOR_BGRA2RGBA))
                imgtk = ImageTk.PhotoImage(image=show_img.resize((384, 384)))
                time.sleep(0.1)
                self.show1.imgtk=imgtk
                self.show1.config(image=imgtk)
                print('waiting ' + str(wait_lig) + ' seconds')
                time.sleep(wait_lig)     # 檢查紅綠燈頻率


class UI:  
    def __init__(self):
        self.run()
      
    def quit(win, S):
        stop_event.set()
        S.stop()
        win.destroy()

    def stop():
        stop_event.set()

    def run(self):
        S = Schedule()
        tkWindow = tk.Tk()
        tkWindow.resizable(False,0)
        tkWindow.iconbitmap("./UI/traffic_lights.ico")
        tkWindow.geometry('768x512')
        tkWindow.title('Traffic')
        # photo 1
        img1 = Image.open('./UI/zidane.jpg')
        img1 = img1.resize((384,384))
        img1 = ImageTk.PhotoImage(img1)
        show1 = tk.Label(tkWindow,width=384,height=384)
        show1.place(x=0,y=0)
        show1.imgtk = img1
        show1.config(image = img1)
        # photo 2
        img2 = Image.open('./UI/bus.jpg')
        img2 = img2.resize((384,384))
        img2 = ImageTk.PhotoImage(img2)
        show2 = tk.Label(tkWindow,width=384,height=384)
        show2.place(x=385,y=0)
        show2.imgtk = img2
        show2.config(image = img2)
        # button
        stop_button = tk.Button(tkWindow,text="Stop", command = lambda:[UI.stop()],width=20,height=1,font=('Arial',10),state=tk.DISABLED)
        stop_button.place(x=300,y=420)
        start_button = tk.Button(tkWindow,text='Start',command=lambda : Threader(args=(show1,show2,start_button,stop_button)),width=20,height=1,font=('Arial',10))
        start_button.place(x=100,y=420)
        #tk.Button(tkWindow, text="Start", command = lambda:[main()],width=20,height=1,font=('Arial',10)).place(x=100,y=420)      
        tk.Button(tkWindow,text="Close", command = lambda:[UI.quit(tkWindow, S)],width=20,height=1,font=('Arial',10)).place(x=500,y=420)
        Threader(args = (show1,show2,start_button,stop_button))
        tkWindow.mainloop()


if __name__ == "__main__":
    UI()