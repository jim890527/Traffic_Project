import pandas as pd
import os
import time
import numpy as np

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import vlc
import threading

ROOT = 'D://CPD_LOG/'
stop_event = threading.Event()      # be used to exit threading
ok_img = cv2.imread('./UI/ok.jpg')
ng_img = cv2.imread('./UI/ng.jpg')

def find_ng(df2, col, machine):
    NG = pd.DataFrame(columns=['Report_Time','CPDX','PanelID','NG_head'], dtype=object)
    for j in range(len(df2)):   # j = rows
        ng_list = []
        ng_area = []
        for i in col:           # i = cols
            if df2[i].values[j] >= int(df2['MIN'].values[j]) and df2[i].values[j] <= int(df2['MAX'].values[j]):     # ok
                continue
            else:               # ng
                ng_area.append(i)
        if len(ng_area)!=0:     # 有任何AREA NG
            ng_list = [df2['Report_Time'].values[j], machine, df2['PanelID'].values[j], ng_area]
            NG.loc[len(NG)] = ng_list   
    if len(NG) == 0:            # 無NG
        return True, df2['PanelID'].values[0]
    else:                       # 匯出NG數據
        print(NG)
        if not os.path.exists(ROOT+'NG.csv'):
            NG.to_csv('NG.csv', index=False)
        else:
            NG_file = pd.read_csv(ROOT+'NG.csv')
            NG_file = NG_file.append(NG)
            NG_file.to_csv('NG.csv', index=False)
        return False, df2['PanelID'].values[0]

def check_report(p_id, month, machine):
    df = pd.read_csv(ROOT+machine+'/CHECK_REPORT/'+machine+'_check_report_'+month+'.csv', error_bad_lines=False, warn_bad_lines=False)
    rep = {}
    df = df[df['PanelID']==p_id]
    if len(df) >= 2:
        rec = df['Golden_File_Loc'].values[0].split('/')[-1].split('.')[-2]     # recipe
        tea = df['Golden_File_Loc'].values[1].split('/')[-1].split('.')[-2]     # teaching
        rep[rec] = df['Compare'].values[0]
        rep[tea] = df['Compare'].values[1]
    elif len(df) == 1:
        rec = df['Golden_File_Loc'].values[0].split('/')[-1].split('.')[-2]
        rep[rec] = df['Compare'].values[0]
        rep['TEACHING'] = 'N/A'
    else:
        print('PanelID not found')
        rep['RECIPE'] = 'N/A'
        rep['TEACHING'] = 'N/A'
    return rep

def check_file(path, machine, file=[]):         # 有新資料才跑
    com = pd.read_csv('RECIPE_SPEC.csv', encoding='latin1')
    for i in reversed(file):       
        df = pd.read_csv(path+i)
        df2 = pd.merge(com, df, on="Recipe_No", how='inner')
        df2.dropna(axis='columns', inplace=True)
        val = df2.columns.values
        col = []
        for j in val:
            if 'AREA' in j and 'GAP' not in j :
                col.append(j)
        is_ok, p_id = find_ng(df2, col, machine)
    month = path.split('/')[-2][0:6]
    rep = check_report(p_id, month, machine)
    report = [*rep]
    for i in range(len(report)):
        report[i] = report[i] +': '+ rep[report[i]]
    if is_ok == True:       # OK
        return True, p_id, report
    else:                   # NG
        return False, p_id, report

def check_folder(path, date):
    machine = path.split('/')[len(ROOT.split('/'))-1]
    df = pd.DataFrame(np.array([]), columns=['file_name'])
    path = path+date+'/'
    print(path)
    try:
        if len(os.listdir(path)) > 0:  # check folder is not empty
            excuted = -1
            for i in (os.listdir(path)):
                f = []
                if i == date+'.csv':
                    df = pd.read_csv(path+date+'.csv')
                    excuted = len(df['file_name'])
                else:
                    count = len(os.listdir(path))-excuted-1        # 需比對的數量
                    #print(count)
                    for j in reversed(os.listdir(path)):           # 反著找比較快
                        if count > 0:
                            if j not in str(df['file_name'].values):    # 不在df內表示為新資料
                                print(j)
                                f.append(j)
                                count -= 1
                        else:
                            break
                    if len(f) != 0:     # 有新資料
                        is_ok, p_id, report = check_file(path, machine, f)
                        df1 = pd.DataFrame(reversed(f), columns=['file_name'])
                        df = df.append(df1, ignore_index = True)
                        df.to_csv(path+date+'.csv', header=['file_name'], index=False)
                        return machine, is_ok, p_id, report
                    else:               # 無新資料
                        is_ok = True
                        return machine, is_ok, '', ['N/A','N/A']
        else:
            print('folder is empty')
            return machine, True, '', ['N/A','N/A']
    except Exception as e:
        print(e)
        return machine, True, '', ['N/A','N/A']
                    
def read(root = ROOT):
    date = time.strftime("%Y%m%d")
    dic = {}
    ID = {}
    report = {}
    for i in os.listdir(root):
        if 'CPD' in i:
            machine, is_ok, p_id , rep = check_folder(root+i+'/'+'CONVERTED_DATA_GAP/', '20220221')
            dic[machine] = is_ok
            ID[machine] = p_id
            report[machine] = rep
    print(dic)
    return dic, ID, report


class Threader(threading.Thread):
    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.show1 = kwargs['args'][0]     # ph1
        self.show2 = kwargs['args'][1]     # ph2
        self.show3 = kwargs['args'][2]     # ph3
        self.show4 = kwargs['args'][3]     # ph4
        self.show5 = kwargs['args'][4]     # ph5
        self.show6 = kwargs['args'][5]     # ph6
        self.show7 = kwargs['args'][6]     # ph7
        self.button = kwargs['args'][7]    # start button
        self.stop_button = kwargs['args'][8]
        self.L1 = kwargs['args'][9]        # label1
        self.L2 = kwargs['args'][10]       # label2
        self.L3 = kwargs['args'][11]       # label3
        self.L4 = kwargs['args'][12]       # label4
        self.L5 = kwargs['args'][13]       # label5
        self.L6 = kwargs['args'][14]       # label6
        self.L7 = kwargs['args'][15]       # label7
        self.start()

    def run(self):
        p = vlc.MediaPlayer('./UI/test.mp3')
        self.button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        lists = {'CPD01':self.show1,'CPD02':self.show2,'CPD03':self.show3,'CPD04':self.show4,'CPD05':self.show5,'CPD06':self.show6,'CPD07':self.show7}
        list_label = {'CPD01':self.L1,'CPD02':self.L2,'CPD03':self.L3,'CPD04':self.L4,'CPD05':self.L5,'CPD06':self.L6,'CPD07':self.L7}
        dic = {}
        while True:
            if stop_event.is_set():
                stop_event.clear()
                self.button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                break
            dic, ID, report = read()
            for i in lists:
                if dic[i] == True:     # ok
                    show_img = Image.fromarray(cv2.cvtColor((ok_img).astype(np.uint8),cv2.COLOR_BGRA2RGBA))
                    imgtk = ImageTk.PhotoImage(image=show_img.resize((240, 240)))
                    time.sleep(0.1)                   
                    lists[i].imgtk=imgtk
                    lists[i].config(image=imgtk)
                    list_label[i].config(text='PanelID:\r'+ID[i]+'\r'+ report[i][0]+'\r'+report[i][1], bg='#00824e', justify='left')
                else:                   # ng
                    p.play()
                    show_img = Image.fromarray(cv2.cvtColor((ng_img).astype(np.uint8),cv2.COLOR_BGRA2RGBA))#00824e
                    imgtk = ImageTk.PhotoImage(image=show_img.resize((240, 240)))
                    time.sleep(0.1)
                    lists[i].imgtk=imgtk
                    lists[i].config(image=imgtk)
                    list_label[i].config(text='PanelID:\r'+ID[i]+'\r'+ report[i][0]+'\r'+report[i][1], bg='#b60000', justify='left')
            time.sleep(10)              # 執行頻率
            p.stop()


class UI:
    def __init__(self):
        self.run()
      
    def quit(win):
        stop_event.set()
        win.destroy()

    def stop():
        stop_event.set()

    def run(self):
        tkWindow = tk.Tk()
        tkWindow.resizable(False,0)
        #tkWindow.iconbitmap("./UI/lcd.ico")
        tkWindow.geometry('780x920')
        tkWindow.title('CPD_LOG')
        # photo 1
        img1 = Image.open('./UI/ok.jpg')
        img1 = img1.resize((240,240))
        img1 = ImageTk.PhotoImage(img1)
        show1 = tk.Label(tkWindow,width=240,height=240)
        show1.place(x=15,y=30)
        show1.imgtk = img1
        show1.config(image = img1)
        # photo 2
        img2 = Image.open('./UI/ok.jpg')
        img2 = img2.resize((240,240))
        img2 = ImageTk.PhotoImage(img2)
        show2 = tk.Label(tkWindow,width=240,height=240)
        show2.place(x=265,y=30)
        show2.imgtk = img2
        show2.config(image = img2)
        # photo 3
        img3 = Image.open('./UI/ok.jpg')
        img3 = img3.resize((240,240))
        img3 = ImageTk.PhotoImage(img3)
        show3 = tk.Label(tkWindow,width=240,height=240)
        show3.place(x=515,y=30)
        show3.imgtk = img3
        show3.config(image = img3)
        # photo 4
        img4 = Image.open('./UI/ok.jpg')
        img4 = img4.resize((240,240))
        img4 = ImageTk.PhotoImage(img4)
        show4 = tk.Label(tkWindow,width=240,height=240)
        show4.place(x=15,y=300)
        show4.imgtk = img4
        show4.config(image = img4)
        # photo 5
        img5 = Image.open('./UI/ok.jpg')
        img5 = img5.resize((240,240))
        img5 = ImageTk.PhotoImage(img5)
        show5 = tk.Label(tkWindow,width=240,height=240)
        show5.place(x=265,y=300)
        show5.imgtk = img5
        show5.config(image = img5)
        # photo 6
        img6 = Image.open('./UI/ok.jpg')
        img6 = img6.resize((240,240))
        img6 = ImageTk.PhotoImage(img6)
        show6 = tk.Label(tkWindow,width=240,height=240)
        show6.place(x=515,y=300)
        show6.imgtk = img6
        show6.config(image = img6)
        # photo 7
        img7 = Image.open('./UI/ok.jpg')
        img7 = img7.resize((240,240))
        img7 = ImageTk.PhotoImage(img7)
        show7 = tk.Label(tkWindow,width=240,height=240)
        show7.place(x=15,y=570)
        show7.imgtk = img7
        show7.config(image = img7)
        # photo 8
        img8 = Image.open('./UI/AUO.png')
        img8 = ImageTk.PhotoImage(img8)
        show8 = tk.Label(tkWindow,width=240,height=240)
        show8.place(x=400,y=570)
        show8.imgtk = img8
        show8.config(image = img8)
        #label
        CPD1 = tk.Label(tkWindow, text='CPD01', font=('Arial', 16))
        CPD1.place(x=15,y=30)
        L1 = tk.Label(tkWindow, text='', font=('Arial', 16), bg = '#00824e')
        L1.place(x=17,y=170)
        CPD2 = tk.Label(tkWindow, text='CPD02', font=('Arial', 16))
        CPD2.place(x=265,y=30)
        L2 = tk.Label(tkWindow, text='', font=('Arial', 16), bg = '#00824e')
        L2.place(x=267,y=170)
        CPD3 = tk.Label(tkWindow, text='CPD03', font=('Arial', 16))
        CPD3.place(x=515,y=30)
        L3 = tk.Label(tkWindow, text='', font=('Arial', 16), bg = '#00824e')
        L3.place(x=517,y=170)
        CPD4 = tk.Label(tkWindow, text='CPD04', font=('Arial', 16))
        CPD4.place(x=15,y=300)
        L4 = tk.Label(tkWindow, text='', font=('Arial', 16), bg = '#00824e')
        L4.place(x=17,y=440)
        CPD5 = tk.Label(tkWindow, text='CPD05', font=('Arial', 16))
        CPD5.place(x=265,y=300)
        L5 = tk.Label(tkWindow, text='', font=('Arial', 16), bg = '#00824e')
        L5.place(x=267,y=440)
        CPD6 = tk.Label(tkWindow, text='CPD06', font=('Arial', 16))
        CPD6.place(x=515,y=300)
        L6 = tk.Label(tkWindow, text='', font=('Arial', 16), bg = '#00824e')
        L6.place(x=517,y=440)
        CPD7 = tk.Label(tkWindow, text='CPD07', font=('Arial', 16))
        CPD7.place(x=15,y=570)
        L7 = tk.Label(tkWindow, text='', font=('Arial', 16), bg = '#00824e')
        L7.place(x=17,y=710)      
        # button
        stop_button = tk.Button(tkWindow,text="Stop", command = lambda:[UI.stop()],width=20,height=1,font=('Arial',10),state=tk.DISABLED)
        stop_button.place(x=300,y=850)
        start_button = tk.Button(tkWindow,text='Start',command=lambda : Threader(args=(show1,show2,show3,show4,show5,show6,show7,start_button,stop_button,L1,L2,L3,L4,L5,L6,L7)),width=20,height=1,font=('Arial',10))
        start_button.place(x=100,y=850)
        tk.Button(tkWindow,text="Close", command = lambda:[UI.quit(tkWindow)],width=20,height=1,font=('Arial',10)).place(x=500,y=850)
        tkWindow.mainloop()

if __name__ == '__main__':
    UI()
    #df = pd.read_csv('D://CPD_LOG/CPD01/CHECK_REPORT/CPD01_check_report_202202.csv', error_bad_lines=False)
    #df = pd.read_excel('U://RECIPE_SPEC.xlsx',engine='openpyxl')
    #df.to_csv('RECIPE_SPEC.csv', index=False)