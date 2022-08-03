import pandas as pd
import time
import sqlite3
import requests
from datetime import datetime, timedelta 
import win32com.client as win32  


def send_mail(reciver = './Result/test.csv', lists = './Result/HR.csv'):
    recive = pd.read_csv(reciver)
    hr = pd.read_csv(lists)
    df2 = pd.merge(recive, hr, on="License_Plate", how='inner')
    for i in df2.index:
        mail = win32.Dispatch('outlook.application').CreateItem(0)
        mail.To = df2.iloc[i]['Mail']
        mail.Subject = df2.iloc[i]['AI_Predict_Time'] + '違規'
        attachment = mail.Attachments.Add(df2.iloc[i]['original'])
        attachment.PropertyAccessor.SetProperty("http://schemas.microsoft.com/mapi/proptag/0x3712001F", "MyId1")
        mail.HTMLBody = "<html><body><img src=""cid:MyId1""></body></html>"
        mail.Send()


def upload_data(path, file):
    data={'site':'L6B','Path':'ML6B01/Arthurliu'}
    file1 = [
        ('files',(file, open(path,'rb')))
    ]
    response = requests.post('http://autceda/files/MultiUpload', files=file1, data=data)
    print(response.text)


def SQLite2csv(db = 'D:/ML6B01/環安專案/SQLite/Environment_Safety.db'):
    dest = './Result/Traffic_result.csv'
    dateRange = datetime.strftime(datetime.now() - timedelta(60), '%Y-%m-%d') 
    conn = sqlite3.connect(db)
    df = pd.read_sql("SELECT * FROM Traffic WHERE DATE > '" + dateRange + "'", conn)
    print(df)
    df.to_csv(dest, index=False)
    conn.close()
    return dest


def df2SQLite(df):
    conn = sqlite3.connect('D:/ML6B01/環安專案/SQLite/Environment_Safety.db')
    df.to_sql('Traffic', conn, if_exists='append', index=False)
    print('Successfully entered')
    conn.close()
	

def Process(path = './Result/2022-06-15.csv'):
    print(path)
    df1 = pd.read_csv(path)
    #path = './Result/test.csv'

    group = df1.groupby(['Date','Red_light_Time','AI_Predict_Time','License_Plate'])
    df2 = (group.size().reset_index(name='License_Plate_cnt'))
    df2.drop(['Date','Red_light_Time','AI_Predict_Time'], axis=1, inplace=True)
    #print(df2)

    df2 = pd.merge(df1, df2, on="License_Plate", how='inner')
    #print(df2)
    
    df2 = df2[df2['License_Plate_cnt']>=2]
    df2.drop_duplicates(subset='License_Plate', keep='first', inplace=True)
    df2['lm_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(df2)

    df2.to_csv(path, index=False)    
    df2SQLite(df2)

    path = SQLite2csv()
    upload_data(path, path.split('/')[-1])
    
    #send_mail(reciver = path, lists = 'HR.csv')

if __name__ == '__main__':
    Process()   # Test