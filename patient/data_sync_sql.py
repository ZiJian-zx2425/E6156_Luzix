import time

import pymysql
import requests
import pymysql.cursors
from io import StringIO
import os
import redivis
import pandas as pd

MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
MYSQL_USER = 'root'
MYSQL_PASSWORD = '123456'
MYSQL_DATABASE = 'bbs'

import os

os.environ["REDIVIS_API_TOKEN"] = "AAACNzi0Gg3t6AEtK82zMIb+wRLXDLfI"

def mysql_data_sync_to_framework(filepath):
    token = 'Bearer eyJhbGciOiJIUzI1NiIsImtpZCI6InhaMVpZN0p4RHlPaUhqRFMiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNjk0MDE1MTQzLCJpYXQiOjE2OTQwMTE1NDMsImlzcyI6Imh0dHBzOi8veGpyd2Vsanhta2ljam9hbGVucWQuc3VwYWJhc2UuY28vYXV0aC92MSIsInN1YiI6IjYwM2M1OTZmLWJmMzctNDY5Mi04NDAxLWVhMDllOGFkZTUzNCIsImVtYWlsIjoieHpqLmN1bWNAZ21haWwuY29tIiwicGhvbmUiOiIiLCJhcHBfbWV0YWRhdGEiOnsicHJvdmlkZXIiOiJlbWFpbCIsInByb3ZpZGVycyI6WyJlbWFpbCJdfSwidXNlcl9tZXRhZGF0YSI6e30sInJvbGUiOiJhdXRoZW50aWNhdGVkIiwiYWFsIjoiYWFsMSIsImFtciI6W3sibWV0aG9kIjoicGFzc3dvcmQiLCJ0aW1lc3RhbXAiOjE2OTM5Njc2Njl9XSwic2Vzc2lvbl9pZCI6ImEzNzM4ZmM5LWE0NjItNDdlNy1iY2Y3LWRlZWQyOTUzN2RmZiJ9.TR7cMzegUhBuw3vCXwaT9ZUjlRzslFjKA33qw2oVvO8'
    brain_id = 'e1889bed-0288-4812-b86e-58a722c4ea27'
    header = {
        'Authorization': token,
        'Connection': 'keep-alive',
        # 'Content-Type': 'multipart/form-data; boundary=----WebKitFormBoundaryFXTT4S1LKA1LUDBd',
        'Cookie': 'SHIROJSESSIONID=75ace860-0f00-4db0-9440-6c6d53cdf101',
        'Host': 'localhost:5050',
        'Origin': 'http://localhost:3000/',
        'Referer': 'http://localhost:3000/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Openai-Api-Key': '',
    }
    ### replace my token
    ### to find my API keys https://platform.openai.com/account/api-keys
    with open(filepath,'rb') as file:
        files = {'uploadFile': file}
        try:
            req = requests.post(f'http://localhost:5050/upload?brain_id={brain_id}', headers=header, files=files)
            print(req.json())
            if req.json()['type'] == 'warning' or req.json()['type'] == 'success':
                os.remove(filepath)
        except Exception as e:
            print(e)




    # with StringIO(conntent) as file:
    #     file.name = '{filename}.txt'.format(filename=filename)
    #     files = {'uploadFile': file}
    #     req = requests.post('http://localhost:5050/upload?brain_id=70dcf1e6-e8b7-4cca-afe2-3173ea1c8067',
    #                         headers=header, files=files)
    #     print(req.text)

def connect_database1():
    conn = pymysql.connect(
        host= MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        # charset='utf-8',
        cursorclass=pymysql.cursors.DictCursor
    )
    sql = 'select * from post'
    cursor = conn.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    for row in result:
        title = row['title']
        content = row['content']
        with open('./tmp/{filename}.txt'.format(filename=title),'w') as f:
            f.write(content)
        # mysql_data_sync_to_framework(content, title+str(count))

    conn.close()

def connect_redivis():
    user = redivis.user("graph_ai")
    dataset = user.dataset("hrs_rand_long:0xrw:next")
    table = dataset.table("longlong_hrs:va4p")

    # Load table as a dataframe
    df = table.to_dataframe(max_results=100)
    df.head()
    # Assuming your dataframe has columns 'title' and 'content'
    for index, row in df.iterrows():
        ID = row['HHIDPN']
        wave = row['WAVE']
        print(ID,wave)
        with open('./tmp/{filename}.txt'.format(filename=ID), 'w') as f:
            f.write(str(wave))

if __name__ == '__main__':
    step = input("请输入步骤1or2")
    print('输入为', step)
    if step == '1':
#        connect_database()
         connect_redivis()
    elif step == '2':
        filePath = './tmp'
        if not os.path.exists(filePath):
            os.makedirs(filePath)
        for file in os.listdir(filePath):
            mysql_data_sync_to_framework('{dir}/{file}'.format(dir=filePath,file=file))
            # time.sleep(10)
    else:
        print("该程序分为两个步骤1：数据先存入本地文件夹 2: 把本地数据导入到知识库")




