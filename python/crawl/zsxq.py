
import json
import re
from urllib import request
from urllib.request import urlretrieve
import os
from urllib.parse import quote
import requests
from datetime import datetime, timedelta
import random

headers = {
    'accept': "application/json, text/plain, */*",
    'origin': "https://wx.zsxq.com",
    'x-version': "1.10.39",
    'user-agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36",
    'x-request-id': "718dc82a5-cb02-6173-6c31-a1fc7af4082",
    'referer': "https://wx.zsxq.com/dweb2",
    'accept-encoding': "gzip, deflate, br",
    'accept-language': "en,zh-CN;q=0.9,zh;q=0.8,zh-TW;q=0.7",
    'cookie': "_bl_uid=evkjp7ykcmjsdLg5btsv703pkL6O; UM_distinctid=170a61f7a7d5e-04d64fa2927669-317c0e5e-1fa400-170a61f7a7eea; _ga=GA1.2.1377753453.1583335310; amplitude_id_fef1e872c952688acd962d30aa545b9ezsxq.com=eyJkZXZpY2VJZCI6IjA5Njc1MzZmLTIyOGQtNDk5My05ZmQ4LWE1OTg3ZWZkYzgyY1IiLCJ1c2VySWQiOm51bGwsIm9wdE91dCI6ZmFsc2UsInNlc3Npb25JZCI6MTU4MzMzNTMxMDIxMiwibGFzdEV2ZW50VGltZSI6MTU4MzMzNTMxMDIyMywiZXZlbnRJZCI6MSwiaWRlbnRpZnlJZCI6MSwic2VxdWVuY2VOdW1iZXIiOjJ9; zsxq_access_token=1B4843D7-F96E-5C60-3C7A-B11D86EA566D",
}


#加入的圈子id
def get_group(headers):
    url = "https://api.zsxq.com/v1.10/groups"
    #对URL发起get请求获取页面内容
    response = requests.request("GET",url,headers=headers)
    #将json对象转化为python对象
    jsonobj = json.loads(response.text)
    # print(jsonobj)
    # json1 = json.loads(jsonobj)
    # print(type(jsonobj))
    try:
        for number in range(len(jsonobj['resp_data']['preferences']['sorts'])):
            print("星球ID：",jsonobj['resp_data']['groups'][number]['group_id'])
            print("星球的名字：",jsonobj['resp_data']['groups'][number]['name'])
            print("星球的星主是：",jsonobj['resp_data']['groups'][number]['owner']['name'])
            print('\n')
    except Exception:
        print("已没有再多的星球")
    # print(type(jsonobj['resp_data']['preferences']['sorts']))
    # print(len(jsonobj['resp_data']['preferences']['sorts']))
    
#获取文件id后下载文件（zip,doc,xls  ....）
def get_gobal_file(file_url,file_name,headers,savepath='./'):
    # #对URL发起get请求获取页面内容
    response = requests.request("GET",file_url,headers=headers)
    #将json对象转化为python对象
    jsonobj = json.loads(response.text)
    # r = requests.get(jsonobj['resp_data']['download_url'])
    # with open(file_name,"wb") as f:
    #     f.write(r.content)
    # print(jsonobj)
    # print(jsonobj['resp_data']['download_url'])
    # with open(file_name,"wb") as f:
    #     f.write()
    # for download_url in jsonobj['resp_data']['download_url']:
    #     pass
    def reporthook(a, b, c):
        print("\r  Downloading: %5.1f%%" % (a * b * 100.0 / c), end="")

    filename = os.path.basename(file_name)
    if not os.path.isfile(os.path.join(savepath, filename)):
        print('Downloading data from %s' % jsonobj['resp_data']['download_url'])
        urlretrieve(jsonobj['resp_data']['download_url'], os.path.join(savepath,filename), reporthook=reporthook)
        print('\nDownloading finished')
    else:
        print('File already exists!')

    filesize = os.path.getsize(os.path.join(savepath,filename))

    print('File size = %.2f Mb' % (filesize/1024/1024))

#生成时间对应的格式
def time():
    now = datetime.now()
    print(now)
    time1 = now.strftime('%Y-%m-%d')
    print(time1)
    time2 = now.strftime('%H:%M:%S')
    print(time2)
    number = str(random.randint(100,999))
    # print (type(random.randint(100,999)))
    end_time = time1+'T'+time2+'.'+number+'+0800'
    return end_time




#获取某个星球下的文件id
def get_download_url(headers,group_id):
    # url = "https://api.zsxq.com/v1.10/groups/"+group_id+"/files?count=3"
    end_time = time()
    url = "https://api.zsxq.com/v1.10/groups/"+group_id+"/files?count=20&end_time="+quote(end_time)
    print(url)
    #对URL发起get请求获取页面内容
    response = requests.request("GET",url,headers=headers)
    jsonobj = json.loads(response.text)
    print(type(jsonobj))
    # print(jsonobj['resp_data']['files'][0]['file']['file_id'])
    # print(len(jsonobj['resp_data']['files']))
    try:
        for file_id in range(len(jsonobj['resp_data']['files'])):
            file_url = 'https://api.zsxq.com/v1.10/files/'+str(jsonobj['resp_data']['files'][file_id]['file']['file_id'])+'/download_url'
            print(file_url)
            file_name = jsonobj['resp_data']['files'][file_id]['file']['name']
            print(file_name)
            get_gobal_file(file_url,file_name,headers,savepath='./')
            file_write(file_name,file_url)
    except IOError as e:
        print('Not have any files!')
    

def file_write(file_name,file_url):
    try:
        f = open('test1.log','a',encoding='utf-8')
        f.write(file_name+'\n')
        f.write(file_url+'\n\n')
        f.close()
    except IOError as e:
        print('读写文件失败！')


if __name__ == "__main__":
    get_group(headers)
    # get_gobal_file(headers)
    group_id = input("请输入星球id：")
    get_download_url(headers,group_id)
