



import requests
import json
import urllib
import csv
import re
import ipdb
import datetime
from pq import PQ

#头信息。网站只提供扫码登陆的方式，没有账号密码。我以为应该比较麻烦，但在header信息里找到了Authorization信息之后，直接可以保持登陆状态了。
# 令一个标志是直接在浏览器里访问内页网址的话，浏览器的报错是“{"succeeded":false,"code":401,"info":"","resp_data":{}}”，这个很像原来node.js的数据中心没有登陆的报错，而数据中心的模拟登陆也是通过在header中添加Authorization来实现的。
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36',
    'Referer': 'https://wx.zsxq.com/dweb2/',
    'Cookie': 'UM_distinctid=170a61f7a7d5e-04d64fa2927669-317c0e5e-1fa400-170a61f7a7eea; _ga=GA1.2.1377753453.1583335310; zsxq_access_token=1B4843D7-F96E-5C60-3C7A-B11D86EA566D; abtest_env=product; _gid=GA1.2.2097390934.1583575945; amplitude_id_fef1e872c952688acd962d30aa545b9ezsxq.com=eyJkZXZpY2VJZCI6IjA5Njc1MzZmLTIyOGQtNDk5My05ZmQ4LWE1OTg3ZWZkYzgyY1IiLCJ1c2VySWQiOm51bGwsIm9wdE91dCI6ZmFsc2UsInNlc3Npb25JZCI6MTU4MzU3NTk0NjU2OCwibGFzdEV2ZW50VGltZSI6MTU4MzU3NTk0NjU4MiwiZXZlbnRJZCI6MSwiaWRlbnRpZnlJZCI6MSwic2VxdWVuY2VOdW1iZXIiOjJ9'
}


def handle_link(text):
    result =  re.findall(r'<e\ [^>]*>', text)
    for i in result:
        html = PQ(i)
        if html.attr('type') == 'web':
            template = '[%s](%s)' % (urllib.parse.unquote(html.attr('title')), urllib.parse.unquote(html.attr('href')))
        elif html.attr('type') == 'hashtag':
            template = ' `%s` ' % urllib.parse.unquote(html.attr('title'))
        elif html.attr('type') == 'mention':
            template = urllib.parse.unquote(html.attr('title'))
        text = text.strip().replace(i, template)
    return text



"""
"resp_data": {
"topics": [
{
"topic_id": 421881845481158,
"group": {
"group_id": 15284285842552,
"name": "OptimalLR深度学习"
},
"type": "talk",
"talk": {
"owner": {
"user_id": 48245481255848,
"name": "Bryce1010",
"avatar_url": "https://images.zsxq.com/FjQ1wWt4VYSZ8aCZs02jUr8_ZGL_?e=1588262399&token=kIxbL07-8jAj8w1n4s9zv64FuZZNEATmlU_Vm6zD:69N0Fe4UkQOtdlZUEnmodUrwBME=",
"description": "在路上的深度学习者"
},
"text": "<e type="hashtag" hid="824522452882" title="%23C%2B%2B%23" /> 
modern-cpp-features  
C++20/17/14/11
[[github]]( <e type="web" href="https%3A%2F%2Fgithub.com%2FAnthonyCalandra%2Fmodern-cpp-features" title="GitHub+-+AnthonyCalandra%2Fmodern-cpp-features%3A+A+ch..." /> )"
},
"likes_count": 0,
"rewards_count": 0,
"comments_count": 0,
"reading_count": 22,
"digested": false,
"sticky": false,
"create_time": "2020-03-07T14:33:11.334+0800",
"user_specific": {
"liked": false,
"subscribed": false
}
},
"""
f=open('./output/out.csv','w+')
writer=csv.writer(f)
writer.writerow(['created_time','talk'])

def get_info(url):
    res=requests.get(url,headers=headers)
    json_data=json.loads(res.text)
    datas=json_data['resp_data']['topics']

    for data in datas:
        if 'talk' in data.keys(): # 判断json中是否包含talk这个key
            text=data['talk']['text']
            result =  re.findall(r'<e\ [^>]*>', text)
            for i in result:
                """
                #html = PQ(i)
                # i=PQ(i)
                if 'web' in i:
                    #template = '[%s](%s)' % (urllib.parse.unquote(i.attr('title')), urllib.parse.unquote(i.attr('href')))

                elif 'hashtag' in i:
                    template = ' `%s` ' % urllib.parse.unquote(i.attr('title'))
                elif i.attr('type') == 'mention':
                    template = urllib.parse.unquote(i.attr('title'))
                text = text.strip().replace(i, template)
                """
                text=text.strip().replace(i,"")
            create_time=data['create_time']
            year=create_time.split('-')[0]
            month=create_time.split('-')[1]
            day=create_time.split('-')[2][:2]
            day_a=datetime.date(int(year),int(month),int(day))
            #b=datetime(int(set_year),int(set_month),int(set_day))
            day_b=datetime.date(2020,3,6)
            if(day_b.__gt__(day_a)): 
                return False
            print(create_time)
            writer.writerow([create_time,text])
    """
    end_time=datas[19]['create_time']
    url_encode=urllib.parse.quote(end_time)
    next_url='https://api.zsxq.com/v1.10/groups/15284285842552/topics?count=20&end_time'+url_encode
    get_info(next_url)
    """

if __name__=="__main__":
    url='https://api.zsxq.com/v1.10/groups/15284285842552/topics?count=20'
    get_info(url)


