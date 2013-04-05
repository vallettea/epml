import urllib2
import datetime, time, json

import json

req = urllib2.Request("http://openweathermap.org/data/2.1/history/city/2988507?type=hour&strart=1293832800")
opener = urllib2.build_opener()
 
f = opener.open(req)
data = json.load(f)

list = data["list"]

for x in list:
    print time.strftime("%D %H:%M", time.localtime(x["dt"]))



start =int(time.mktime(time.strptime('2010-12-31 23:00:00', '%Y-%m-%d %H:%M:%S')))

