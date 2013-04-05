from datetime import datetime, timedelta
import sys, json, glob, re, pickle

# stations

names = { 
"Finland": "FI",
"Ireland": "IR",
"Iceland": "IC",
"Denmark": "DK",
"Belgium": "BE",
"France": "FR",
"Germany": "DE",
"Austria": "AT",
"Czech-republic": "CZ",
"Hungary": "HU",
"Albania": "AL",
"Croatia": "HR",
"Bosnia-and-herzegovina": "BA",
"Bulgaria": "BU",
"Italy": "IT",
"Greece": "GR",
"Estonia": "EE",
"Belarus": "no_europe",
"Georgia": "no_europe",
"Armenia": "no_europe",
"Norway": "NO",
"Sweden": "SE",
"United-kingdom": "GB",
"Netherlands": "NL",
"Switzerland": "CH",
"Spain": "ES",
"Portugal": "PT",
"Slovakia": "SK",
"Poland": "PL",
"Serbia": "CS",
"Montenegro": "no_europe",
"Macedonia": "MK",
"Slovenia": "SI",
"Romania": "RO",
"Latvia": "LV",
"Lithuania": "LT",
"Moldova": "no_europe",
"Ukraine" : "UA"
}
stations = {}
infile = open("../dataset_meteo/stations.txt", "r")
infile.readline()
infile.readline()
for line in infile.readlines():
	tab = re.split("\s+", line)
	usaf = tab[0]
	country = tab[3]
	country = country[0]+country[1:].lower()
	stations[usaf] = names[country]


infile.close()


# data

def good_temp(temp):
	try:
		return int(temp)
	except:
		return False

meteo = {}
begin = datetime.strptime("2010-12-31T23:00Z", "%Y-%m-%dT%H:%MZ")
infile = open("../dataset_meteo/data.txt", "r")
infile.readline()
num_line = 1
prev_delta = 0
prev_hour = begin
for line in infile.readlines():
	num_line+=1
	usaf = line[0:6]
	country = stations[usaf]
	tt = line[13:25]
	hour = datetime.strptime(tt, "%Y%m%d%H%M")
	if country != "no_europe":           # if it is an european country
		if hour.minute != 0:
			hour = hour - timedelta(minutes = hour.minute)
		if hour.minute == 0:
			try:
				temp = int(line[84:87])
			except:
				temp = prev_temp
				print "Problem 4: replacing by older temp."                  

			delta_days = str((hour-begin).days)
			if meteo.has_key(country):
				delta_hours = (hour-prev_hour).seconds/3600
				if meteo[country].has_key(delta_days):
					if delta_hours == 1:
						meteo[country][delta_days] += [temp]
					elif delta_hours == 0:
						pass
					else:
						meteo[country][delta_days] += [temp]*delta_hours
						print "Problem 2: filling a %s h gap." % str(delta_hours-1)
				else:
					if delta_hours == 0 and delta_days != delta_days_prev:
						meteo[country][delta_days_prev] += [temp]*(22 - prev_hour.hour)
					if delta_hours == 1:
						meteo[country][delta_days] = [temp,]
					else:
						if hour.hour == 23:
							meteo[country][delta_days_prev] += [temp]*(delta_hours-1)
							meteo[country][delta_days] = [temp,]
							print "Problem 3: filling a %s h gap." % str(delta_hours-1)
						else:
							step = (23 + hour.hour) % 23 + 2
							meteo[country][delta_days_prev] += [temp]*(delta_hours - step)
							meteo[country][delta_days] = [temp]*step
							print "Problem 5: filling a %s h gap, line %s" % (str(delta_hours-1), str(num_line))
					if len(meteo[country][delta_days_prev]) != 24:
						print num_line
						print line
						sys.exit(0)

			else:
				if delta_days == "0":
					if hour.hour == 23:
						meteo[country] = {delta_days : [temp]}
					else:
						step = (23 + hour.hour) % 23 + 2
						meteo[country] = {delta_days : [temp]*step}
				else:
					print "Problem 1"
					sys.exit(0)
			prev_temp = temp
			delta_days_prev = delta_days
			prev_hour = hour


# corret
for pays in meteo.keys():
	for day in meteo[pays].keys():
		if len(meteo[pays][day]) == 1 and day == '365':
			del meteo[pays][day]

	# correct missing day
	l = [int(x) for x in meteo[pays].keys()]
	l.sort()
	xprev = 0
	for x in l:
		if x-xprev >1:
			meteo[pays][str(x-1)] = [int(sum(meteo[pays][str(xprev)])/float(len(meteo[pays][str(xprev)])))]*24
		xprev = x
	if l[-1] == 363:
		meteo[pays]["364"] = [int(sum(meteo[pays]["363"])/float(len(meteo[pays]["363"])))]*24
	# correct incoplete days
	for day in meteo[pays].keys():
		if len(meteo[pays][day]) != 24:
			meteo[pays][day] = meteo[pays][day] + [meteo[pays][day][-1]]*(24-len(meteo[pays][day]))


# check
def check():
	for pays in meteo.keys():
		if len(meteo[pays].keys()) != 365:
			print  pays,len(meteo[pays].keys())
		for day in meteo[pays].keys():
			if len(meteo[pays][day]) != 24:
				print pays, day, len(meteo[pays][day])
check()

pickle.dump( meteo, open( "../outputs/meteo.p", "wb" ) )























