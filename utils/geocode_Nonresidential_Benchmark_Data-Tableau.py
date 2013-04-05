import time, sys, csv, random, json
from geopy import geocoders



csvfile = open("../data/Nonresidential_Benchmark_Data-Tableau_1.csv", "rb")
spamreader = csv.reader(csvfile, delimiter=";")
next(spamreader) # skip first line
geocoder1 = geocoders.Google()
geocoder2 = geocoders.Yahoo("RV0rLODV34HFs15lnhvC_uvrcNtjDoLam8B8gyebMu1SjfF24O4RQemocQRxnjQRb1isHL7NQPFchw")  

input = open("../data/Nonresidential_Benchmark_Data-Tableau_geocoded.json", "r")
geocoded = json.loads(input.read())
input.close()

for row in spamreader:
    bbl = row[0]
    house_num = row[1]
    street = row[2]
    district = row[3]

    if geocoded.has_key(str(bbl)):
        print "done already"
    else:
        # resolve adress
        print house_num, street, district
        try:
            place, (lat, lon) = geocoder.geocode("%s %s, %s" % (house_num, street.lower(), district))  
            print lat,lon
            geocoded[bbl] = (lon, lat)
        except:
            try:
                place, (lat, lon) = geocoder2.geocode("%s %s, %s" % (house_num, street.lower(), district))
                print lat,lon
                geocoded[bbl] = (lon, lat)
            except:
                print "missed adress"

output = open("../data/Nonresidential_Benchmark_Data-Tableau_geocoded.json", "w")
output.write(json.dumps(geocoded))
output.close()

