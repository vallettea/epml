from datetime import datetime, timedelta
import sys, json, glob, re, pickle
from matplotlib import finance

d1 = datetime(2011, 1, 1)
d2 = datetime(2011, 12, 31)


quotes = {}
indicators = ["OIL", "KOL"]
for indicator in indicators:
	quotes[indicator] = {}
	yquotes = finance.quotes_historical_yahoo(indicator, d1, d2)

	begin = datetime.strptime("2011-01-1T00:00Z", "%Y-%m-%dT%H:%MZ")
	for q in yquotes:

		date = datetime.fromordinal(int(q[0]))
		delta_days = (date-begin).days
		quotes[indicator][delta_days] = q[2]

for indicator in quotes.keys():
	quotes[indicator][0] = quotes[indicator][quotes[indicator].keys()[0]]
	for day in range(365):
		if not quotes[indicator].has_key(day):
			j = 0
			while quotes[indicator].has_key(day - j) == False:
				j += 1
			quotes[indicator][day] = quotes[indicator][day - j]

pickle.dump(quotes, open( "../outputs/oil.p", "wb" ) )