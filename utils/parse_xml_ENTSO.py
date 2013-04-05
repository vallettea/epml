# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from datetime import datetime
import sys, json, glob, re
import igraph

# # fetch the correspondance in eic
# eicnames = {}
# tree = ET.parse("dataset_ENTSO/allocated-eic-codes.xml")
# root = tree.getroot()

# infos = root.findall("EicInformation")
# for info in infos:
# 	iden = info.find("Identification")
# 	type = iden.find("EicType").get("v")
# 	if type == "Y":
# 		code = iden.find("EicCode").get("v")
# 		coord = info.find("Coordinates")
# 		name = coord.find("EicName").get("v")
# 		eicnames[code] = name


# national correspondance
nodes = {
			'10Y1001A1001A45N' : "SE",
			'10YFR-RTE------C' : "FR",
			'10YRO-TEL------P' : "RO",
			'10YDK-2--------M' : "DK",
			'10Y1001A1001A47J' : "SE",
			'10YCA-BULGARIA-R' : "BU",
			'10YSI-ELES-----O' : "SI",
			'10YDK-1--------W' : "DK",
			'10YGB----------A' : "GB",
			'10YCB-GERMANY--8' : "DE",
			'10YHU-MAVIR----U' : "HU",
			'10YDE-VE-------2' : "DE",
			'10YNO-5-------1W' : "NO",
			'10YCH-SWISSGRIDZ' : "CH",
			'10YAL-KESH-----5' : "AL",
			'10YBE----------2' : "BE",
			'10Y1001A1001A016' : "IR",
			'10YHR-HEP------M' : "HR",
			'10YFI-1--------U' : "FI",
			'10YUA-WEPS-----0' : "UA",
			'10YIT-GRTN-----B' : "IT",
			'10Y1001A1001A39I' : "EE",
			'10YLV-1001A00074' : "LV",
			'10Y1001A1001A46L' : "SE",
			'10YPT-REN------W' : "PT",
			'10YNO-4--------9' : "NO",
			'10YSE-1--------K' : "SE",
			'10YAT-APG------L' : "AT",
			'10YES-REE------0' : "ES",
			'10YGR-HTSO-----Y' : "GR",
			'10YBA-JPCC-----D' : "BA",
			'10Y1001A1001A44P' : "SE",
			'10YNO-1--------2' : "NO",
			'10YNO-3--------J' : "NO",
			'10YNO-2--------T' : "NO",
			'10YLT-1001A0008Q' : "LT",
			'10YNL----------L' : "NL",
			'10YMK-MEPSO----8' : "MK",
			'10YDE-EON------1' : "DE",
			'10YSK-SEPS-----K' : "SK",
			'10YPL-AREA-----S' : "PL",
			'10YCS-SERBIATSOV' : "CS",
			'10YCZ-CEPS-----N' : "CZ"}



G = igraph.Graph(directed=True)
G.add_vertices(len(list(set(nodes.values()))))
G.vs["name"] = list(set(nodes.values()))

begin = datetime.strptime("2010-12-31T23:00Z", "%Y-%m-%dT%H:%MZ")

for file in glob.glob("../dataset/*-1.xml"):

	name = re.sub("../dataset_ENTSO/ETSOVista-PhysicalFlow-(\w+)-2011-1\.xml","\g<1>" , file)
	print name

	tree = ET.parse(file)
	root = tree.getroot()

	for timeserie in root.findall("ScheduleTimeSeries"):
		inArea = nodes[timeserie.find("InArea").get("v")]
		outArea = nodes[timeserie.find("OutArea").get("v")]

		period = timeserie.find("Period")
		interval = period.find("TimeInterval").get("v")
		start , end = interval.split("/")
		start, end = datetime.strptime(start, "%Y-%m-%dT%H:%MZ"), datetime.strptime(end, "%Y-%m-%dT%H:%MZ")
		delta = (start-begin).days
		quantities = []
		for elem in period.findall("Interval"):
			if elem.find("Qty").get("v") != "":
				quantities += [int(elem.find("Qty").get("v"))]
			else:
				# attention la il n y a pas de data !!!! 0 est pas terrible
				quantities += [0]
		
		toCountry = G.vs.select(name = inArea)[0].index
		fromCountry = G.vs.select(name = outArea)[0].index

		
		try:
			idEdge = G.get_eid(fromCountry, toCountry)
		except:
			G.add_edge(fromCountry, toCountry)
			idEdge = G.get_eid(fromCountry, toCountry)
		G.es[idEdge][str(delta)] = quantities


G.write_pickle("../outputs/graph.p")

