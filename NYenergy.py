import time, sys, os, csv, random, json,sys, logging
import matplotlib
import numpy as np
import pylab as plt
from scipy.stats.stats import pearsonr
from sklearn import svm, metrics, preprocessing
from sklearn import cross_validation
from imposm.parser import OSMParser

import citygraph

cityname = "NewYork"
original_pbf = "/Users/vallette/projects/DATA/citygraph/pbf/new-york.osm.pbf"

families = {
    "bank":{"amenity":["bank", "atm", "bureau_de_change"]},
    "entertainment":{"amenity":["cinema","community_center","nightclub","stripclub","theatre"],
            "historic":"*",
            "leisure":"*",
            "sport":"*",
            "tourism":"*",
            },
    "storage":{
            "building":["industrial","warehouse"],
            },
    "health":{
            "amenity":["baby_hatch","clinic","dentist","doctors","hospital","social_facility"]
            },
    "education":{
            "amenity":["colledge","kindergarten","library","school","university"]
            },
    "hotel":{
            "building":["hotel",]
            },
    "living":{
            "building":["apartments","dormitory","house","residential"]
            },
    "commercial":{"shop":"*",
            "amenity":["car_rental","marketplace"],
            "building":["commercial","retail"],
            "landuse":["commercial",]
            },
    "office":{"office":"*", 
            }
            }

logger = logging.getLogger('root')
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
handler = logging.FileHandler("/Users/vallette/projects/DATA/epml/outputs/log_%s.log" % cityname)
handler.setFormatter(formatter)
logger.addHandler(handler)


class Building(citygraph.Feature):
    def __init__(self, id, lon, lat, btype, conso):
        self.bbl = id
        self.btype = btype
        self.conso = conso
        if conso != None:
            super(Building, self).__init__(lon, lat, "monitored_building", "building")
        else:
            super(Building, self).__init__(lon, lat, "unmonitored_building", "building")

class ElecSection(object):
    def __init__(self, section, input):
        self.section = section
        self.input = input
        self.result = False

class NewCity(citygraph.City):
    def __init__(self, name, top, left, bottom, right):
        super(NewCity, self).__init__(name, top, left, bottom, right)
        self.classifier = None
        self.elecsections = []


    def load_osm_features(self, pbf_path , families):
        logger.info("    Adding feature from OSM.")
        pbf_path, pbf_name = os.path.split(pbf_path)
        categories = {}
        for family in families:
            locations = {}

            class FamilyParser(object):
                ways_objects = {}
                nodes_objects = {}
                all_nodes = {}

                def ways(self, ways):
                    for osmid, tags, refs in ways:
                        for key in families[family].keys():
                            if key in tags.keys():
                                if families[family][key] == "*" or tags[key] in families[family][key]:
                                    self.ways_objects[int(osmid)] = {"nd" : refs, "tag" : tags[key]}

                def nodes(self, nodes):
                    for osmid, tags, coords in nodes:
                        for key in families[family].keys():
                            if key in tags.keys():
                                if families[family][key] == "*" or tags[key] in families[family][key]:
                                    self.nodes_objects[int(osmid)] = {"lon" : coords[0], "lat" : coords[1], "tag" : tags[key]}

                def coords(self, coords):
                    for coords in coords:
                        self.all_nodes[int(coords[0])] = (coords[1], coords[2])

            parser = FamilyParser()
            p = OSMParser(concurrency = 2, ways_callback = parser.ways, nodes_callback = parser.nodes, coords_callback = parser.coords)
            p.parse("%s/%s.osm.pbf" % (pbf_path, self.name))

            for k in parser.nodes_objects.keys():
                locations[k] = (parser.nodes_objects[k]["lon"], parser.nodes_objects[k]["lat"], parser.nodes_objects[k]["tag"])
            for k in parser.ways_objects.keys():
                # compute mean postion
                m_lon, m_lat = 0, 0
                for nd in parser.ways_objects[k]["nd"]:
                    node = parser.all_nodes[int(nd)]
                    m_lon += node[0]
                    m_lat += node[1]
                m_lon /= float(len(parser.ways_objects[k]["nd"]))
                m_lat /= float(len(parser.ways_objects[k]["nd"]))
                locations[nd] = (m_lon, m_lat, parser.ways_objects[k]["tag"]) #a way can have the same id as a node so we take the id of the last node

            logger.info("Number of elements in family %s : %s" % (family, str(len(locations.keys()))))

            categories[family] = locations

        for category in categories.keys():
            for id in categories[category].keys():
                building = Building(int(id), categories[category][id][0], categories[category][id][1], category, None)
                self.features += [building]

    def load_electricity_consumption(self):
        category_map = {
            '': "unknown", 
            'Other': "unknown",
            'Data Center': "unknown",
            'Bank/Financial Institution': "bank", 
            'Entertainment/Culture': "entertainment", 
            'Self-Storage': "storage", 
            'Health Care: Outpatient': "health", 
            'College/University (Campus-Level)': "education", 
            'Senior Care Facility': "health", 
            'Hotel': "hotel", 
            'Multifamily Housing': "living", 
            'Hospital (General Medical and Surgical)': "health", 
            'Recreation': "entertainment", 
            'Medical Office': "health", 
            'Warehouse (Unrefrigerated)': "storage", 
            'Residence Hall/Dormitory': "living", 
            'Retail': "commercial", 
            'Storage/Shipping/Non-Refrigerated Warehouse': "storage", 
            'Office': "office", 
            'Retail (Misc)': "commercial", 
            'Service (Vehicle Repair/Service, Postal Service)': "commercial", 
            'Clinic/Other Outpatient Health': "health", 
            'Lodging': "living", 
            'K-12 School': "education", 
            'Education': "education", 
            'House of Worship': "entertainment", 
            "Health Care: Inpatient (Specialty Hospitals, Excluding Children's)": "health",
            'Automobile Dealership': "commercial",
            'Public Assembly': "entertainment", 
            'Social/Meeting': "entertainment",
            'Warehouse (Refrigerated)': "storage",
            'Supermarket/Grocery': "commercial"
        }
        logger.info("    Adding features from Nonresidential Benchmark Data-Tableau 1")
        with open("/Users/vallette/projects/DATA/epml/csv/Nonresidential_Benchmark_Data-Tableau_1.csv", "rb") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=";")
            next(spamreader) # skip first line
            geofile = open("/Users/vallette/projects/DATA/epml/json/Nonresidential_Benchmark_Data-Tableau_geocoded.json", "r")
            geocoded = json.loads(geofile.read())
            n_id = 0
            types = []
            for row in spamreader:
                bbl = int(row[0])
                house_num = row[1]
                street = row[2]
                district = row[3]
                if geocoded.has_key(str(bbl)):
                    lon, lat = geocoded[str(bbl)]
                    if self.point_inside_polygon(lon, lat):
                        try:
                            conso = float(row[10].replace(",","."))
                            btype = row[13]
                            types += [btype]
                            if category_map.has_key(btype):
                                if category_map[btype] != "unknown":
                                    building = Building(bbl, lon, lat, category_map[btype], conso)
                                    self.features += [building]
                                    n_id += 1
                            else:
                                print "Missing mapping for %s" % btype
                                f=lambda s,d={}:([d.__setitem__(i,d.get(i,0)+1) for i in s],d)[-1]
                                print f(types)
                                sys.exit(0)
                        except:
                            pass
            
        logger.info("     %s buildings with conso" % str(n_id))

    def create_ElecSections(self):
        self.elecsections = []
        maximums = np.zeros(13)
        minimums = np.zeros(13)
        resmax = 0
        resmin = 0
        for section in self.sections.values():
            x,y = self.map.map(np.mean([n.lon for n in section.nodes]), np.mean([n.lat for n in section.nodes]))

            type_dict = {'living_street' : 1.,'residential' : 2.,'primary': 6.,'unclassified': 3.,'tertiary' : 4.,'secondary': 5.}
            section_type = type_dict[section.type]

            list = [x, y, section.Vmax, section_type]

            cat_index = {"bank":0, "entertainment":1, "storage":2, "health":3, "education":4, "hotel":5, "living":6, "commercial":7, "office":8}
            tab = np.zeros(9)
            for building in section.features:
                tab[cat_index[building.btype]] += 1
            if sum(tab) > 0:
                input_vec = np.append(np.array(list), tab)
                elecsection = ElecSection(section, input_vec)

                maximums = np.maximum(maximums, input_vec)
                minimums = np.minimum(minimums, input_vec)

                buildings = [f for f in section.features if f.tag == "monitored_building"]
                if len(buildings) > 0:
                    mean_conso = sum([float(b.conso) for b in buildings])
                    resmax = np.max([resmax, mean_conso])
                    resmin = np.min([resmin, mean_conso])
                    elecsection.result = float(mean_conso)
                
                self.elecsections += [elecsection]

        # normalization
        for elecsection in self.elecsections:
            elecsection.input = (elecsection.input-minimums)/(maximums-minimums)
            if elecsection.result:
                elecsection.result = (elecsection.result - resmin)/(resmax - resmin)

    def learn(self):
        #make data
        data = np.array([], dtype="float64", order="C")
        result = np.array([], dtype="float64", order="C")
        for elecsection in self.elecsections:
            if elecsection.result:
                result = np.append(result, elecsection.result)
                if len(data) > 0:
                    data = np.vstack((data, elecsection.input))
                else:
                    data = np.array(elecsection.input)

        n_samples = len(result)
        logger.info("%s samples" % str(n_samples))
        indexer = np.arange(n_samples)
        random.shuffle(indexer)
        result_s = result[indexer]
        data_s = data[indexer]

        # learn
        size = n_samples/2
        self.classifier = svm.NuSVR()
        self.classifier.fit(data_s[:size], result_s[:size])
        expected = result_s[size:]
        predicted = self.classifier.predict(data_s[size:])
        logger.info("    Pearson coefficient: %s" % str(pearsonr(expected,predicted)[0]))
        scores = cross_validation.cross_val_score(self.classifier, data_s, result_s, cv=5)
        logger.info("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))

    def plot(self, list):
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)

        if "highways" in list:
            for highway in self.highways:
                lons = [node.lon for node in highway.nodes]
                lats = [node.lat for node in highway.nodes]
                x, y = self.map.map(lons, lats)
                ax.plot(x, y, "-", linewidth = 0.5, color = "k")

        if "buildings" in list:
            for feature in self.features:
                x, y = self.map.map(feature.lon, feature.lat)
                if feature.tag == "monitored_building":
                    ax.scatter(x, y, marker = "o", color = "r", s = 3)
                else:
                    ax.scatter(x, y, marker = "o", color = "b", s = 2)

        
        if "heat" in list:
            xall, yall, call = [], [], []
            cm = plt.get_cmap('YlOrRd') 
            cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=0.1)
            scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
            for elecsection in self.elecsections:
                section = elecsection.section
                lons = [node.lon for node in section.nodes]
                lats = [node.lat for node in section.nodes]
                x, y = self.map.map(lons, lats)
                colorVal = scalarMap.to_rgba(self.classifier.predict(elecsection.input)[0])
                ax.plot(x, y, "-", linewidth = 5, color=colorVal, alpha=0.5)
        
        ax.axis("equal")
        plt.xticks([])
        plt.yticks([])
        if "pdf" in list:
            fig = plt.gcf()
            fig.set_size_inches(20,20)
            plt.savefig("fig.pdf")
        else:
            plt.show()

    def feature_distribution(self):
        hist_families = {"bank":[], "entertainment":[], "storage":[], "health":[], "education":[], "hotel":[], "living":[], "commercial":[], "office":[]}

        for building in self.features:
            hist_families[building.btype] += [building.conso]
        importances = np.array([np.mean(hist_families[family]) for family in hist_families.keys()])
        std = np.array([np.std(hist_families[family]) for family in hist_families.keys()])
        indices = np.argsort(importances)[::-1]

        fig = plt.figure()
        plt.title("Feature importances")
        ax = fig.add_subplot(111)
        ax.bar(xrange(len(hist_families.keys())), importances[indices], color="r", yerr=std[indices], align="center")
        plt.xticks(xrange(len(hist_families.keys())), [hist_families.keys()[i] for i in indices],rotation=90)
        plt.xlim([-1, len(hist_families.keys())])
        plt.show()


NY = NewCity(cityname, top=41.162, left=-74.593, bottom=40.229, right=-73.365)
NY.load_highways_from_osm(original_pbf, [189005738, 189020941, 46936809, 37788309, 37788311, 46945175, 37788079, 37788080, 37788405, 37788412, 37788493, 37811053, 37787073,
                           37787072, 37787122, 37811067, 37787976, 37787975, 37787971, 37788522, 61603751, 61928627, 61922976, 61603744, 61923037, 61909165, 37565950,
                           37565866, 37565868, 37565564, 37565158, 37565053, 37565052, 61603753, 37565051, 37569166, 61602967, 61602968, 61602961, 61602961, 61602969,
                           61602963, 61602964, 61602965, 37567463, 37567690, 189026085, 37567199, 37567197], reverse = True, redo_pbf = False)

NY.make_sections()
NY.compute_connectivity()
NY.clean_sections(plot_check = False)

NY.load_graph()
NY.load_osm_features(original_pbf, families)

NY.load_electricity_consumption()
NY.associate_features(precision = 5)

NY.create_ElecSections()
NY.learn()



# import json
# out = []

# for elecsection in NY.elecsections:
#     section = elecsection.section
#     lons = [node.lon for node in section.nodes]
#     lats = [node.lat for node in section.nodes]
#     lonlat = [[lon,lat] for lon, lat in zip(lons, lats)]
#     val = NY.classifier.predict(elecsection.input)[0]
#     s = {"coords" : lonlat, "id" : section.id, "value" : val}
#     out += [s]

# outfile = open("empl.json", "w")
# outfile.write(json.dumps(out))
# outfile.close()





