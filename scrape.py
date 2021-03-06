import string
import os

from IPython.core.release import description
from lxml import html
import requests
from queue import Queue
from enum import Enum
from threading import Thread
import queue
import time
import bs4
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder

SYMPTOM_DETAILS_CSV = "symptomDetails.csv"

CAUSEDETAILS_CSV = "causedetails.csv"

CAUSESYM_CSV = "causesym.csv"

DEFAULT_EXPORT_PATH = "mayoexport"


class JobType(Enum):
    SYMPTOM_LIST = 0
    SYMPTOM = 1
    CAUSES = 2
    DISEASE = 3

class CauseType(Enum):
    DISEASE = 0 # and conditions
    PROCEDURE = 1
    HEALTHY_LIFESTYLE = 2
    SYMPTOM = 3
    FIRST_AID = 4
    OTHER = 5


class MayoBase:
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    BASE_DIR = "mayo"
    causeType = {"Diseases and Conditions": CauseType.DISEASE,
                 "Diseases & Conditions": CauseType.DISEASE,
                 "Tests and Procedures": CauseType.PROCEDURE,
                 "Tests & Procedures": CauseType.PROCEDURE,
                 "Healthy Lifestyle": CauseType.HEALTHY_LIFESTYLE,
                 "Symptoms": CauseType.SYMPTOM,
                 "First aid": CauseType.FIRST_AID,
                 "Other": CauseType.OTHER
                 }

    @staticmethod
    def cause_name_to_type(causeName):
        return MayoBase.causeType[causeName]

    @staticmethod
    def request(address, prefixDir):
        path = MayoBase.address_to_path(address, prefixDir)
        content = ""
        if os.path.exists(path) and os.stat(path).st_size != 0:
            with open(path, 'r') as f:
                content = f.read()

        else:
            print("Requesting:", address)
            content = MayoBase._request_address(address)

            with open(path, 'w', encoding="utf-8") as f:
                f.write(content)


        return content

    @staticmethod
    def _request_address(address):
        content = ""
        start = time.clock()
        timeout = 10
        while content == "" :
            page = requests.get(address)
            page.raise_for_status()
            content = page.content.decode("utf-8")
            if content == "" and time.clock() - timeout < start:
                os.sleep(1000)
                print("Retrying")
            else:
                break
        return content

    @staticmethod
    def address_to_path(address, prefixDir):
        dir = os.path.join(MayoBase.BASE_DIR, prefixDir)
        if not os.path.exists(dir):
            os.makedirs(dir)
        name = address.split('/')[-1]
        name = "".join(c for c in name if c in MayoBase.valid_chars)
        path = os.path.join(dir, name)
        return path

    @staticmethod
    def is_scraped(address, prefixDir):
        path = MayoBase.address_to_path(address, prefixDir)
        content = ""
        if os.path.exists(path):
            return True
        return False



class MayoWorker(Thread):
    '''
    '''
    BASE_ADDRESS = "http://www.mayoclinic.org"

    def __init__(self, queue):
        super(MayoWorker, self).__init__()
        self.queue = queue
        self.jobHandlers = {
            JobType.SYMPTOM_LIST: self.scrape_symptom_list,
            JobType.SYMPTOM : self.scrape_symptom,
            JobType.CAUSES : self.scrape_causes,
            JobType.DISEASE : self.scrape_disease
        }

    def run(self):
        while True:
            if self.queue.empty():
                print("Worker exited normally")
                return # todo: stopping indication
            try:
                job = self.queue.get()
                handler = self.jobHandlers[job["type"]]
                job["address"] = self.make_address(job["address"])
                handler( **job)
            except queue.Empty:
                return # todo: as above


    def scrape_symptom_list(self, **kwargs):
        '''
        Scrapes a list of symptoms on a page.
        :return: a list of new jobs generated by this scrape
        '''

        content = MayoBase.request(kwargs["address"], "symptom_list")

        tree = bs4.BeautifulSoup(content)
        symptomLinks = tree.select("#index > ol > li > a")
        for symptom in symptomLinks:
            href = symptom.attrs["href"]
            causesHref = href.replace("definition", "causes")
            self.queue.put({"type": JobType.SYMPTOM, "address" : href})
            self.queue.put({"type": JobType.CAUSES, "address" : causesHref})


    def scrape_symptom(self, **kwargs):
        content = MayoBase.request(kwargs["address"], "symptom")

    def scrape_causes(self, **kwargs):
        content = MayoBase.request(kwargs["address"], "symcauses")
        tree = bs4.BeautifulSoup(content, "lxml")
        causeLinks = tree.select("div#main-content > ol > li > a")
        for cause in causeLinks:
            # self.queue.put({"disease": JobType.})
            href = cause["href"]
            self.queue.put({"type" : JobType.DISEASE, "address" : href})


    def scrape_disease(self, **kwargs):
        content = MayoBase.request(kwargs["address"], "disease")
        tree = bs4.BeautifulSoup(content, "lxml")


    def make_address(self, *args):
        return MayoWorker.BASE_ADDRESS + "".join(args)

class MayoScraper(MayoBase):
    '''
    '''
    BASE_ADDRESS = "http://www.mayoclinic.org"
    LINKS_ADDRESS = "/symptoms/index?letter="
    def __init__(self, numWorkers=1):
        self.jobs = Queue()
        self.numWorkers = numWorkers

    def scrape(self):
        for letter in string.ascii_uppercase:
            listAddress = MayoScraper.LINKS_ADDRESS + letter.lower()
            self.jobs.put({"type": JobType.SYMPTOM_LIST, "address": listAddress})
        self.start_workers()

    def start_workers(self):
        for i in range(self.numWorkers):
            worker = MayoWorker(self.jobs)
            worker.start()

def id_from_str(value):
    mo = re.search(r"([0-9]+)$", value)
    if mo == None:
        return None
    return mo.group(0)

def _collect_symcause_data(basedir):
    '''
    Collects the data from basedir in unprocessed form.
    :param basedir:
    :return: a list. Each entry is a tuple comprising of:
    id - the id of a symptom
    name - the name of a symptom (may contain a synonym in parentheses
    causes - a list of cause-tuples, each tuple comprising of
        id - the id of the cause
        name - the name of the cause (may contain a synonym in parentheses
    '''
    causespath = os.path.join(basedir, "symcauses")
    data = []
    for fn in os.listdir(causespath):
        path = os.path.join(causespath, fn)
        if os.path.isfile(path):

            with open(path, mode="r") as f:
                tree = bs4.BeautifulSoup(f, "lxml")

                causeLinks = tree.select("div#main-content > ol > li > a")
                causes = []
                for causel in causeLinks:
                    causeId = id_from_str(causel["href"])
                    causes.append((causeId, causel.getText().strip()))

                symptom = tree.select("div.headers > h1 > a")[0].getText()
                symptomId = id_from_str(fn)
                data.append((symptomId, symptom, causes))


    return data

class Cause:
    def __init__(self, id, causeType, names, description=None):
        self.id = id
        self.type = causeType
        self.names = names
        self.symptoms = []
        self.description = description

    def add_symptom(self, symptom):
        self.symptoms.append(symptom)

def _collect_cause_details(basedir):
    '''

    :param basedir:
    :return: Dictionary with causeIds as keys, and Cause as value.
    '''
    causespath = os.path.join(basedir, "disease")
    data = {}
    for fn in os.listdir(causespath):
        path = os.path.join(causespath, fn)
        if os.path.isfile(path):
            with open(path, mode="r") as f:
                causeId = id_from_str(fn)
                tree = bs4.BeautifulSoup(f, "lxml")
                causetLink = tree.select("div.headers > a")
                nameLink = tree.select("div.headers > h1 > a")

                if len(causetLink) == 0: # There's exactly one known faq-page where these two if-statements have different evaluation result
                    causetLink = tree.select("header > div.row > div.breadcrumbs > ul > li > a")
                if len(nameLink) == 0:
                    nameLink = tree.select("div.main > header > div.row > h1 > a")

                if len(causetLink) > 0:
                    causetLink = causetLink[-1]

                    causeTypeName = causetLink.getText()
                else:
                    causeTypeName = "Other"
                if len(nameLink) > 0:
                    name = nameLink[0].getText()
                    names = _name_to_names(name)
                else:
                    print("unknown name: ", fn)
                descElement = tree.select("#main-content > p:nth-of-type(1)")
                if len(descElement) > 0:
                    description = descElement[0].getText()
                    description = description.replace(";", ".").replace("\n", "")

                else:
                    description = "No description available"
                try:
                    causeType = MayoBase.cause_name_to_type(causeTypeName)
                    data[causeId] = Cause(causeId, causeType, names=names, description=description)
                except KeyError:
                    print("CauseType not found:", causeTypeName, "\nSkipping...")

    return data

def _name_to_names(rawname, excludeList=["male", "female", "body", "scalp"]):
    '''
    A name parameter of the following forms:
    name1 (name3)
    name2
    :param rawname:
    :return: A list of names in the string. For example, for rawname="name1 (name3)" returns [name1, name3]
    '''
    names = []
    mo = re.match(r"([^(]+)(\([^)]*\))?", rawname)

    for group in mo.groups():
        if group:
            group = re.sub("[()]", "", group)
            if group not in excludeList:
                names.append(group.strip())
            else:
                return [rawname.strip()]
    return names


def process_dataset(basedir):
    '''

    :param basedir: base dir to process. It should have subdirectories symptom and symcauses
    :return: a dictionary of disease-symptomlist pairs
    '''
    symptomData = _collect_symcause_data(basedir)
    causeData = _collect_cause_details(basedir)
    symptomIds = {}
    causeIds = {}
    for entry in symptomData:
        symptomId = entry[0]
        symptomraw = entry[1]

        symptomIds[symptomId] = _name_to_names(symptomraw)
    for entry in symptomData:
        symId = entry[0]
        for cause in entry[2]:
            id = cause[0]
            name = cause[1]
            causeIds[id] = name
            causeData[id].add_symptom(symId)

    return causeData, symptomIds

def mayo_to_csv(dataBaseDir, exportPath):
    if not os.path.exists(exportPath):
        os.mkdir(exportPath)
    causeData, symptomIds = process_dataset(dataBaseDir)
    causeSymptoms = {cause.id : cause.symptoms for cause in causeData.values()
                     if len(cause.symptoms) > 0}

    causeVecPath = os.path.join(exportPath, CAUSESYM_CSV)
    with open(causeVecPath, mode="w") as f:
        f.write("Id;Symptoms\n")
        for cause,symptoms in causeSymptoms.items():
            f.write("%s;%s\n" % (cause, ",".join(symptoms)))

    causeDetailsPath = os.path.join(exportPath, CAUSEDETAILS_CSV)
    with open(causeDetailsPath, mode="w") as f:
        f.write(";".join(["Id", "Type", "Name", "Description"]) + "\n")
        for cause in causeData.values():
            f.write("%s;%s;%s;%s\n" % (cause.id, cause.type.value, ",".join(cause.names), cause.description))


    symptomDetailsPath = os.path.join(exportPath, SYMPTOM_DETAILS_CSV)
    with open(symptomDetailsPath, mode="w") as f:
        f.write(";".join(["Id", "Name"]) + "\n")
        for symptomId, symptomNames in symptomIds.items():
            f.write("%s;%s\n" % (symptomId, ",".join(symptomNames)))




def load_data(reuse = True, basedir=DEFAULT_EXPORT_PATH):

    if not reuse or not os.path.exists(basedir):
        mayo_to_csv(MayoBase.BASE_DIR, basedir)


class MayoDataProvider:
    '''
    Provides symptom-cause data in indicator matrix format, and performs the appropriate (inverse) transforms.
    Attributes: X, y.
    '''
    def __init__(self, basedir=DEFAULT_EXPORT_PATH, causeTypeFilter=None):
        load_data(reuse=True)
        self._load_cause_details(basedir)
        self._load_causesym(basedir, causeTypeFilter=causeTypeFilter)
        self._load_symptom_details(basedir)
        self._filter_cause_types(causeTypeFilter)

    def _load_causesym(self, basedir, causeTypeFilter=None):
        path = os.path.join(basedir, CAUSESYM_CSV)
        df = pd.read_csv(path, sep=";")
        temp = {id: value[0] for id,value in self.causes.items()}

        df["Type"] = df.Id.map(temp)

        # df["Type"] = pd.Series(index=[np.int64(t[0]) for t in temp], data=[np.int64(t[1]) for t in temp])
        if causeTypeFilter != None:
            df = df[df.Type == causeTypeFilter.value]

        symptoms = df[["Symptoms"]].values
        symptoms = [vec[0].split(",") for vec in symptoms]
        symptoms = [tuple(np.int64(vec)) for vec in symptoms]

        self.causesym = dict(zip(df.Id, symptoms))

        self.mlb = MultiLabelBinarizer()

        self.X = self.mlb.fit_transform(symptoms)

        self.lv = LabelEncoder()

        self.y = self.lv.fit_transform(df["Id"].values)

    def _load_symptom_details(self, basedir):
        path = os.path.join(basedir, SYMPTOM_DETAILS_CSV)
        df = pd.read_csv(path, sep=";")
        self.symptoms = dict(zip(df.Id, df.Name))

    def _load_cause_details(self, basedir):
        path = os.path.join(basedir, CAUSEDETAILS_CSV)
        df = pd.read_csv(path, sep=";")
        self.causes = dict(zip(df.Id, zip(df.Type, df.Name, df.Description)))

    def _filter_cause_types(self, causeTypeFilter):
        if causeTypeFilter == None:
            return
        # self.causes = {id : value for id,value in self.causes if value[1] == causeTypeFilter}
        # self.causesym = {id : value for id,value in self.causesym if id in self.causes}

    def inverse_transform_symptoms(self, X):
        return self.mlb.inverse_transform(X)

    def inverse_transform_causes(self, y):
        return self.lv.inverse_transform(y)

    def transform_symptoms(self, X):
        return self.mlb.transform(X)

    def transform_causes(self, y):
        return self.lv.transform(y)

    def explain_symptoms(self, X):
        '''
        :param X: A transformed indicator matrix
        :return: Matrix where each indicator element has been transformed to symptom string
        '''
        X = self.inverse_transform_symptoms(X)

        return [[self.symptoms[id] for id in x] for x in X]


    def explain_causes(self, y):
        '''

        :param y: Encoded labels of causes
        :return: List of cause names
        '''
        y = self.inverse_transform_causes(y)
        return [self.causes[id][1] for id in y]

    def describe(self, cause):
        return self.causes[self.inverse_transform_causes(cause)][-1]

    def explain(self, X, y):
        return self.explain_symptoms(X), self.explain_causes(y)

def main():
    load_data(reuse=False)
    dp = MayoDataProvider("mayoexport", causeTypeFilter=CauseType.DISEASE)

    # print(dp.explain_symptoms(dp.X))
    # print(dp.explain_causes(dp.y))
    # m = MayoScraper()
    # m.scrape()

if __name__ == "__main__":
    main()