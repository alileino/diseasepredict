from lxml import html
import requests

class MayoSymptomWorker:
    '''
    '''
    BASE_DIR = "mayo"
    def __init__(self, alphabet):
        self.listAddress = MayoClinicScraper.BASE_ADDRESS + alphabet.lower()

class MayoClinicScraper:
    '''
    '''

    BASE_ADDRESS = "http://www.mayoclinic.org/symptoms/index?letter="

m = MayoSymptomWorker("b")
#
# page = requests.get("http://econpy.pythonanywhere.com/ex/001.html")
# tree=html.fromstring(page.content)
print()