#importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
from bs4 import BeautifulSoup
import seaborn as sns
import requests

#creating a request object
r = requests.get("https://www.daft.ie/property-for-sale/dublin", headers = {'User-agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0'})

#storing the content
content = r.content

print(content)
