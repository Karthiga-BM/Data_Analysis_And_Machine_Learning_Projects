#importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
from bs4 import BeautifulSoup
import seaborn as sns
import requests

pin = input("Enter the pincode:")
if len(pin) != 5:
    print("invalid pin")
    sys.exit(1)

url = "https://philadelphia.craigslist.org/search/hhh?availabilityMode=0&hasPic=1&query={0}&sort=rel".format(pin)


#creating empty list to contain information like rent amount, date posted, link, description, etc
#and setting counter variable(s)=0
s = 0
itern = 0
Rent_amount = []
Date_posted = []
Link = []
Description = []
Bedroom = []
Location = []


for i in range(0,100000):
    x = urlopen(url)
    y = soup(x.read(), "html.parser")
    k = y.find_all('li', class_='result-row')
    if len(k)==0: break
    for i in range(0,len(k)):
        Rent_amount.append(k[i].find('span', class_='result-price').text)
        Date_posted.append(k[i].find('time', class_='result-date')['title'])
        Link.append(k[i].find('a', class_='result-title hdrlnk')['href'])
        Description.append(k[i].find('a', class_='result-title hdrlnk').text)
        if k[i] and k[i].find('span', class_='result-hood'):
            Location.append(k[i].find('span', class_='result-hood').text)
        else:
            Location.append("")
        if k[i] and k[i].find('span', class_='housing'):
            Bedroom.append(k[i].find('span', class_='housing').text.strip())
        else:
            Bedroom.append("")
    s+=120
    url = "https://philadelphia.craigslist.org/search/hhh?availabilityMode=0&hasPic=1&query={0}&sort=rel".format(pin+"&s="+str(s))
    itern += 1
    print("Page " + str(itern) + " scraped successfully!")

print("\n")

print("Scrape complete!")

eb_apts = pd.DataFrame({'Price': Rent_amount,
                       'Posting_Date': Date_posted,
                       'URL': Link,
                       'Description': Description,
                        'Rooms_and_Sqft': Bedroom,
                       'Nearby_Location': Location})
print(eb_apts.info())
eb_apts.head(10)

eb_apts.to_csv("data/craig_list.csv")


#
#
# #creating a request object
# r = requests.get("https://www.daft.ie/property-for-sale/dublin", headers = {'User-agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0'})
#
# #storing the content
# content = r.content
#
# print(content)
#
# #make the content more readable using beautifulsoup
# content = BeautifulSoup(content, "html.parser")
# content.prettify()
# len(content)
# print("the length",(len(content)))
#
#
#
#
#
# #number of webpages available showing the result
# total_pages =content.find_all("div", {"Class": "Pagination__StyledPagination-sc-13f8db0-0 bpAWkj"})
# print("the value is", total_pages)
# print("total number of pages:", type(total_pages))
#
# property_json = {'Details_Broad': {}, 'Address': {}}
#
# #extracting the title of the property listing
# for title in content.find_all("title"):
#     property_json['Title'] = title.text.strip()
#     print(property_json)
#     break
#
# price = content.find_all('span', class_="price primary")[0].text
#
# #create a list to store the attribute values
