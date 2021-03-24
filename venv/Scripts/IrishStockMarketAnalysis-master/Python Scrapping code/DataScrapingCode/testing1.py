from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import bs4
import re
import requests
from bs4 import BeautifulSoup
import csv

csv_file= open('isrish_all_stock.csv' , 'w')
csv_writer= csv.writer(csv_file)
csv_writer.writerow(['Stock Name','Date','Headline','Description','Body'])
share_list=['ABBEY','AIB','AMINEX','AMRYT+PHARMA','APPLEGREEN','ARYZTA','BANK+OF+IRELAND','C+and+C+GROUP','CAIRN+HOMES','CPL+RESOURCES,CRH','DALATA+HOTEL+GROUP','DATALEX','DIAGEO','DONEGAL+INVESTMENT+GROUP','DRAPER+ESPRIT','FALCON+OIL+AND+GAS','FBD+HOLDINGS','FIRST+DERIVATIVES','FLUTTER+ENTERTAINMENT','GAN','GLANBIA','GLENVEAGH+PROPERTIES','GREAT+WESTERN+MINING+ORPORATION','GREEN+REIT'+'PUBLIC+LIMITED+COMPANY','GREENCOAT+RENEWABLES','HIBERNIA+REIT','HOSTELWORLD+GROUP','IFG+GROUP','INDEPENDENT+NEWS+AND+MEDIA,IRISH'+'CONTINENTAL+GROUP,IRISH'+'RESIDENTIAL+PROPERTIES+REIT','KENMARE+RESOURCES','KERRY+GROUP','KINGSPAN+GROUP','MAINSTAY+MEDICAL+INTERNATIONAL','MALIN+CORPORATION','MINCON+GROUP','OPEN+ORPHAN','ORIGIN+ENTERPRISES','ORMONDE+MINING','OVOCA+BIO','PERMANENT+TSB+GROUP','PETRONEFT','PROVIDENCE,RYANAIR','SCISYS+GROUP','SMURFIT+KAPPA+GROUP','TESCO','TOTAL+PRODUCE','TULLOW+OIL,UNIPHAR','VR+EDUCATION+HOLDINGS','WISDOMTREE+ISSUER','YEW+GROVE+REIT']
for shares in share_list:
    shares=shares
    try:
        source= requests.get('https://www.irishtimes.com/search/search-7.4195619?q='+shares+'+stock+&fromDate=01%2F01%2F2009&toDate=24%2F10%2F2019').text
        #print(source)
        soup= BeautifulSoup(source, 'lxml' )
        check=soup.find_all('a', class_='button-link')
        print(check)

        length=len(check)-1

        last=check[length]['href'].split('=')
        #print(last)
        leng=len(last)-1
        #print(leng)
        loop_end=int(last[leng])+1
        #print(loop_end)
    except Exception:
        pass
    page_list=[]
    for j in range(0,loop_end):
        page_link="https://www.irishtimes.com/search/search-7.4195619?q="+shares+"+stock+&fromDate=01%2F01%2F2009&toDate=23%2F10%2F2019&page="+str(j)+""
        page_list.append(page_link)
    print(len(page_list))
    article_list=[]
    for k in range(1,len(page_list)):
        source1= requests.get(page_list[k]).text
        soup1= BeautifulSoup(source1, 'lxml' )
        for a1 in soup1.find_all('div', class_='search_items_title'):
                link1=a1.span.a['href']

                article_list.append('https://www.irishtimes.com/'+link1+'')
    print(len(article_list))
    print(article_list)
    for i in range(1,len(article_list)):
        try:
            print(article_list[i])
            data= requests.get(article_list[i]).text
            soup2 = bs4.BeautifulSoup(data, 'html.parser')
            date=soup2.find('div', class_='time-metadata').time.text
                            
                                
            results = soup2.body.find_all(string=re.compile('.*{0}.*'.format(shares)), recursive=True)
            row =  soup2.find_all('div', string=shares)
                            

                            
                       
            article=soup2.find(name='hgroup')
            headline= article.h1.text
            description=article.h2.text
            pData1=''
            body=soup2.find_all("div", class_="article_bodycopy", id=False)
            for tag in body:
                for element in tag.find_all("p"):
                    pData = element.text
                    pData1= pData1+'\n'+pData
            print(date)
            print(headline)
            print(description)
                    #print(pData1)
            csv_writer.writerow([shares,date,headline,description,pData1])
        except Exception:
            pass
        except ConnectionError:
            pass
        except:
            pass
csv_file.close() 
                                
    
    
            
