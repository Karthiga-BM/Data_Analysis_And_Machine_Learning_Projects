# The Google Maps library
import googlemaps
# Date time for easy computations between dates
from datetime import datetime
# JSON handling
import json
# Pandas
import pandas as pd
# Regular expressions
import re
# TQDM for fancy loading bars
from tqdm import tqdm
import time
import random

# !!! Define the main access point to the Google APIs.
# !!! This object will contain all the functions needed
geolocator = googlemaps.Client(key="<YOUR API KEY>")

WORK_LAT_LNG = (< LATITUDE >, < LONGITUDE >)
# You can set this parameter to decide the time from which
# Google needs to calculate the directions
# Different times affect public transport
DEPARTURE_TIME = datetime.now

# Load the source data
data = pd.read_csv("/path/to/raw/data/data.csv")

# Define the columns that we want in the geocoded dataframe
geo_columns = ["_link", "lat", "lng", "_time_to_work_seconds_transit", "_time_to_work_seconds_walking"]

# Create an array where we'll store the geocoded data
geo_data = []
# For each element of the raw dataframe, start the geocoding
for index,
    in tqdm(data.iterrows()):
# Google Geo coding
_location = ""
_location_json = ""
try:
    # Try to retrieve the base location,
    # i.e. the Latitude and Longitude given the address
    _location = geolocator.geocode(row._address)
    _location_json = json.dumps(_location[0])
except:
    pass

_time_to_work_seconds_transit = 0
_directions_json = ""
_lat_lon = {"lat": 0, "lng": 0}
try:
    # Given the work latitude and longitude, plus the property latitude and longitude,
    # retrieve the distance with PUBLIC TRANSPORT (`mode=transit`)
    _lat_lon = _location[0]["geometry"]["location"]
    _directions = geolocator.directions(WORK_LAT_LNG,
                                        (_lat_lon["lat"], _lat_lon["lng"]), mode="transit")
    _time_to_work_seconds_transit = _directions[0]["legs"][0]["duration"]["value"]
    _directions_json = json.dumps(_directions[0])
except:
    pass

_time_to_work_seconds_walking = 0
try:
    # Given the work latitude and longitude, plus the property latitude and longitude,
    # retrieve the WALKING distance (`mode=walking`)
    _lat_lon = _location[0]["geometry"]["location"]
    _directions = geolocator.directions(WORK_LAT_LNG, (_lat_lon["lat"], _lat_lon["lng"]), mode="walking")
    _time_to_work_seconds_walking = _directions[0]["legs"][0]["duration"]["value"]
except:
    pass

#  This block retrieves the number of SUPERMARKETS arount the property
'''
_supermarket_nr = 0
_supermarket = ""
try:
    # _supermarket = geolocator.places_nearby((_lat_lon["lat"],_lat_lon["lng"]), radius=750, type="supermarket")
    _supermarket_nr = len(_supermarket["results"])
except:
    pass
'''

#  This block retrieves the number of PHARMACIES arount the property
'''
_pharmacy_nr = 0
_pharmacy = ""
try:
    # _pharmacy = geolocator.places_nearby((_lat_lon["lat"],_lat_lon["lng"]), radius=750, type="pharmacy")
    _pharmacy_nr = len(_pharmacy["results"])
except:
    pass
'''

#  This block retrieves the number of RESTAURANTS arount the property
'''
_restaurant_nr = 0
_restaurant = ""
try:
    # _restaurant = geolocator.places_nearby((_lat_lon["lat"],_lat_lon["lng"]), radius=750, type="restaurant")
    _restaurant_nr = len(_restaurant["results"])
except:
    pass
'''

geo_data.append([row._link, _lat_lon["lat"], _lat_lon["lng"], _time_to_work_seconds_transit,
                 _time_to_work_seconds_walking])
geo_data_df = pd.DataFrame(geo_data)
geo_data_df.columns = geo_columns
geo_data_df.to_csv("geo_data_houses.csv", index=False)