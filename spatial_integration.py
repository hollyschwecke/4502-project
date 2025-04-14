import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

#load data
#fill in file path once all data is uploaded/files joined 
wildfires = pd.read_csv("")
weather = pd.read_csv("")

#convert to GeoDataFrames
wildfires['geometry'] = wildfires.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
weather['geometry'] = weather.apply((lambda row: Point(row['lon'], row['lat']), axis=1)

gdf_fires = gpd.GeoDateFrame(wildfires, geometry='geometry', crs='EPSG:4326')
gdf_weather = gpd.GeoDateFrame(weather, geometry='geometry', crs='EPSG:4326')

# use spatial join to match wildfires to the nearest weather station
joined = gpd.sjoin_nearest(gdf_fires, gdf_weather, how='left', distance_col="dist_km')
