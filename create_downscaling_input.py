import pickle
import argparse
import pandas as pd
import numpy as np
from astropy import units as u
from astropy.constants import R_earth

if __name__ == "__main__":
    # argparse for input filepath
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, 
                        help='path to input metrics file', 
                        default="Data/kelp_metrics_26_36.pkl")
    args = parser.parse_args()

    # load data from disk
    with open(args.file_path, 'rb') as f:
        data = pickle.load(f)

    # # Access the latitude and longitude variables from the dataset
    # lat_lon_all = (data['lat'], data['lon'])
    # lat_lon = np.unique(lat_lon_all, axis=1) # degrees

    # # print set of unique lat/lon values
    # print("Unique Lat/Lon Values:")
    # print("Unique Latitudes: ", np.unique(lat_lon[0]).shape)
    # print("Unique Longitudes: ", np.unique(lat_lon[1]).shape)

    # print bounding box of data
    print("Bounding Box:")
    print("Min Latitude: ", np.min(data['lat']))
    print("Max Latitude: ", np.max(data['lat']))
    print("Min Longitude: ", np.min(data['lon']))
    print("Max Longitude: ", np.max(data['lon']))

    # regrid data every 375m
    ddeg = (R_earth * np.pi / 180.0).to(u.meter).value # meters per degree
    dx = 375 # meters
    dlat = dx / ddeg
    dlon = dx / (ddeg * np.cos(np.pi * np.mean(data['lat']) / 180.0))
    lat = np.arange(np.min(data['lat']), np.max(data['lat']), dlat)
    lon = np.arange(np.min(data['lon']), np.max(data['lon']), dlon)
    # meshgrid then flatten
    lat, lon = np.meshgrid(lat, lon)
    lat = lat.flatten()
    lon = lon.flatten()

    # Create a DataFrame with latitude and longitude as columns
    df = pd.DataFrame({'Latitude': lat, 'Longitude': lon})

    # Save the DataFrame to a CSV file
    df.to_csv(args.file_path.replace('.pkl', '.csv').replace("metrics","locations"), index=False)
    print(f"Saved CSV file to {args.file_path.replace('.pkl', '.csv').replace('metrics','locations')}")