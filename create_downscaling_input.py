import pickle
import argparse
import pandas as pd
import numpy as np
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

    # Access the latitude and longitude variables from the dataset
    lat_lon = (data['lat'], data['lon'])

    # print set of unique lat/lon values
    print("Unique Lat/Lon Values:")
    print("Unique Latitudes: ", np.unique(lat_lon[0]).shape)
    print("Unique Longitudes: ", np.unique(lat_lon[1]).shape)

    # print bounding box of data
    print("Bounding Box:")
    print("Min Latitude: ", np.min(lat_lon[0]))
    print("Max Latitude: ", np.max(lat_lon[0]))
    print("Min Longitude: ", np.min(lat_lon[1]))
    print("Max Longitude: ", np.max(lat_lon[1]))

    # randomly choose 1000 points without replacement
    idx = np.random.choice(len(lat_lon[0]), 1000, replace=False)

    # Create a DataFrame with latitude and longitude as columns
    df = pd.DataFrame({'Latitude': lat_lon[0][idx], 'Longitude': lat_lon[1][idx]})

    # Save the DataFrame to a CSV file
    df.to_csv(args.file_path.replace('.pkl', '.csv'), index=False)
