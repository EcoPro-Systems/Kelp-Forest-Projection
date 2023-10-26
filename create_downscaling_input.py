import pickle
import argparse
import pandas as pd

if __name__ == "__main__":
    # argparse for input filepath
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, 
                        help='path to input metrics file', 
                        default="Data/kelp_metrics_31_36.pkl")
    args = parser.parse_args()

    # load data from disk
    with open(args.file_path, 'rb') as f:
        data = pickle.load(f)

    # Access the latitude and longitude variables from the dataset
    latitude = data['lat'][::1000]
    longitude = data['lon'][::1000]

    # Create a DataFrame with latitude and longitude as columns
    df = pd.DataFrame({'Latitude': latitude, 'Longitude': longitude})

    # Save the DataFrame to a CSV file
    df.to_csv(args.file_path.replace('.pkl', '.csv'), index=False)
