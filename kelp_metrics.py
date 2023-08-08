import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from statsmodels.regression.linear_model import OLS

# make function to extract kelp data and temperatures
def extract_kelp_metrics(data, bathymetry, lowerBound, upperBound):
    """
    Extract various metrics like correlation coefficients and slopes from kelp data and differences in temperature

    Parameters
    ----------
    data : list
        list of dictionaries containing kelp data

    bathymetry : xarray
        bathymetry data

    lowerBound : float
        lower bound of latitude

    upperBound : float
        upper bound of latitude 
    """

    # initialize dictionary of data
    kelp_data = {
        'dkelp':[],          # one quarter difference in kelp area
        'dkelp_kelp':[],     # kelp area at same time as difference (average of neighboring quarters)
        'dtemp':[],          # one quarter difference in temperature
        'dtemp_temp':[],     # temperature at same time as difference (average of neighboring quarters)
        'dtemp_temp_lag':[], # temperature one quarter before 
        'dtemp_temp_lag2':[],# temperature two quarter before 
        'kelp':[],           # total kelp area
        'temp':[],           # temperature
        'temp_lag':[],       # temperature one quarter before
        'temp_lag2':[],      # temperature two quarters before
        'time':[],           # time
        'dtime':[],          # time of difference
        'lat':[],            # latitude
        'lon':[],            # longitude
        'dlat':[],           # latitude corresponding to difference measurements
        'dlon':[],           # longitude corresponding to difference measurements
        'elevation':[],      # elevation [m]
        'delevation':[],     # elevation corresponding to difference measurements [m]
    }

    # loop over all locations
    for d in tqdm(data):

        # skip data that is not between the upper and lower bounds
        if (d['lat'] >= upperBound) or (d['lat'] < lowerBound): 
            continue 

        # filter out areas with no measurements and nans
        bad_mask = (d['kelp_area'] == 0) | (np.isnan(d['kelp_area'])) | (np.isnan(d['kelp_temp']))

        # compute temperature one quarter before
        d['kelp_temp_lag'] = np.roll(d['kelp_temp'],1)
        d['kelp_temp_lag'][0] = np.nan

        # compute temperature two quarters before
        d['kelp_temp_lag2'] = np.roll(d['kelp_temp'],2)
        d['kelp_temp_lag2'][0] = np.nan
        d['kelp_temp_lag2'][1] = np.nan

        # save data
        kelp_data['kelp'].extend(d['kelp_area'][~bad_mask])
        kelp_data['temp'].extend(d['kelp_temp'][~bad_mask])
        kelp_data['temp_lag'].extend(d['kelp_temp_lag'][~bad_mask])
        kelp_data['temp_lag2'].extend(d['kelp_temp_lag2'][~bad_mask])
        kelp_data['time'].extend(d['kelp_time'][~bad_mask])

        # save latitude and longitude
        kelp_data['lat'].extend(d['lat']*np.ones(len(d['kelp_area'][~bad_mask])))
        kelp_data['lon'].extend(d['long']*np.ones(len(d['kelp_area'][~bad_mask])))

        # use linear interpolation
        elevation = bathymetry.interp(lat=d['lat'],lon=d['long']).elevation.values
        kelp_data['elevation'].extend(elevation*np.ones(len(d['kelp_area'][~bad_mask])))

        # properly estimate the difference data
        nanmask = d['kelp_area'] == 0 # mask out areas with no kelp
        nandata = d['kelp_area'].copy()
        nandata[nanmask] = np.nan # set areas with no kelp to nan
        nandiff = np.diff(nandata) # take quarterly difference
        nonnanmask_kelp = ~np.isnan(nandiff) # mask out nans to ensure difference between sequential quarters
        nandiff_temp = np.diff(d['kelp_temp']) # take quarterly difference of temperature
        nonnanmask_temp = ~np.isnan(nandiff_temp) # mask out nans to ensure difference between sequential quarters
        nonnanmask = nonnanmask_kelp & nonnanmask_temp # mask out nans in both kelp and temperature data

        kelp_data['dkelp'].extend(nandiff[nonnanmask]) # save kelp difference data
        kelp_data['dkelp_kelp'].extend((d['kelp_area'][1:][nonnanmask] + d['kelp_area'][:-1][nonnanmask])/2) # average kelp area at same time as difference

        # average temperature at same time as difference
        kelp_data['dtemp_temp'].extend((d['kelp_temp'][1:][nonnanmask] + d['kelp_temp'][:-1][nonnanmask])/2)
        kelp_data['dtemp_temp_lag'].extend((d['kelp_temp_lag'][1:][nonnanmask] + d['kelp_temp_lag'][:-1][nonnanmask])/2)
        kelp_data['dtemp_temp_lag2'].extend((d['kelp_temp_lag2'][1:][nonnanmask] + d['kelp_temp_lag2'][:-1][nonnanmask])/2)
        
        kelp_data['dtemp'].extend(nandiff_temp[nonnanmask]) # save temperature difference data

        # save time data for differences by averaging time of sequential quartersq
        times_prev = d['kelp_time'][1:][nonnanmask].astype('datetime64[ns]')
        times_next = d['kelp_time'][:-1][nonnanmask].astype('datetime64[ns]')
        delta = times_next - times_prev
        times_mid = times_prev + delta/2
        kelp_data['dtime'].extend(times_mid.astype(str))

        # save latitude and longitude
        kelp_data['dlat'].extend(d['lat']*np.ones(len(nandiff[nonnanmask])))
        kelp_data['dlon'].extend(d['long']*np.ones(len(nandiff[nonnanmask])))
        
        # add delevation
        kelp_data['delevation'].extend(elevation*np.ones(len(nandiff[nonnanmask])))

        # free up memory
        del nanmask, nandata, nandiff, nonnanmask, nandiff_temp, bad_mask

    # convert to numpy arrays
    for k in kelp_data.keys():
            # convert to numpy arrays
        for k in kelp_data.keys():
            try:
                kelp_data[k] = np.array(kelp_data[k])
            except:
                print("could not convert", k)
        kelp_data[k] = np.array(kelp_data[k])

    # convert time to datetime
    kelp_data['time'] = kelp_data['time'].astype('datetime64[ns]')
    kelp_data['dtime'] = kelp_data['dtime'].astype('datetime64[ns]')

    # measure correlation and trend line
    A = np.vstack([kelp_data['dtemp_temp']-273.15, np.ones(len(kelp_data['dtemp_temp']))]).T
    res = OLS(kelp_data['dkelp'], A).fit()
    m,b = res.params[0], res.params[1]

    # define characteristic temperature where change in kelp is 0
    # just solve for x above to get the characteristic temperature at each location
    kelp_data['temp_char'] = -b/m
    kelp_data['average_temp'] = np.mean(kelp_data['dtemp_temp'])-273

    # saving the slope + error
    kelp_data['slope_dkelp_temp_char'] = m
    kelp_data['slope_dkelp_temp_char_err'] = res.bse[0]
    
    return kelp_data

if __name__ == "__main__":
    # load data
    with open('Data/kelp_interpolated_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # load bathymetry data
    #bathymetry = xr.open_dataset('Data/GEBCO_2022_sub_ice_topo.nc')
    #lower_lat = 37
    #upper_lat = 50

    # if using noaa dem: limit: 31-37
    bathymetry = xr.open_dataset('Data/crm_socal_1as_vers2.nc')
    bathymetry = bathymetry.rename({'Band1':'elevation'})
    lower_lat = 31
    upper_lat = 36

    kelp_data = extract_kelp_metrics(data, bathymetry, lower_lat, upper_lat)

    #save to pkl file
    with open(f"Data/kelp_metrics_{lower_lat}_{upper_lat}.pkl", "wb") as f:
        pickle.dump(kelp_data, f)