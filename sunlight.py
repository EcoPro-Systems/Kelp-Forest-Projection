import ephem
from datetime import datetime, timedelta

def calculate_daylight(lat, lon, date):
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.date = date
    
    sunrise = observer.previous_rising(ephem.Sun())
    sunset = observer.next_setting(ephem.Sun())
    
    daylight_duration = sunset - sunrise
    
    return daylight_duration

# Example usage:
latitude = 37.7749  # Replace with your latitude
longitude = -122.4194  # Replace with your longitude
date = datetime(2023, 9, 1)  # Replace with your desired date

daylight_duration = calculate_daylight(latitude, longitude, date)
print(f"Daylight duration on {date}: {daylight_duration} (HH:MM:SS)")
