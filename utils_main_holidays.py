import os
import json
import requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def extract_public_Holiday(year, iso):
    holidays_json = f'public_holidays_{year}.json'
    if os.path.exists(holidays_json):
        with open(holidays_json, 'r') as file:
            holidays = json.load(file)
    else:
        url = f'https://date.nager.at/Api/v3/PublicHolidays/{year}/{iso}'
        response = requests.get(url)
        if response.status_code == 200:
            holidays = response.json()
            with open(holidays_json, 'w') as file:
                json.dump(holidays, file)
    holiday_dates = []
    for holiday in holidays:
        holiday_dates.append(holiday['date'])
    holiday_data = pd.DataFrame(holidays, columns=['date', 'name'])
    holiday_data['date'] = pd.to_datetime(holiday_data['date'])
    holiday_data = holiday_data.rename(columns={'date': 'date', 'name': 'Holiday'})
    start_date = pd.to_datetime(f'{year}-01-01')
    end_date = pd.to_datetime(f'{year}-12-31')
    # calendar_df = pd.DataFrame(pd.date_range(start=start_date, end=end_date, inclusive='left'), columns=['date'])
    calendar_df = pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=['date'])
    calendar_df = calendar_df.merge(holiday_data, on='date', how='left')
    calendar_df['Holiday'].fillna('Not a Holiday', inplace=True)
    calendar_df['Day_Type_2'] = np.where((calendar_df['date'].dt.weekday < 5) & (~calendar_df['date'].isin(holiday_dates)),
                                        'Weekday', 'Public_Holiday/Weekend')
    return calendar_df

def fetch_from_nager_date(year, iso):
    """
    Fetch public holidays from Nager.Date API.
    """
    url = f"https://date.nager.at/Api/v3/PublicHolidays/{year}/{iso}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None  # If API fails, return None

def fetch_from_calendarific(year, iso):
    """
    Fetch public holidays from Calendarific API (free tier).
    Requires an API key.
    """
    api_key = "DVzAjiD7QBPeh4LUkGWjVDDoLhJFPEJ3"  # Replace with your Calendarific API key
    url = f"https://calendarific.com/api/v2/holidays?api_key={api_key}&country={iso}&year={year}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        holidays = data['response']['holidays']
        return [{'date': holiday['date']['iso'], 'name': holiday['name']} for holiday in holidays]
    else:
        return None  # If API fails, return None

def fetch_from_github_jp(year):
    """
    Fetch public holidays for Japan from the GitHub holidays-jp repository.
    """
    url = f"https://raw.githubusercontent.com/shogo82148/holidays-jp/master/json/{year}.json"
    response = requests.get(url)
    if response.status_code == 200:
        holidays = response.json()
        # Convert GitHub format to match Nager.Date's output
        return [{'date': date, 'name': name} for date, name in holidays.items()]
    else:
        return None  # If GitHub API fails, return None
# Add a column for Day_Type based on holidays and dates
def determine_day_type(row):
    # Check if it's a holiday or a regular weekday/weekend
    if row['Holiday'] != "Not a Holiday":
        return "Public_Holiday/Weekend"
    else:
        # Weekends: Saturday (5) or Sunday (6)
        weekday = pd.Timestamp(row['date']).weekday()
        return "Public_Holiday/Weekend" if weekday in [5, 6] else "Weekday"

def extract_public_Holiday_v2(year, iso):
    holidays_json = f'public_holidays_{year}_v5.json'
    
    # Check if the local file exists
    if os.path.exists(holidays_json):
        with open(holidays_json, 'r') as file:
            holidays = json.load(file)
    else:
        # Try fetching from Nager.Date API first
        holidays = fetch_from_nager_date(year, iso)
        
        # If Nager.Date fails, try Calendarific API
        if holidays is None:
            print("No Nager")
            holidays = fetch_from_calendarific(year, iso)
        
        # If both Nager.Date and Calendarific fail, and the country is Japan, switch to GitHub
        if holidays is None and iso == "JP":
            holidays = fetch_from_github_jp(year)
        
        if holidays is None or len(holidays) == 0:
            print(f"No holiday data available for {year} and country {iso}. Returning an empty DataFrame.")
            return pd.DataFrame(columns=['date', 'Holiday', 'Day_Type_2'])
        
        # Save the holidays data locally
        with open(holidays_json, 'w') as file:
            json.dump(holidays, file)
    
    # Extract holiday dates and create a DataFrame
    holiday_data = pd.DataFrame(holidays, columns=['date', 'name'])
    # Convert the 'date' column to datetime, forcing errors to be NaT if there are any parsing issues
    holiday_data['date'] = pd.to_datetime(holiday_data['date'], errors='coerce')
    
    # Now, format the 'date' column to ensure it is in "YYYY-MM-DD" format
    holiday_data['date'] = holiday_data['date'].dt.strftime('%Y-%m-%d')
    
    # If there are any invalid date entries after conversion, we can drop them (optional)
    holiday_data = holiday_data.dropna(subset=['date'])
    # holiday_data['date'] = pd.to_datetime(holiday_data['date'])
    holiday_data = holiday_data.rename(columns={'date': 'date', 'name': 'Holiday'})
    
    tmp1 = holiday_data.copy()
    tmp1['date'] = pd.to_datetime(tmp1['date'], errors='coerce')
    start_date = pd.to_datetime(f'{year}-01-01')
    end_date = pd.to_datetime(f'{year}-12-31')
    calendar_df = pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=['date'])
    
    # Now, 'calendar_df' should also be of type datetime
    calendar_df['date'] = pd.to_datetime(calendar_df['date'], errors='coerce')
    
    # Merge holiday data with the calendar DataFrame
    calendar_df = calendar_df.merge(tmp1, on='date', how='left')
    
    # Fill missing values in the 'Holiday' column
    calendar_df['Holiday'].fillna('Not a Holiday', inplace=True)
    
    calendar_df = calendar_df.groupby("date", as_index=False).agg({
        "Holiday": lambda x: "_".join(sorted(set(x))),
    })
    calendar_df['Day_Type_2'] = calendar_df.apply(determine_day_type, axis=1)
    
    return calendar_df 

# Define the years and ISO code for Japan
years = [2024,2025]
holidays_list = []

# Process each year
for year in years:
    tmp = extract_public_Holiday_v2(year, "JP")
    holidays_list.append(tmp)

# Concatenate results for all years
if holidays_list:
    holiday = pd.concat(holidays_list)
    # print(all_holidays)
else:
    print("No holiday data found for the specified years.")
