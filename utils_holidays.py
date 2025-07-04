import os
import json
import requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def fetch_from_nager_date(year: int, iso: str):
    """Fetch public holidays from Nager.Date API."""
    url = f"https://date.nager.at/Api/v3/PublicHolidays/{year}/{iso}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None


def fetch_from_calendarific(year: int, iso: str):
    """Fetch public holidays from Calendarific API."""
    api_key = "DVzAjiD7QBPeh4LUkGWjVDDoLhJFPEJ3"  # Free tier key
    url = f"https://calendarific.com/api/v2/holidays?api_key={api_key}&country={iso}&year={year}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        holidays = data['response']['holidays']
        return [{'date': h['date']['iso'], 'name': h['name']} for h in holidays]
    return None


def fetch_from_github_jp(year: int):
    """Fetch public holidays for Japan from GitHub repository."""
    url = f"https://raw.githubusercontent.com/shogo82148/holidays-jp/master/json/{year}.json"
    response = requests.get(url)
    if response.status_code == 200:
        holidays = response.json()
        return [{'date': d, 'name': n} for d, n in holidays.items()]
    return None


def determine_day_type(row):
    if row['Holiday'] != "Not a Holiday":
        return "Public_Holiday/Weekend"
    weekday = pd.Timestamp(row['date']).weekday()
    return "Public_Holiday/Weekend" if weekday in [5, 6] else "Weekday"


def extract_public_Holiday_v2(year: int, iso: str):
    """Get public holiday calendar for a given year and country."""
    holidays_json = f"public_holidays_{year}_v5.json"

    if os.path.exists(holidays_json):
        with open(holidays_json, 'r') as f:
            holidays = json.load(f)
    else:
        holidays = fetch_from_nager_date(year, iso)
        if holidays is None:
            holidays = fetch_from_calendarific(year, iso)
        if holidays is None and iso == "JP":
            holidays = fetch_from_github_jp(year)
        if not holidays:
            return pd.DataFrame(columns=['date', 'Holiday', 'Day_Type_2'])
        with open(holidays_json, 'w') as f:
            json.dump(holidays, f)

    holiday_data = pd.DataFrame(holidays, columns=['date', 'name'])
    holiday_data['date'] = pd.to_datetime(holiday_data['date'], errors='coerce')
    holiday_data['date'] = holiday_data['date'].dt.strftime('%Y-%m-%d')
    holiday_data = holiday_data.dropna(subset=['date'])
    holiday_data = holiday_data.rename(columns={'date': 'date', 'name': 'Holiday'})

    tmp1 = holiday_data.copy()
    tmp1['date'] = pd.to_datetime(tmp1['date'], errors='coerce')
    start_date = pd.to_datetime(f"{year}-01-01")
    end_date = pd.to_datetime(f"{year}-12-31")
    calendar_df = pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=['date'])
    calendar_df['date'] = pd.to_datetime(calendar_df['date'], errors='coerce')
    calendar_df = calendar_df.merge(tmp1, on='date', how='left')
    calendar_df['Holiday'].fillna('Not a Holiday', inplace=True)
    calendar_df = calendar_df.groupby('date', as_index=False).agg({'Holiday': lambda x: '_'.join(sorted(set(x)))})
    calendar_df['Day_Type_2'] = calendar_df.apply(determine_day_type, axis=1)
    return calendar_df


def load_holidays(years, iso="JP"):
    """Load holidays for multiple years and concatenate."""
    dfs = [extract_public_Holiday_v2(year, iso) for year in years]
    dfs = [df for df in dfs if not df.empty]
    return pd.concat(dfs) if dfs else pd.DataFrame(columns=['date', 'Holiday', 'Day_Type_2'])
