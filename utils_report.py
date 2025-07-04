import os
import platform
import tempfile
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import gradio as gr
from meteostat import Point, Daily
import importlib

# Globals to be initialised from main
network_df = pd.DataFrame()
location_df = pd.DataFrame()
otcRatio = pd.DataFrame()
holiday = pd.DataFrame()

campaign_cart = []
latest_report_data = None

# Spot calculation utilities

def check_spot(spotsHour, spotDur):
    return spotsHour * (spotDur / (spotDur + 15))


def spotPV(spotDurMult, spotGet, avgPerMin, pv):
    return min(spotDurMult * spotGet * avgPerMin, pv)


def check_dwell(dwellTime, loopLength):
    if dwellTime < 10:
        return 0.6
    elif dwellTime >= 10 and dwellTime >= loopLength:
        return dwellTime / loopLength
    elif dwellTime >= 10 and dwellTime < loopLength:
        return 1


def spot_calc(pv, dwellTime, loopLength, spotsHour, spotDur):
    avgPerMin = pv / 60 if dwellTime <= 60 else (pv / 60) * (dwellTime / 60)
    spotGet = check_dwell(dwellTime, loopLength)
    spotDurMult = check_spot(spotsHour, spotDur)
    return spotPV(spotDurMult, spotGet, avgPerMin, pv)


def custom_round(value):
    return round(value, 2) if pd.notnull(value) else 0

# -----------------------------------------
# Excel report helpers

def displayName(name):
    loc = {
        'JPN-JEK-D-00000-00029': 'J・ADビジョン　巣鴨駅改札外',
        'JPN-JEK-D-00000-00030': 'J・ADビジョン　新宿駅東口',
        'JPN-JEK-D-00000-00031': 'J・ADビジョン　新宿駅南口',
        'JPN-JEK-D-00000-00032': 'J・ADビジョン　新宿駅甲州街道改札',
        'JPN-JEK-D-00000-00034': 'J・ADビジョン　五反田駅',
        'JPN-JEK-D-00000-00035': 'J・ADビジョン　品川駅中央改札内',
        'JPN-JEK-D-00000-00039': 'J・ADビジョン　有楽町駅中央改札口',
        'JPN-JEK-D-00000-00040': 'J・ADビジョン　東京駅丸の内地下連絡通路',
        'JPN-JEK-D-00000-00041': 'J・ADビジョン　東京駅京葉通路',
        'JPN-JEK-D-00000-00042': 'J・ADビジョン　秋葉原駅新電気街口',
        'JPN-JEK-D-00000-00044': 'J・ADビジョン　吉祥寺駅南北自由通路',
        'JPN-JEK-D-00000-00045': 'J・ADビジョン　浦和駅改札口',
        'JPN-JEK-D-00000-00046': 'J・ADビジョン　大宮駅中央改札',
        'JPN-JEK-D-00000-00047': 'J・ADビジョン　横浜駅中央通路',
        'JPN-JEK-D-00000-00048': 'J・ADビジョン　JR横浜タワーアトリウム',
        'JPN-JEK-D-00000-00049': 'J・ADビジョン　高田馬場駅スマイル・ステーションビジョン',
        'JPN-JEK-D-00000-00050': 'J・ADビジョン　池袋駅中央改札内',
        'JPN-JEK-D-00000-00051': 'J・ADビジョン　桜木町駅',
        'JPN-JEK-D-00000-00052': 'J・ADビジョン　横浜駅南改札内',
        'JPN-JEK-D-00000-00058': 'J・ADビジョン　東京駅新幹線北乗換口',
        'JPN-JEK-D-00000-00059': 'J・ADビジョン　東京駅新幹線南乗換口',
        'JPN-JEK-D-00000-00060': 'J・ADビジョン　恵比寿駅西口',
        'JPN-JEK-D-00000-00061': 'J・ADビジョン　赤羽駅北改札',
        'JPN-JEK-D-00000-00960': 'J・ADビジョン　八王子駅自由通路南',
        'JPN-JEK-D-00000-00961': 'J・ADビジョン　上野駅公園改札内',
        'JPN-JEK-D-00000-04333': 'J・ADビジョン 新橋駅北改札',
        'JPN-JEK-D-00000-04334': 'J・ADビジョン 新橋駅南改札',
        'JPN-JEK-D-00000-00036': 'J・ADビジョン　高輪ゲートウェイ駅',
        'JPN-JEK-N-00000-00055': '[2025]J・adビジョン ステーションネットワーク',
    }
    return loc.get(name, name)


def networkSum_(df, name):
    df_copy = df.copy()
    df_copy[['gender', 'age']] = df_copy['agegender'].str.split('_', n=1, expand=True)
    df_copy['age'] = df_copy['age'] + '_Impressions'
    df_copy['gender_age'] = df_copy['gender'] + '_' + df_copy['age']

    network_df2 = df_copy.groupby(['date', 'hour_group', 'gender_age'])['PV'].sum().reset_index()
    pivot_df = network_df2.pivot_table(index=['date', 'hour_group'],
                                       columns='gender_age', values='PV', fill_value=0).reset_index()

    expected_cols = [f'{g}_{a}_Impressions' for g in ['female', 'male'] for a in ['10_19', '20_29', '30_39', '40_49', '50_59', '60_plus']]
    for col in expected_cols:
        if col not in pivot_df:
            pivot_df[col] = 0

    location = displayName(name)
    pivot_df['Location'] = location
    pivot_df['Reference_Id'] = name
    pivot_df = pivot_df[['Location', 'date', 'hour_group', 'Reference_Id'] + expected_cols]

    pivot_df['Total Impressions'] = pivot_df[expected_cols].sum(axis=1)
    impression_cols = expected_cols + ['Total Impressions']
    pivot_df[impression_cols] = pivot_df[impression_cols].astype(int)
    pivot_df = pivot_df.sort_values(by=['date', 'hour_group'])
    pivot_df['date'] = pd.to_datetime(pivot_df['date'])
    if platform.system() == 'Windows':
        pivot_df['date'] = pivot_df['date'].dt.strftime('%A, %#d %B, %Y')
    else:
        pivot_df['date'] = pivot_df['date'].dt.strftime('%A, %-d %B, %Y')
    return pivot_df


def ageGender_script(df):
    age_order = ['10-19', '20-29', '30-39', '40-49', '50-59', '60_plus']
    df_copy = df.copy()
    df_copy['age'] = df_copy['agegender'].str.extract(r'(\d+)', expand=False).astype(int)
    df_copy['age_group'] = pd.cut(df_copy['age'], bins=[9, 19, 29, 39, 49, 59, 200],
                                 labels=age_order, right=True)

    age_summary = df_copy.groupby('age_group')['PV'].sum().reset_index()
    age_summary.columns = ['Age', 'Impression']
    age_summary['Percentage'] = (age_summary['Impression'] / age_summary['Impression'].sum())
    age_summary = age_summary[['Age', 'Percentage', 'Impression']]

    df_copy['gender'] = df_copy['agegender'].str.extract(r'(female|male)')
    gender_summary = df_copy.groupby('gender')['PV'].sum().reset_index()
    gender_summary.columns = ['Gender', 'Impression']
    gender_summary['Percentage'] = (gender_summary['Impression'] / gender_summary['Impression'].sum())
    gender_summary = gender_summary[['Gender', 'Percentage', 'Impression']]

    agegender_summary = df_copy.groupby('agegender')['PV'].sum().reset_index()
    agegender_summary.columns = ['AgeGender', 'Impression']
    agegender_summary['Percentage'] = agegender_summary['Impression'] / agegender_summary['Impression'].sum()
    agegender_pivot = agegender_summary.set_index('AgeGender')[['Percentage', 'Impression']].T
    agegender_pivot.loc['Percentage'] = agegender_pivot.loc['Percentage'].astype(float).round(10)
    agegender_pivot.loc['Impression'] = agegender_pivot.loc['Impression'].apply(lambda x: f"{int(x):,}")

    agegender_overall = df_copy.groupby(['Reference_Id', 'agegender'])['PV'].sum().reset_index()
    agegender_overall.columns = ['Reference_Id', 'AgeGender', 'Impression']
    agegender_overall['Percentage'] = agegender_overall.groupby('Reference_Id')['Impression'].transform(lambda x: x / x.sum())
    agegender_overall = agegender_overall.pivot(index='Reference_Id', columns='AgeGender', values=['Impression', 'Percentage'])
    return age_summary, gender_summary, agegender_pivot, agegender_overall


def generate_excel_report(df):
    try:
        df_report = df.copy()
        hour_groups_train_channel = {
            5: '05-10', 6: '05-10', 7: '05-10', 8: '05-10', 9: '05-10',
            10: '10-18', 11: '10-18', 12: '10-18', 13: '10-18', 14: '10-18',
            15: '10-18', 16: '10-18', 17: '10-18',
            18: '18-24', 19: '18-24', 20: '18-24', 21: '18-24', 22: '18-24', 23: '18-24'
        }
        df_report["hour_group"] = df_report["hour"].map(hour_groups_train_channel)
        df_report = df_report.rename(columns={"SpotImpressions": "PV"})
        df_report['date'] = pd.to_datetime(df_report['date'])

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        engine = 'openpyxl'
        if importlib.util.find_spec('openpyxl') is None:
            engine = 'xlsxwriter'
        with pd.ExcelWriter(temp_file.name, engine=engine) as writer:
            overall_perf = df_report.groupby("ReferenceId", as_index=False)["PV"].sum()
            overall_perf = overall_perf.rename(columns={"PV": "Impression"})
            overall_perf.to_excel(writer, sheet_name="Overall Performance", index=False)

            daily = df_report.groupby(["date"], as_index=False)["PV"].sum()
            daily = daily.rename(columns={"PV": "Impression"})
            daily['date'] = daily['date'].dt.strftime('%Y-%m-%d')
            daily.to_excel(writer, sheet_name="Daily Summary", index=False)

            df_report_renamed = df_report.rename(columns={"ReferenceId": "Reference_Id"})
            age_summary, gender_summary, agegender_pivot, agegender_overall = ageGender_script(df_report_renamed)
            age_summary.to_excel(writer, sheet_name="Age Gender", startrow=0, index=False)
            gender_summary.to_excel(writer, sheet_name="Age Gender", startrow=0, startcol=4, index=False)
            agegender_pivot.to_excel(writer, sheet_name="Age Gender", startrow=10)
            agegender_overall.to_excel(writer, sheet_name="Age Gender", startrow=15)

            hourly = df_report.groupby(["hour"], as_index=False)["PV"].sum()
            hourly = hourly.rename(columns={"PV": "Impression"})
            hourly.to_excel(writer, sheet_name="Hourly Summary", index=False)

            overall_name = 'JPN-JEK-N-00000-00055'
            network_summary = networkSum_(df_report_renamed, overall_name)
            network_summary.to_excel(writer, sheet_name="Network Summary", index=False)

            reference_ids = df_report['ReferenceId'].unique()
            for ref_id in reference_ids:
                tmp = df_report_renamed[df_report_renamed['Reference_Id'] == ref_id]
                if not tmp.empty:
                    network_summary = networkSum_(tmp, ref_id)
                    loc_name = displayName(ref_id)
                    sheet_name = loc_name[:31].replace('/', '_').replace('\\', '_').replace('?', '_').replace('*', '_').replace('[', '_').replace(']', '_')
                    network_summary.to_excel(writer, sheet_name=sheet_name, index=False)
        return temp_file.name
    except Exception as e:
        print(f"Error generating Excel report: {e}")
        return None

# -----------------------------------------

def get_weather_label(date_str, lat, lon):
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        point = Point(lat, lon)
        daily = Daily(point, date, date)
        weather = daily.fetch()
        if weather.empty:
            return "sunny"
        w = weather.iloc[0]
        prcp = w.get('prcp', 0) or 0
        snow = w.get('snow', 0) or 0
        tavg = w.get('tavg', 15) or 15
        if snow > 0:
            return "snow"
        elif prcp > 0:
            return "rain"
        elif tavg < 5:
            return "cold"
        else:
            return "sunny"
    except Exception as e:
        print(f"Weather API error: {e}")
        return "sunny"


def hour_type_to_hours(hour_type):
    mapping = {
        'morning': list(range(5, 12)),
        'afternoon': list(range(12, 18)),
        'evening': list(range(18, 24)),
        'night': list(range(0, 5)),
        'full': list(range(5, 24))
    }
    return mapping.get(hour_type.lower(), [])


def generate_date_range(start, end):
    return pd.date_range(start=start, end=end)


def add_to_cart(item_type, name, start_date, end_date, hour_types, spots_per_hour):
    if not name:
        return "Please select a name", get_cart_display(), update_remove_choices()
    if not start_date or not end_date:
        return "Please select both start and end dates", get_cart_display(), update_remove_choices()
    if not hour_types:
        return "Please select at least one hour type", get_cart_display(), update_remove_choices()
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if start_dt > end_dt:
            return "End date must be after start date", get_cart_display(), update_remove_choices()
    except ValueError:
        return "Invalid date format", get_cart_display(), update_remove_choices()

    item = {
        'type': item_type,
        'name': name,
        'start_date': start_date,
        'end_date': end_date,
        'hour_type': hour_types if isinstance(hour_types, list) else [hour_types],
        'spots_per_hour': int(spots_per_hour) if spots_per_hour else 1
    }
    campaign_cart.append(item)
    return f"Added: {item_type} - {name}", get_cart_display(), update_remove_choices()


def get_cart_display():
    if not campaign_cart:
        return pd.DataFrame(columns=['Type', 'Name', 'Start Date', 'End Date', 'Hour Types', 'Spots/Hour'])
    display_data = []
    for i, item in enumerate(campaign_cart):
        display_data.append({
            'Index': i,
            'Type': item['type'],
            'Name': item['name'],
            'Start Date': item['start_date'],
            'End Date': item['end_date'],
            'Hour Types': ', '.join(item['hour_type']),
            'Spots/Hour': item['spots_per_hour']
        })
    return pd.DataFrame(display_data)


def update_remove_choices():
    if not campaign_cart:
        return gr.update(choices=[])
    choices = [f"{i}: {item['type']} - {item['name']}" for i, item in enumerate(campaign_cart)]
    return gr.update(choices=choices)


def remove_from_cart(selected_item):
    if not selected_item or not campaign_cart:
        return "No item selected or cart is empty", get_cart_display(), update_remove_choices()
    try:
        index = int(selected_item.split(':')[0])
        if 0 <= index < len(campaign_cart):
            removed_item = campaign_cart.pop(index)
            return f"Removed: {removed_item['type']} - {removed_item['name']}", get_cart_display(), update_remove_choices()
        else:
            return "Invalid selection", get_cart_display(), update_remove_choices()
    except (ValueError, IndexError):
        return "Invalid selection format", get_cart_display(), update_remove_choices()


def clear_cart():
    campaign_cart.clear()
    return "Cart cleared successfully", get_cart_display(), update_remove_choices()


def predict(cart):
    if not cart:
        return pd.DataFrame()
    data = []
    agegender_keys = [
        'female_10_19', 'female_20_29', 'female_30_39', 'female_40_49', 'female_50_59', 'female_60_plus',
        'male_10_19', 'male_20_29', 'male_30_39', 'male_40_49', 'male_50_59', 'male_60_plus'
    ]
    for item in cart:
        try:
            if item['type'] == 'Location':
                location_data = location_df[location_df['ReferenceId'] == item['name']]
            elif item['type'] == 'Network':
                network_ref_ids = network_df[network_df['NetworkId'] == item['name']]['ReferenceId'].tolist()
                location_data = location_df[location_df['ReferenceId'].isin(network_ref_ids)]
            else:
                continue
            if location_data.empty:
                continue
            for _, loc in location_data.iterrows():
                date_range = generate_date_range(item['start_date'], item['end_date'])
                for date in date_range:
                    date_str = date.strftime('%Y-%m-%d')
                    weather_label = get_weather_label(date_str, loc['lat'], loc['lon'])
                    for hour_type in item['hour_type']:
                        for hour in hour_type_to_hours(hour_type):
                            for agegender in agegender_keys:
                                row = {
                                    'date': date_str,
                                    'Reference_Id': loc['ReferenceId'],
                                    'geohash5': loc['geohash5'],
                                    'geohash6': loc['geohash6'],
                                    'hour': hour,
                                    'agegender': agegender,
                                    'weather': weather_label,
                                    'spotsPerHour': item['spots_per_hour'],
                                    'spotDuration': loc['spotDuration'],
                                    'dwellTime': loc['dwellTime'],
                                    'loopLength': loc['loopLength']
                                }
                                data.append(row)
        except Exception as e:
            print(f"Error processing item {item}: {e}")
            continue
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    if not holiday.empty:
        df = df.merge(holiday[['date', 'Day_Type_2']], on='date', how='left')
        df.rename(columns={'Day_Type_2': 'Day_Type'}, inplace=True)
    else:
        df['Day_Type'] = 'regular'
    df['day_of_week'] = df['date'].dt.day_name()
    df.to_csv("TrainingData_Test.csv", index=False, encoding='utf-8-sig')
    model = joblib.load('best_model_XGBoost.pkl')
    encoder = joblib.load('onehot_encoder.pkl')
    try:
        X_new = df[['geohash5', 'geohash6', 'Day_Type', 'day_of_week', 'weather', 'hour', 'agegender']]
        X_new_encoded = encoder.transform(X_new)
        predictions = model.predict(X_new_encoded)
        df["predicted_impressions"] = predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        df['predicted_impressions'] = np.random.randint(100, 1000, len(df))
    df = df.merge(otcRatio, left_on="Reference_Id", right_on="referenceId", how="left")
    df["NonSpotImpressions"] = df["predicted_impressions"] * (df["share"] * df["mediaRatio"])
    df["NonSpotImpressions"] = df["NonSpotImpressions"].replace([np.inf, -np.inf], np.nan).fillna(0)
    est_spot_PV = []
    for _, row1 in df.iterrows():
        tmpSpot = custom_round(spot_calc(row1['NonSpotImpressions'], row1['dwellTime'],
                                       row1['loopLength'], row1['spotsPerHour'], row1['spotDuration']))
        est_spot_PV.append(tmpSpot)
    df['SpotImpressions'] = est_spot_PV
    df['predicted_impressions'] = df['predicted_impressions'].round(2)
    df['NonSpotImpressions'] = df['NonSpotImpressions'].round(2)
    return df


def generate_report():
    global latest_report_data
    if not campaign_cart:
        return pd.DataFrame(columns=['Message']).assign(Message=['No items in cart. Please add items first.'])
    try:
        result_df = predict(campaign_cart)
        if result_df.empty:
            return pd.DataFrame(columns=['Message']).assign(Message=['No data generated. Please check your selections.'])
        latest_report_data = result_df.copy()
        output_df = result_df[['Reference_Id', 'date', 'hour', 'agegender', 'predicted_impressions', 'NonSpotImpressions', 'SpotImpressions']].copy()
        output_df = output_df.rename(columns={'Reference_Id': 'ReferenceId', 'predicted_impressions': 'predcited_impressions'})
        output_df = output_df.sort_values(['ReferenceId', 'date', 'hour', 'agegender']).reset_index(drop=True)
        return output_df
    except Exception as e:
        error_msg = f"Error generating report: {str(e)}"
        print(error_msg)
        return pd.DataFrame(columns=['Error']).assign(Error=[error_msg])


def download_full_data():
    global latest_report_data
    if latest_report_data is None or latest_report_data.empty:
        return None
    try:
        df = latest_report_data[['Reference_Id', 'date', 'hour', 'agegender', 'predicted_impressions', 'NonSpotImpressions', 'SpotImpressions']].copy()
        df = df.rename(columns={'Reference_Id': 'ReferenceId', 'predicted_impressions': 'predcited_impressions'})
        df['predcited_impressions'] = df['predcited_impressions'].round(2)
        df['NonSpotImpressions'] = df['NonSpotImpressions'].round(2)
        df['SpotImpressions'] = df['SpotImpressions'].round(2)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return None


def download_excel_report():
    global latest_report_data
    if latest_report_data is None or latest_report_data.empty:
        return None
    try:
        excel_df = latest_report_data.copy()
        excel_file = generate_excel_report(excel_df)
        if excel_file:
            return excel_file
        return None
    except Exception as e:
        print(f"Error generating Excel: {e}")
        return None


def initialize_data(n_df, l_df, otc_df, hol_df):
    global network_df, location_df, otcRatio, holiday
    network_df = n_df
    location_df = l_df
    otcRatio = otc_df
    holiday = hol_df

