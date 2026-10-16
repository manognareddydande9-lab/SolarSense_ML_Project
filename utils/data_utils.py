import os
import pandas as pd

FEATURES = [
    "DC_POWER","DAILY_YIELD","AMBIENT_TEMPERATURE","MODULE_TEMPERATURE",
    "IRRADIATION","hour","minute","day","month","day_of_year","is_daylight"
]

def load_solar_data(base_dir):
    data_dir = os.path.join(base_dir, "data")
    gen = pd.read_csv(os.path.join(data_dir, "Plant_1_Generation_Data.csv"))
    weather = pd.read_csv(os.path.join(data_dir, "Plant_1_Weather_Sensor_Data.csv"))
    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], dayfirst=True, errors="coerce")
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], dayfirst=True, errors="coerce")
    g = gen.groupby("DATE_TIME", as_index=False).agg(
        DC_POWER=("DC_POWER","sum"), AC_POWER=("AC_POWER","sum"),
        DAILY_YIELD=("DAILY_YIELD","sum"), TOTAL_YIELD=("TOTAL_YIELD","sum"))
    w = weather.groupby("DATE_TIME", as_index=False).agg(
        AMBIENT_TEMPERATURE=("AMBIENT_TEMPERATURE","mean"),
        MODULE_TEMPERATURE=("MODULE_TEMPERATURE","mean"),
        IRRADIATION=("IRRADIATION","mean"))
    df = pd.merge_asof(g.sort_values("DATE_TIME"), w.sort_values("DATE_TIME"),
                       on="DATE_TIME", direction="nearest",
                       tolerance=pd.Timedelta("15min"))
    df["hour"]=df["DATE_TIME"].dt.hour
    df["minute"]=df["DATE_TIME"].dt.minute
    df["day"]=df["DATE_TIME"].dt.day
    df["month"]=df["DATE_TIME"].dt.month
    df["day_of_year"]=df["DATE_TIME"].dt.dayofyear
    df["is_daylight"]=(df["IRRADIATION"].fillna(0)>5).astype(int)
    return df
