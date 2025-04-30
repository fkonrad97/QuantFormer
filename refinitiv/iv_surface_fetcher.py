import os
import json
import pandas as pd
import refinitiv.dataplatform as rdp
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ====== AUTH CONFIG ======
load_dotenv()
APP_KEY = os.getenv('APP_KEY')
REFINITIVE_USERNAME = os.getenv('REFINITIVE_USERNAME')
REFINITIVE_PASSWORD = os.getenv('REFINITIVE_PASSWORD')

# ====== SESSION SETUP ======
session = rdp.PlatformSession(
    APP_KEY,
    rdp.GrantPassword(username=REFINITIVE_USERNAME, password=REFINITIVE_PASSWORD)
)
session.open()
print("Session opened.")

vs_endpoint = rdp.Endpoint(session, "https://api.refinitiv.com/data/quantitative-analytics-curves-and-surfaces/v1/surfaces")

# ====== FLATTENER ======
def flatten_json_surface(json_data, valuation_date_str):
    valuation_date = pd.to_datetime(valuation_date_str)
    surface_data = json_data['data'][0]['surface']
    strikes = list(map(float, surface_data[0][1:]))
    records = []

    for row in surface_data[1:]:
        expiry_str = row[0]
        maturity = (pd.to_datetime(expiry_str) - valuation_date).days / 365.0
        vols = row[1:]
        for strike, iv in zip(strikes, vols):
            if iv is not None:
                records.append({'strike': strike, 'maturity': maturity, 'iv': iv / 100})

    return pd.DataFrame(records)

# ====== BATCH FETCH ======
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 4, 30)
instrumentCode = "TSM"

all_surfaces = {}
request_body_template = {
    "universe": [
        {
            "surfaceTag": "1",
            "underlyingType": "Eti",
            "underlyingDefinition": {
                "instrumentCode": f"{instrumentCode}"
            },
            "surfaceParameters": {
                "priceSide": "Mid",
                "volatilityModel": "SVI",
                "xAxis": "Date",
                "yAxis": "Strike"
            },
            "surfaceLayout": {
                "format": "Matrix",
                "yPointCount": 10
            }
        }
    ],
    "outputs": ["ForwardCurve"]
}

for i in range((end_date - start_date).days + 1):
    date = start_date + timedelta(days=i)
    date_str = date.strftime("%Y-%m-%d")
    
    try:
        response = vs_endpoint.send_request(
            method=rdp.Endpoint.RequestMethod.POST,
            body_parameters=request_body_template
        )
        all_surfaces[date_str] = response.data.raw
        print(f"Collected {date_str}")
    except Exception as e:
        print(f"Failed to fetch {date_str}: {e}")

# ====== SAVE TO FILES ======
for date_str, json_data in all_surfaces.items():
    try:
        df = flatten_json_surface(json_data, valuation_date_str=date_str)
        folder_path = f"data/real_world/iv_data/{instrumentCode}/"
        os.makedirs(folder_path, exist_ok=True)
        df.to_csv(os.path.join(folder_path, f"{instrumentCode}_{date_str}_iv_surface.csv"), index=False)

        print(f"Saved surface for {date_str}")
    except Exception as e:
        print(f"Failed to process {date_str}: {e}")

session.close()
print("Session closed.")