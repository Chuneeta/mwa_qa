from argparse import ArgumentParser
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import pandas as pd
import os

parser = ArgumentParser(
    description="Exporting google sheet to csv file")
parser.add_argument('--id', dest='id', type=str,
                    help='Google spreadsheet ID',
                    default='1jauEVvv_uT37ED8OTFh3MffPpPPwGPvGjDyC57kkhH8')
parser.add_argument('--crd', dest='crd', type=str, help='Credential file',
                    default='credentials.json')
parser.add_argument('--gn', dest='gsheet_name', type=str, help='Name of google spreadsheet',
                    default='WS_PASS_PH1')
parser.add_argument('--out', dest='outfile', default=None,
                    help='Name of ouput csvfile')
args = parser.parse_args()


# Load credentials from JSON file
credentials = Credentials.from_service_account_file(args.crd)

scoped_credentials = credentials.with_scopes(
    ['https://www.googleapis.com/auth/spreadsheets'])

# Authorize the client
gc = gspread.authorize(scoped_credentials)

# test access to a spreadsheet
SPREADSHEET_ID = args.id

spreadsheet = gc.open_by_key(SPREADSHEET_ID)
worksheet = spreadsheet.worksheet(args.gsheet_name)


# Retrieve all values from the worksheet
res = worksheet.get_values('A:BE')
# print(f"{len(res)=} {res=}")

# Create a DataFrame from the retrieved data
df = pd.DataFrame(res)

# set the first row as the column headers
df.columns = df.iloc[0]
df = df[1:]  # Remove the first row (column headers) from the DataFrame

print(df)

if args.outfile is None:
    out_name = args.gsheet_name + '.csv'
else:
    out_name = args.outfile
df.to_csv(out_name, index=False)
