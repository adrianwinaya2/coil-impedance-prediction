import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection

st.title('Prediction History')

# ! LOCAL
# def get_data():
#     conn = st.connection('history_db', type='sql', ttl=0)
#     data = pd.DataFrame(conn.query('SELECT * from HISTORY'))
#     return data.set_index('id')

# st.cache_data.clear()
# data = get_data()
# st.write(data)

# ! ORIGINAL GSPREAD
# gc = gspread.service_account(filename='service_account.json')
# sh = gc.open("Coil Impedance Prediction Result")
# data = pd.DataFrame(sh.sheet1.get_all_records())

# ! STREAMLIT GSPREAD
conn = st.experimental_connection("gsheets", type=GSheetsConnection)
data = conn.read(ttl=0, usecols=[range(0, 12)])

st.write(data)

# dialect = "mysql"
# host = "localhost"
# port = 3306
# database = "xxx"
# username = "xxx"
# password = "xxx"