import streamlit as st
import pandas as pd

st.title('Prediction History')

def get_data():
    conn = st.connection('history_db', type='sql', ttl=0)
    data = pd.DataFrame(conn.query('SELECT * from HISTORY'))
    return data.set_index('id')

st.cache_data.clear()
history = get_data()
st.write(history)

# dialect = "mysql"
# host = "localhost"
# port = 3306
# database = "xxx"
# username = "xxx"
# password = "xxx"