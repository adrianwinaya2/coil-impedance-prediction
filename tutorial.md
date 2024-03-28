import streamlit as st
import pandas as pd
import numpy as np
import time

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon']
)

st.write("Hello, world!")
st.table(df)
st.line_chart(df)
st.map(map_data)

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

st.text_input("Your name", key="name")
if st.session_state.name:
    st.write("Hello", st.session_state.name)

if st.checkbox("Show dataframe"):
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c']
    )
    chart_data

option = st.selectbox(
    'Which number do you like best?',
    df['first column']
)
'You selected: ', option

add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

add_slider = st.sidebar.slider(
    "Select a range of values",
    0.0, 100.0, (25.0, 75.0)
)

left_column, right_column = st.columns(2)
left_column.button("Press me!")
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin")
    )
    st.write(f"You are in {chosen} house!")

'Starting a long computation...'
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    latest_iteration.text(f'Iteration {i+1}')
    bar.progress(i + 1)
    time.sleep(0.1)
'...and now we\'re done!'   