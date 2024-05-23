import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import shap
import pickle
import sqlite3
from streamlit_gsheets import GSheetsConnection

import xgboost as xgb

def get_data() -> pd.DataFrame:
    return conn.read(ttl=0, usecols=[i for i in range(0, 14)]).dropna()

def insert_data(new_data, impedance):
    with status:
        st.write('Inserting to Database.')
        time.sleep(1)
    # with st.connection('history_db', type='sql').session as s:
    #     s.execute(f'''
    #         INSERT INTO history (
    #             pid_lv, lid_lv, tid_lv, pod_lv, lod_lv, tod_lv,
    #             pid_hv, lid_hv, tid_hv, pod_hv, lod_hv, tod_hv, impedance
    #         )
    #         VALUES (
    #             {params['PID LV'][0]}, {params['LID LV'][0]}, {params['TID LV'][0]}, 
    #             {params['POD LV'][0]}, {params['LOD LV'][0]}, {params['TOD LV'][0]},
    #             {params['PID HV'][0]}, {params['LID HV'][0]}, {params['TID HV'][0]},
    #             {params['POD HV'][0]}, {params['LOD HV'][0]}, {params['TOD HV'][0]}, {impedance}
    #         )
    #     ''')
    #     s.commit()
    # query = f'''INSERT INTO History (
    #                 "PID LV", "LID LV", "TID LV", "POD LV", "LOD LV", "TOD LV",
    #                 "PID HV", "LID HV", "TID HV", "POD HV", "LOD HV", "TOD HV", "Impedance"
    #             )
    #             VALUES (
    #                 {data['PID LV'][0]}, {data['LID LV'][0]}, {data['TID LV'][0]}, 
    #                 {data['POD LV'][0]}, {data['LOD LV'][0]}, {data['TOD LV'][0]},
    #                 {data['PID HV'][0]}, {data['LID HV'][0]}, {data['TID HV'][0]},
    #                 {data['POD HV'][0]}, {data['LOD HV'][0]}, {data['TOD HV'][0]}, {data['Impedance'][0]}
    #             ) '''

    old_data = get_data()
    new_data['Predict Time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    new_data['Impedance'] = impedance

    data = old_data.append(new_data, ignore_index=True)

    df = conn.update(
        worksheet=0,
        data=data,
    )
    # st.cache_data.clear()
    # st.experimental_rerun()

def predict(test_data):
    with status:
        st.write('Calculating Impedance')
        time.sleep(1)

    # model = pd.read_pickle('coil_impedance_model.pkl')
    impedance = model.predict(test_data)[0]

    return impedance

def visualize(model, test_data):

    @st.cache_resource
    def init_shap_js():
        shap.initjs()
    
    with status:
        st.write('Calculating SHAP Values.')
        time.sleep(1)

    # ! SHAP PREPARATION
    init_shap_js()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_data)

    shap_values_df = pd.DataFrame(shap_values, columns=[str(col+' Shap') for col in test_data.columns])
    shap_values_df.index = test_data.index
    st.write(shap_values_df)
    explanation = shap.Explanation(shap_values, data=test_data, feature_names=test_data.columns)

    # ! SHAP PLOT
    shap_plot(shap_values, test_data, explainer=explainer, explanation=explanation, show=False)
    
    return shap_values, explanation

def shap_plot(shap_val, feature_data, explainer=None, explanation=None, show=True):
    with status:
        st.write('Visualizing Result.')
        time.sleep(1)
    
    summary, decision, force, bar, waterfall = st.tabs(['Summary', 'Decision', 'Force', 'Bar', 'Waterfall'])

    with summary:
        st.header('Summary Plot')
        shap.summary_plot(shap_val, features=feature_data, feature_names=feature_data.columns, show=show)
        st.pyplot(plt.gcf())
    
    with decision:
        st.header('Decision Plot')
        shap.decision_plot(explainer.expected_value, shap_val, features=feature_data)
        st.pyplot(plt.gcf())
    
    # with dependence:
    #     st.header('Dependence Plot - Not Compatible')
    #     select1, select2 = st.columns(2)
    #     feature1 = select1.selectbox('Feature 1', options=feature_data.columns, key='dependence_feature1')
    #     feature2 = select2.selectbox('Feature 2', options=feature_data.columns, key='dependence_feature2')

    #     shap.dependence_plot(feature1, interaction_index=feature2, shap_values=shap_val, features=feature_data)
    #     st.pyplot(plt.gcf())
    
    with force:
        st.header('Force Plot')
        shap.force_plot(explainer.expected_value, shap_val[0], feature_data.iloc[0], matplotlib=True)
        st.pyplot(plt.gcf())

    with bar:
        st.header('Bar Plot')
        print(explanation)
        plt.figure(figsize=(10, 8))

        shap.bar_plot(shap_val[0], feature_data.values[0], feature_data.columns)
        # shap.plots.bar(explanation, max_display=3, show=False)

        st.pyplot(plt.gcf())
    
    # with embed:
    #     st.header('Embedding Plot')
    #     feature1 = st.selectbox('Select Feature', options=feature_data.columns)
    #     shap.embedding_plot(feature1, shap_val, feature_data.columns)
    #     st.pyplot(plt.gcf())
    
    with waterfall:
        st.header('Waterfall Plot')
        plt.figure(figsize=(10, 8))

        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_val[0])
        # shap.plots.waterfall(explanation[0])
        
        st.pyplot(plt.gcf())

st.set_page_config(
    page_title="Predict",
    page_icon="📊",
)

# conn = sqlite3.connect('database.db')
# cursor = conn.cursor()
conn = st.connection("gsheets", type=GSheetsConnection)
        
with open('models/xgb_fs.pkl', 'rb') as file:
    model = pickle.load(file)
# model = xgb.Booster(model_file='models/xgb_fs.xgb')

st.title('Coil Impedance Prediction')
if 'error_msg' not in st.session_state:
    st.session_state['error_msg'] = "Please fill required inputs"

# Validation function
def validate_inputs():
    if st.session_state['PID LV'] >= st.session_state['POD LV']:
        st.session_state['error_msg'] = "'PID LV' must be smaller than 'POD LV'"
    elif st.session_state['PID HV'] >= st.session_state['POD LV']:
        st.session_state['error_msg'] = "'PID HV' must be smaller than 'POD HV'"
    elif st.session_state['LID LV'] >= st.session_state['LOD LV']:
        st.session_state['error_msg'] = "'LID LV' must be smaller than 'LOD LV'"
    elif st.session_state['LID HV'] >= st.session_state['LOD HV']:
        st.session_state['error_msg'] = "'LID HV' must be smaller than 'LOD HV'"
    elif st.session_state['TID LV'] >= st.session_state['TOD LV']:
        st.session_state['error_msg'] = "'TID LV' must be smaller than 'TOD LV'"
    elif st.session_state['TID HV'] >= st.session_state['TOD HV']:
        st.session_state['error_msg'] = "'TID HV' must be smaller than 'TOD HV'"
    else:
        st.session_state['error_msg'] = ""

# Create the input form
left, right = st.columns(2)
with st.form(key='input_form'):
    left.number_input('PID LV', min_value=0, on_change=validate_inputs, key='PID LV')
    left.number_input('LID LV', min_value=0, on_change=validate_inputs, key='LID LV')
    left.number_input('TID LV', min_value=0, on_change=validate_inputs, key='TID LV')

    left.number_input('POD LV', min_value=0, on_change=validate_inputs, key='POD LV')
    left.number_input('LOD LV', min_value=0, on_change=validate_inputs, key='LOD LV')
    left.number_input('TOD LV', min_value=0, on_change=validate_inputs, key='TOD LV')

    right.number_input('PID HV', min_value=0, on_change=validate_inputs, key='PID HV')
    right.number_input('LID HV', min_value=0, on_change=validate_inputs, key='LID HV')
    right.number_input('TID HV', min_value=0, on_change=validate_inputs, key='TID HV')

    right.number_input('POD HV', min_value=0, on_change=validate_inputs, key='POD HV')
    right.number_input('LOD HV', min_value=0, on_change=validate_inputs, key='LOD HV')
    right.number_input('TOD HV', min_value=0, on_change=validate_inputs, key='TOD HV')

    # Initial placeholder for the submit button
    if st.session_state['error_msg']:
        submitted = st.form_submit_button(label='Predict Impedance', disabled=True)
    else:
        submitted = st.form_submit_button(label='Predict Impedance')

# Display error message if validation fails
if st.session_state['error_msg']:
    st.error(st.session_state['error_msg'])

# Successful form submission
if submitted and not st.session_state['error_msg']:
    st.success("Form submitted successfully!")
    status = st.status('Processing...', expanded=True)

    # ! INPUT DATA
    input_data = pd.DataFrame({
        'PID LV': st.session_state['PID LV'], 'LID LV': st.session_state['LID LV'], 'TID LV': st.session_state['TID LV'],
        'POD LV': st.session_state['POD LV'], 'LOD LV': st.session_state['LOD LV'], 'TOD LV': st.session_state['TOD LV'],
        'PID HV': st.session_state['PID HV'], 'LID HV': st.session_state['LID HV'], 'TID HV': st.session_state['TID HV'],
        'POD HV': st.session_state['POD HV'], 'LOD HV': st.session_state['LOD HV'], 'TOD HV': st.session_state['TOD HV'],
    }, index=[0])
    test_data = input_data[model.feature_names_in_]
    st.subheader('Input Data')
    st.dataframe(test_data)

    # ! PREDICT IMPEDANCE
    impedance = predict(test_data)
    st.subheader('Result')
    st.write(f'The predicted impedance is {impedance:.2f} ohms')
    
    # ! INSERT DATA TO DB
    insert_data(input_data, impedance)

    # ! SHAP VISUALIZATION
    st.subheader('SHAP Visualization')
    visualize(model, test_data)

    with status:
        status.update(label="Process Complete!", state="complete", expanded=False)
        time.sleep(1)