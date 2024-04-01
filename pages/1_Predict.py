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

def insert_data(data):
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
    conn = st.experimental_connection("gsheets", type=GSheetsConnection)
    df = conn.update(
        spreadsheet="Coil Impedance Prediction Result",
        worksheet=0,
        data=data,
    )
    # st.cache_data.clear()
    # st.experimental_rerun()

def predict(test_data):
    # model = pd.read_pickle('coil_impedance_model.pkl')
    impedance = model.predict(test_data[model.feature_names_in_])[0]

    st.write(f'The predicted impedance is {impedance:.2f} ohms')
    df = test_data.copy()
    df['Impedance'] = impedance
    insert_data(df)

    visualize_button = st.button('Visualize')

    if visualize_button:
        visualize()

    return impedance

def visualize(model, test_data):

    @st.cache
    def init_shap_js():
        shap.initjs()
    
    print("masuk visualize")
    # ! SHAP PREPARATION
    init_shap_js()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_data)

    shap_values = pd.DataFrame(explainer.shap_values(test_data), columns=[str(col+' Shap') for col in test_data.columns])
    shap_values.index = test_data.index
    explanation = shap.Explanation(shap_values.values, data=test_data, feature_names=test_data.columns)

    # ! SHAP PLOT
    shap_plot(shap_values, test_data, explainer, explanation, index1='PID LV', index2='LID LV', show=False)
    
    return shap_values, explanation

def shap_plot(shap_val, feature_data, explainer=None, explanation=None, show=True):
    print("masuk shap_plot")
    
    summary, decision, dependence, force, bar, embed, waterfall = st.tabs(['Summary', 'Decision', 'Dependence', 'Force', 'Bar', 'Embedding', 'Waterfall'])

    with summary:
        st.header('Summary Plot')
        shap.summary_plot(shap_val, features=feature_data, feature_names=feature_data.columns, show=show)
        st.pyplot(plt.gcf())
    
    with decision:
        st.header('Decision Plot')
        shap.decision_plot(explainer.expected_value, shap_val, features=feature_data)
        st.pyplot(plt.gcf())
    
    with dependence:
        st.header('Dependence Plot')
        feature1 = st.selectbox('Feature 1', options=feature_data.columns, key='feature1')
        feature2 = st.selectbox('Feature 2', options=feature_data.columns, key='feature2')

        shap.dependence_plot(feature1, interaction_index=feature2, shap_values=shap_val, features=feature_data)
        st.pyplot(plt.gcf())
    
    with force:
        st.header('Force Plot')
        shap.force_plot(explainer.expected_value, shap_val, feature_data)
        st.pyplot(plt.gcf())

    with bar:
        st.header('Bar Plot')
        feature1 = st.selectbox('Select Feature', options=feature_data.columns)
        shap.bar_plot(shap_val, feature_data.values[feature1], feature_data.columns)
        st.pyplot(plt.gcf())
    
    with embed:
        st.header('Embedding Plot')
        feature1 = st.selectbox('Select Feature', options=feature_data.columns)
        shap.embedding_plot(feature1, shap_val.values, feature_data.columns)
        st.pyplot(plt.gcf())
    
    with waterfall:
        st.header('Waterfall Plot')
        shap.plots.waterfall(explanation) # max_display=14
        st.pyplot(plt.gcf())


# conn = sqlite3.connect('database.db')
# cursor = conn.cursor()
        
# with open('models/xgb_fs.pkl', 'rb') as file:
#     model = pickle.load(file)
model = xgb.Booster(model_file='xgb_fs.xgb')

st.title('Coil Impedance Prediction')
with st.form(key='input_form'):
    left, right = st.columns(2)
    params = {
        'PID LV': int(left.number_input('PID LV', format='%.0f')),
        'LID LV': int(left.number_input('LID LV', format='%.0f')),
        'TID LV': int(left.number_input('TID LV', format='%.0f')),

        'POD LV': int(left.number_input('POD LV', format='%.0f')),
        'LOD LV': int(left.number_input('LOD LV', format='%.0f')),
        'TOD LV': int(left.number_input('TOD LV', format='%.0f')),

        'PID HV': int(right.number_input('PID HV', format='%.0f')),
        'LID HV': int(right.number_input('LID HV', format='%.0f')),
        'TID HV': int(right.number_input('TID HV', format='%.0f')),

        'POD HV': int(right.number_input('POD HV', format='%.0f')),
        'LOD HV': int(right.number_input('LOD HV', format='%.0f')),
        'TOD HV': int(right.number_input('TOD HV', format='%.0f')),
    }

    submitted = st.form_submit_button(label='Predict Impedance')

if submitted:
    with st.spinner('Predicting...'):
        time.sleep(1)

        test_data = pd.DataFrame(params, index=[0])
        st.dataframe(test_data)
        predict(test_data)

        st.success('Done')

