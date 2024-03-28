import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import shap
import pickle
import sqlite3

# conn = sqlite3.connect('database.db')
# cursor = conn.cursor()
conn = st.connection('history_db', type='sql')
with open('xgb_fs.pkl', 'rb') as file:
    model = pickle.load(file)

def insert_data(params, impedance):
    with conn.session as s:
        s.execute('''
            INSERT INTO coil_impedance (
                pid_lv, lid_lv, tid_lv, pod_lv, lod_lv, tod_lv,
                pid_hv, lid_hv, tid_hv, pod_hv, lod_hv, tod_hv, impedance
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            params['PID LV'], params['LID LV'], params['TID LV'],
            params['POD LV'], params['LOD LV'], params['TOD LV'],
            params['PID HV'], params['LID HV'], params['TID HV'],
            params['POD HV'], params['LOD HV'], params['TOD HV'],
            impedance
        ))
        s.commit()

def get_data():
    return conn.query('select * from pet_owners')

def close_connection():
    conn.close()

def predict(test_data):
    # TODO
    # model = pd.read_pickle('coil_impedance_model.pkl')
    
    impedance = model.predict(test_data)[0]
    st.write(f'The predicted impedance is {impedance:.2f} ohms')
    insert_data(test_data, impedance)

    visualize_button = st.button('Visualize')

    if visualize_button:
        visualize()

    return impedance[0]

def visualize(model, test_data):

    @st.cache
    def init_shap_js():
        shap.initjs()
    
    init_shap_js()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_data)

    shap_values = pd.DataFrame(explainer.shap_values(test_data), columns=[str(col+' Shap') for col in test_data.columns])
    shap_values.index = test_data.index
    explanation = shap.Explanation(shap_values.values, data=test_data, feature_names=test_data.columns)

    shap_plot(shap_values, test_data, explainer, explanation, index1='PID LV', index2='LID LV', show=False)
    
    return shap_values, explanation

def shap_plot(shap_val, feature_data, explainer=None, explanation=None, index1=None, index2=None, show=True):
    
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
        shap.dependence_plot(index1, interaction_index=index2, shap_values=shap_val, features=feature_data)
        st.pyplot(plt.gcf())
    
    with force:
        st.header('Force Plot')
        shap.force_plot(explainer.expected_value, shap_val, feature_data)
        st.pyplot(plt.gcf())

    with bar:
        st.header('Bar Plot')
        shap.bar_plot(shap_val, feature_data.values[index1], feature_data.columns)
        st.pyplot(plt.gcf())
    
    with embed:
        st.header('Embedding Plot')
        shap.embedding_plot(index1, shap_val.values, feature_data.columns)
        st.pyplot(plt.gcf())
    
    with waterfall:
        st.header('Waterfall Plot')
        shap.plots.waterfall(explanation) # max_display=14
        st.pyplot(plt.gcf())


st.title('Coil Impedance Prediction')
with st.form(key='input_form'):
    left, right = st.columns(2)
    params = {
        'pid_lv': left.number_input('PID LV'),
        'lid_lv': left.number_input('LID LV'),
        'tid_lv': left.number_input('TID LV'),

        'pod_lv': left.number_input('POD LV'),
        'lod_lv': left.number_input('LOD LV'),
        'tod_lv': left.number_input('TOD LV'),

        'pid_hv': right.number_input('PID HV'),
        'lid_hv': right.number_input('LID HV'),
        'tid_hv': right.number_input('TID HV'),

        'pod_hv': right.number_input('POD HV'),
        'lod_hv': right.number_input('LOD HV'),
        'tod_hv': right.number_input('TOD HV'),
    }

    submitted = st.form_submit_button(label='Predict Impedance')

if submitted:
    with st.spinner('Predicting...'):
        time.sleep(1)
        test_data = pd.DataFrame(params, index=[0])
        predict(test_data)
        st.success('Done')

