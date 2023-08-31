import streamlit as st 
import pandas as pd
from Home import face_rec

st.set_page_config(page_title='Reporting', layout='wide')
st.subheader('Reporting')

# Recuperar datos de registros y mostrar en Report.py
# extraer datos de la lista redis
name = 'attendance:logs'

def load_logs(name, end=-1):
    logs_list = face_rec.r.lrange(name, start=0, end=end)  # extraer todos los datos de la base de datos redis
    decoded_logs = [log.decode('utf-8') for log in logs_list]  # decodificar bytes a string
    
    # Dividir cada entrada de registro
    logs_data = []
    for log in decoded_logs:
        name, role, datetime = log.split('@')
        logs_data.append({'Name': name, 'Role': role, 'Time': datetime})
    
    return logs_data

# pestañas para mostrar la información
tab1, tab2 = st.tabs(['Registered Data', 'Logs'])

with tab1:
    if st.button('Refresh Data'):
        # Recuperar los datos de la base de datos Redis
        with st.spinner('Retrieving Data from Redis DB ...'):    
            redis_face_db = face_rec.retrieve_data(name='academy:register')
            st.dataframe(redis_face_db[['Name', 'Role']])

with tab2:
    if st.button('Refresh Logs'):
        logs_data = load_logs(name=name)
        # Convertir datos de registros a DataFrame para una mejor visualización
        df_logs = pd.DataFrame(logs_data)
        st.dataframe(df_logs)

