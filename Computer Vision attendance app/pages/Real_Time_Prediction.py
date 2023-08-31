import streamlit as st 
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

st.set_page_config(page_title='Predictions')
st.subheader('Real-Time Attendance System')

# Recuperar los datos de la base de datos Redis
with st.spinner('Retrieving Data from Redis DB ...'):    
    redis_face_db = face_rec.retrieve_data(name='academy:register')
    st.dataframe(redis_face_db.reset_index())
    
st.success("Data sucessfully retrieved from Redis")

# Eliminar entrada de la base de datos Redis
delete_index = st.number_input("Enter the index of the entry you want to delete:", min_value=0, max_value=len(redis_face_db)-1, step=1, format="%i")
if st.button("Delete Entry"):
    # Obtener el nombre y el rol del índice seleccionado
    name_to_delete = redis_face_db.iloc[delete_index]['Name']
    role_to_delete = redis_face_db.iloc[delete_index]['Role']
    name_role_key = f"{name_to_delete}@{role_to_delete}"

    # Eliminar de Redis
    face_rec.r.hdel('academy:register', name_role_key)
    st.success(f"Entry {name_role_key} deleted successfully!")
    
    # Refrescar la página
    st.experimental_rerun()

waitTime = 30 # tiempo en segundos
setTime = time.time()
realtimepred = face_rec.RealTimePred() 


def video_frame_callback(frame):
    global setTime
    
    img = frame.to_ndarray(format="bgr24") 
    
    pred_img = realtimepred.face_prediction(img,redis_face_db,
                                        'facial_features',['Name','Role'],thresh=0.5)
    
    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
        realtimepred.saveLogs_redis()
        setTime = time.time() # reiniciar tiempo       
        print('Save Data to redis database')
    

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback)
