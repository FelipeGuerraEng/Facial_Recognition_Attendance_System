import numpy as np
import pandas as pd
import cv2
import redis
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
import time
from datetime import datetime
import os


# Conectar al Cliente Redis
hostname = '' #Ejemplo: 'redis-11240.c91.us-east-2-5.ec2.cloud.redislabs.com'
portnumber =   # Ejemplo: 11240
password = ''

r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)

# Recuperar datos de la base de datos
def retrieve_data(name):
    retrieve_dict= r.hgetall(name)
    retrieve_series = pd.Series(retrieve_dict)
    retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrieve_series.index
    index = list(map(lambda x: x.decode(), index))
    retrieve_series.index = index
    retrieve_df =  retrieve_series.to_frame().reset_index()
    retrieve_df.columns = ['name_role','facial_features']
    retrieve_df[['Name','Role']] = retrieve_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrieve_df[['Name','Role','facial_features']]


#Configurar FaceAnalysis de InsightFace
faceapp = FaceAnalysis(name='buffalo_sc',root='insightface_model', providers = ['CPUExecutionProvider'])
faceapp.prepare(ctx_id = 0, det_size=(640,640), det_thresh = 0.5)

# Algoritmo de Comparación usando la similitud coseno
def comparison_algorithm(dataframe,feature_column,test_vector,
                        name_role=['Name','Role'],thresh=0.5):
   
    # paso-1: tomar el dataframe (colección de datos)
    dataframe = dataframe.copy()
    # paso-2: Indexar los datos del embedding (caracteristicas) del dataframe y convertir en array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)
    
    # paso-3: Calcular la similitud coseno
    similar = pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    # paso-4: filtrar los datos
    data_filter = dataframe.query(f'cosine >= {thresh}')
    #En este algoritmo, thresh se establece en 0.5, lo que significa que solo se considerarán las coincidencias que tengan una similitud coseno de 0.5 o superior. Es una forma de filtrar coincidencias débiles o irrelevantes. El valor de 0.5 es una elección común en muchos contextos, pero podría ajustarse según las necesidades específicas de la aplicación.
    if len(data_filter) > 0:
        # paso-5: obtener el nombre de la persona
        data_filter.reset_index(drop=True,inplace=True)
        argmax = data_filter['cosine'].argmax()
        #Después de filtrar los datos para mantener solo aquellos con una similitud superior al umbral, argmax identifica cuál de esos vectores filtrados es el más similar al vector de prueba.
        person_name, person_role = data_filter.loc[argmax][name_role]
        
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
        
    return person_name, person_role


### Predicción en Tiempo Real
# Guardamos registros cada 1 minuto
class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[],role=[],current_time=[])
        
    def reset_dict(self):
        self.logs = dict(name=[],role=[],current_time=[])
        
    def saveLogs_redis(self):
        # paso-1: crear un dataframe de registros
        dataframe = pd.DataFrame(self.logs)        
        # paso-2: eliminar la información duplicada
        dataframe.drop_duplicates('name',inplace=True) 
        # paso-3: hacer push a la base de datos redis
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []
        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)
                
        if len(encoded_data) >0:
            r.lpush('attendance:logs',*encoded_data)
        
                    
        self.reset_dict()     
        
        
    def face_prediction(self,image, dataframe,feature_column,
                            name_role=['Name','Role'],thresh=0.5):
        # paso-1: hallar la fecha y hora actual
        current_time = str(datetime.now())
        
        # paso-2: tomar la imagen y aplicarla a insightface
        results = faceapp.get(image)
        test_copy = image.copy()
        # paso-3: usar un bucle for para extraer los datos del embedding (caracteristicas) y pasar al algoritmo de 
        #comparación
        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = comparison_algorithm(dataframe,
                                                        feature_column,
                                                        test_vector=embeddings,
                                                        name_role=name_role,
                                                        thresh=thresh)
            if person_name == 'Unknown':
                color =(0,0,255) # bgr
            else:
                color = (0,255,0)

            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)

            text_gen = person_name
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            # save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)
            

        return test_copy


#### Formulario de Registro
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0
        
    def get_embedding(self,frame):
        # obtener resultados del modelo insightface
        results = faceapp.get(frame,max_num=1)
        #Al establecer max_num=1, se está indicando que solo queremos detectar una única cara en la imagen frame. 
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
            # muestras
            text = f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            
            # caracteristicas faciales
            embeddings = res['embedding']
            
        return frame, embeddings
    
    def save_data_in_redis_db(self,name,role):
        # validacion del nombre
        if name is not None:
            if name.strip() != '':
                key = f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'name_false'
        
        # si face_embedding.txt existe
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'
        
        
        # paso-1: cargar "face_embedding.txt"
        x_array = np.loadtxt('face_embedding.txt',dtype=np.float32)         
        
        # paso-2: convertir en array
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples,512)
        x_array = np.asarray(x_array)       
        
        # paso-3: calcular la media de las caracteristicas
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()
        
        # paso-4: guardar esto en la base de datos redis
        r.hset(name='academy:register',key=key,value=x_mean_bytes)
        
        # 
        os.remove('face_embedding.txt')
        self.reset()
        
        return True
