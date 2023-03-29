import pickle

datalist_dir='D:/keras_tpgan/landmarks_file.pkl'
#datalist_dir='D:/latest_dltpgan/dataset/side_face_landmarks.pkl'
#datalist_dir='D:/latest_dltpgan/datalist_side_face.pkl'


with open(datalist_dir, 'rb') as f:
     data = pickle.load(f)
