import os
import cv2
from keras.models import load_model 
from keras.models import Model
import numpy as np
import face_alignment
from skimage import img_as_float
import multipie_gen
import tensorflow as tf
import pickle
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

# In[] : define position & size
'''
LEYE_Y, LEYE_X = 40, 42 
REYE_Y, REYE_X = 40, 86
NOSE_Y, NOSE_X = 71, 64
MOUTH_Y, MOUTH_X = 87, 64
'''
EYE_H, EYE_W = 40, 40
NOSE_H, NOSE_W = 32, 40
MOUTH_H, MOUTH_W = 32, 48

# In[] : reshape 4D
 
def reshape_4d(img):
    size = img.shape
    img_4d = np.reshape(img, (1 ,size[0] ,size[1] ,size[2]))
    
    return img_4d

# In[] : reshape 3D 
def reshape_3d(img):
    size = img.shape
    img_3d = np.reshape(img, (size[1] ,size[2] ,size[3]))
    
    return img_3d

# In[]: uint8 to float
    
def img2float(img):
    out_img = img_as_float(img)
    return out_img

# In[] : load model  
    
def build_model(model_path):
    model = load_model(model_path,custom_objects= {'CloudableModel':CloudableModel,'tf':tf,'multipie_gen':multipie_gen})
    
    return model 

# In[] : None

class CloudableModel(Model):
    """
    wrapper of keras model. this class override some functions to be available for Google Cloud Strorage.
    """
    
    def load_weights(self, filepath, by_name=False):
        print('begin loading weights file. target file: {}'.format(filepath))

        super().load_weights(filepath=filepath, by_name=by_name)
        
        print('end loading weights file. target file: {}'.format(filepath))
    
    def save_weights(self, filepath, overwrite=True):
        print('begin saving weights file. target file: {}'.format(filepath))
        

        Model.save(filepath, overwrite=overwrite)
            
        print('end saving weights file. target file: {}'.format(filepath))


# In[] : crop image (need landmarks)
    
def crop(img, landmarks, size): #, angle
        eye_y = 40/size
        mouth_y = 88/size
        
        reye = np.average(np.array((landmarks[37], landmarks[38], landmarks[40], landmarks[41])), axis=0)
        leye = np.average(np.array((landmarks[43], landmarks[44], landmarks[46], landmarks[47])), axis=0)
        mouth = np.average(np.array((landmarks[48], landmarks[54])), axis=0)
        nose_tip = landmarks[30]
        
        vec_mouth2reye = reye - mouth
        vec_mouth2leye = leye - mouth
        phi = np.arccos(vec_mouth2reye.dot(vec_mouth2leye) / (np.linalg.norm(vec_mouth2reye) * np.linalg.norm(vec_mouth2leye)))/np.pi * 180
        
        if phi < 15: # consider the profile image is 90 deg.
    
            # in case of 90 deg. set invisible eye with copy of visible eye.
            eye_center = (reye + leye) / 2
            if nose_tip[0] > eye_center[0]:
                leye = reye
            else:
                reye = leye
        
         # calc angle eyes against horizontal as theta
        if np.array_equal(reye, leye) or phi < 38: # in case of 90 deg. avoid rotation
            theta = 0
        else: 
            vec_leye2reye = reye - leye
            if vec_leye2reye[0] < 0:
                vec_leye2reye = -vec_leye2reye
            theta = np.arctan(vec_leye2reye[1]/vec_leye2reye[0])/np.pi*180
        
        imgcenter = (img.shape[1]/2, img.shape[0]/2)
        rotmat = cv2.getRotationMatrix2D(imgcenter, theta, 1)
        rot_img = cv2.warpAffine(img, rotmat, (img.shape[1], img.shape[0])) 
        rot_landmarks = np.transpose(rotmat[:, :2].dot(np.transpose(landmarks)) + np.repeat(rotmat[:, 2].reshape((2,1)), landmarks.shape[0], axis=1))
        
        rot_reye = np.average(np.array((rot_landmarks[37], rot_landmarks[38], rot_landmarks[40], rot_landmarks[41])), axis=0)
        rot_leye = np.average(np.array((rot_landmarks[43], rot_landmarks[44], rot_landmarks[46], rot_landmarks[47])), axis=0)
        rot_mouth = np.average(np.array((rot_landmarks[48], rot_landmarks[54])), axis=0)
        
        crop_size = int((rot_mouth[1] - rot_reye[1])/(mouth_y - eye_y) + 0.5)
        crop_up = int(rot_reye[1] - crop_size * eye_y + 0.5)
        crop_left = int((rot_reye[0] + rot_leye[0]) / 2 - crop_size / 2 + 0.5)
      
        up_size_fixed=0
        left_size_fixed=0
        
        if crop_up < 0 :
            up_size_fixed=abs(crop_up)
            crop_up = 0
        
        if crop_left < 0 :
            left_size_fixed=abs(crop_left)
            crop_left = 0
                
        crop_img = rot_img[crop_up:crop_up+crop_size+up_size_fixed, crop_left:crop_left+crop_size+left_size_fixed]
        crop_landmarks = rot_landmarks - np.array([crop_left, crop_up])
        
        crop_img = cv2.resize(crop_img, (size, size))
        crop_landmarks *= size / crop_size
    
        leye_points = crop_landmarks[42:48]
        leye_center = (np.max(leye_points, axis=0) + np.min(leye_points, axis=0)) / 2
        leye_left = int(leye_center[0] - EYE_W / 2 + 0.5)
        leye_up = int(leye_center[1] - EYE_H / 2 + 0.5)
        leye_img = crop_img[leye_up:leye_up + EYE_H, leye_left:leye_left + EYE_W]
        
        reye_points = crop_landmarks[36:42]
        reye_center = (np.max(reye_points, axis=0) + np.min(reye_points, axis=0)) / 2
        reye_left = int(reye_center[0] - EYE_W / 2 + 0.5)
        reye_up = int(reye_center[1] - EYE_H / 2 + 0.5)
        reye_img = crop_img[reye_up:reye_up + EYE_H, reye_left:reye_left + EYE_W]
        
        nose_points = crop_landmarks[31:36]
        nose_center = (np.max(nose_points, axis=0) + np.min(nose_points, axis=0)) / 2
        nose_left = int(nose_center[0] - NOSE_W / 2 + 0.5)
        nose_up = int(nose_center[1] - 10 - NOSE_H / 2 + 0.5)
        nose_img = crop_img[nose_up:nose_up + NOSE_H, nose_left:nose_left + NOSE_W]
        
        mouth_points = crop_landmarks[48:60]
        mouth_center = (np.max(mouth_points, axis=0) + np.min(mouth_points, axis=0)) / 2
        mouth_left = int(mouth_center[0] - MOUTH_W / 2 + 0.5)
        mouth_up = int(mouth_center[1] - MOUTH_H / 2 + 0.5)
        mouth_img = crop_img[mouth_up:mouth_up + MOUTH_H, mouth_left:mouth_left + MOUTH_W]
        
        return crop_img, leye_img, reye_img, nose_img, mouth_img

if __name__ == "__main__":


    model_dir ='D:/latest_dltpgan/generator/epoch7000_loss0.356.hdf5' #gen_img3
    src_dataset_dir = 'D:/latest_dltpgan/dataset/side_face/'
    src_datalist_file = 'D:/latest_dltpgan/datalist_side_face.pkl'
    dest_gen_image_dir = 'D:/latest_dltpgan/gen_img3/'
    os.makedirs(dest_gen_image_dir, exist_ok = True)
    
    
    model = build_model(model_dir)
    valid_count=360
  
    with open(src_datalist_file,'rb') as f:
         datalist_temp = pickle.load(f) 
    
    datalist = datalist_temp[:valid_count]
    
    for i, x_data_path in enumerate(datalist):
        img_path = os.path.join(src_dataset_dir, x_data_path+'.jpg')
        print(i," ",img_path)
        
        side_face_image = cv2.imread(img_path)

        landmarks =fa.get_landmarks(cv2.cvtColor(side_face_image,cv2.COLOR_BGR2RGB))
        crop_im, leye_im, reye_im, nose_im, mouth_im =crop(side_face_image, landmarks[0],size=128)
        #img2tensor
        crop_img = img2float(crop_im)
        leye_img = img2float(leye_im)
        reye_img = img2float(reye_im)
        nose_img = img2float(nose_im)
        mouth_img = img2float(mouth_im)
        
        inputs=[reshape_4d(crop_img), reshape_4d(leye_img), reshape_4d(reye_img), reshape_4d(nose_img), reshape_4d(mouth_img), np.random.normal(size=(1, 100))]
        generate_image, _, _, _, _, _, _, _ = model.predict(inputs, batch_size=1)
        generate_image = (generate_image*np.iinfo(np.uint8).max).astype(np.uint8)
        out_image = reshape_3d(generate_image)
        
        out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
        
        gen_img_path = os.path.join(dest_gen_image_dir, x_data_path+'.jpg')
        create_folder_path = os.path.join(dest_gen_image_dir, x_data_path[:3])
        
        os.makedirs(create_folder_path,exist_ok=True)
        cv2.imwrite(gen_img_path,out_image)

            