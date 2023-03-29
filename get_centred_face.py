import os
#from  generate_image_tf1 import *
import shutil
from matplotlib import pyplot as plt
from keras.models import load_model 
from keras.models import Model
import tensorflow as tf
import cv2
import multipie_gen
import face_alignment
from skimage import img_as_float

import numpy as np
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

EYE_H, EYE_W = 40, 40
NOSE_H, NOSE_W = 32, 40
MOUTH_H, MOUTH_W = 32, 48

def build_model(model_path):
    model = load_model(model_path,custom_objects= {'CloudableModel':CloudableModel,'tf':tf,'multipie_gen':multipie_gen})
    
    return model 

def reshape_4d(img):
    size = img.shape
    img_4d = np.reshape(img, (1 ,size[0] ,size[1] ,size[2]))
    
    return img_4d

def img2float(img):
    out_img = img_as_float(img)
    return out_img

def reshape_3d(img):
    size = img.shape
    img_3d = np.reshape(img, (size[1] ,size[2] ,size[3]))
    
    return img_3d
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



def crop_face(img):
    face_cascade = cv2.CascadeClassifier('D:/demo system/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi_color = img[y:y + h, x:x + w]

        return roi_color
    else:
        return "None"



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

def get_centred_face(side_face_image,dest,model_path):

    print("loading model")
    model = build_model(model_path)
    valid_count=360
    print("model loaded")
    

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
        
    return out_image



if __name__ == "__main__":
    src = "G:/Mahmood/test_images/201/90.jpg"
    flage = False
    live_floder_path = "D:/demo system/live_test/"
    files = os.listdir(live_floder_path)
    if flage and len(files) > 0:
        src = live_floder_path + files[0]
        side_face_image = crop_face(cv2.imread(src))
    else:
        side_face_image = cv2.imread(src)
    dest = "generated_output/"
    #updated_gen_path ='C:/Users/E322/Desktop/updated_model/weights/generator/updated_gen.hdf5'
    updated_gen_path ='G:/Mahmood/models/updated/epoch7000_loss0.356.hdf5'
    
    #old_gen_path ='C:/Users/E322/Desktop/old_model/weights/generator/old_model.hdf5'
    
    old_gen_path ='G:/Mahmood/models/old/epoch1046_loss0.357.hdf5'
    
    #old_gen_path ='G:/Mahmood/models/epoch1570_loss0.874.hdf5'

    
    os.makedirs(dest, exist_ok = True)
    updated_out_image = get_centred_face(side_face_image,dest,updated_gen_path)
    updated_out_image = cv2.cvtColor(updated_out_image, cv2.COLOR_BGR2RGB)
    
    
    img = updated_out_image.copy()
    landmark = fa.get_landmarks(img)[0]
    
    c = 1
    #print(landmark[0][0])
    for i in range(len(landmark)):
        
        img[int(landmark[i][1]), int(landmark[i][0]), c] = 255
        img[int(landmark[i][1]) - 1, int(landmark[i][0]), c] = 255
        img[int(landmark[i][1]), int(landmark[i][0]) - 1, c] = 255
        img[int(landmark[i][1]) - 1, int(landmark[i][0]) - 1, c] = 255
        img[int(landmark[i][1]) + 1, int(landmark[i][0]), c] = 255
        img[int(landmark[i][1]), int(landmark[i][0]) + 1, c] = 255
        img[int(landmark[i][1]) + 1, int(landmark[i][0]) + 1, c] = 255
        img[int(landmark[i][1]) - 1, int(landmark[i][0]) + 1, c] = 255
        img[int(landmark[i][1]) + 1, int(landmark[i][0]) - 1, c] = 255
    
    old_out_image = get_centred_face(side_face_image,dest,old_gen_path)
    
    
    old_out_image = cv2.cvtColor(old_out_image, cv2.COLOR_BGR2RGB)
    
    fig = plt.figure(figsize=(10, 7))
      
    # setting values to rows and column variables
    rows = 2
    columns = 2
    
    fig.add_subplot(rows, columns, 1)

      
    # showing image
    plt.imshow(old_out_image)
    plt.axis('off')
    plt.title("TP-GAN_Gen" )
    
    fig.add_subplot(rows, columns, 2)
      
    # showing image
    plt.imshow(img)
    plt.axis('off')
    plt.title("landmarks" )
    
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 3)
      
    # showing image
    plt.imshow(updated_out_image)
    plt.axis('off')
    plt.title("Our Method_Gen" )
    
    
     # fig.add_subplot(rows, columns, 4)
      
    # showing image
     # plt.imshow(img)
    #  plt.axis('off')
     # plt.title("landmarks" )


