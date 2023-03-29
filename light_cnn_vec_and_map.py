import os
import cv2
import pickle
from keras.models import load_model 
from skimage import img_as_float
import numpy as np
from keras import backend as K
import tensorflow as tf
from tpgan import LightCNN

from keras.layers import Lambda

# In[]: load icnn model

def build_lcnn_model(model_path):
    
    #lcnn = load_model(model_path)
    lcnn = LightCNN(extractor_type='29v2', extractor_weights=model_path)
    
    return lcnn

# In[]: reshape for input predict model
    
def reshape_4d(img):
    size = img.shape
    img_4d = np.reshape(img, (1,size[0] ,size[1] ,1))
    #    img_4d = np.reshape(img, (1,size[0] ,size[1] ,size[2]))
    
    return img_4d

# In[]: img2tensor
    
def img2float(img):
    
    out_img = img_as_float(img)
    
    return out_img

# In[]: build map&vec dict
    
def build_lcnn_dict(IS_FRONT,data_path):
    
    map_dict = {}
    vec_dict = {}
    
    os.chdir(data_path)
    
    subjects = os.listdir('.')
    
    for subject in subjects:
        
        map_dict[subject] ={}
        vec_dict[subject] ={}
        
        images = os.listdir(subject)
        
        for image in images:
            if IS_FRONT == True:
                if image[-6:-4] == '11':
                    map_dict[subject][image] ={}
                    vec_dict[subject][image] ={}
                    
                    image_path = os.path.join(data_path,subject,image)
                    
                    tmp_img_2d = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                    
                    tmp_img_3d = reshape_3d(tmp_img_2d)
                    
                    tmp_tensor = img2float(tmp_img_3d)
                    
                    map_mat ,vec_mat = lcnn.predict(tmp_tensor)
                    
                    map_dict[subject][image] = map_mat
                    vec_dict[subject][image] = vec_mat
                    
            elif IS_FRONT == False:
                if image[-6:-4] != '11' :
                    map_dict[subject][image] ={}
                    vec_dict[subject][image] ={}
                    
                    image_path = os.path.join(data_path,subject,image)
                    
                    tmp_img_2d = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                    
                    tmp_img_3d = reshape_3d(tmp_img_2d)
                    
                    tmp_tensor = img2float(tmp_img_3d)
                    
                    map_mat ,vec_mat = lcnn.predict(tmp_tensor)
                    
                    map_dict[subject][image] = map_mat
                    vec_dict[subject][image] = vec_mat
                    
    return map_dict,vec_dict

# In[]:save as pkl
    
def save_dict(save_dict ,save_path ,save_name):
    
    os.chdir(save_path)
    
    with open(save_name, 'wb') as f:
        pickle.dump(save_dict, f) 
        
# In[]: compute distance for accuracy
        
def smallest_distance(map_pred ,vec_pred ,true_map_dict ,true_vec_dict, subject, image_name):
    
    vec_true = true_vec_dict[subject][image_name]
    map_true = true_map_dict[subject][image_name]
    

    
    loss_val = K.mean(K.abs(vec_pred - vec_true)) + K.mean(K.abs(map_pred - map_true))
    
    show_tensor = tf.Session() 
    min_loss_val = show_tensor.run(loss_val)
    
    for i in true_map_dict.keys():
        for j in true_map_dict[i].keys():
            
            if i != subject and j !=image_name:
                vec_true = true_vec_dict[i][j]
                map_true = true_map_dict[i][j]
                loss_val = K.mean(K.abs(vec_pred - vec_true)) + K.mean(K.abs(map_pred - map_true))
                show_tensor = tf.Session() 
                cmp_loss_val = show_tensor.run(loss_val)
                
                if cmp_loss_val <= min_loss_val:
                    return 0
    return 1

# In[]: side2front
    
def FEI_front_img(img_path):
    front_path = img_path[:-6]+'11.jpg'
    
    return front_path  
  
# In[]: main function
        
if __name__ == "__main__":
    
    count=0
    total=0
    save_path = 'D:/latest_dltpgan/lcnn_id'
    os.makedirs(save_path,exist_ok=True)
    #model_path = 'E:/tpgan_keras(for FEI)/lcnn/epoch1210_lr0.00100_loss0.295_valacc1.000.hdf5'
    #model_path = 'E:/identity/light_cnn_model/epoch1300_lr0.00100_loss0.378_valacc1.000.hdf5'
    model_path = 'D:/TPGAN MultiPIE/latest_tpgan supp/lcnn_weights/extract29v2_lr0.00010_loss0.997_valacc1.000_epoch1110.hdf5'
    
    data_path = 'D:/latest_dltpgan/gen_img3/'    
    #data_path = 'E:/latest_dltpgan/front_img/'  
    lcnn = build_lcnn_model(model_path)
    
    out_dict = {} #landmark in one    
    os.chdir(data_path)  #change working pathway        
    subjects = os.listdir('.')
    
    for subject in subjects:
        os.chdir(data_path)
        #print(subject) 
        
        out_dict[subject] = {}
        images = os.listdir(subject)
        src_subject_dir = os.path.join(data_path,subject)
        
        for image in images: 
            print(subject+" "+image)
            #change pathway
            os.chdir(src_subject_dir)
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = reshape_4d(img)
            tf_img = img2float(img)
            
            t = tf.convert_to_tensor(tf_img, tf.float32, name='t')
            #img_gray = Lambda(lambda x: tf.image.rgb_to_grayscale(x))(t)
            map_val,vec_val = lcnn.extractor()(t)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            

            #show_tensor = tf.Session() 
            c1=map_val
            c2=vec_val
            show_map_val = sess.run(c1)
            show_vec_val = sess.run(c2)
            
            out_dict[subject][image] = {}
            out_dict[subject][image]['map_val'] = show_map_val
            out_dict[subject][image]['vec_val'] = show_vec_val
            
    
    os.chdir(save_path)
    
    with open('light_cnn_vec_map(gen_img2).pkl', 'wb') as f:
        pickle.dump(out_dict, f)  #save pkl
    #lcnn.summary()
    #IS_FRONT = True
    #true_map_dict,true_vec_dict = build_lcnn_dict(IS_FRONT ,data_path)
    
    #save_dict(true_map_dict ,save_path ,'front_map_dict.pkl')
    #save_dict(true_vec_dict ,save_path ,'front_vec_dict.pkl')
    
    #print("front face is OK.")
    '''
     add model predict
    '''
    #IS_FRONT = False
    #pred_map_dict,pred_vec_dict = build_lcnn_dict(IS_FRONT ,data_path)
    
    #save_dict(pred_map_dict ,save_path ,'side_map_dict.pkl')
    #save_dict(pred_vec_dict ,save_path ,'side_vec_dict.pkl')
    
    #print("side face is OK.")
    
    #datalist_path = 'E:/identity/datalist/datalist_fei.pkl'
    #datalist_path = 'E:/identity/datalist/datalist_fei(03_08).pkl'
    #datalist_path = 'E:/identity/datalist/datalist_fei(04_07).pkl'
    #datalist_path = 'E:/identity/datalist/datalist_fei(05_06).pkl'
    #datalist_path = 'E:/identity/datalist/datalist_fei(front).pkl'
    
    '''
    with open(datalist_path, 'rb') as f:
        datalist = pickle.load(f)
        
    for i, pred_data_path in enumerate(datalist):
        total=total+1
        subject,image_name = pred_data_path.split(os.sep)
        image_name = image_name + '.jpg'
        
        pred_map = pred_map_dict[subject][image_name]
        pred_vec = pred_vec_dict[subject][image_name]
        
        print(subject," ",image_name)
        
        image_name = FEI_front_img(image_name)
        
        count = count + smallest_distance(pred_map ,pred_vec ,true_map_dict ,true_vec_dict, subject, image_name)

    acc = count/total
    
    print(" acc : ",acc)
    '''