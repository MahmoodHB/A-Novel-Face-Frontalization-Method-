import os
import pickle
#from skimage import img_as_float
#import numpy as np
#from keras import backend as K
import tensorflow as tf

import time
    
# In[]: compute distance for accuracy
        
def smallest_distance(pred_map ,pred_vec ,id_datalist, gen_img):
    
    start = time.process_time()            
    subject = gen_img[:3]
    temp_id = subject
    image_name = gen_img[:-9]+gen_img[-4:]
    map_true = id_datalist[subject][image_name]['map_val']
    vec_true = id_datalist[subject][image_name]['vec_val']
    

    from keras import backend as K
    loss_val = K.mean(K.abs(pred_vec - vec_true)) + K.mean(K.abs(pred_map - map_true))
    #show_tensor = tf.Session()   #session.close()
    #min_loss_val = show_tensor.run(loss_val)
    
    with tf.Session() as show_tensor:
        min_loss_val = show_tensor.run(loss_val)
    
    start = time.process_time()    
    for i, id_no in enumerate(id_datalist):
        map_true = 0
        vec_true = 0
        loss_val = 0
        cmp_loss_val = 0
        
        for j, id_img in enumerate(id_datalist[id_no]):
            #print(subject,' vs ',id_no ,' ',id_img)
            map_true = id_datalist[id_no][id_img]['map_val']
            vec_true = id_datalist[id_no][id_img]['vec_val']
            
            
            loss_val = K.mean(K.abs(pred_vec - vec_true)) + K.mean(K.abs(pred_map - map_true))
            with tf.Session() as show_tensor:
                cmp_loss_val = show_tensor.run(loss_val)
            
            #show_tensor = tf.Session() 
            #cmp_loss_val = show_tensor.run(loss_val)
            
            if cmp_loss_val <= min_loss_val:
                min_loss_val = cmp_loss_val
                temp_id = id_no
    end = time.process_time()

    print("                        Execution time: %f seconds" % (end - start))
    #del show_tensor    
    #del show_tensor    
    #del show_tensor
    #gc.collect()
    return temp_id
  
# In[]: main function
        
if __name__ == "__main__":
    
    src_path='D:/latest_dltpgan/lcnn_id'
    gen_file = 'landmarks(gen_img).pkl'
    id_file = 'landmarks(front_img).pkl'
    
    gen_path = os.path.join(src_path,gen_file)
    id_path = os.path.join(src_path,id_file)
    
    with open(gen_path, 'rb') as f:
        gen_datalist = pickle.load(f)
    
    with open(id_path, 'rb') as f:
        id_datalist = pickle.load(f)
        
    total=0    
    
    id_img_acc = {}
    
    
    for i, gen_id in enumerate(gen_datalist):
        
        #if gen_id == '201':
        gen_id = '201'   
        for j, gen_img in enumerate(gen_datalist[gen_id]):
            
            total=total+1
            
            print(total,' ',gen_id,' ',gen_img)
            
            id_img_acc[gen_img] = {} 
            
            pred_map = gen_datalist[gen_id][gen_img]['map_val']
            pred_vec = gen_datalist[gen_id][gen_img]['vec_val']
            
            result = smallest_distance(pred_map ,pred_vec ,id_datalist, gen_img)
            id_img_acc[gen_img]= result
            if j == 5:
                break
            

    with open('id201_acc(all_angle).pkl', 'wb') as f:
        pickle.dump(id_img_acc, f)  #save pkl

    
   