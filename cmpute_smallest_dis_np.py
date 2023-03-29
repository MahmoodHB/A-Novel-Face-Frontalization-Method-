import os
import pickle
import numpy as np

import time

ANGLE_DIR = {
        '110':'90',
        '120':'75',
        '090':'60',
        '080':'45',
        '130':'30',
        '140':'15',
        '050':'15',
        '041':'30',
        '190':'45',
        '200':'60',
        '010':'75',
        '240':'90',
        }
# In[]:compute distance
def cmp_dis(pred_vec ,vec_true ,pred_map ,map_true):
    #dis =  np.sum(pred_vec - vec_true)
    #dis =  np.mean(pred_vec - vec_true)
    #dis =  np.sum(pred_map - map_true)
    #dis =  np.mean(pred_map - map_true)
    dis =  np.mean(np.abs(pred_vec - vec_true)) + np.mean(np.abs(pred_map - map_true))
    #dis =  np.mean((pred_vec - vec_true)*(pred_vec - vec_true)) + np.mean((pred_map - map_true)*(pred_map - map_true))
    
    return dis
    
# In[]: compute distance for accuracy
        
def smallest_distance(pred_map ,pred_vec ,id_datalist, gen_img):
    
    start = time.process_time()            
    subject = gen_img[:3]
    temp_id = subject
    image_name = gen_img[:-9]+gen_img[-4:]
    map_true_ = id_datalist[subject][image_name]['map_val']
    vec_true_ = id_datalist[subject][image_name]['vec_val']
    
    #min_loss_val = np.mean(np.abs(pred_vec - vec_true_)) + np.mean(np.abs(pred_map - map_true_))
    #min_loss_val = np.abs(pred_vec - vec_true_) + np.abs(pred_map - map_true_)
    min_loss_val = cmp_dis(pred_vec,vec_true_,pred_map,map_true_)
    print(min_loss_val)
    
    
    start = time.process_time()    
    for i, id_no in enumerate(id_datalist):
        map_true = 0
        vec_true = 0
        cmp_loss_val = 0
        
        for j, id_img in enumerate(id_datalist[id_no]):
            #print(subject,' vs ',id_no ,' ',id_img)
            map_true = id_datalist[id_no][id_img]['map_val']
            vec_true = id_datalist[id_no][id_img]['vec_val']
            
            cmp_loss_val =cmp_dis(pred_vec,vec_true,pred_map,map_true)
            
            #show_tensor = tf.Session() 
            #cmp_loss_val = show_tensor.run(loss_val)
            
            if cmp_loss_val <= min_loss_val:
                min_loss_val = cmp_loss_val
                temp_id = id_no
    end = time.process_time()
    
    print("                        Execution time: %f seconds" % (end - start))
    
    return temp_id
  
# In[]: main function
        
if __name__ == "__main__":
    
    src_path='D:/latest_dltpgan/lcnn_id/'
    #gen_file = 'landmarks(gen_img).pkl'
    gen_file = 'light_cnn_vec_map(gen_img2).pkl'
    id_file = 'landmarks(front_img).pkl'
    
#    gen_path = os.path.join(src_path,gen_file)
#    id_path = os.path.join(src_path,id_file)
    
    gen_path = src_path+gen_file
    id_path = src_path+id_file
    
    with open(gen_path, 'rb') as f:
        gen_datalist = pickle.load(f)
    
    with open(id_path, 'rb') as f:
        id_datalist = pickle.load(f)
        
    total=0    
    
    id_img_acc = {}
    
    
    for i, gen_id in enumerate(gen_datalist):
        
        #if gen_id == '201':
        #gen_id = '201'   
            for j, gen_img in enumerate(gen_datalist[gen_id]):
                
                total=total+1
                
                print(total,' ',gen_id,' ',gen_img)
                
                id_img_acc[gen_img] = {} 
                
                pred_map = gen_datalist[gen_id][gen_img]['map_val']
                pred_vec = gen_datalist[gen_id][gen_img]['vec_val']
                
                result = smallest_distance(pred_map ,pred_vec ,id_datalist, gen_img)
                id_img_acc[gen_img]= result

    with open('id_acc(all_angle)_L1_distance.pkl', 'wb') as f:
        pickle.dump(id_img_acc, f)  #save pkl
    
    angle = ['15','30','45','60','75','90']

    acc_list={}
    angle_count={}
    
    for i in angle:
        acc_list[i] = 0
        angle_count[i] = 0
    
    for i, gen_img in enumerate(id_img_acc):
        
        print(i," ",gen_img,": ",id_img_acc[gen_img])
        
        id_val = gen_img[10:13]
        id_angle = ANGLE_DIR[id_val]
        
        if gen_img[:3] == id_img_acc[gen_img]:
            
            acc_list[id_angle] = acc_list[id_angle] + 1
            
        angle_count[id_angle] = angle_count[id_angle] + 1 
    
    total_acc = 0
    total_no = 0
    
    for i in angle:
        print('angle ',i,' acc :',acc_list[i]/angle_count[i])
        total_acc = total_acc + acc_list[i]
        total_no = total_no + angle_count[i]
        
    print('total acc :',total_acc/total_no)

    
   