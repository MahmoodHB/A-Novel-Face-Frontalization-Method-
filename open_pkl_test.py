import pickle

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


load_acc_list= 'id_acc(all_angle).pkl'
   
with open(load_acc_list, 'rb') as f:
    acc_datalist = pickle.load(f)

angle = ['15','30','45','60','75','90']

acc_list={}
angle_count={}

for i in angle:
    acc_list[i] = 0
    angle_count[i] = 0

for i, gen_img in enumerate(acc_datalist):
    
    print(i," ",gen_img,": ",acc_datalist[gen_img])
    
    if gen_img[:3] == acc_datalist[gen_img]:
        id_val = gen_img[10:13]
        id_angle = ANGLE_DIR[id_val]
        acc_list[id_angle] = acc_list[id_angle] + 1
    angle_count[id_angle] = angle_count[id_angle] + 1 

total_acc = 0
total_no = 0

for i in angle:
    print('angle ',i,' acc :',acc_list[i]/angle_count[i])
    total_acc = total_acc + acc_list[i]
    total_no = total_no + angle_count[i]
    
print('total acc :',total_acc/total_no)
    