import tensorflow as tf

from tpgan import LightCNN
import pickle

    
if __name__ == "__main__":
    
    lcnn_extractor_weights='D:/latest_dltpgan/lcnn_weights/extract29v2_lr0.00010_loss0.997_valacc1.000_epoch1110.hdf5'
    datalist_file ='D:/latest_dltpgan/datalist_test.pkl'
    
    if tf.gfile.Exists(datalist_file):
        with open(datalist_file, 'rb') as f:
            datalist = pickle.load(f)
    else:
        print("No datalist file")
        
    lcnn = LightCNN(extractor_type='29v2', extractor_weights=lcnn_extractor_weights)
    lcnn.extractor()
    