from tpgan import TPGAN, multipie_gen
from keras.optimizers import SGD 
from keras.optimizers import Adam

if __name__ == '__main__':
    
    #K.clear_session()
    op = 'Adam'

    gan = TPGAN(base_filters=64, gpus=1,
                lcnn_extractor_weights='D:/latest_dltpgan/lcnn_weights/extract29v2_lr0.00010_loss0.997_valacc1.000_epoch1110.hdf5',
                generator_weights='',
                classifier_weights='',   
                discriminator_weights='')
    
    datagen = multipie_gen.Datagen(dataset_dir='D:/latest_dltpgan/dataset/', landmarks_dict_file='D:/latest_dltpgan/landmark/landmarks_file.pkl', 
                                   datalist_dir='D:/latest_dltpgan/datalist_side_face.pkl', min_angle=-90, max_angle=90, valid_count=130)

    if op == 'Adam':
        optimizer = Adam(lr=0.0001, beta_1=0.9)#, beta_2=0.999 # n=4
    elif op == 'SGD':
        optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True) # n=2
                  
    #print('gan.discriminator():',gan.discriminator().output)
    gan.train_gan(gen_datagen_creator=datagen.get_generator, 
                  gen_train_batch_size=4, 
                  gen_valid_batch_size=4,  
                  disc_datagen_creator=datagen.get_discriminator_generator, 
                  disc_batch_size=10,  
                  disc_gt_shape=gan.discriminator().output_shape[1:3],
                  optimizer=optimizer,
                  gen_steps_per_epoch=100, disc_steps_per_epoch=100,  
                  epochs=5000, out_dir='D:/latest_dltpgan/out_data/', out_period=1, is_output_img=True,
                  lr=0.0001, decay=0, lambda_128=1, lambda_64=1, lambda_32=1,
                  lambda_sym=1e-1, lambda_ip=1e-3, lambda_adv=5e-3, lambda_tv=1e-5,
                  lambda_class=1, lambda_parts=3)

'''
lr=0.001, decay=0, lambda_128=1, lambda_64=1, lambda_32=1,
                  lambda_sym=3e-1, lambda_ip=3e-3, lambda_adv=1e-3, lambda_tv=1e-4,
                  lambda_class=4e-1, lambda_parts=3)
'''
