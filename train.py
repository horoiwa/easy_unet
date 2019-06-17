import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import glob
from PIL import Image
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from sample256 import *
from model import *
import argparse





def main(n_sample,steps_per_epoch=2000,epochs=20):
    print("Start processing")
    prep_data(n_sample=n_sample)
    
    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')


    myGene = trainGenerator(batch_size=2,train_path='data',image_folder='image',
                            mask_folder='mask', aug_dict=data_gen_args, save_to_dir = None)

    model = unet()
    
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")
    
    os.makedirs("checkpoints")
    
    
    model_checkpoint = ModelCheckpoint('checkpoints/unet.hdf5', 
                                       monitor='loss',verbose=1, save_best_only=True)

    model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,
                        epochs=epochs,callbacks=[model_checkpoint])
    
    print("Unet finished gracefully")



def commandline_input():
    parser = argparse.ArgumentParser(description='Training Unet for semantic segmentation ')
    parser.add_argument('-n_sample', required=True, type=int,
                        help='sample n images[250*250] per original images')

    parser.add_argument('-steps', type=int, default=500,
                        help='steps per epochs, default 500')

    parser.add_argument('-epochs', type=int, default=1,
                        help='num_epochs, default 1')


    args = parser.parse_args()
    return args




def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256), seed=1):
    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


if __name__ == "__main__":
    args = commandline_input()
    
    n_sample = args.n_sample
    steps_per_epoch = args.steps
    epochs = args.epochs
    
    print("User configuration")
    print("250*250 sample per image:",n_sample)
    print("steps per epoch:",steps_per_epoch)
    print("epochs:",epochs)
    
    main(n_sample=n_sample,steps_per_epoch=steps_per_epoch, epochs=epochs)
    