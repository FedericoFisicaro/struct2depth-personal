""" Offline data generation for the UMONS dataset."""

import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import cv2
import os, glob

import alignment
from alignment import compute_overlap
from alignment import align

def run_all():
    WIDTH = 416
    HEIGHT = 128
    DATASET_DIR = "/home/FisicaroF/2/dataset/nyu_depth_v2/sync/"
    INPUT_FILE = '/home/FisicaroF/2/bts/train_test_inputs/nyudepthv2_train_files_with_gt.txt'
    OUTPUT_DIR = '/home/FisicaroF/5/struct2depth/NYU_Processed/'
    old_seqname = ""
    ct = 1


    f = open(INPUT_FILE)
    lines = sorted(f.read().splitlines())
    f.close()
    

    if not OUTPUT_DIR.endswith('/'):
        OUTPUT_DIR = OUTPUT_DIR + '/'
    
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    processed_files_list = open(OUTPUT_DIR + "splits2/" + data_split + "_" + SPLIT + '.txt', 'w')


    for sequence_path in sorted(glob.glob(DATASET_DIR + "*/")):

        seqname = sequence_path.split('/')[-2]

        print('Processing sequence', seqname)
        ct = 1
        if not os.path.exists(OUTPUT_DIR + seqname):
            os.mkdir(OUTPUT_DIR + seqname)

        fx = float(518.8579)
        fy = float(518.8579)
        cx = float(320)
        cy = float(240)

        for frame_path in sorted(glob.glob(sequence_path + "rgb_*.jpg")):
            
            # print(frame_path)
            frame_nbr = int(frame_path[-9:-4])

            imgnum = str(ct).zfill(10)
            # print(imgnum)

            # if os.path.exists(OUTPUT_DIR + seqname + '/' + imgnum + '.png'):
            #     ct+=1
            #     continue
        
            # big_img = np.zeros(shape=(HEIGHT, WIDTH*3, 3))  
        
            frame_past_path = os.path.dirname(frame_path) + "/rgb_" + str(frame_nbr -1).zfill(5) + ".jpg"
            frame_future_path = os.path.dirname(frame_path) + "/rgb_" + str(frame_nbr +1).zfill(5) + ".jpg"

            if os.path.isfile(frame_past_path) and os.path.isfile(frame_future_path):

                frames_seq_path = [frame_past_path,frame_path,frame_future_path]
                            
                for j in range(len(frames_seq_path)): 
                
                    img = cv2.imread(frames_seq_path[j])
                    ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img.shape
                    
                    zoom_x = WIDTH/ORIGINAL_WIDTH
                    zoom_y = HEIGHT/ORIGINAL_HEIGHT

                    fx_current = fx
                    cx_current = cx
                    fy_current = fy
                    cy_current = cy

                    fx_current *= zoom_x
                    cx_current *= zoom_x
                    fy_current *= zoom_y
                    cy_current *= zoom_y            
                    calib_representation = str(fx_current) + ',0.0,' + str(cx_current) + ',0.0,' + str(fy_current) + ',' + str(cy_current) + ',0.0,0.0,1.0'

                    img = cv2.resize(img, (WIDTH, HEIGHT))            
                    big_img[:,j*WIDTH:(j+1)*WIDTH] = img
                                
                cv2.imwrite(OUTPUT_DIR + seqname + '/' + imgnum + '.png', big_img)            

                f = open(OUTPUT_DIR + seqname + '/' + imgnum + '_cam.txt', 'w')
                f.write(calib_representation)
                f.close()
                ct+=1


                if lines[0].split()[0] in frame_path:
                    processed_files_list.write(seqname + " " + imgnum + "\n")
                    del lines[0]

    
    processed_files_list.close()
                    
          
                

def main(_):
  run_all()


if __name__ == '__main__':
  app.run(main)
