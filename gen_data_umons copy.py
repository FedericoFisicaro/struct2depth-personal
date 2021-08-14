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
    SEQ_LENGTH = 3
    WIDTH = 416
    HEIGHT = 128
    DATASET_DIR = "/home/FisicaroF/UmonsIndoorDataset/dataset/"
    INPUT_DIR = '/home/FisicaroF/UmonsIndoorDataset/splits/ALL/'
    OUTPUT_DIR = '/home/FisicaroF/5/struct2depth/UMONS_ALL_Processed/'
    SPLIT ="train"
    old_seqname = ""
    ct = 1

    UH = 0

    if not OUTPUT_DIR.endswith('/'):
        OUTPUT_DIR = OUTPUT_DIR + '/'
    
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    processed_files_list = open(OUTPUT_DIR + '/' + SPLIT + '.txt', 'w')

    f = open(INPUT_DIR + SPLIT + "_files.txt", "r")
    lines = sorted(f.read().splitlines())
    f.close()

     
    for line in lines:
        print(line)
        frame_path = DATASET_DIR + line.split()[0]

        building = frame_path.split('/')[-5]
        height = frame_path.split('/')[-4]
        room = frame_path.split('/')[-3]
        data_split = frame_path.split('/')[-2][4:]
        frame_nbr = int(frame_path[-10:-4])

        seqname = building + "_" + height + "_" + room
        if len(data_split) > 0 : 
            seqname = seqname + "_" + data_split

        if seqname != old_seqname:
            print('Processing sequence', seqname)
            # print('OLD sequence', old_seqname)
            ct = 1
            if not os.path.exists(OUTPUT_DIR + seqname):
                os.mkdir(OUTPUT_DIR + seqname)

        cameraParamsFile = open(os.path.join(DATASET_DIR, building, height, room, "camera_params" + data_split + '.txt'),"r")
        cameraParamsLines = cameraParamsFile.readlines()
        cameraParamsFile.close()
        cameraParamsLines = [cameraParamsLines.rstrip() for cameraParamsLines in cameraParamsLines]
        fx = float(cameraParamsLines[1].split(":")[-1])
        fy = float(cameraParamsLines[2].split(":")[-1])
        cx = float(cameraParamsLines[3].split(":")[-1])
        cy = float(cameraParamsLines[4].split(":")[-1])

        
        imgnum = str(ct).zfill(10)
        print(imgnum)
        if os.path.exists(OUTPUT_DIR + seqname + '/' + imgnum + '.png'):
            ct+=1
            continue
        
        big_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))  
        
        frame_past_path = os.path.dirname(frame_path) + "/left" + str(frame_nbr -1).zfill(6) + ".png"
        frame_future_path = os.path.dirname(frame_path) + "/left" + str(frame_nbr +1).zfill(6) + ".png"

        frames_seq_path = [frame_past_path,frame_path,frame_future_path]
                    
        for j in range(SEQ_LENGTH): 
        
            img = cv2.imread(frames_seq_path[j])
            ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img.shape
            
            zoom_x = WIDTH/ORIGINAL_WIDTH
            zoom_y = HEIGHT/ORIGINAL_HEIGHT

            fx *= zoom_x
            cx *= zoom_x
            fy *= zoom_y
            cy *= zoom_y            
            calib_representation = str(fx) + ',0.0,' + str(cx) + ',0.0,' + str(fy) + ',' + str(cy) + ',0.0,0.0,1.0'

            img = cv2.resize(img, (WIDTH, HEIGHT))            
            big_img[:,j*WIDTH:(j+1)*WIDTH] = img
                        
        cv2.imwrite(OUTPUT_DIR + seqname + '/' + imgnum + '.png', big_img)            

        f = open(OUTPUT_DIR + seqname + '/' + imgnum + '_cam.txt', 'w')
        f.write(calib_representation)
        f.close()
        ct+=1

        processed_files_list.write(seqname + " " + imgnum + "\n")       

        old_seqname = seqname 

        
        if seqname != "CLICK_H1_BigRoom":
            break

    processed_files_list.close()                    
                    
          
                

def main(_):
  run_all()


if __name__ == '__main__':
  app.run(main)
