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
    DATASET_DIR = "/home/FisicaroF/UmonsIndoorDataset/dataset/"
    INPUT_DIR = '/home/FisicaroF/UmonsIndoorDataset/splits/'
    OUTPUT_DIR = '/home/FisicaroF/5/struct2depth/UMONS_Processed/'
    SPLIT ="train"
    DATA_SPLITS = ["ALL","H1","H2","H3","H1-H2","H1-H3","H2-H3"]
    old_seqname = ""
    ct = 1
    data_lists = []
    processed_files_lists = []

    for data_split in DATA_SPLITS:
        f = open(INPUT_DIR + data_split + "/" + SPLIT + "_files.txt", "r")
        lines = sorted(f.read().splitlines())
        f.close()
        data_lists.append(lines)

    if not OUTPUT_DIR.endswith('/'):
        OUTPUT_DIR = OUTPUT_DIR + '/'
    
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if not os.path.exists(OUTPUT_DIR + "splits2"):
        os.mkdir(OUTPUT_DIR + "splits2")

    for data_split in DATA_SPLITS:

        processed_files_lists.append(open(OUTPUT_DIR + "splits2/" + data_split + "_" + SPLIT + '.txt', 'w'))


    for sequence_path in sorted(glob.glob(DATASET_DIR + "*/*/*/*/")):

        building = sequence_path.split('/')[-5]
        height = sequence_path.split('/')[-4]
        room = sequence_path.split('/')[-3]
        data_split = sequence_path.split('/')[-2][4:]

        seqname = building + "_" + height + "_" + room
        if len(data_split) > 0 : 
            seqname = seqname + "_" + data_split

        print('Processing sequence', seqname)
        ct = 1
        if not os.path.exists(OUTPUT_DIR + seqname):
            os.mkdir(OUTPUT_DIR + seqname)

        # cameraParamsFile = open(os.path.join(os.path.dirname(sequence_path[0:-2]), "camera_params" + data_split + '.txt'),"r")
        # cameraParamsLines = cameraParamsFile.readlines()
        # cameraParamsFile.close()
        # cameraParamsLines = [cameraParamsLines.rstrip() for cameraParamsLines in cameraParamsLines]
        # fx = float(cameraParamsLines[1].split(":")[-1])
        # fy = float(cameraParamsLines[2].split(":")[-1])
        # cx = float(cameraParamsLines[3].split(":")[-1])
        # cy = float(cameraParamsLines[4].split(":")[-1])

        for frame_path in sorted(glob.glob(sequence_path + "left*.png")):
            if "depth" in frame_path:
                continue
            
            # print(frame_path)
            frame_nbr = int(frame_path[-10:-4])

            imgnum = str(ct).zfill(10)
            # print(imgnum)

            # if os.path.exists(OUTPUT_DIR + seqname + '/' + imgnum + '.png'):
            #     ct+=1
            #     continue
        
            # big_img = np.zeros(shape=(HEIGHT, WIDTH*3, 3))  
        
            frame_past_path = os.path.dirname(frame_path) + "/left" + str(frame_nbr -1).zfill(6) + ".png"
            frame_future_path = os.path.dirname(frame_path) + "/left" + str(frame_nbr +1).zfill(6) + ".png"

            if os.path.isfile(frame_past_path) and os.path.isfile(frame_future_path):

                # frames_seq_path = [frame_past_path,frame_path,frame_future_path]
                            
                # for j in range(len(frames_seq_path)): 
                
                #     img = cv2.imread(frames_seq_path[j])
                #     ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img.shape
                    
                #     zoom_x = WIDTH/ORIGINAL_WIDTH
                #     zoom_y = HEIGHT/ORIGINAL_HEIGHT

                #     fx_current = fx
                #     cx_current = cx
                #     fy_current = fy
                #     cy_current = cy

                #     fx_current *= zoom_x
                #     cx_current *= zoom_x
                #     fy_current *= zoom_y
                #     cy_current *= zoom_y            
                #     calib_representation = str(fx_current) + ',0.0,' + str(cx_current) + ',0.0,' + str(fy_current) + ',' + str(cy_current) + ',0.0,0.0,1.0'

                #     img = cv2.resize(img, (WIDTH, HEIGHT))            
                #     big_img[:,j*WIDTH:(j+1)*WIDTH] = img
                                
                # cv2.imwrite(OUTPUT_DIR + seqname + '/' + imgnum + '.png', big_img)            

                # f = open(OUTPUT_DIR + seqname + '/' + imgnum + '_cam.txt', 'w')
                # f.write(calib_representation)
                # f.close()
                ct+=1


                for i,data_list in enumerate(data_lists):
                    if len(data_list) == 0:
                        continue
                    if data_list[0].split()[0] in frame_path:
                        processed_files_lists[i].write(seqname + " " + imgnum + "\n")
                        del data_list[0]

    
    for file_list in processed_files_lists:
        file_list.close()
                    
          
                

def main(_):
  run_all()


if __name__ == '__main__':
  app.run(main)
