
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Offline data generation for the KITTI dataset."""

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


def get_line(file, start):
    file = open(file, 'r')
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    ret = None
    for line in lines:
        nline = line.split(': ')
        if nline[0]==start:
            ret = nline[1].split(' ')
            ret = np.array([float(r) for r in ret], dtype=float)
            ret = ret.reshape((3,4))[0:3, 0:3]
            break
    file.close()
    return ret


def run_all():
    SEQ_LENGTH = 3
    WIDTH = 416
    HEIGHT = 128
    INPUT_DIR = '/home/FisicaroF/3/DNet/splits/eigen_zhou/'
    OUTPUT_DIR = '/home/FisicaroF/5/struct2depth/KITTI_Processed/'
    SPLIT ="train"
    old_seqname = ""
    ct = 1

    if not OUTPUT_DIR.endswith('/'):
        OUTPUT_DIR = OUTPUT_DIR + '/'

    processed_files_list = open(OUTPUT_DIR + '/' + SPLIT + '.txt', 'w')

    f = open(INPUT_DIR + SPLIT + "_files.txt", "r")
    lines = sorted(f.read().splitlines())
    f.close()

    print("Start separation")
    files_right = [line for line in lines if line.split()[2] == "r"]
    files_left = [line for line in lines if line.split()[2] == "l"]

    files_lists = [files_left, files_right]

    print("End separation")

    for lines in files_lists : 
        for line in lines:
            print(line)
            side_folder = "/image_02" if line.split()[2] == "l" else "/image_03"
            frame_path = "/databases/kittiMonodepth/" + line.split()[0] + side_folder + "/data/" + str(line.split()[1]).zfill(10) + ".jpg"


            date_folder = "/databases/kittiMonodepth/" + line.split()[0].split('/')[0]
            file_calibration = date_folder + '/calib_cam_to_cam.txt'

            calib_camera = get_line(file_calibration, 'P_rect_02') if line.split()[2] == "l" else get_line(file_calibration, 'P_rect_03')

            seqname = line.split()[0].split('/')[1]
            
            
            seqname = seqname + "_02" if line.split()[2] == "l" else seqname + "_03"

            if seqname != old_seqname:
                print('Processing sequence', seqname)
                print('OLD sequence', old_seqname)
                ct = 1
                if not os.path.exists(OUTPUT_DIR + seqname):
                    os.mkdir(OUTPUT_DIR + seqname)

            imgnum = str(ct).zfill(10)
            if os.path.exists(OUTPUT_DIR + seqname + '/' + imgnum + '.png'):
                ct+=1
                continue
            
            big_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))  
            
            frame_past_path = "/databases/kittiMonodepth/" + line.split()[0] + side_folder + "/data/" + str(int(line.split()[1]) - 1 ).zfill(10) + ".jpg"
            frame_future_path = "/databases/kittiMonodepth/" + line.split()[0] + side_folder + "/data/" + str(int(line.split()[1]) + 1 ).zfill(10) + ".jpg"

            frames_seq_path = [frame_past_path,frame_path,frame_future_path]
                        
            for j in range(SEQ_LENGTH): 
            
                img = cv2.imread(frames_seq_path[j])
                ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img.shape
                
                zoom_x = WIDTH/ORIGINAL_WIDTH
                zoom_y = HEIGHT/ORIGINAL_HEIGHT

                calib_current = calib_camera.copy()
                calib_current[0, 0] *= zoom_x
                calib_current[0, 2] *= zoom_x
                calib_current[1, 1] *= zoom_y
                calib_current[1, 2] *= zoom_y            
                calib_representation = ','.join([str(c) for c in calib_current.flatten()])

                img = cv2.resize(img, (WIDTH, HEIGHT))            
                big_img[:,j*WIDTH:(j+1)*WIDTH] = img
                            
            cv2.imwrite(OUTPUT_DIR + seqname + '/' + imgnum + '.png', big_img)            

            f = open(OUTPUT_DIR + seqname + '/' + imgnum + '_cam.txt', 'w')
            f.write(calib_representation)
            f.close()
            ct+=1

            processed_files_list.write(seqname + " " + imgnum + "\n")       

            old_seqname = seqname 
        

    processed_files_list.close()                    
                    
          
                

def main(_):
  run_all()


if __name__ == '__main__':
  app.run(main)
