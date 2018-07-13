# Robert Cooper
# 9-15-2017
# This script exports the distortion applied to each frame of a video
# from a DeMotion strip-registered dataset to a [dmpname]_transforms.csv file.
#
# Copyright (C) 2018 Robert Cooper
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os,pickle,csv
import Tkinter as tk
import Tkconstants, tkFileDialog, tkMessageBox
import numpy as np

root = tk.Tk()

options = {}
options['title'] = 'Select the tif and dmp file folder'
options['parent'] = root

folder_path = tkFileDialog.askdirectory(**options)


for thisfile in os.listdir(folder_path):
     if thisfile.endswith(".dmp"):
          print thisfile
          try:
               pickle_path = os.path.join(folder_path, thisfile)

               #Fix the fact that it was done originally in Windows...
               pickle_file = open(pickle_path, 'rb')
               text = pickle_file.read().replace('\r\n', '\n')
               pickle_file.close()

               pickle_file = open(pickle_path, 'wb')
               pickle_file.write(text)
               pickle_file.close()
	          
               pickle_file = open(pickle_path,'r')
               pick = pickle.load(pickle_file)
               
               ff_translation_info_rowshift = pick['full_frame_ncc']['row_shifts']
               ff_translation_info_colshift = pick['full_frame_ncc']['column_shifts']
               strip_translation_info = pick['sequence_interval_data_list']
               
               firsttime=True
               
               pickle_file.close()

               matches=list()
               #Find all of the tifs that match this dmp file, and record them.
               for thatfile in os.listdir(folder_path):
                    if thatfile.endswith(".tif") and thatfile.startswith(thisfile[0:-4]):
                         matches.append(thatfile)
                         print "Matches transform file: "+thatfile

               minmaxpix = np.empty([1,2])
               		
               for frame in strip_translation_info:
                    for frame_contents in frame:
                         ref_pixels = frame_contents['slow_axis_pixels_in_current_frame_interpolated']
                         minmaxpix = np.append(minmaxpix,[[ref_pixels[0], ref_pixels[-1]]], axis=0)

               minmaxpix=minmaxpix[1:,:]
               #print minmaxpix[:,1].min()
               #print minmaxpix[:,0].max()
               topmostrow = minmaxpix[:,0].max()
               bottommostrow= minmaxpix[:,1].min()

               #print np.array([pick['strip_cropping_ROI_2'][-1]]).shape
               # The first row is the crop ROI.
               for i in range(0,len(matches)):
                    np.savetxt(os.path.join(folder_path, matches[i][0:-4])+ "_transforms.csv" , np.array([pick['strip_cropping_ROI_2'][i]]), delimiter=",", newline="\n", fmt="%f")

               for frame in strip_translation_info:
                    if len(frame) > 0:                    
                         print "************************ Frame " +str(frame[0]['frame_index']+1) + "************************"
                         frame_ind = frame[0]['frame_index']
                         slow_axis_pixels=np.zeros([1])
                         all_col_shifts=np.zeros([1])
                         all_row_shifts=np.zeros([1])

                         for frame_contents in frame:                             
                              slow_axis_pixels = np.append(slow_axis_pixels,frame_contents['slow_axis_pixels_in_reference_frame'])
                             
                              ff_row_shift = ff_translation_info_rowshift[frame_ind]
                              ff_col_shift = ff_translation_info_colshift[frame_ind]
			               
			                  #First set the relative shifts
                              row_shift = (np.subtract(frame_contents['slow_axis_pixels_in_reference_frame'],\
			                              frame_contents['slow_axis_pixels_in_current_frame_interpolated']))
                              col_shift = (frame_contents['fast_axis_pixels_in_reference_frame_interpolated'])

			                  #These will contain all of the motion, not the relative motion between the aligned frames-
			                  #So then subtract the full frame row shift
                              row_shift = np.add(row_shift, ff_row_shift)
                              col_shift = np.add(col_shift, ff_col_shift)
                              all_col_shifts = np.append(all_col_shifts,col_shift)
                              all_row_shifts = np.append(all_row_shifts,row_shift)

                         slow_axis_pixels = slow_axis_pixels[1:]
                         all_col_shifts = all_col_shifts[1:]
                         all_row_shifts = all_row_shifts[1:]

                         for matchedfile in matches:
                              transhandle = open( os.path.join(folder_path, matchedfile[0:-4]) + "_transforms.csv", 'a')
                              np.savetxt(transhandle, np.vstack( (slow_axis_pixels, all_col_shifts, all_row_shifts) ),delimiter=',',fmt='%f')
                              transhandle.close()

          except(ValueError, RuntimeError):
               tkMessageBox.showwarning("DMP failed to process.","Failed to process DMP ("+thisfile+")! This file may be corrupted. Re-process the DMP, or contact your local RFC.")

root.destroy()
