# Robert Cooper
# 9-15-2017
# This script removes residual distortion from both the vertical and horizontal
# (slow and fast scanners in this case, respectively) from a DeMotion strip-registered dataset.
#
# It requires:
#    * A *functioning* MATLAB runtime, that has been set up to link to Python (instructions are on MATLAB's website).
#    * the MATLAB curve fitting toolbox
#    * The .dmp file output from Alfredo Dubra's Demotion software suite. **I realize this makes it VERY specific-
#      I do not promise any amazing things happening as result of using this software!
#    * The 'mat' file corresponding to the grid calibration- also using Alf Dubra's script.
#    * An folder of images that you wish de-distorted.
#
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


# This fork has been updated for use with python3 by J.D. Rogers jdrogers42@github
# Note: update was implemented with the goal to minimize changes from original code
# Summary of changes: fixed tkinter imports, added () to print statements, added modalities to work with WAIVS channel names
#
# python 3 tk renaming, see: https://stackoverflow.com/questions/673174/which-tkinter-modules-were-renamed-in-python-3:
# Tkinter → tkinter
# tkMessageBox → tkinter.messagebox
# tkFileDialog → tkinter.filedialog

try:
    import matlab.engine # This needs to be imported first for some stupid reason.
except:
    import tkinter as tk #import Tkinter as tk
    #import Tkconstants, tkFileDialog, tkMessageBox
    import tkinter.constants as Tkconstants
    import tkinter.filedialog as tkFileDialog
    import tkinter.messagebox as tkMessageBox
    import os, sys, ctypes
    import subprocess
    import socket
    
    options = {}
    options['title'] = 'Please select your [MATLABROOT]\extern\engines\python folder to link to MATLAB.'
    matlab_folder_path = tkFileDialog.askdirectory(**options)

    ctypes.windll.shell32.ShellExecuteW(None, u"runas",  unicode("C:\\Python27\\python.exe"), u"setup.py install", unicode(matlab_folder_path), 1)   

    try:
        import matlab.engine
    except:
        tkMessageBox.showerror("Linking (should be) successful!", "If the console did not display any errors, then linking successful! Please restart this script.")
        sys.exit(0)
        
import os, pickle
import tkinter as tk #import Tkinter as tk
#import Tkconstants, tkFileDialog, tkMessageBox
import tkinter.constants as Tkconstants
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox
import numpy as np

root = tk.Tk()

try:
    mat_engi = matlab.engine.start_matlab()
except:
    tkMessageBox.showerror("Unable to start MATLAB! Ensure you have a valid copy of MATLAB installed AND it has been linked with python.")
    quit(1)


options = {}
options['title'] = 'Select the folder containing the DESINUSOID files, or CANCEL to ignore:'
options['parent'] = root

desinsoid_folder = tkFileDialog.askdirectory(**options)

options = {}
options['title'] = 'Select the folder containing the DMP files:'
options['parent'] = root
options['initialdir'] = desinsoid_folder

dmp_folder_path = tkFileDialog.askdirectory(**options)

options = {}
options['title'] = 'Select the folder containing the IMAGE or MOVIE files:'
options['parent'] = root
options['initialdir'] = dmp_folder_path

image_folder_path = tkFileDialog.askdirectory(**options)

# progo = ttk.Progressbar(root, length=len(os.listdir(dmp_folder_path)))
# progo.pack()


for thisfile in os.listdir(dmp_folder_path):
    if thisfile.endswith(".dmp"):

        try:
            print("Using DMP file: " + thisfile)
            pickle_path = os.path.join(dmp_folder_path, thisfile)

            # Fix the fact that it was done originally in Windows...
            # pickle_file = open(pickle_path, 'rb')
            # text = pickle_file.read().replace('\r\n', '\n')
            # pickle_file.close()
            #
            # pickle_file = open(pickle_path, 'wb')
            # pickle_file.write(text)
            # pickle_file.close()
            #
            # pickle_file = open(pickle_path, 'r')
            # pick = pickle.load(pickle_file)  

            # changed to python3 compat -- JDR
            # Fix the fact that it was done originally in Windows...            
            with open(pickle_path, 'rb') as pickle_file:
                text = pickle_file.read().replace(b'\r\n', b'\n') # the b before '\r\n' and '\n' is key -- JDR
            
            with open(pickle_path, 'wb') as pickle_file:
                pickle_file.write(text)

            with open(pickle_path, 'rb') as pickle_file:
                pick = pickle.load(pickle_file, encoding="latin1") # for python 3 --  jDR

            ff_translation_info_rowshift = pick['full_frame_ncc']['row_shifts']
            ff_translation_info_colshift = pick['full_frame_ncc']['column_shifts']
            strip_translation_info = pick['sequence_interval_data_list']

            # @TODO: Add this repair for Dubra system style data.
            if desinsoid_folder != "":
                static_distortion = mat_engi.Static_Distortion_Repair(os.path.join(desinsoid_folder, pick['desinusoid_data_filename'].split("//")[-1])) # this uses filename instead of absolute path to avoid errors in finding the file
            else:
                static_distortion = []
            firsttime = True

            pickle_file.close()

            # Find the dmp's matching image(s).
            modalities = ('confocal', 'split_det', 'avg', 'visible', 'PMT1CF', 'PMT2NW', 'PMT3NE', 'PMT4SE', 'PMT5SW', 'PMT1C', 'PMT2N', 'PMT3E', 'PMT4S', 'PMT5W') # added WAIVS modes


            images_to_fix =[]
            # Find all images in our folder that this dmp applies to.
            for thismode in modalities:
                if thismode in thisfile:

                    for mode in modalities:
                        checkfile = thisfile[0:-4].replace(thismode, mode)
                        for imagefile in os.listdir(image_folder_path):
                            if (checkfile in imagefile) and (imagefile.endswith(".tif") or imagefile.endswith(".avi")):
                                images_to_fix.append(imagefile)

            if not images_to_fix:# If we don't have any accompanying modality images, just find the image this dmp applies to.
                checkfile = thisfile[0:-4]
                for imagefile in os.listdir(image_folder_path):
                    if (checkfile in imagefile) and (imagefile.endswith(".tif") or imagefile.endswith(".avi")):
                        images_to_fix.append(imagefile)
            
            if images_to_fix:


                minmaxpix = np.empty([1,2])
               		
                for frame in strip_translation_info:
                    for frame_contents in frame:
                        ref_pixels = frame_contents['slow_axis_pixels_in_current_frame_interpolated']
                        minmaxpix = np.append(minmaxpix,[[ref_pixels[0], ref_pixels[-1]]], axis=0)

                minmaxpix=minmaxpix[1:,:]
                topmostrow = minmaxpix[:, 0].max()
                bottommostrow = minmaxpix[:, 1].min()

                # print np.array([pick['strip_cropping_ROI_2'][-1]])
                # The first row is the crop ROI.
                # np.savetxt(pickle_path[0:-4] + "_transforms.csv", np.array([pick['strip_cropping_ROI_2'][-1]]),
                #            delimiter=",", newline="\n", fmt="%f")

                shift_array = np.zeros([len(strip_translation_info)*3, 1000])
                shift_ind = 0
                for frame in strip_translation_info:
                    if len(frame) > 0:
                        # print "************************ Frame " + str(frame[0]['frame_index'] + 1) + "************************"
                        # print "Adjusting the rows...."
                        frame_ind = frame[0]['frame_index']
                        slow_axis_pixels=np.zeros([1])
                        all_col_shifts=np.zeros([1])
                        all_row_shifts=np.zeros([1])

                        for frame_contents in frame:                             
                            slow_axis_pixels = np.append(slow_axis_pixels,frame_contents['slow_axis_pixels_in_reference_frame'])
                             
                            ff_row_shift = ff_translation_info_rowshift[frame_ind]
                            ff_col_shift = ff_translation_info_colshift[frame_ind]

                            #First set the relative shifts
                            row_shift = (np.subtract(frame_contents['slow_axis_pixels_in_reference_frame'],
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

                        shift_array[shift_ind*3,   0:len(slow_axis_pixels)] = slow_axis_pixels
                        shift_array[shift_ind*3+1, 0:len(all_col_shifts)] = all_col_shifts
                        shift_array[shift_ind*3+2, 0:len(all_row_shifts)] = all_row_shifts

                        shift_ind += 1

                # progo.configure("Extracted the eye motion from the dmp file...")

                rois = np.array(pick['strip_cropping_ROI_2'][0])

                for i in range(1, len(pick['strip_cropping_ROI_2'])):
                    rois = np.append(rois, pick['strip_cropping_ROI_2'][i],axis=0)

                for image in images_to_fix:
                    # progo.configure("Removing distortion from :"+image +"...")
                    print("Removing distortion from :"+image +"...")
                    mat_engi.Eye_Motion_Distortion_Repair(image_folder_path, image, rois.tolist(),
                                                          shift_array.tolist(), static_distortion, nargout=0)

                # progo.step()

        except:
            tkMessageBox.showwarning("DMP failed to process.",
                                     "Failed to process DMP (" + thisfile + ")! This file may be corrupted. Re-process the DMP, or contact your local RFC.")

root.destroy()
# shiftT = np.transpose(shift_array)
# transhandle = open(pickle_path[0:-4] + "_transforms.csv", 'w')
# np.savetxt(transhandle, shift_array, delimiter=',', fmt='%f')
# transhandle.close()
