import glob
import logging
import os
import datetime
import pickle
import sys

from os import walk
from os.path import splitext
from pathlib import Path
from tkinter import ttk, filedialog, Tk
import tkinter.messagebox as tkMessageBox
from tkinter.constants import HORIZONTAL

import numpy as np

if __name__ == "__main__":

    now = datetime.datetime.now().isoformat(timespec='minutes')
    now = now.replace(":", "_")
    logging.basicConfig(filename="dubra_dewarp_" + now + ".log", level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Make a python set for checking our filenames against.
    modalities = {
        'confocal', 'avg', 'visible', 'PMT1CF', 'PMT2NW', 'PMT3NE', 'PMT4SE', 'PMT5SW', 'PMT1C', 'PMT2N',
        'PMT3E', 'PMT4S',
        'PMT5W'}  # added WAIVS modes - removed 'split_det', because it can't be quickly checked... bad fname- SIT.

    root = Tk()
    root.lift()
    w = 1
    h = 1
    x = root.winfo_screenwidth() / 4
    y = root.winfo_screenheight() / 4
    root.geometry(
        '%dx%d+%d+%d' % (
            w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.

    pName = filedialog.askdirectory(title="Select the folder containing the DMP files of interest.", parent=root)

    if not pName:
        quit()

    sr_avi_path = filedialog.askdirectory(title="Select the folder containing the SR_AVI files.", parent=root)

    if not sr_avi_path:
        quit()

    x = root.winfo_screenwidth() / 2 - 128
    y = root.winfo_screenheight() / 2 - 128
    root.geometry(
        '%dx%d+%d+%d' % (
            w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.
    root.update()

    allFiles = dict()
    totFiles = 0
    # Parse out the dmp and their associated filenames, store them in a hash table.
    dmp_search_path = Path(pName)
    avi_search_path = Path(sr_avi_path)
    for path in dmp_search_path.glob("*.dmp"):
        dmp_fName = path.name

        dmpmode = modalities.intersection(dmp_fName.split("_"))  # Determines the modality, and thus the reference,
        # of the dmp

        if len(dmpmode) == 1:
            dmpmode = dmpmode.pop()
        elif len(
                dmpmode) == 0 and "split_det" in dmp_fName:  # If we find no elements, its either finding split_det, or nothing.
            dmpmode = "split_det"
        else:
            logging.warning("No matching modality found for " + dmp_fName + ".")
            continue

        allFiles[path.as_posix()] = []

        # For this dmp, find all of the associated AVIs in the folder specified.
        globby_fName = dmp_fName.replace(dmpmode, "*")
        globby_fName = globby_fName[:-4] + "*.avi"
        modality_set = set()
        for assoc_fName in avi_search_path.glob(globby_fName):
            logging.info(dmp_fName + " == associated with ==> " + assoc_fName.name)

            alamode = modalities.intersection(assoc_fName.name.split("_"))
            if len(alamode) != 0 and alamode not in modality_set:  # Track which modalities we have for this file.
                modality_set.union(alamode)
                allFiles[path.as_posix()].append(assoc_fName)
            elif "split_det" in assoc_fName.name and "split_det" not in modality_set:
                modality_set.add("split_det")
                allFiles[path.as_posix()].append(assoc_fName)
            else:
                logging.error(
                    dmp_fName + "'s association with " + assoc_fName.name + " causes multiple files to associate with a single dmp!")
                logging.error("Removing " + dmp_fName + " from the processing list!!!")
                allFiles[path.as_posix()] = []
                break

        if len(allFiles[path.as_posix()]) == 0:
            logging.warning("No matching avi files for " + dmp_fName + ".")
            del allFiles[path.as_posix()]

        totFiles += 1

    pb = ttk.Progressbar(root, orient=HORIZONTAL, length=512)
    pb.grid(column=0, row=0, columnspan=2, padx=3, pady=5)
    pb_label = ttk.Label(root, text="Initializing setup...")
    pb_label.grid(column=0, row=1, columnspan=2)
    pb.start()

    # Resize our root to show our progress bar.
    w = 512
    h = 64
    x = root.winfo_screenwidth() / 2 - 256
    y = root.winfo_screenheight() / 2 - 64
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    root.update()

    r = 0
    pb["maximum"] = totFiles
    for associations in allFiles.items():
        try:
            dmpPath = associations[0]
            aviPaths = associations[1]

            # changed to python3 compat -- JDR
            # Fix the fact that it was done originally in Windows...
            with open(dmpPath, 'rb') as pickle_file:
                text = pickle_file.read().replace(b'\r\n', b'\n')  # the b before '\r\n' and '\n' is key -- JDR

            with open(dmpPath, 'wb') as pickle_file:
                pickle_file.write(text)

            with open(dmpPath, 'rb') as pickle_file:
                pick = pickle.load(pickle_file, encoding="latin1")  # for python 3 --  jDR

            ff_translation_info_rowshift = pick['full_frame_ncc']['row_shifts']
            ff_translation_info_colshift = pick['full_frame_ncc']['column_shifts']
            strip_translation_info = pick['sequence_interval_data_list']

            minmaxpix = np.empty([1, 2])

            for frame in strip_translation_info:
                for frame_contents in frame:
                    ref_pixels = frame_contents['slow_axis_pixels_in_current_frame_interpolated']
                    minmaxpix = np.append(minmaxpix, [[ref_pixels[0], ref_pixels[-1]]], axis=0)

            minmaxpix = minmaxpix[1:, :]
            topmostrow = minmaxpix[:, 0].max()
            bottommostrow = minmaxpix[:, 1].min()

            shift_array = np.zeros([len(strip_translation_info) * 3, 1000])
            shift_ind = 0
            for frame in strip_translation_info:
                if len(frame) > 0:
                    frame_ind = frame[0]['frame_index']
                    slow_axis_pixels = np.zeros([1])
                    all_col_shifts = np.zeros([1])
                    all_row_shifts = np.zeros([1])

                    if len(frame) > 1:
                        print("wtfbbq")

                    for frame_contents in frame:
                        slow_axis_pixels = np.append(slow_axis_pixels,
                                                     frame_contents['slow_axis_pixels_in_reference_frame'])

                        ff_row_shift = ff_translation_info_rowshift[frame_ind]
                        ff_col_shift = ff_translation_info_colshift[frame_ind]

                        # First set the relative shifts
                        row_shift = (np.subtract(frame_contents['slow_axis_pixels_in_reference_frame'],
                                                 frame_contents['slow_axis_pixels_in_current_frame_interpolated']))
                        col_shift = (frame_contents['fast_axis_pixels_in_reference_frame_interpolated'])

                        # These will contain all of the motion, not the relative motion between the aligned frames-
                        # So then subtract the full frame row shift
                        row_shift = np.add(row_shift, ff_row_shift)
                        col_shift = np.add(col_shift, ff_col_shift)
                        all_col_shifts = np.append(all_col_shifts, col_shift)
                        all_row_shifts = np.append(all_row_shifts, row_shift)

                    slow_axis_pixels = slow_axis_pixels[1:]
                    all_col_shifts = all_col_shifts[1:]
                    all_row_shifts = all_row_shifts[1:]

                    shift_array[shift_ind * 3, 0:len(slow_axis_pixels)] = slow_axis_pixels
                    shift_array[shift_ind * 3 + 1, 0:len(all_col_shifts)] = all_col_shifts
                    shift_array[shift_ind * 3 + 2, 0:len(all_row_shifts)] = all_row_shifts

                    shift_ind += 1

            # progo.configure("Extracted the eye motion from the dmp file...")

            rois = np.array(pick['strip_cropping_ROI_2'][0])

            for i in range(1, len(pick['strip_cropping_ROI_2'])):
                rois = np.append(rois, pick['strip_cropping_ROI_2'][i], axis=0)

        except Exception as inst:
            logging.warning(inst)
            logging.warning("DMP failed to process. Failed to process DMP (" + dmpPath + ")! " \
                            + "This file may be corrupted. Re-process the DMP, or contact your local RFC.")
