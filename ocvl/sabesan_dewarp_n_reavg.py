import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from PIL.Image import Image

from ocvl.preprocessing.improc import optimizer_stack_align, weighted_z_projection
from ocvl.utility.resources import load_video



if __name__ == "__main__":
    # Load our video data.

    res = load_video("M:\\Dropbox (Personal)\\Research\\Torsion_Distortion_Correction\\COST_rawvideo.avi")
    num_frames = res.data.shape[-1]
    width = res.data.shape[1]
    height = res.data.shape[0]
    video_data = res.data.astype("float32") / 255

    matcontents = scipy.io.loadmat("M:\\Dropbox (Personal)\\Research\\Torsion_Distortion_Correction\\shifts_all.mat")

    all_shifts = matcontents["shifts_all"]

    all_shifts = np.zeros((height, 2, num_frames))
    for f in range(num_frames):
        all_shifts[..., f] = matcontents["shifts_all"][f * height:(f + 1) * height, :]

    median_col_shifts = np.nanmedian(all_shifts[:, 0, :], axis=-1)
    median_row_shifts = np.nanmedian(all_shifts[:, 1, :], axis=-1)

    col_base = np.tile(np.arange(width, dtype=np.float32)[np.newaxis, :], [height, 1])
    row_base = np.tile(np.arange(height, dtype=np.float32)[:, np.newaxis], [1, width])

    shifted = np.zeros(video_data.shape)

    # Best guess for reference frame, since we can't tell from the data.
    reference_frame_idx = np.sum(np.sum(np.abs(all_shifts) < 2, axis=0), axis=0).argmax()

    f=0
    for f in range(num_frames):
        print(f)
        colshifts = all_shifts[:, 0, f] - median_col_shifts
        rowshifts = all_shifts[:, 1, f] - median_row_shifts

        centered_col_shifts = col_base - np.tile(colshifts[:, np.newaxis], [1, width]).astype("float32")
        centered_row_shifts = row_base - np.tile(rowshifts[:, np.newaxis], [1, width]).astype("float32")

        shifted[..., f] = cv2.remap(video_data[..., f], centered_col_shifts, centered_row_shifts,
                                    interpolation=cv2.INTER_CUBIC)

        # plt.figure(0)
        # plt.clf()
        # plt.imshow(shifted[..., f])
        # plt.show(block=False)
        # plt.pause(0.01)

        f += 1

    # Determine and remove residual torsion.
    mask_data = (shifted>0).astype("float32")
    shifted, xforms, inliers = optimizer_stack_align(shifted, mask_data,
                                                        reference_idx=reference_frame_idx)

    # Clamp our data.
    mask_data[mask_data < 0] = 0
    mask_data[mask_data >= 1] = 1


    overlap_map, sum_map = weighted_z_projection(mask_data, mask_data)
    avg_im, sum_map = weighted_z_projection(shifted, mask_data)

    # cv2.imwrite(avg_path, (avg_im*255).astype("uint8"))
    im_conf = Image.fromarray((avg_im * 255).astype("uint8"), "L")
    im_conf.putalpha(Image.fromarray((overlap_map * 255).astype("uint8"), "L"))
    im_conf.save("M:\\Dropbox (Personal)\\Research\\Torsion_Distortion_Correction\\COST_rawvideo_reavg.png")