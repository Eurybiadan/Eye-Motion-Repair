import warnings

import cv2
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot
from scipy.ndimage import binary_erosion
from scipy.signal import fftconvolve
from numpy.polynomial import Polynomial

from ocvl.utility.resources import save_video


def flat_field_frame(dataframe, sigma):
    kernelsize = 3 * sigma
    if (kernelsize % 2) == 0:
        kernelsize += 1

    mask = np.ones(dataframe.shape, dtype=dataframe.dtype)
    mask[dataframe == 0] = 0

    dataframe[dataframe == 0] = 1
    blurred_frame = cv2.GaussianBlur(dataframe.astype("float64"), (kernelsize, kernelsize),
                                     sigmaX=sigma, sigmaY=sigma)
    flat_fielded = (dataframe.astype("float64") / blurred_frame)

    flat_fielded *= mask
    flat_fielded -= np.amin(flat_fielded)
    flat_fielded = np.divide(flat_fielded, np.amax(flat_fielded), where=flat_fielded != 0)
    if dataframe.dtype == "uint8":
        flat_fielded *= 255
    elif dataframe.dtype == "uint16":
        flat_fielded *= 65535

    return flat_fielded.astype(dataframe.dtype)


def flat_field(dataset, sigma=20):

    if len(dataset.shape) > 2:
        flat_fielded_dataset = np.zeros(dataset.shape, dtype=dataset.dtype)
        for i in range(dataset.shape[-1]):
            flat_fielded_dataset[..., i] = flat_field_frame(dataset[..., i], sigma)

        return flat_fielded_dataset.astype(dataset.dtype)
    else:
        return flat_field_frame(dataset, sigma)


def norm_video(video_data, norm_method="mean", rescaled=False):
    """
    This function normalizes a video (a single sample of all cells) using a method supplied by the user.

    :param video_data: A NxMxF numpy matrix with N rows, M columns, and F frames of some video.
    :param norm_method: The normalization method chosen by the user. Default is "mean". Options: "mean", "median"
    :param rescaled: Whether or not to keep the data at the original scale (only modulate the numbers in place). Useful
                     if you want the data to stay in the same units. Default: False. Options: True/False

    :return: a NxMxF numpy matrix of normalized video data.
    """

    if norm_method == "mean":
        # Determine each frame's mean.
        framewise_norm = np.empty([video_data.shape[-1]])
        for f in range(video_data.shape[-1]):
            frm = video_data[:, :, f].flatten().astype("float32")
            frm[frm == 0] = np.nan
            framewise_norm[f] = np.nanmean(frm)

        all_norm = np.nanmean(framewise_norm)
        #plt.plot(framewise_norm/np.amax(framewise_norm))
    # plt.show()
    elif norm_method == "median":
        # Determine each frame's median.
        framewise_norm = np.empty([video_data.shape[-1]])
        for f in range(video_data.shape[-1]):
            frm = video_data[:, :, f].flatten().astype("float32")
            frm[frm == 0] = np.nan
            framewise_norm[f] = np.nanmedian(frm)
        all_norm = np.nanmean(framewise_norm)

    else:
        # Determine each frame's mean.
        framewise_norm = np.empty([video_data.shape[-1]])
        for f in range(video_data.shape[-1]):
            frm = video_data[:, :, f].flatten().astype("float32")
            frm[frm == 0] = np.nan
            framewise_norm[f] = np.nanmean(frm)
        all_norm = np.nanmean(framewise_norm)
        warnings.warn("The \"" + norm_method + "\" normalization type is not recognized. Defaulting to mean.")

    if rescaled: # Provide the option to simply scale the data, instead of keeping it in relative terms
        ratio = framewise_norm / all_norm

        rescaled_vid = np.empty(video_data.shape)
        for f in range(video_data.shape[-1]):
            rescaled_vid[:, :, f] = video_data[:, :, f].astype("float32") / ratio[f]

        return rescaled_vid
    else:
        rescaled_vid = np.zeros(video_data.shape)
        for f in range(video_data.shape[-1]):
            rescaled_vid[:, :, f] = video_data[:, :, f].astype("float32") / framewise_norm[f]

        return rescaled_vid




def dewarp_2D_data(image_data, row_shifts, col_shifts, method="median"):
    '''
    # Where the image data is N rows x M cols and F frames
    # and the row_shifts and col_shifts are F x N.
    # Assumes a row-wise distortion/a row-wise fast scan ("distortionless" along each row)
    # Returns a float image (spans from 0-1).
    :param image_data:
    :param row_shifts:
    :param col_shifts:
    :param method:
    :return:
    '''
    numstrips = row_shifts.shape[1]
    height = image_data.shape[0]
    width = image_data.shape[1]
    num_frames = image_data.shape[-1]

    allrows = np.linspace(0, numstrips - 1, num=height)  # Make a linspace for all of our images' rows.
    substrip = np.linspace(0, numstrips - 1, num=numstrips)

    indiv_colshift = np.zeros([num_frames, height])
    indiv_rowshift = np.zeros([num_frames, height])

    for f in range(num_frames):
        # Fit across rows, in order to capture all strips for a given dataset
        finite = np.isfinite(col_shifts[f, :])
        col_strip_fit = Polynomial.fit(substrip[finite], col_shifts[f, finite], deg=12)
        indiv_colshift[f, :] = col_strip_fit(allrows)
        # Fit across rows, in order to capture all strips for a given dataset
        finite = np.isfinite(row_shifts[f, :])
        row_strip_fit = Polynomial.fit(substrip[finite], row_shifts[f, finite], deg=12)
        indiv_rowshift[f, :] = row_strip_fit(allrows)

    if method == "median":
        centered_col_shifts = -np.nanmedian(indiv_colshift, axis=0)
        centered_row_shifts = -np.nanmedian(indiv_rowshift, axis=0)

    dewarped = np.zeros(image_data.shape)

    col_base = np.tile(np.arange(width, dtype=np.float32)[np.newaxis, :], [height, 1])
    row_base = np.tile(np.arange(height, dtype=np.float32)[:, np.newaxis], [1, width])

    centered_col_shifts = col_base + np.tile(centered_col_shifts[:, np.newaxis], [1, width]).astype("float32")
    centered_row_shifts = row_base + np.tile(centered_row_shifts[:, np.newaxis], [1, width]).astype("float32")

    if image_data.dtype == np.uint8:
        for f in range(num_frames):
            dewarped[..., f] = cv2.remap(image_data[..., f].astype("float32") / 255, centered_col_shifts,
                                         centered_row_shifts,
                                         interpolation=cv2.INTER_CUBIC)

        # Clamp our values.
        dewarped[dewarped < 0] = 0
        dewarped[dewarped > 1] = 1

        return (dewarped * 255).astype("uint8"), centered_col_shifts, centered_row_shifts
    else:
        datmax = np.amax(image_data[:])
        for f in range(num_frames):
            dewarped[..., f] = cv2.remap(image_data[..., f] / datmax,
                                         centered_col_shifts,
                                         centered_row_shifts,
                                         interpolation=cv2.INTER_CUBIC)

        # Clamp our values.
        dewarped[dewarped < 0] = 0
        dewarped[dewarped > 1] = 1

        return dewarped*datmax, centered_col_shifts, centered_row_shifts

    # save_video("C:\\Users\\rober\\Documents\\temp\\test.avi", (dewarped*255).astype("uint8"), 30)


# Calculate a running sum in all four directions - apparently more memory efficient
# Used for determining the total energy at every point that an image contains.
def local_sum(matrix, overlap_shape):

    if matrix.shape[0] == overlap_shape[0] and matrix.shape[1] == overlap_shape[1]:
        energy_mat = np.cumsum(matrix, axis=0)
        energy_mat = np.pad(energy_mat, ((0, energy_mat.shape[0]-1), (0, 0)),  mode="reflect") # This is wrong- needs to be a pad with end value
                # and a subtraction of the above nrg matrix. Not sure why intuitively. Maybe FD wrapping?
        energy_mat = np.cumsum(energy_mat, axis=1)
        energy_mat = np.pad(energy_mat, ((0, 0), (0, energy_mat.shape[1]-1)), mode="reflect")

    else:
        print("This code does not yet support unequal image sizes!")

    return energy_mat


# Using Dirk Padfield's Masked FFT Registration approach (Normalized xcorr for arbitrary masks).
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5540032
def general_normxcorr2(template_im, reference_im, template_mask=None, reference_mask=None, required_overlap=None):
    temp_size = template_im.shape
    ref_size = reference_im.shape

    template = template_im.astype("float32")
    reference = reference_im.astype("float32")

    if template_mask is None:
        template_mask = np.ones(temp_size)
    if reference_mask is None:
        reference_mask = np.ones(ref_size)

    # First, cross correlate our two images (but this isn't normalized, yet!)
    # The templates should be rotated by 90 degrees. So...
    template_mask = np.rot90(template_mask.copy(), k=2)
    template = np.rot90(template, k=2)
    base_xcorr = fftconvolve(template, reference)

    # Fulfill equations 10-12 from the paper.
    # First get the overlapping energy...
    pixelwise_overlap = fftconvolve(template_mask, reference_mask)  # Eq 10
    pixelwise_overlap[pixelwise_overlap <= 0] = 1
    # For the template frame denominator portion.
    ref_corrw_one = fftconvolve(reference, reference_mask)  # Eq 11
    ref_sq_corrw_one = fftconvolve(reference * reference, reference_mask)  # Eq 12

    ref_denom = ref_sq_corrw_one - ((ref_corrw_one * ref_corrw_one) / pixelwise_overlap)
    ref_denom[ref_denom < 0] = 0  # Clamp these values to 0.

    # For the reference frame denominator portion.
    temp_corrw_one = fftconvolve(template, template_mask)  # Eq 11
    temp_sq_corrw_one = fftconvolve(template * template, template_mask)  # Eq 12

    temp_denom = temp_sq_corrw_one - ((temp_corrw_one * temp_corrw_one) / pixelwise_overlap)
    temp_denom[temp_denom < 0] = 0  # Clamp these values to 0.

    # Construct our numerator
    numerator = base_xcorr - ((temp_corrw_one*ref_corrw_one)/pixelwise_overlap)
    denom = np.sqrt(temp_denom*ref_denom)

    # Need this bit to avoid dividing by zero.
    tolerance = 1000*np.finfo(np.amax(denom)).eps

    xcorr_out = np.zeros(numerator.shape, dtype=np.float)
    xcorr_out[denom > tolerance] = numerator[denom > tolerance] / denom[denom > tolerance]

    # By default, the images have to overlap by more than 20% of their maximal overlap.
    if not required_overlap:
        required_overlap = np.amax(pixelwise_overlap)*.5
    xcorr_out[pixelwise_overlap < required_overlap ] = 0

    maxval = np.amax(xcorr_out[:])
    maxloc = np.unravel_index(np.argmax(xcorr_out[:]), xcorr_out.shape)
    maxshift = (-float(maxloc[1]-ref_size[1]), -float(maxloc[0]-ref_size[0])) #Output as X and Y.

    return maxshift, maxval, xcorr_out


def simple_image_stack_align(im_stack, mask_stack, ref_idx):
    num_frames = im_stack.shape[-1]
    shifts = [None] * num_frames
    # flattened = flat_field(im_stack)
    flattened = (im_stack)
    print("Aligning to frame " + str(ref_idx))
    for f2 in range(0, num_frames):
        shift, val, xcorrmap = general_normxcorr2(flattened[..., f2], flattened[..., ref_idx],
                                                  template_mask=mask_stack[..., f2],
                                                  reference_mask=mask_stack[..., ref_idx])
        # print("Found shift of: " + str(shift) + ", value of " + str(val))
        shifts[f2] = shift

    return shifts


def optimizer_stack_align(im_stack, mask_stack, reference_idx, determine_initial_shifts=False, dropthresh=None, transformtype="affine"):
    num_frames = im_stack.shape[-1]

    reg_stack = np.zeros(im_stack.shape)
    eroded_mask = np.zeros(mask_stack.shape)

    # Erode our masks a bit to help with stability.
    for f in range(0, num_frames):
        print(f)
        eroded_mask[..., f] = binary_erosion(mask_stack[..., f], structure=np.ones((21, 21)))
    #   #  im_stack[..., f] *= mask_stack[..., f]

    if determine_initial_shifts:
        initial_shifts = simple_image_stack_align(im_stack * eroded_mask, eroded_mask, reference_idx)
    else:
        initial_shifts = [(0.0, 0.0)] * num_frames


    imreg_method = sitk.ImageRegistrationMethod()
    imreg_method.SetMetricAsCorrelation()
    #imreg_method.SetMetricAsMeanSquares() #Equivalent to MATLAB results ?
    #imreg_method.SetMetricAsANTSNeighborhoodCorrelation(16) # Similar to, but not better than the above
    imreg_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.0625, minStep=1e-5,
                                                          numberOfIterations=500,
                                                          relaxationFactor=0.6, gradientMagnitudeTolerance=1e-5)
    #imreg_method.SetOptimizerAsConjugateGradientLineSearch()
    imreg_method.SetOptimizerScalesFromPhysicalShift() #This apparently allows parameters to change independently of one another.
                                                      # And is incredibly important.
    # #https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/61_Registration_Introduction_Continued.html#Final-registration
    #imreg_method.SetInterpolator(sitk.sitkLanczosWindowedSinc) # Adding this just makes it dog slow.

    ref_im = sitk.GetImageFromArray(im_stack[..., reference_idx])
    # ref_im = sitk.Cast(ref_im, sitk.sitkFloat64)
    ref_im = sitk.Normalize(ref_im)

    dims = ref_im.GetDimension()

    imreg_method.SetMetricFixedMask(sitk.GetImageFromArray(eroded_mask[..., reference_idx], sitk.sitkInt8))
    xforms = [None] * num_frames
    inliers = np.zeros(num_frames, dtype=bool)
    for f in range(0, num_frames):
        print(f)
        if transformtype == "rigid":
            xForm = sitk.Euler2DTransform()
        elif transformtype == "affine":
            xForm = sitk.AffineTransform(2)

        xForm.SetCenter((np.array(im_stack[..., reference_idx].shape, dtype="float")-1) / 2)
        xForm.SetTranslation(initial_shifts[f])

        imreg_method.SetInitialTransform(xForm)
        imreg_method.SetInterpolator(sitk.sitkLinear)

        moving_im = sitk.GetImageFromArray(im_stack[..., f])
        moving_im = sitk.Normalize(moving_im)
        imreg_method.SetMetricMovingMask(sitk.GetImageFromArray(eroded_mask[..., f], sitk.sitkInt8))

        outXform = imreg_method.Execute(ref_im, moving_im)

        if dropthresh is not None and imreg_method.GetMetricValue() > -dropthresh:
            # print("Excluded: " + str(imreg_method.GetMetricValue()))
            inliers[f] = False
        else:
            # print("INCLUDED: " + str(imreg_method.GetMetricValue()))
            inliers[f] = True

        if transformtype == "rigid":
            outXform = sitk.Euler2DTransform(outXform)
        elif transformtype == "affine":
            outXform = sitk.AffineTransform(outXform)

        # ITK's "true" transforms are found as follows: T(x)=A(x−c)+t+c
        A = np.array(outXform.GetMatrix()).reshape(2, 2)
        c = np.array(outXform.GetCenter())
        t = np.array(outXform.GetTranslation())

        Tx = np.eye(3)
        Tx[:2, :2] = A
        Tx[0:2, 2] = -np.dot(A, c)+t+c
        xforms[f] = Tx[0:2, :]

        out_im = sitk.Resample(sitk.GetImageFromArray(im_stack[..., f]), ref_im, outXform, sitk.sitkLanczosWindowedSinc)
        reg_stack[..., f] = sitk.GetArrayFromImage(out_im)

    save_video(
        "M:\\Dropbox (Personal)\\Research\\Torsion_Distortion_Correction\\testalign.avi",
        (reg_stack * 255).astype("uint8"), 30)
    return reg_stack, xforms, inliers


def relativize_image_stack(image_data, mask_data, reference_idx=0, numkeypoints=5000, method="affine", dropthresh=None):
    num_frames = image_data.shape[-1]

    xform = [None] * num_frames
    corrcoeff = np.empty((num_frames, 1))
    corrcoeff[:] = np.NAN
    corrected_stk = np.zeros(image_data.shape)

    sift = cv2.SIFT_create(numkeypoints, nOctaveLayers=55, contrastThreshold=0)

    keypoints = []
    descriptors = []

    for f in range(num_frames):
        kp, des = sift.detectAndCompute(image_data[..., f], mask_data[..., f], None)
        # if numkeypoints > 8000:
        #     print("Found "+ str(len(kp)) + " keypoints")
        # Normalize the features by L1; (make this RootSIFT) instead.
        des /= (des.sum(axis=1, keepdims=True) + np.finfo(np.float).eps)
        des = np.sqrt(des)
        keypoints.append(kp)
        descriptors.append(des)


    # Set up FLANN parameters (feature matching)... review these.
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=64)

    flan = cv2.FlannBasedMatcher(index_params, search_params)

    # Specify the number of iterations.
    for f in range(num_frames):
        matches = flan.knnMatch(descriptors[f], descriptors[reference_idx], k=2)

        good_matches = []
        for f1, f2 in matches:
            if f1.distance < 0.75 * f2.distance:
                good_matches.append(f1)

        if len(good_matches) >= 4:
            src_pts = np.float32([keypoints[f][f1.queryIdx].pt for f1 in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[reference_idx][f1.trainIdx].pt for f1 in good_matches]).reshape(-1, 1, 2)

            # img_matches = np.empty((max(image_data[..., f].shape[0], image_data[..., f].shape[0]), image_data[..., f].shape[1] + image_data[..., f].shape[1], 3),
            #                        dtype=np.uint8)
            # cv2.drawMatches( image_data[..., f], keypoints[f], image_data[..., reference_idx], keypoints[reference_idx], good_matches, img_matches)
            # cv2.imshow("meh", img_matches)
            # cv2.waitKey()

            if method == "affine":
                M, inliers = cv2.estimateAffine2D(dst_pts, src_pts) # More stable- also means we have to set the inverse flag below.
            else:
                M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts)

            if M is not None and np.sum(inliers) >= 4:
                xform[f] = M

                corrected_stk[..., f] = cv2.warpAffine(image_data[..., f], xform[f], image_data[..., f].shape,
                                                      flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
                warped_mask = cv2.warpAffine(mask_data[..., f], xform[f], mask_data[..., f].shape,
                                             flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)

                # Calculate and store the final correlation. It should be decent, if the transform was.
                res = cv2.matchTemplate(image_data[..., reference_idx], corrected_stk[..., f].astype("uint8"),
                                    cv2.TM_CCOEFF_NORMED, mask=warped_mask)

                corrcoeff[f] = res.max()

                # print("Found " + str(np.sum(inliers)) + " matches between frame " + str(f) + " and the reference, for a"
                #                                         " normalized correlation of " + str(corrcoeff[f]))
            else:
                pass
                #print("Not enough inliers were found: " + str(np.sum(inliers)))
        else:
            pass
            #print("Not enough matches were found: " + str(len(good_matches)))

    if not dropthresh:
        print("No drop threshold detected, auto-generating...")
        dropthresh = np.nanquantile(corrcoeff, 0.01)


    corrcoeff[np.isnan(corrcoeff)] = 0  # Make all nans into zero for easy tracking.

    inliers = np.squeeze(corrcoeff >= dropthresh)
    corrected_stk = corrected_stk[..., inliers]
    # save_video(
    #     "\\\\134.48.93.176\\Raw Study Data\\00-64774\\MEAOSLO1\\20210824\\Processed\\Functional Pipeline\\test_corrected_stk.avi",
    #     corrected_stk, 29.4)
    for i in range(len(inliers)):
        if not inliers[i]:
            xform[i] = None # If we drop a frame, eradicate its xform. It's meaningless anyway.

    print("Using a threshold of "+ str(dropthresh) +", we kept " + str(np.sum(corrcoeff >= dropthresh)) + " frames. (of " + str(num_frames) + ")")


    return corrected_stk, xform, inliers


def weighted_z_projection(image_data, weights=None, projection_axis=-1, type="average"):
    num_frames = image_data.shape[-1]

    origtype = image_data.dtype

    if origtype == np.float32 or origtype == np.float64:
        minpossval = 0
        maxpossval = 1
    else:
        minpossval = np.iinfo(origtype).min
        maxpossval = np.iinfo(origtype).max

    if weights is None:
        weights = image_data > 0

    image_data = image_data.astype("float32")
    weights = weights.astype("float32")

    image_projection = image_data * weights
    image_projection = np.nansum(image_projection, axis=projection_axis)
    weight_projection = np.nansum(weights, axis=projection_axis)
    weight_projection[weight_projection == 0] = np.nan

    image_projection /= weight_projection

    weight_projection[np.isnan(weight_projection)] = minpossval

    image_projection[image_projection < minpossval] = minpossval
    image_projection[image_projection >= maxpossval] = maxpossval
    #pyplot.imshow(image_projection, cmap='gray')
    #pyplot.show()

    return image_projection.astype(origtype), (weight_projection / np.amax(weight_projection))
