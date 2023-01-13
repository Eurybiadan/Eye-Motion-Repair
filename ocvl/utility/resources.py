import os
import warnings
from enum import Enum
from os.path import exists, splitext

import cv2
import numpy as np


class ResourceType(Enum):
    CUSTOM = 1
    IMAGE2D = 2
    IMAGE3D = 3
    IMAGE4D = 4
    TEXT = 5
    COORDINATES = 6


def load_video(video_path):
    # Load the video data.
    vid = cv2.VideoCapture(video_path)

    framerate = -1

#    warnings.warn("Videos are currently only loaded as grayscale.")
    if vid.isOpened():
        framerate = vid.get(cv2.CAP_PROP_FPS)
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Grab one frame, so we can find out the datatype for our array.
        ret, frm = vid.read()

        video_data = np.empty([height, width, num_frames], dtype=frm.dtype)
        video_data[..., 0] = frm[..., 0]
    else:
        warnings.warn("Failed to open video: "+video_path)

    i = 1
    while vid.isOpened():
        ret, frm = vid.read()
        if ret:
            video_data[..., i] = frm[..., 0]
            i += 1
        else:
            break

    vid.release()

    meta = {"framerate": framerate}

    return Resource(dattype=ResourceType.IMAGE3D, name=os.path.basename(video_path), data=video_data, metadict=meta)

def save_tiff_stack(stack_path, stack_data, scalar_mapper=None):

    framelist = np.empty((stack_data.shape[2], stack_data.shape[0], stack_data.shape[1]))
    for f in range(stack_data.shape[-1]):
        framelist[f, :, :] = stack_data[..., f].astype("uint8")

    cv2.imwritemulti(str(stack_path), framelist)

def save_video(video_path, video_data, framerate = 30, scalar_mapper=None):
    if scalar_mapper is None:
        vidout = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"Y800"), framerate,
                                 (video_data.shape[1], video_data.shape[0]), isColor=False )

        if vidout.isOpened():
            i=0
            while(True):
                vidout.write(video_data[..., i].astype("uint8"))

                i+=1

                if i >= video_data.shape[-1]:
                    break

            vidout.release()
    else:
        vidout = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"FFV1"), framerate,
                                 (video_data.shape[1], video_data.shape[0]), isColor=True)

        if vidout.isOpened():
            i = 0
            while (True):
                frm = scalar_mapper.to_rgba(video_data[..., i]) * 255
                vidout.write(cv2.cvtColor(frm[..., 0:3].astype("uint8"), cv2.COLOR_RGB2BGR))

                i += 1

                if i >= video_data.shape[-1]:
                    break

            vidout.release()


class ResourceLoader:
    def __init__(self, file_paths):

        if not file_paths:
            raise FileNotFoundError("File path " + file_paths + " doesn't exist!")

        # If this isn't a list, then make it one.
        if not hasattr(file_paths, "__len__") or isinstance(file_paths, str):
            self.file_paths = np.array([file_paths])
        else:
            self.file_paths = file_paths

        for file in self.file_paths:
            if not exists(file):
                raise FileNotFoundError("File path " + file + " doesn't exist! Load failed.")

    def load_by_extension(self):
        loaded_data = []

        for file in self.file_paths:
            extension = splitext(file)

            ext_switch = {
                ".avi": load_video,
            }

            loaded_data.append(ext_switch[extension[1]](file))

        return loaded_data

class ResourceSaver:
    def __init__(self, file_paths, dataset, metadata):

        # If this isn't a list, then make it one.
        if not hasattr(file_paths, "__len__") or isinstance(file_paths, str):
            self.file_paths = np.array([file_paths])
        else:
            self.file_paths = file_paths

        for file in self.file_paths:
            if not exists(file):
                raise FileNotFoundError("File path " + file + " doesn't exist! Load failed.")



class Resource:
    def __init__(self, dattype=ResourceType.CUSTOM, name="", data=np.empty([1]), metadict={}):
        self.type = dattype
        self.name = name
        self.data = data
        self.metadict = metadict
