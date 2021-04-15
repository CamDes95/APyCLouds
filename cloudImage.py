# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:57:50 2021
Class used for manage image, build masks, visualize and load mask and clous images, save (resized if any) 
mask and images
@author: luc eglin, camille desjardin, toufik saddik
"""

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


class cloudImage:
    def __init__(self, path="", mask_path="", height=1400, width=2100,fileName="", dataFrame=pd.DataFrame(data=None), new_size=[0, 0]):
        self.h = height  # original image height
        self.w = width  # original image width
        self.new_size = new_size
        self.resize_jpg = new_size[::-1] # la convention est w x h pour resize dans PILLOW
        self.fileName = fileName
        # Builds full path name
        self.fileNameWithPath = path + fileName
        self.mask_path = mask_path
        # list of the patterns
        self.patternList = ['Fish', 'Flower', 'Gravel', 'Sugar']
        self.fileNameMaskWithPath = []
        for p in self.patternList:
            self.fileNameMaskWithPath.append(mask_path+fileName.split('.')[0] + '_' + p + '.jpg')
        # Colors of the masks
        self.ColorList = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [.5, 0, .3]]
        self.boxes = {'pattern': [], 'mask': []}  # masks per class
        self.data = []
        self.taskImage = [] # estimation of the task, present in some images

        if not dataFrame.empty:
            # Reduces the data frame to 4 lines corresponding to fileName
            self.df_image = dataFrame[dataFrame['FileName'] == self.fileName]

            # Builds the boolean array of the presence of clouds type
            self.containsPattern = []
            for p in self.patternList:
                self.containsPattern.append(self.df_image[self.df_image.PatternId == p].PatternPresence)

    def indexPattern(self, pattern):
        """
        :param pattern: string corresponding to the pattern ('Fish', ...)
        :return: index of the pattern in the list of pattern
        """
        return self.patternList.index(pattern)

    def hasPattern(self, pattern):
        """
        :param pattern: string corresponding to the pattern ('Fish', ...)
        :return: boolean if pattern is present in current image
        """
        return self.containsPattern[self.indexPattern(pattern)]


    def load(self, is_mask=False, augmented=False):
        """
        Open jpg image, resize if any, return an np array
        :param is_mask: True if mask, False if cloud image
        :return: image as numpy array
        """
        if is_mask==False:
            fileNameWithPath = [self.fileNameWithPath]
        else:
            fileNameWithPath = self.fileNameMaskWithPath
            if self.new_size != [0, 0]:
                im_array = np.zeros((*new_size, 4))
            else:
                im_array = np.zeros((self.h, self.w, 4))

        for ind,f in enumerate(fileNameWithPath):
            pil_im = Image.open(f, 'r')
            if self.new_size != [0, 0]:
                if is_mask == False:
                    im_array = np.resize(np.asarray(pil_im), self.new_size)
                else:
                    im_array[:,:,ind] = np.resize(np.asarray(pil_im), self.new_size, resample=Image.NEAREST)
                    im_array[:,:,ind] = np.where(im_array[:,:,ind] > 1, 1, im_array[:,:,ind])
            else:
                if is_mask == False:
                    im_array = np.asarray(pil_im)
                else:
                    arr = np.asarray(pil_im)
                    arr = np.where(arr > 1, 1, arr)
                    im_array[:,:,ind] = np.asarray(arr)


        return im_array

    # def transformImage(self, im):


    def visualize(self):
        """
        show cloud image
        :return: show the image
        """
        data = [load(self) if self.data==[] else self.data]
        plt.imshow(data, alpha=3.)
        plt.axis(False)
        plt.title(self.fileName)

    def visualizeBoxes(self):
        title = self.fileName
        for i_mask in range(len(self.boxes['mask'])):
            plt.imshow(self.boxes['mask'][i_mask], alpha=.3, );
            title = title + ' / ' + self.boxes['pattern'][i_mask]
        plt.axis(False)
        plt.title(title)

    def computeBoxCoordinates(self):
        """
        For each present pattern, we extract the encodedPixels,
        then the indexes of each starting vertical line, and the length of this line
        From that we build all the masks corresponding to the current image
        """

        arrayEncodedPixels = np.array(self.df_image.EncodedPixels)
        self.boxes['mask'] = []

        # if we have build object with a new size, the masks will have this new size
        if (self.new_size != [0, 0]):
            new_size = self.new_size
        # otherwise we keep the original size
        else:
            new_size = [self.h, self.w]

        for index_pattern, pattern in enumerate(self.patternList):
            if self.hasPattern(pattern).item():
                # initialisation mask
                self.boxes['mask'].append(np.uint8(np.zeros(shape=(self.h, self.w))))
                # extract the corresponding encodedPixels
                encodedPixels = np.fromstring(arrayEncodedPixels[index_pattern], sep=" ")

                # one point over two, from the first element
                indexStartPixels = encodedPixels[::2]
                # one point over two, from the second element
                lengthLine = encodedPixels[1::2]

                # loop to build, line by line, the masks, froms encodedPixels data of the dataframe
                for ind_ind, start_ind in enumerate(indexStartPixels):
                    row_start = int((start_ind - 1) % self.h)
                    col       = int((start_ind - 1) // self.h)
                    # fill mask, vertical line by vertical line
                    index_row = np.arange(row_start, min(self.h, row_start + int(lengthLine[ind_ind])))
                    self.boxes['mask'][index_pattern][index_row, col] = 1

            else:
                self.boxes['mask'].append(np.uint8(np.zeros(shape=new_size)))

            self.boxes['pattern'].append(pattern)
            index_pattern += 1

        # self.boxes['mask'].append()

    def saveMaskAsJPG(self):
        """
        Save masks in jpeg format, maybe by resizing them

        Returns 
        -------
        None.

        """
        try:
            if not (os.path.isdir(self.mask_path)):
                # Création répertoire
                os.mkdir(self.mask_path)
        except OSError:
            print("Creation of the directory %s failed" % self.mask_path)

        for index_pattern, pattern in enumerate(self.patternList):
            im_array = self.boxes['mask'][index_pattern]
            im = Image.fromarray(im_array)

            if (self.resize_jpg!=[0,0]):
                im = im.resize(self.resize_jpg, resample=Image.NEAREST)
                im.convert("L")
                #im = im.point(lambda i: i <= 1 and 1)

            name_image = self.mask_path + self.fileName.split('.')[0] + "_" + self.boxes['pattern'][index_pattern] + ".jpg"
            im.save(name_image)

    def saveReducedImageAsJPG(self, images_dir):
        """
        Save images in jpeg format, maybe by resizing them

        Returns 
        -------
        None.

        """
        try:
            if not (os.path.isdir(images_dir)):
                # Création répertoire
                os.mkdir(images_dir)
        except OSError:
            print("Creation of the directory %s failed" % images_dir)

        if self.resize_jpg!=[0,0]:
            im = Image.open(self.fileNameWithPath, 'r').convert('L')

            """
            # estimation of the task
            mask = Image.eval(im.filter(ImageFilter.MedianFilter).filter(ImageFilter.MedianFilter),
                              (lambda x: 255 if x<10 else 0)).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
            mask = mask.resize(self.resize_jpg, resample=Image.NEAREST)
            mask = mask.convert('L')
            mask = Image.eval(mask, (lambda x: 1 if x >= 128 else 0))
            mask.save(self.mask_path + self.fileName.split('.')[0] + "_task.png")"""

            im = im.resize(self.resize_jpg)
            im.save(images_dir + self.fileName)
