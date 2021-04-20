from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

df_train = pd.read_csv("train.csv")
df_train['FileName'] = df_train['Image_Label'].apply(lambda col: col.split('_')[0])
df_train['PatternId'] = df_train['Image_Label'].apply(lambda col: col.split('_')[1])
df_train['PatternPresence'] = ~ df_train['EncodedPixels'].isna()


class cloudImage2:
    def __init__(self, path = "", fileName = "", dataFrame = df_train):
        self.h = 1400 # image height
        self.w = 2100 # image width
        self.fileName = fileName
        # Builds full path name
        self.fileNameWithPath = path + fileName
        # list of the patterns
        self.patternList = ['Fish', 'Flower', 'Gravel', 'Sugar']
        # Colors of the masks
        self.ColorList = [[0,0,1], [0,1,0], [1,0,0], [.5,0,.3]]
        self.boxes = {'pattern':[], 'mask':[]} # masks per class
        
        # Reduces the data frame to 4 lines corresponding to fileName
        self.df_image = dataFrame[dataFrame['FileName'] == self.fileName]
        
        # Builds the boolean array of the presence of clouds type
        self.containsPattern = []
        for p in self.patternList:
            self.containsPattern.append(self.df_image[self.df_image.PatternId == p].PatternPresence) 
        
    def indexPattern(self, pattern):
        return self.patternList.index(pattern)
        
    def hasPattern(self, pattern):
        return self.containsPattern[self.indexPattern(pattern)]
        
    def visualize(self):
        pil_im = Image.open(self.fileNameWithPath,'r')
        plt.imshow(np.asarray(pil_im), alpha = 3.)
        plt.axis(False)
        plt.title(self.fileName)
        
    def visualizeBoxes(self):
        title = self.fileName
        for i_mask in range(len(self.boxes['mask'])):
            plt.imshow(self.boxes['mask'][i_mask], alpha = .3);
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
        
        for index_pattern, pattern in enumerate(self.patternList):
            # initialisation mask
            self.boxes['mask'].append( np.uint8( np.zeros(shape = (self.h, self.w, 3))) )
            
            if self.hasPattern(pattern).item():
                # extract the corresponding encodedPixels
                encodedPixels = np.fromstring( arrayEncodedPixels[index_pattern], sep = " ")
                
                indexStartPixels    = encodedPixels[::2]
                lengthLine          = encodedPixels[1::2]
                
                for ind_ind, start_ind in enumerate(indexStartPixels):
                    row_start = int((start_ind-1)%self.h)
                    col_start = int((start_ind-1)//self.h)
                    # fill mask, vertical line by vertical line
                    self.boxes['mask'][index_pattern][row_start:row_start+int(lengthLine[ind_ind]) - 1,col_start, :] += np.uint8(self.ColorList[self.indexPattern(pattern)])
            
            self.boxes['pattern'].append(pattern) 
            index_pattern+=1



















