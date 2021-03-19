from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np

class cloudImage:
    def __init__(self, dataFrame,  path = "", fileName = ""):
        self.h = 1400 # image height
        self.w = 2100 # image width
        self.fileName = fileName
        # Builds full path name
        self.fileNameWithPath = path + fileName
        # list of the patterns
        self.patternList = ['Fish', 'Flower', 'Gravel', 'Sugar']
        # Colors of the masks
        self.ColorList = [[0,0,1], [0,1,0], [1,0,0], [.5,0,.3]]
        self.boxes = {'pattern':[], 'mask':[]}
        
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
        self.boxes['mask']
        arrayEncodedPixels = np.array(self.df_image[self.df_image['PatternPresence']].EncodedPixels)
        
        for _ in self.patternList:
            if self.hasPattern(_).item():
                # index of this new entry (0 the first time)
                index_pattern = len(self.boxes['pattern'])
                # initialisation mask
                self.boxes['mask'].append(np.zeros(shape = (self.h, self.w, 3)))
                # extract the corresponding encodedPixels
                encodedPixels = np.fromstring( arrayEncodedPixels[index_pattern], sep = " ")
                
                indexStartPixels    = encodedPixels[::2]
                lengthLine          = encodedPixels[1::2]
                
                for ind_ind, start_ind in enumerate(indexStartPixels):
                    row_start = int((start_ind-1)%self.h)
                    col_start = int((start_ind-1)//self.h)
                    # fill mask, vertical line by vertical line
                    self.boxes['mask'][index_pattern][row_start:row_start+int(lengthLine[ind_ind]) - 1,col_start, :] += self.ColorList[self.indexPattern(_)]
                
                self.boxes['pattern'].append(_)
                


