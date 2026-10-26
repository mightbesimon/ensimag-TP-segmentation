# USAGE
# python predict.py

# import the necessary packages
from model import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

from PyQt5.QtCore import QLibraryInfo
# from PySide2.QtCore import QLibraryInfo


def get_img_name(imagePath):
        #print(imagePath)
        imagePathR = "".join(reversed(imagePath))
        pos=imagePathR.find('/')
        #print(imagePath[len(imagePath)-pos:len(imagePath)-4])
        return imagePath[len(imagePath)-pos:len(imagePath)-4]

def prepare_plot(origImage, origMask, predMask, predProb, imagePath):
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        #fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
        axs = axs.flatten()

        # plot the original image, its mask, and the predicted mask
        axs[0].imshow(origImage)
        axs[1].imshow(origMask)
        axs[2].imshow(predMask)
        axs[3].imshow(predProb)

        # set the titles of the subplots
        axs[0].set_title("Image")
        axs[1].set_title("Original Mask")
        axs[2].set_title("Predicted Mask")
        axs[3].set_title("Predicted Probability")

        # set the layout of the figure and display it
        fig.tight_layout()
        fig.show()
        plotPath=config.PLOT_PATH.replace(get_img_name(config.PLOT_PATH),'predict_plot_' + get_img_name(imagePath))
        fig.savefig(plotPath)

def make_predictions(model, imagePath):
        # set model to evaluation mode
        model.eval()

        # turn off gradient tracking
        with torch.no_grad():
                # load the image from disk, swap its color channels, cast it
                # to float data type, and scale its pixel values
                image = cv2.imread(imagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype("float32") / 255.0
                
                # resize the image and make a copy of it for visualization
                image = cv2.resize(image, (128, 128))
                orig = image.copy()
                
                # find the filename and generate the path to ground truth
                # mask
                filename = imagePath.split(os.path.sep)[-1]
                groundTruthPath = os.path.join(config.MASKS_PATH,filename)
                
                # load the ground-truth segmentation mask in grayscale mode
                # and resize it
                gtMask = cv2.imread(groundTruthPath, 0)
                gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_HEIGHT))
                
                # make the channel axis to be the leading one, add a batch
                # dimension, create a PyTorch tensor, and flash it to the
                # current device
                image = np.transpose(image, (2, 0, 1))
                image = np.expand_dims(image, 0)
                image = torch.from_numpy(image).to(config.DEVICE)
                
                # make the prediction, pass the results through the sigmoid
                # function, and convert the result to a NumPy array
                predMask = model(image).squeeze()
                predMask = torch.sigmoid(predMask)
                predMask = predMask.cpu().numpy()
                
                # filter out the weak predictions and convert them to integers
                predProb = predMask * 255
                predMask = (predMask > config.THRESHOLD) * 255
                predMask = predMask.astype(np.uint8)

                # prepare a plot for visualization
                prepare_plot(orig, gtMask, predMask, predProb, imagePath)


if __name__ == '__main__':
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
                
        # load the image paths in our testing file and randomly select 10
        # image paths
        print("[INFO] loading up test image paths...")
        imagePaths = open(config.TEST_PATH).read().strip().split("\n")
        imagePaths = np.random.choice(imagePaths, size=10)
        
        # load our model from disk and flash it to the current device
        print("[INFO] load up model...")
        unet = torch.load(config.BEST_MODEL_PATH).to(config.DEVICE)

        # iterate over the randomly selected test image paths
        for path in imagePaths:
	        # make predictions and visualize the results
	        make_predictions(unet, path)
