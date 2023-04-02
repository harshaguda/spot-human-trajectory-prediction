import cv2
import numpy as np

class ImageStitcher:
    """
    Class which stitches images using Intrinsic camera matrix for the raw (distorted) images, Projection/camera matrix.
    """
    def __init__(self):
        self.K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
                            [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self.K1 = np.array([[255.8085174560547, 0.0, 314.3297424316406],
                            [0.0, 255.2574462890625, 240.5596466064453],
                            [0.0, 0.0, 1.0]])
        self.K2 = np.array([[256.9691467285156, 0.0, 320.87396240234375], 
                            [0.0, 256.55633544921875,240.7810516357422], 
                            [0.0, 0.0, 1.0]])

        self.P1 = np.array([[255.8085174560547, 0.0, 314.3297424316406, 0.0], 
                            [0.0, 255.2574462890625, 240.5596466064453, 0.0], 
                            [0.0, 0.0, 1.0, 0.0]])
        self.P2 = np.array([[256.9691467285156, 0.0, 320.87396240234375, 0.0], 
                            [0.0, 256.55633544921875, 240.7810516357422, 0.0], 
                            [0.0, 0.0, 1.0, 0.0]])
        self.sticher = cv2.Stitcher_create()
        
    def stitch(self, image1, image2):
        """
        Stitch the images together using the intrinsic camera matrix and projection matrix.
        """
        stiched_image = self.sticher.stitch([image1, image2], cv2.STITCHER_PANORAMA)
        print(stiched_image)
        return stiched_image

def main():
    image_stitcher = ImageStitcher()
    image1 = cv2.imread('10.jpg')
    image2 = cv2.imread('11.jpg')
    stiched_image = image_stitcher.stitch(image1, image2)
    cv2.imwrite('stiched_image.png', stiched_image)
if __name__ == '__main__':
    main()