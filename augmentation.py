import cv2
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
import argparse

def args_parse():
    parser = argparse.ArgumentParser(description='Augmentation')
    parser.add_argument('--image_path', '-p' ,type=str, help='Path to the image')
    parser.add_argument('--darker', '-d', type=bool, help='Make image darker')
    parser.add_argument('--save_path', '-s', type=str, help='where to save the image')
    
    return parser.parse_args()

def hist_equa(image_path, darker=False, save_path=False):
    
    image = cv2.imread(image_path)
    
    image = cv2.resize(image, (800, 600))

    image = cv2.GaussianBlur(image, (3,3), 0)
    
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(6,6))

    final_img  = clahe.apply(image_bw)
    
    final_img = cv2.normalize(final_img, None, 0, 255, cv2.NORM_MINMAX)
    
    if darker:
        final_img = adjust_gamma(final_img, gamma=0.5)
   
    cv2.imwrite(save_path, final_img)
        
    return image, final_img

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)  # Áp dụng bảng tra cứu (lookup table)

def visualize_hist(image, final_image):
    hist = cv2.calcHist([final_image], [0], None, [256], [0, 256])

    cv2.imshow("ordinary threshold", image)
    cv2.imshow("CLAHE image", final_image)
    
    
    plt.plot(hist, color='black')
    plt.title('Histogram of an image')
    plt.xlabel('gray level')
    plt.ylabel('number of pixels')
    plt.show()
    
    cv2.waitKey(0)
    
if __name__ == '__main__':
    args = args_parse()
    
    image_path = args.image_path
    darker = args.darker
    save_path = args.save_path
    
    image, final_image = hist_equa(image_path, darker=darker, save_path=save_path)
    
    visualize_hist(image,final_image)
