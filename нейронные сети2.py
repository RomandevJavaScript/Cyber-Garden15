import numpy as np
import cv2
#read image img_src = cv2.imread('sample.jpg')
 #prepare the 5x5 shaped filter kernel = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]) kernel = kernel/sum(kernel) #filter the source image img_rst = cv2.filter2D(img_src,-1,kernel)
 #save result image cv2.imwrite('result.jpg',img_rst)
import numpy as np
import cv2
#read image img_src = cv2.imread('sample.jpg') #edge detection filter kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]) kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1) #filter the source image img_rst = cv2.filter2D(img_src,-1,kernel) #save result image cv2.imwrite('result.jpg',img_rst)
kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]])
n
import numpy as np
 import cv2 #read image img_src = cv2.imread('sample.jpg') #kernal sensitive to horizontal lines kernel = np.array([[-1.0, -1.0], [2.0, 2.0], [-1.0, -1.0]]) kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1) #filter the source image img_rst = cv2.filter2D(img_src,-1,kernel) #save result image cv2.imwrite('result.jpg',img_rst)
@misc { shonenkov2021emojich,
      title={Emojich -- zero-shot emoji generation using Russian language: a technical report},
      author={Alex Shonenkov and Daria Bakshandaeva and Denis Dimitrov and Aleksandr Nikolich},
      year={2021},
      eprint={2112.02448},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}