import numpy as np
import cv2
from PIL import Image as im
from numpy.ma.core import asanyarray


def nothing():
    pass

np.set_printoptions(threshold=np.inf)

cv2.namedWindow('Slider')
cv2.createTrackbar('t1', 'Slider', 0, 255, nothing)
cv2.createTrackbar('t2', 'Slider', 0, 255, nothing)

def func():
    while True:
        # grid = np.genfromtxt('/home/hetal/catkin_ws/src/drl_test/src/log1.csv', delimiter=',')
        grid = np.genfromtxt('log1.csv', delimiter=',')
        grid = np.reshape(grid, (384,384))
        grid = grid[0:320, 0:320]

        empty = np.zeros((32,32))
        w = {}
        w[-1] = 0.02
        w[0] = 0.08
        w[100] = 0.9

        for i in range(0,320,10):
            for j in range(0,320,10):
                submatrix = grid[i:i+10, j:j+10]
                #print(i, j)
                # print(submatrix)
                (unique, counts) = np.unique(submatrix, return_counts=True)
                frequencies = np.asarray((unique, counts))
                if unique.shape[0] == 1 and int(unique[0]) == -1:
                    empty[i//10][j//10] = -1.0
                else:
                    d = {}
                    los = {}
                    for i in range(unique.shape[0]):
                        d[int(unique[i])] = counts[i]
                    for key in list(d.keys()):
                        multiplier = w[key]
                        sum = d[key] * multiplier
                        los[key] = sum
                    
                    empty[i//10][j//10] = list(los.keys())[list(los.values()).index(max(list(los.values())))]
                # print(frequencies)

        # Nearest neighbor scaling

        scaled_empty = empty.repeat(2, axis=0).repeat(2, axis=1)



        # print(grid.shape)
        # print(grid)
        # print(scaled_empty)
        # empty_img = np.copy(empty + 155.0*np.ones((32,32)))
        # # for i in range(32):
        # #     for j in range(32):
        # #         if empty_img[i][j] == 154.0:
        # #             empty_img[i][j] == 0.0


        # empty_img[empty_img==154] = 0
        # empty_img[empty_img==155] = 50

        new_empty = np.copy(empty)
        new_empty[new_empty==0] = 0.5 # Open
        new_empty[new_empty==-1] = 0  # Unknown
        new_empty[new_empty==100] = 1 # Closed

        # print(empty_img)

        empty_img = cv2.resize(new_empty,(256,256))
        new_empty[new_empty==1] = 0
        new_empty[new_empty==0.5] = 1
        empty_img_mod = cv2.resize(new_empty, (256,256))
        
        print(new_empty)
        # _,final = cv2.threshold(empty_img, 154, 255, type=cv2.THRESH_BINARY)
        # temp = cv2.cvtColor(np.asanyarray(cv2.resize(empty_img,(256,256))), cv2.COLOR_GRAY2BGR)


        # final = cv2.applyColorMap(temp, cv2.COLORMAP_BONE)
        # empty_img = cv2.cvtColor(empty_img, cv2.COLOR_GRAY2BGR)
        # final = cv2.applyColorMap(empty_img, cv2.COLORMAP_BONE)
        
        
        cv2.imshow("image", empty_img)
        cv2.imshow("image_mod", empty_img_mod)
        # img_blank = np.zeros((256,256,3))
        # for i in range(256):
            # for j in range(256):
                # for k in range(3):
                    # img_blank[i][j][k] = empty_img_mod[i][j]
        # np.reshape(empty_img_mod, (256,256,1))
        empty_img_mod = empty_img_mod.astype('uint8')
        print(np.shape(empty_img_mod))
        # img_blank = cv2.cvtColor(img_blank, cv2.COLOR_BGR2GRAY)
        tr1 = cv2.getTrackbarPos('t1', 'Slider')
        tr2 = cv2.getTrackbarPos('t2', 'Slider')
        canny = cv2.Canny(empty_img_mod,tr1,tr2)
        cv2.imshow("edges", canny)

        key = cv2.waitKey(1)

        if key == 27:
            break


if __name__ == '__main__':
    func()