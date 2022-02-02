import numpy as np
from matplotlib import pyplot as plt
if __name__ == "__main__":
    data = np.load("data.npy", mmap_mode='r')
    for i in range(data.shape[0]):
        plt.imsave("img\\img{}.png".format(i), np.hstack((data[i,0],data[i,1],data[i,2])))
        print(i)
    print("done")
