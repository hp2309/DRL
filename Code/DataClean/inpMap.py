import numpy as np
grid = np.zeros((384, 384))
weights = {-1: 0.02, 0: 0.08, 100: 0.9} #unknown -1, open 0, occupied 100; in order
#I will get the grid somehow
grid = np.reshape(grid, (384, 384))
small_grid = np.ones((64, 64))
for i in range(0, 384, 6):
    for j in range(0, 384, 6):
        sub_grid = grid[i:i+6,j:j+6]
        info, freq = np.unique(sub_grid, return_counts = True)
        d = {-1: 0, 0: 0, 100: 0}
        weighted_sum = {}
        for k in range(len(info)):
            d[info[k]] = freq[k]
        weighted_sum[-1] = weights[-1]*d[-1]
        weighted_sum[0] = weights[0]*d[0]
        weighted_sum[100] = weights[100]*d[100]
        key_list = list(weighted_sum.keys())
        val_list = list(weighted_sum.values())
        state = key_list[val_list.index(max(val_list))]
        small_grid[i//6][j//6] = state
        
        