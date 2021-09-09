import torch.multiprocessing as mp
import os

# class Hello():
#     def __init__(self):
#         print("class initiated")

#     def f(self):
#         n = 2
#         if n<GG:
#             print(GG)
        


# if __name__ == '__main__':
#     GG = 43
#     hell = Hello()
#     hell.f()

print(mp.cpu_count())
print(os.cpu_count())