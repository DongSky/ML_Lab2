import argparse
import math
import numpy as np
import os
import random
if __name__=="__main__":
    f_read=open("transfusion.data","r")
    data_x=[]
    data_label=[]
    data=f_read.readlines()[1:]
    random.shuffle(data)
    for line in data:
        piece=eval(line)
        data_label.append(piece[-1])
        data_x.append(piece[:len(piece)-1])
    test_x=data_x[:200]
    test_label=data_label[:200]
    train_x=data_x[200:]
    train_label=data_label[200:]
    f1=open("data.txt","w")
    f2=open("label.txt","w")
    for i in range(len(train_label)):
        f2.write(str(train_label[i])+"\n")
        line=""
        for j in range(len(train_x[0])):
            if j==0:
                line+=str(train_x[i][j])
            else:
                line+=" "+str(train_x[i][j])
        f1.write(line+"\n")
    f1.close()
    f2.close()
    f1=open("test_data.txt","w")
    f2=open("test_label.txt","w")
    for i in range(len(test_label)):
        f2.write(str(test_label[i])+"\n")
        line=""
        for j in range(len(test_x[0])):
            if j==0:
                line+=str(test_x[i][j])
            else:
                line+=" "+str(test_x[i][j])
        f1.write(line+"\n")
    f1.close()
    f2.close()
