import argparse
import math
import numpy as np
import os
import random

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Data Generate")
    parser.add_argument('-n',action='store',dest='arg_n',default=200,type=int)
    parser.add_argument('-t',action='store',dest='arg_t',default=200,type=int)
    parser.add_argument('-a',action='store',dest='arg_a',default="[1.3, -1.5, 0.0]",type=str)
    parser.add_argument('-x',action='store',dest='arg_x',default="[0, 10]",type=str)
    result=parser.parse_args()
    w=eval(result.arg_a)
    rx=eval(result.arg_x)
    n=int(result.arg_n)
    t=int(result.arg_t)
    len_w=len(w)
    var_lst=[]
    label_lst=[]
    for i in range(n):
        t_x=[]
        for j in range(len_w-1):
            t_x.append(random.randint(rx[0],rx[1])+random.random())
        var_lst.append(t_x)
        s=0
        for j in range(len_w-1):
            s+=t_x[j]*w[j]
        s+=1.0*w[len_w-1]
        if s>0:
            s=1
        else:
            s=0
        label_lst.append(s)
    f1=open("data.txt","w")
    f2=open("label.txt","w")
    for i in range(len(label_lst)):
        f2.write(str(label_lst[i])+"\n")
        line=""
        for j in range(len(var_lst[0])):
            if j==0:
                line+=str(var_lst[i][j])
            else:
                line+=" "+str(var_lst[i][j])
        f1.write(line+"\n")
    f1.close()
    f2.close()
    var_lst=[]
    label_lst=[]
    for i in range(t):
        t_x=[]
        for j in range(len_w-1):
            t_x.append(random.randint(rx[0],rx[1])+random.random())
        var_lst.append(t_x)
        s=0
        for j in range(len_w-1):
            s+=t_x[j]*w[j]
        s+=1.0*w[len_w-1]
        if s>0:
            s=1
        else:
            s=0
        label_lst.append(s)
    f1=open("test_data.txt","w")
    f2=open("test_label.txt","w")
    for i in range(len(label_lst)):
        f2.write(str(label_lst[i])+"\n")
        line=""
        for j in range(len(var_lst[0])):
            if j==0:
                line+=str(var_lst[i][j])
            else:
                line+=" "+str(var_lst[i][j])
        f1.write(line+"\n")
    f1.close()
    f2.close()
