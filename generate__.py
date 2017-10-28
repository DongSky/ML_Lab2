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
    sigma1=np.array([[1,0.5],[0.5,1]])
    sigma_=np.array([[1,0.3],[0.3,1]])
    mu=np.array([[4.5,6.5]])
    mu1=np.array([[6,5]])
    ran=random.randint(0,n//4)
    a=np.random.multivariate_normal(mu[0],sigma1,n//2+ran)
    b=np.random.multivariate_normal(mu1[0],sigma_,n-(n//2+ran))
    var_lst=np.concatenate((a,b))
    label_lst=[1 for i in range(n//2+ran)]+[0 for i in range(n-(n//2+ran))]
    #print(var_lst)
    #print(label_lst)
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
    ran=random.randint(0,t//4)
    a=np.random.multivariate_normal(mu[0],sigma1,t//2+ran)
    b=np.random.multivariate_normal(mu1[0],sigma_,t-(t//2+ran))
    var_lst=np.concatenate((a,b))
    label_lst=[1 for i in range(t//2+ran)]+[0 for i in range(t-(t//2+ran))]
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
