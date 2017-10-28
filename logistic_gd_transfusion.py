import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import math
eps=1e-6
alpha=2e-4
iteration=1000
train_size=1000
panel=1
def sigmoid(x):
    return 1.0/(np.exp(-x)+1.0)
def Eriri(y,calc_y):
    loss=0.0
    for i in range(0,len(y)):
        if y[i]==1:
            loss+=-math.log(calc_y[i])
        else:
            loss+=-math.log(1-calc_y[i])
    return loss
def Eriri_with_panelty(y,calc_y,theta):
    theta_=np.array(theta).reshape(1,theta.shape[0])[0]
    loss=0.0
    for i in range(0,len(y)):
        if y[i]==1:
            loss+=-math.log(calc_y[i])
        else:
            loss+=-math.log(1-calc_y[i])
    res=0.0
    for i in range(0,len(theta_)):
        res+=theta_[i]*theta_[i]*panel
    return loss+math.sqrt(res)
def BGD(matX,y):
    X=np.array(matX)
    theta=np.mat(np.array([random.uniform(-1,1) for i in range(X.shape[1])]).reshape((X.shape[1],1)))
    eriri=0.0;cnt=1
    y=np.array(y).reshape(len(y),1)
    while True:
        #update w(named as theta)
        temp_x=sigmoid(X*theta)
        theta=theta-alpha*(X.T*(temp_x-y))
        #calculate the new loss
        temp_x=sigmoid(X*theta)
        #print(temp_x)
        new_eriri=Eriri(y,temp_x)
        #output the current loss
        if cnt%1000==0:
            print("%06d    %.8f"%(cnt,new_eriri))
        if abs(eriri-new_eriri)<=eps:
            break
        eriri=new_eriri;cnt+=1
    return theta
def BGD_penalty(matX,y):
    X=np.array(matX)
    theta=np.mat(np.array([random.uniform(-1,1) for i in range(X.shape[1])]).reshape((X.shape[1],1)))
    eriri=0.0;cnt=1
    y=np.array(y).reshape(len(y),1)
    while True:
        #update w(named as theta)
        temp_x=sigmoid(X*theta)
        theta=theta-alpha*(X.T*(temp_x-y))
        #calculate the new loss
        temp_x=sigmoid(X*theta)
        #print(temp_x)
        new_eriri=Eriri_with_panelty(y,temp_x,theta)
        #output the current loss
        if cnt%1000==0:
            print("%06d    %.8f"%(cnt,new_eriri))
        if abs(eriri-new_eriri)<=eps:
            break
        eriri=new_eriri;cnt+=1
    return theta

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="logistic with BGD")
    parser.add_argument("--data",action="store",type=str,default="data.txt",dest="arg_data")
    parser.add_argument("--label",action="store",type=str,default="label.txt",dest="arg_label")
    parser.add_argument("--tdata",action="store",type=str,default="test_data.txt",dest="arg_test_data")
    parser.add_argument("--tlabel",action="store",type=str,default="test_label.txt",dest="arg_test_label")
    result=parser.parse_args()
    data_path=result.arg_data
    label_path=result.arg_label
    tdata_path=result.arg_test_data
    tlabel_path=result.arg_test_label
    f1=open(data_path,"r")
    f2=open(label_path,"r")
    label=[int(i) for i in f2.readlines()]
    X=[]
    lines=f1.readlines()
    for line in lines:
        piece=line.split()
        item_x=[float(j) for j in piece]
        item_x[0]/=5.0
        item_x[1]/=5.0
        item_x[2]/=500.0
        item_x[3]/=20.0
        item_x.append(1.0)
        X.append(item_x)
    X=np.mat(X)
    #print(X)
    label=np.array(label)
    #print(label)
    theta=BGD(X,label)
    theta_p=BGD_penalty(X,label)
    print(theta)
    print(theta_p)
    f1.close()
    f2.close()
    f1=open(tdata_path,"r")
    f2=open(tlabel_path,"r")
    tlabel=[int(i) for i in f2.readlines()]
    t_X=[]
    lines=f1.readlines()
    for line in lines:
        piece=line.split()
        item_x=[float(j) for j in piece]
        item_x[0]/=5.0
        item_x[1]/=5.0
        item_x[2]/=500.0
        item_x[3]/=20.0
        item_x.append(1.0)
        t_X.append(item_x)
    tX=np.mat(t_X)
    #print(tX)
    label=np.array(label)
    result_0=sigmoid(tX*theta)
    result_1=sigmoid(tX*theta_p)
    cnt=0
    for i in range(len(tlabel)):
        if (result_0[i]>0.5 and tlabel[i]==1)or(result_0[i]<0.5 and tlabel[i]==0)or(result_0[i]==0.5):
            cnt+=1
    acc_0=(cnt+0.0)/len(tlabel)
    cnt=0
    for i in range(len(tlabel)):
        if (result_1[i]>0.5 and tlabel[i]==1)or(result_1[i]<0.5 and tlabel[i]==0)or(result_1[i]==0.5):
            cnt+=1
    acc_1=(cnt+0.0)/len(tlabel)
    print("Accurency without penalty:")
    print("%.8f"%acc_0)
    print("Accurency with penalty:")
    print("%.8f"%acc_1)
