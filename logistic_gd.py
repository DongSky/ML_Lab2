import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import math
eps=1e-5
alpha=1e-3
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
    theta=np.mat(np.ones((X.shape[1],1)).reshape((X.shape[1],1)))
    eriri=0.0;cnt=1
    y=np.array(y).reshape(len(y),1)
    while True:
        #update w(named as theta)
        temp_x=sigmoid(X*theta)
        theta=theta-alpha*(X.T*(temp_x-y))
        #calculate the new loss
        temp_x=sigmoid(X*theta)
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
    theta=np.mat(np.ones((X.shape[1],1)).reshape((X.shape[1],1)))
    eriri=0.0;cnt=1
    y=np.array(y).reshape(len(y),1)
    while True:
        #update w(named as theta)
        temp_x=sigmoid(X*theta)
        theta=theta-alpha*(X.T*(temp_x-y))
        #calculate the new loss
        temp_x=sigmoid(X*theta)
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
        item_x.append(1.0)
        t_X.append(item_x)
    tX=np.mat(t_X)
    #print(X)
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
    plt.figure("MachineLearningProjectTwo")
    one_data_x=[]
    one_data_y=[]
    zero_data_x=[]
    zero_data_y=[]
    for i in range(len(tlabel)):
        if tlabel[i]==1:
            one_data_x.append(t_X[i][0]);one_data_y.append(t_X[i][1])
        else:
            zero_data_x.append(t_X[i][0]);zero_data_y.append(t_X[i][1])
    plt.plot(one_data_x,one_data_y,color='r',linestyle='',marker='.')
    plt.plot(zero_data_x,zero_data_y,color='b',linestyle='',marker='.')
    xxa=[]
    yya=[]
    p=0.01
    while p<9.99:
        xxa.append(p)
        yya.append(-float((p*theta[0]+theta[2])/theta[1]))
        p+=0.01
    plt.plot(xxa,yya,color='g',linestyle='-',marker='')
    xxa=[]
    yya=[]
    p=0.01
    while p<9.99:
        xxa.append(p)
        yya.append(-float((p*theta_p[0]+theta_p[2])/theta_p[1]))
        p+=0.01
    plt.plot(xxa,yya,color='m',linestyle='-',marker='')
    plt.show()
