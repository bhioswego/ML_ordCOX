#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#NO RISK,epochs =1000, c_index=0.7047503219344684 ？ remove log_var[i] seems as no difference.
"""
Demonstrates how the partial likelihood from a Cox proportional hazards
model can be used in a NN loss function. An example shows how a NN with
one linear-activation layer and the (negative) log partial likelihood as
loss function produces approximately the same predictor weights as a Cox
model fit in a more conventional way.
"""
import datetime
import pandas as pd
import numpy as np
import keras
#from lifelines import CoxPHFitter
#from lifelines.datasets import load_kidney_transplant

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import theano
from keras.layers import Dropout, Activation, Lambda
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Input,Embedding

from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.wrappers.scikit_learn import KerasRegressor,KerasClassifier
from lifelines.utils import concordance_index
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from keras.layers.core import Reshape
from keras.utils.vis_utils import plot_model
from keras.initializers import Constant
from keras.layers import  Layer
from sklearn.preprocessing import minmax_scale
import tensorflow.compat.v1 as tf1

import keras.backend as K
from keras.callbacks import LearningRateScheduler

tf1.disable_v2_behavior()
#

kidtx = pd.read_csv('brca_Surv_data_methylation_mRNA_all_lmqcm.csv')
dataX1 =kidtx.drop(["Unnamed: 0","ID","V2.x","V3.x"], axis = 1).values
y = np.transpose(np.array((kidtx["V2.x"], kidtx["V3.x"]))) # V1=time; erged_data33=status



[ m0,n0] = dataX1.shape
dataX = np.asarray(dataX1)
#dataX =minmax_scale(dataX ) 
data_methylation=dataX1[:,0:17]
data_mRNA=dataX1[:,17:n0]
#dataX=data_mRNA
dataX =minmax_scale(dataX )
[ m,n] = dataX.shape
[ m1,n1] = data_methylation.shape
[ m2,n2] = data_mRNA.shape

 
dataX = dataX.reshape(m,1,n)
x=dataX
data_methylation = data_methylation.reshape(m1,1,n1)
data_mRNA = data_mRNA.reshape(m2,1,n2)

ytime=np.transpose(np.array(kidtx["V2.x"])) # only V1=time;
ystatus= np.transpose(np.array(kidtx["V3.x"])) #only erged_data33=status

from keras.utils import np_utils
ystatus2= np_utils.to_categorical(ystatus)





def neg_log_pl(y_true, y_pred):
    # Sort by survival time (descending) so that
    # - If there are no tied survival times, the risk set
    #   for event i is individuals 0 through i
    # - If there are ties, and time[i - k] through time[i]
    #   represent all times equal to time[i], then the risk set
    #   for events i - k through i is individuals 0 through i
    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
    time = K.gather(y_true[:, 0], indices = sorting.indices)
    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices) #
    risk = K.exp(xbeta)

    event = K.gather(y_true[:, 1], indices = sorting.indices)
    denom = K.cumsum(risk) #
    terms = xbeta - K.log(denom)
    loglik = K.cast(event, dtype = terms.dtype) * terms   #
#    neg_log=-K.sum(loglik)
#    
#    ##############################################
#    diff = K.categorical_crossentropy(y_pred[i]-event)
#    
#    def Loss2(y_pred, y_true, log_vars):
#   loss = 0
#   for i in range(len(y_pred)):
#       precision = K.exp(-log_vars[i])
#       diff = (y_pred[i]-y_true[i])**2.
#       loss += K.sum(precision * diff + log_vars[i], -1)
#   return K.mean(loss)
    
    
    
    
#    return -(loglik)
    return -K.sum(loglik)


## C_index metric function

def c_index3(month,risk, status):

    c_index = concordance_index(np.reshape(month, -1), -np.reshape(risk, -1), np.reshape(status, -1))

    return c_index#def get_bi_lstm_model():  

###########################################################################################
def ordinal_loss (y_true, y_pred):
     

    
    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
    ytime = K.gather(y_true[:, 0], indices = sorting.indices)
    yevent = K.gather(y_true[:, 1], indices = sorting.indices)
    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
    risk = K.exp(xbeta)
    matrix_risk = tf.zeros([int(k_n.get_value())],tf.float32)
#    matrix_I = tf.zeros([int(k_n.get_value())],tf.float32)
    matrix_max= tf.zeros([int(k_n.get_value())],tf.float32)
    
    kk_t =k_ordinal_t.get_value()
    kk_n =k_ordinal_n.get_value()
    Hj =k_ordinal_H.get_value()
    a1=tf.constant(1,dtype=tf.float32) 
    for j in range(len(Hj)):
#        print('j:',j)
#        m=kk_n[j]
#        Hj=sum(H[j:],[])
        matrix_j = tf.zeros([int(k_n.get_value())],tf.float32)
        for i in range(1):
         # 生成一个one_hot张量，长度与tensor_1相同，修改位置为1
            for ii in  range(j,len(Hj)):
#                print('ii:',ii)
                risk_more_j=xbeta[Hj[ii]]
                risk_j=xbeta[Hj[j]]
            
                rec= a1-K.exp(risk_j-risk_more_j)
#                rec2=tf.maximum(0.,rec)
            
                shape = risk.get_shape().as_list()
#                one_hot_j = tf.one_hot(H[j],shape[0],dtype=tf.float32)
                one_hot_more_j = tf.one_hot(Hj[ii],shape[0],dtype=tf.float32)
#                one_hot_more_j =tf.reduce_sum(one_hot_more_j0,axis=0)
               
#                new_tensor = matrix_risk+risk_j * one_hot_j
                matrix_j = matrix_j+ rec * one_hot_more_j
        #        tf.reduce_sum(tf.one_hot(sum(H[13:],[]),n1,dtype=tf.float32),axis=0)
        matrix_risk= matrix_risk+ matrix_j 
    cost2 = K.sum(matrix_risk)/(len(Hj))
    return cost2 
    
    

#      
###############################################################################################
def unique_set(Y_hazard):

    a1 = Y_hazard#.numpy()
#    print('Y_hazard:',Y_hazard)
    # Get unique times
    t, idx = np.unique(a1, return_inverse=True)

    # Get indexes of sorted array
    sort_idx = np.argsort(a1)
#    print(sort_idx)
    # Sort the array using the index
    a_sorted =a1[sort_idx]# a1[np.int(sort_idx)]# a[tf.to_int32(sort_idx)]#
#    print('a_sorted:', a_sorted)
    # Find duplicates and make them 0
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))

    # Difference a[n+1] - a[n] of non zero indexes (Gives index ranges of patients with same timesteps)
    unq_count = np.diff(np.nonzero(unq_first)[0])

    # Split all index from single array to multiple arrays where each contains all indexes having same timestep
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))

    return t, unq_idx

###########################################################################################

###################################################################################################
###########################################################################################

############################################################################################
############################################################ 

def py_func(func, inp, Tout, stateful=True, name=None, grad_func=None):

    grad_name = 'PyFuncGrad_' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(grad_name)(grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        func1=tf.py_func(func, inp, Tout, stateful=stateful, name=name)
        return func1
###########################################################################################        

###########################################################################################        
 
#########################################################################################################################
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        log_var=self.log_vars
#        for y_true, y_pred, log_var ,ii in zip(ys_true, ys_pred, self.log_vars,self.nb_outputs):
        for i in range(self.nb_outputs):    
            precision = (K.exp(-log_var[i]))**0.5
#            precision= tf.clip_by_value(precision, 0., 1.)
            if i==0:
                lossA=neg_log_pl(ys_true[i], ys_pred[i])
                
            if i==1: 
#                 lossA = py_func(ordinal_loss,
#                     [ys_true[i-1], ys_pred[i]], [tf.float32],
#                     name = "ordinal_loss",
#                     grad_func = ordinal_loss_grad )[0]
#                 lossA.set_shape((ys_true[i].shape[0],))
#                lossA=LOSS_L2(ys_true[i], ys_pred[i])
                lossA=ordinal_loss0(ys_true[i-1], ys_pred[i])
#                lossA=tf.py_function(func=ordinal_loss, inp=[ys_true[i], ys_pred[i]], Tout=tf.float32) 
#                lossA=ordinal_loss(Y_trueM, model_aux(tf.to_float(x_train0M)))
#                lossA=neg_log_pl_1(ys_true[i-1], ys_pred[i])
#            loss += K.sum(precision * (y_true - y_pred)**2. + log_var[0], -1)
            loss += K.sum(precision * lossA , -1)
#            loss += K.sum(precision * lossA + log_var[i], -1)
        
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)
    #######################################################################
   
#############################################################################################################
def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0:# and epoch != 0:
        lr2=[0.0005,0.0004,0.0004,0.0002,0.0001,0.00005,0.00004,0.00002,0.00001,0.000005,0.000002]
        learning_rate=lr2[int(epoch/100)]
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, (lr/lr )* learning_rate)
        print("lr changed to {}".format((lr/lr )* learning_rate))
    return K.get_value(model.optimizer.lr)

####################################################################################################################    
seed = 63
np.random.seed(seed)
kf = KFold(n_splits=10, shuffle=True, random_state=seed)
ypred=[]
ypred_train=[]
xtest_original=[]
status_new=[]
time_new=[]
index2=[]
iFold = 0
EPOCH =1000
for train_index, val_index in kf.split(x):
    iFold = iFold+1
#    train_x, test_x, train_y, test_y,= X[train_index], X[val_index], y[train_index], y[val_index] # 这里的X_train，y_train为第iFold个fold的训练集，X_val，y_val为validation set
    x_train, x_test, y_train, y_test, ytime_train, ytime_test, ystatus_train, ystatus_test, ystatus2_train, ystatus2_test =\
        dataX[train_index], dataX[val_index], y[train_index], y[val_index], ytime[train_index], ytime[val_index], ystatus[train_index],ystatus[val_index],\
                           ystatus2[train_index],ystatus2[val_index]
    
    input_dim =x_train.shape[2]
    output_dimM = y_train.shape[1]
    output_dimA = 1
    n1 = y_train.shape[0]
    
#    k_n = theano.shared(np.asarray(n,dtype=theano.config.floatX),borrow=True)
    k_n = theano.shared(n1,borrow=True)
    k_ytime_train = theano.shared(ytime_train,borrow=True)
    k_ystatus_train = theano.shared(ystatus_train,borrow=True)
    N = theano.shared(n1,borrow=True)
    R_matrix = np.zeros([n1, n1], dtype=int)
    R_matrix =theano.shared(R_matrix,borrow=True)
##############################################3    
    
    Y_hazard0=y_train[:,0]
    Y_survival=y_train[:,1]


    
    t0, H0 = unique_set(Y_hazard0) # t:unique time. H original index.
    
    actual_event_index = np.nonzero(Y_survival)[0]
    H0 = [list(set(h) & set(actual_event_index)) for h in H0]
    ordinal_n = np.asarray([len(h) for h in H0])
    Hj=sum(H0[0:],[])
    
    k_ordinal_H = theano.shared(np.asarray(Hj),borrow=True)
    k_ordinal_t = theano.shared(t0,borrow=True)
    k_ordinal_n = theano.shared(ordinal_n,borrow=True)
 #########################################################################################################################   

    class varHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.lambdas0 = []
            self.lambdas1 = []
        def on_epoch_end(self, batch, logs={}):
            self.lambdas0.append( K.get_value(self.model.layers[-1].log_vars[0]))
            self.lambdas1.append( K.get_value(self.model.layers[-1].log_vars[1]))
 #############################################################################################################################    
#    input_dim0 =theano.shared(input_dim,borrow=True)
# Build model structure
    # gene Only
    gene_input = Input(name='gene_input', shape=(1,input_dim))
#    out1=Bidirectional(LSTM(55,activation='linear',return_sequences=True,kernel_initializer=glorot_uniform(),kernel_regularizer=l2(reg),activity_regularizer=l2(0.001)), merge_mode='concat')(title_input)
#    out_gene=Bidirectional(LSTM(55,return_sequences=False), merge_mode='concat')(gene_input)
    
    out_gene=Bidirectional(LSTM(55,activation='tanh', return_sequences=False,\
        kernel_initializer=glorot_uniform(),kernel_regularizer=l2(0.0005),activity_regularizer=l2(0.001)),  merge_mode='concat')(gene_input)
    auxiliary_output= Dense(1,activation='linear',name='auxiliary_output')(out_gene)

    model = Model(inputs=[gene_input],outputs=[auxiliary_output])
    model.summary()
    
    model.compile(optimizer='adam', loss=ordinal_loss)

    
    reduce_lr = LearningRateScheduler(scheduler)
    Lambdas = varHistory() 
    hist = model.fit(x_train, y_train, batch_size = n1, epochs =EPOCH,callbacks=[reduce_lr])
    
   
#    lambda0=np.asarray( Lambdas.lambdas0)
#    lambda1=np.asarray( Lambdas.lambdas1)
    
       
    plt.plot(range(EPOCH), hist.history['loss'])
    plt.title('Training Loss Curves only auxiliary loss')
    plt.ylabel('train-loss Value')
    plt.xlabel('epoch')
#    plt.legend(['${lambda0}$','${lambda1}$'], loc='upper right')
    plt.savefig('Loss Curves only auxiliary loss.jpg', dpi=300) #
    plt.show()
    
    
    np.savetxt("hist.history['loss1000_ml_ordCOX20201220']_without_penalty_term_only_auxiliary_loss.csv",hist.history['loss'], delimiter=",")
    

    prediction = model.predict(x_test)

    
    c_index2=c_index3( np.asarray(ytime_test),np.asarray(prediction), np.asarray(ystatus_test))
    
    print( c_index2)
    
#############################################################################################################################    
   
    ypred.extend(prediction)
#    ypred_train.extend(prediction_train_median)
#    xtest_original.extend(x_test)
    index2.extend(val_index)
    status_new.extend(ystatus[val_index])
    time_new.extend(ytime[val_index])
#    print(ypred.shape)
    
    K.clear_session()
    tf1.reset_default_graph()
    print(iFold)
    nowTime = datetime.datetime.now()
    print("nowTime: ",nowTime)
np.savetxt("brca_prediction1204_18lstm2222_epoch400_drop01_resnet_without penalty term_only_auxiliaryloss.csv", ypred, delimiter=",")
np.savetxt("brca_ytime_test1204_18lstm2222_epoch400_drop01_resnet_without penalty term_only_auxiliaryloss.csv", time_new, delimiter=",")
np.savetxt("brca_ystatus_test1204_18lstm2222_epoch400_drop01_resnet_without penalty term_only_auxiliaryloss.csv", status_new, delimiter=",")
np.savetxt("brca_ypred_train_median1204_18lstm2222_epoch400_drop01_resnet_without penalty term_only_auxiliaryloss.csv", ypred_train, delimiter=",")

df = pd.read_csv("brca_prediction1204_18lstm2222_epoch400_drop01_resnet_without penalty term_only_auxiliaryloss.csv",header=None)    
month=np.asarray(pd.read_csv("brca_ytime_test1204_18lstm2222_epoch400_drop01_resnet_without penalty term_only_auxiliaryloss.csv",header=None)) 
status=np.asarray(pd.read_csv("brca_ystatus_test1204_18lstm2222_epoch400_drop01_resnet_without penalty term_only_auxiliaryloss.csv",header=None)) 



risk=np.asarray(df)
c_indices_mlp = c_index3(month, risk,status)
#np.savetxt("c_indices_nn827.txt", c_indices_mlp, delimiter=",")
np.save("c_indices_without penalty term_only_auxiliaryloss",c_indices_mlp) 
print("brca_methylation_mRNA_lmqcm_lr_backend2233_only_auxiliary_loss20210106.py") 
print("brca only_auxiliary:dataX=data_rna+methylation:",c_indices_mlp)
data_a=np.load('c_indices_without penalty term_only_auxiliaryloss.npy')
aa=0