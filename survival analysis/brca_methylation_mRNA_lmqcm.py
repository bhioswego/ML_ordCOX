#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#epochs =1000, c_index=0.7182000286163972 remove log_var[i] seems as no difference.
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
tf1.disable_v2_behavior()
##############################
#kidtx = pd.read_csv('brca_Surv_data_stage_age.csv',usecols=['MEgreenyellow',	'MEgreen','MEturquoise'	,'MEmagenta','MEbrown',	'MEred',	'MEpink',	'MEblack',	'MEpurple',	'MEblue',	'MEyellow',	'MEgrey','erged_data5_stage','erged_data6_Age','V1','erged_data33'])
#kidtx = pd.read_csv('brca_Surv_data_methylation_mRNA_lmqcm.csv')
#dataX1 =kidtx.drop(["Unnamed: 0","ID","V1","erged_data33"], axis = 1).values
#y = np.transpose(np.array((kidtx["V1"], kidtx["erged_data33"]))) # V1=time; erged_data33=status

kidtx = pd.read_csv('brca_Surv_data_methylation_mRNA_all_lmqcm.csv')
dataX1 =kidtx.drop(["Unnamed: 0","ID","V2.x","V3.x"], axis = 1).values
y = np.transpose(np.array((kidtx["V2.x"], kidtx["V3.x"]))) # V1=time; erged_data33=status



[ m0,n0] = dataX1.shape
dataX = np.asarray(dataX1)
dataX =minmax_scale(dataX ) 
data_methylation=dataX1[:,0:17]
data_mRNA=dataX1[:,17:n0-1]
#dataX=data_mRNA
[ m,n] = dataX.shape
[ m1,n1] = data_methylation.shape
[ m2,n2] = data_mRNA.shape

 
dataX = dataX.reshape(m,1,n)
x=dataX
data_methylation = data_methylation.reshape(m1,1,n1)
data_mRNA = data_mRNA.reshape(m2,1,n2)

ytime=np.transpose(np.array(kidtx["V2.x"])) # only V1=time;
ystatus= np.transpose(np.array(kidtx["V3.x"])) #only erged_data33=status
## Build model structure
#model = Sequential()
#model.add(Dense(units = 20, activation = "tanh", use_bias = False, input_shape=[12]))
#model.add(Dense(1, kernel_initializer='normal'))
# Define loss function
# y_true = (n x 2) array with y_true[i, 0] the survival time
#          for individual i and y_true[i, 1] the event indicator
# y_pred = (n x 1) array of linear predictor (x * beta) values
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
    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
    risk = K.exp(xbeta)
#    print('time:',time)
    # For each set of tied survival times, put the sum of the
    # corresponding risk (exp[x * beta]) values at the first
    # position in the sorted array of times while setting other
    # positions to 0 so that the cumsum operation will result                         tied 关联的
    # in each of the positions having the same sum of risks
#    for i in range(time.shape[0] - 1, 0, -1):
##         Going from smallest survival times to largest
#        if time[i] == time[i - 1]:
#            # Push risk to the later time (earlier in array position)
#            xbeta[i - 1] = xbeta[i - 1] + xbeta[i]
#            xbeta[i] = 0
#        
#    risk = K.exp(xbeta)        
#    for i in range(time.shape[0] - 1, 0, -1):
###         Going from smallest survival times to largest
#        if time[i] == time[i - 1]:
#            # Push risk to the later time (earlier in array position)
#            risk[i - 1] = risk[i - 1] + risk[i]
#            risk[i] = 0        
    event = K.gather(y_true[:, 1], indices = sorting.indices)
    denom = K.cumsum(risk) #这个函数的功能是返回给定axis上的累计和
    terms = xbeta - K.log(denom)
    loglik = K.cast(event, dtype = terms.dtype) * terms   #cast将x的数据格式转化成dtype
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
    
    
    
    
    return -(loglik)
#    return -K.sum(loglik)

def LOSS_L2(y_true, y_pred):
#    MAX_SEQ_LEN=1
    BATCH_SIZE=int(k_n.get_value())
    L2_NORM = 0.001
    
   
    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
#    time = K.gather(y_true[:, 0], indices = sorting.indices)
    xbeta = K.gather(y_pred, indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
    risk = K.exp(xbeta)
    event =K.gather(y_true[:,1], indices = sorting.indices)
    
#    self.preds = preds
    final_dead_rate = xbeta
    final_survival_rate=1.0-final_dead_rate
    predict=K.stack([final_survival_rate, final_dead_rate])
    cross_entropy = - K.cumsum( event*K.log(final_dead_rate))
    cost=cross_entropy
    
#    final_survival_rate=tf.subtract(tf.constant(1.0, dtype=tf.float32), final_dead_rate)
#    predict = tf.transpose(tf.stack([final_survival_rate, final_dead_rate]), name="predict")
##    predict =predict[-1,:,:]
#    cross_entropy = -tf.reduce_sum( event*tf.log(tf.clip_by_value(predict,1e-10,1.0)))
#    tvars = tf.trainable_variables()  #tf.trainable_variables 返回所有 当前计算图中 在获取变量时未标记 trainable=False 的变量集合
#    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars ]) * L2_NORM
#    cost = tf.add(cross_entropy, lossL2, name = "cost")  / BATCH_SIZE

    
#    Loss2=K.categorical_crossentropy( event, xbeta)
    Loss=cost
    return Loss






def R_set(x):
	'''Create an indicator matrix of risk sets, where T_j >= T_i.
	Note that the input data have been sorted in descending order.
	Input:
		x: a PyTorch tensor that the number of rows is equal to the number of samples.
	Output:
		indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
	'''
	n_sample =(x.shape[0]).value
	matrix_ones = tf.ones([n_sample, n_sample],tf.int32)
	indicator_matrix = tf.linalg.band_part(matrix_ones,-1,0) #下三角形

	return(indicator_matrix)
    
def neg_log_pl_1(y_true, y_pred):
    
    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
    ytime = K.gather(y_true[:, 0], indices = sorting.indices)
    yevent = K.gather(y_true[:, 1], indices = sorting.indices)
    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
    risk = K.exp(xbeta)
    matrix_risk = tf.zeros([int(k_n.get_value())],tf.float32)
    matrix_I = tf.zeros([int(k_n.get_value())],tf.float32)
    matrix_max= tf.zeros([int(k_n.get_value())],tf.float32)
     
#    weights = K.cast(K.abs(K.argmax(ytime, axis=0) - K.argmax(risk, axis=0))/(int(k_n.get_value()) - 1), dtype='float32')
    
    kk_ytime_train =k_ytime_train.get_value()
    for i in range(ytime.shape[0] - 1, 0, -1):
        # Going from smallest survival times to largest
#        tf.cond(tf.greater_equal(ytime[i-1], ytime[i]),  risk[i - 1]: (risk[i - 1] + risk[i]),  risk[i - 1]: 0)
#        risk[i - 1]=tf.where(tf.equal(ytime[i-1], ytime[i]), risk[i - 1] + risk[i],  risk[i - 1])
#        risk_i,risk_i-1 = risk_ab( risk[i], risk[i - 1])
#        sess = tf.Session()
#        array_i = np.array(kk_ytime_train[i])
#        array_i_1 = np.array(kk_ytime_train[i-1])
#        feed_dict = {a: array_i, b: array_i_1}
#        tile_b_value = sess.run(tile_tensor_b, feed_dict = feed_dict)
#        print(tile_b_value)
#        def true_proc():
#            risk[i - 1] = risk[i - 1] + risk[i]
#            return risk[i - 1]
#		
#        def false_proc():
#	         risk[i] = 0
#	         return risk[i]
#        def true_proc():
#            result=2
#            return result
#		
#        def false_proc():
#            result=3
#            return result
        a0=tf.constant(0,dtype=tf.float32)      
        a1=K.cast(risk[i - 1],dtype=tf.float32)
        a2=K.cast(risk[i],dtype=tf.float32)
        a3=K.cast(a1+a2,dtype=tf.float32)
        risk_i_1= tf.cond(tf.less( kk_ytime_train[i],  kk_ytime_train[i-1]), lambda: a3, lambda:a1)
        risk_i= tf.cond(tf.less( kk_ytime_train[i],  kk_ytime_train[i-1]), lambda: a0, lambda:a2)
        
#        part1 = matrix_risk[:i-1]
#              
#        part2 = matrix_risk[i+1:]
#        val_i_1= risk_i_1
#        val_i = risk_i
#        matrix_risk2=tf.concat([part1,val_i_1,val_i,part2], axis=0)
#        matrix_risk=K.cast(risk,dtype=tf.float32)
        # 生成一个one_hot张量，长度与tensor_1相同，修改位置为1
        shape = risk.get_shape().as_list()
        one_hot_i = tf.one_hot(i,shape[0],dtype=tf.float32)
        one_hot_i_1 = tf.one_hot(i-1,shape[0],dtype=tf.float32)
       # 做一个减法运算，将one_hot为一的变为原张量该位置的值进行相减
        new_tensor = matrix_risk+risk_i_1 * one_hot_i_1
        matrix_risk = new_tensor+ risk_i * one_hot_i
        
        
        
#        rec_ij = a2 / a1
#        b1=tf.constant(1,dtype=tf.float32) 
#        max_rec_ij= tf.cond(tf.less( rec_ij,  b1), lambda:1-rec_ij, lambda:a0)
#        
#        Iij=tf.cond(tf.less( kk_ytime_train[i],  kk_ytime_train[i-1]), lambda: b1, lambda:a0)
#        mul_ij=Iij * max_rec_ij
#        
#        one_hot_I = tf.one_hot(i,shape[0],dtype=tf.float32)
#        
#        matrix_I = matrix_I+mul_ij*one_hot_I
#        matrix_max=matrix_max+max_rec_ij*one_hot_I
#
#            
##    cost2 = ( K.sum(K.dot(matrix_I,matrix_max)) )  
#    cost2 =( 1+weights) * ( K.sum(matrix_I) )    
#    cost3 = (1+weights) *( K.sum(matrix_risk) )  
    
#    cost2 =(1.0 + weights) * ( K.sum(matrix_I) )    
#    cost3 =(1.0 + weights) * ( K.sum(matrix_risk) )       
#        new_tensor = tf.concat([part1,val,part2], axis=0)

#        a1_list = []
#        a1_list.append(risk_i_1)
#        matrix_risk[i - 1] = tf.stack(a1_list)
#        risk[i - 1]=K.cast( risk_i_1,dtype=risk_i_1.dtype)
#        
#        risk[i]=K.cast( risk_i_1,dtype=risk_i.dtype)
#        risk_i_1= tf.cond(tf.less( kk_ytime_train[i],  kk_ytime_train[i-1]), lambda: risk[i - 1], lambda:(risk[i - 1] + risk[i]))
#        with tf.Session() as sess: 
#            sess.run(tf.global_variables_initializer()) 
#            y1=sess.run([risk_i_1]) 
#            print(y1)
#        risk_i_1= tf.cond(tf.greater_equal( kk_ytime_train[i],  kk_ytime_train[i-1]), true_fn = true_proc, false_fn = false_proc)
#        n1 = ytime[i]!=tf.constant(0)
#        n2 = ytime[i-1]!=tf.constant(0)
#        if kk_ytime_train[i]<=kk_ytime_train[i-1]:
#            # Push risk to the later time (earlier in array position)
#            risk[i - 1] = risk[i - 1] + risk[i]
#            risk[i] = 0
    
 
#    N=ytime.shape[0]
#    R_matrix = tf.zeros([N,N],tf.int32)
#    
#   
#    
#    for i in range(N - 1, 0, -1):
#        # Going from smallest survival times to largest
#        if ytime[i] == ytime[i - 1]:
#            # Push risk to the later time (earlier in array position)
#            risk[i - 1] = risk[i - 1] + risk[i]
#            risk[i] = 0
    n_observed = tf.reduce_sum(yevent,0)
    ytime_indicator = R_set(ytime)
    
    
#    R_matrix1=add_rankLoss_numba(ytime)
    
#    pool = ThreadPool()
#    R_matrix1=pool.map(add_rankLoss_numba(ytime))
#    pool.close()
#    pool.join()
    
#    for i in range(N):
#        for j in range(N):
#            R_matrix[i,j] = y_true[j] >= y_true[i]
            
   
	###if gpu is being used
#    if torch.cuda.is_available():
#        ytime_indicator = ytime_indicator.cuda()
	###
#    risk_set_sum = (ytime_indicator)*(K.exp(y_pred)) 
#    risk_set_sum =  K.sum(K.cast(ytime_indicator, dtype = y_pred.dtype)*(K.exp(xbeta)),axis=1)
    risk_set_sum =  K.sum(matrix_risk)
#    risk_set_sum =  K.sum(K.cast(K.exp(xbeta), dtype = ytime_indicator.dtype)*(ytime_indicator),axis=1,dtype=float32)
    
    diff = xbeta - K.log(risk_set_sum)
#    sum_diff_in_observed = K.cast(K.transpose(diff), dtype = yevent.dtype)*(yevent)
    sum_diff_in_observed = K.cast(yevent, dtype = diff.dtype)*(diff)
#    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
    cost = (- K.sum(sum_diff_in_observed))
#    cost = (- K.sum(sum_diff_in_observed / n_observed))
    
    
#    loss = (cost + regularize_layer_params(self.network,l1) * L1_reg
#            + regularize_layer_params(self.network, l2) * L2_reg
#        )
    
    λ1=1
    λ2=0.0
    λ3=0.0
    
#    return(λ1*cost+λ2*cost2+λ3*cost3)
    return(λ1*cost)


## C_index metric function

def c_index3(month,risk, status):

    c_index = concordance_index(np.reshape(month, -1), -np.reshape(risk, -1), np.reshape(status, -1))

    return c_index#def get_bi_lstm_model():  
#    model=Sequential()
#    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(1,input_dim0), merge_mode='concat'))
#
#    model.add(TimeDistributed(Dense(50, activation='tanh')))
#    model.add(Bidirectional(LSTM(50)))
#    model.add(Dropout(0.2))
#    model.add(Dense(1, activation='linear'))
#  
## Compile model
##    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, \
##        epsilon=None, decay=0.0, amsgrad=False), loss = neg_log_pl)
#    model.compile(optimizer='adam',  loss=neg_log_pl)
#    return model
############################################################################################
#def py_func(func, inp, Tout, stateful=True, name=None, grad_func=None):
#
#    grad_name = 'PyFuncGrad_' + str(np.random.randint(0, 1E+8))
#    tf.RegisterGradient(grad_name)(grad_func)
#    g = tf.get_default_graph()
#    with g.gradient_override_map({"PyFunc": grad_name}):
#        func1=tf.py_func(func, inp, Tout, stateful=stateful, name=name)
#        return func1    
###########################################################################################
def ordinal_loss0 (y_true, y_pred):
     
   
#    Y_hazard=k_ytime_train.get_value()
#    Y_survival=k_ystatus_train.get_value()
#   
#    t, H = unique_set(Y_hazard) # t:unique time. H original index.
#    
##    Y_survival=Y_survival.numpy()
##    risk=np.exp(score)
##    Y_hazard=Y_hazard.numpy()
#    actual_event_index = np.nonzero(Y_survival)[0]
#    H = [list(set(h) & set(actual_event_index)) for h in H]
#    n = [len(h) for h in H]
    
    
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
               # 做一个减法运算，将one_hot为一的变为原张量该位置的值进行相减
#                new_tensor = matrix_risk+risk_j * one_hot_j
                matrix_j = matrix_j+ rec * one_hot_more_j
        #        tf.reduce_sum(tf.one_hot(sum(H[13:],[]),n1,dtype=tf.float32),axis=0)
    matrix_risk= matrix_risk+ matrix_j 
    cost2 = K.sum(matrix_risk)#/(len(Hj))
    return cost2 
    
    
#    Y_true=Y_true.numpy()
#    Y_hazard0=Y_true[:,0]
#    Y_survival=Y_true[:,1]
##            Y_survival_train1=tf.reshape(Y_survival_train, [-1, batch_size,1])
##            Y_survival_train2=Y_survival_train1[-1,:,:] 
#    Y_hazard1M=tf.reshape(Y_hazard0, [-1, Y_true.shape[0],1])
#    Y_hazard=Y_hazard1M[-1,:,-1]
#    
#    t, H = unique_set(Y_hazard) # t:unique time. H original index.
#    score=score.numpy()
##    Y_survival=Y_survival.numpy()
##    risk=np.exp(score)
#    Y_hazard=Y_hazard.numpy()
#    actual_event_index = np.nonzero(Y_survival)[0]
#    H = [list(set(h) & set(actual_event_index)) for h in H]
#    n = [len(h) for h in H]
#    
#    matrix_ones = np.zeros([Y_hazard.shape[0], Y_hazard.shape[0]])
#    total = 0.0
#    for j in range(len(t)):
##        total_1 = np.sum(np.log(np.absolute(score[H[j]] + epsilon)))
#        total_1 = score[H[j]]
#        m = n[j]
#        total_2 = 0.0
#        for i in range(m):
#            matrix_ones[H[j],sum(H[j:],[])]=1
#            risk_more_j=np.exp(score[sum(H[j:],[])])
#            risk_j=np.exp(score[H[j]])
#            
#            rec=risk_j-risk_more_j
#            rec2=np.maximum(0,1-rec)
#            matrix_ones[H[j],sum(H[j:],[])]=rec2[:,-1]
#            subtotal = np.sum(matrix_ones[H[j],sum(H[j:],[])])
#            
##            subtotal = np.sum(np.exp(score[sum(H[j:],[])]) )
##            subtotal = np.log(np.absolute(subtotal + epsilon))
#            total_2 = total_2 + subtotal
#        total = total + total_2
#    return tf.to_float(total)  
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
def ordinal_loss (Y_true, score, epsilon=1e-8):
    Y_true=Y_true#.numpy()
    print('Y_true:',Y_true)
    Y_hazard=Y_true[:,0]
    print('Y_hazarde:',Y_hazard)
    Y_survival=Y_true[:,1]
#            Y_survival_train1=tf.reshape(Y_survival_train, [-1, batch_size,1])
#            Y_survival_train2=Y_survival_train1[-1,:,:] 
#    Y_hazard1M=tf.reshape(Y_hazard0, [-1, Y_true.shape[0],1])
#    Y_hazard=Y_hazard1M[-1,:,-1]
    
    t, H = unique_set(Y_hazard) # t:unique time. H original index.
    score=score#.numpy()
#    Y_survival=Y_survival.numpy()
#    risk=np.exp(score)
    Y_hazard=Y_hazard#.numpy()
    actual_event_index = np.nonzero(Y_survival)[0]
    H = [list(set(h) & set(actual_event_index)) for h in H]
    n = [len(h) for h in H]
    
    matrix_ones = np.zeros([Y_hazard.shape[0], Y_hazard.shape[0]])
    total = 0.0
    for j in range(len(t)):
#        total_1 = np.sum(np.log(np.absolute(score[H[j]] + epsilon)))
#        total_1 = score[H[j]]
        m = n[j]
        total_2 = 0.0
        for i in range(m):
            matrix_ones[H[j],sum(H[j:],[])]=1
            risk_more_j=np.exp(score[sum(H[j:],[])])
            risk_j=np.exp(score[H[j]])
            
            rec=risk_j-risk_more_j
            rec2=np.maximum(0,1-rec)
            matrix_ones[H[j],sum(H[j:],[])]=rec2[:,-1]
            subtotal = np.sum(matrix_ones[H[j],sum(H[j:],[])])
            
#            subtotal = np.sum(np.exp(score[sum(H[j:],[])]) )
#            subtotal = np.log(np.absolute(subtotal + epsilon))
            total_2 = total_2 + subtotal
        total = total + total_2
    return tf.to_float(total)  
###################################################################################################
###########################################################################################
def ordinal_loss_grad_numpy (Y_true, score, grad, epsilon=1e-8):
    Y_true=Y_true#.numpy()
    Y_hazard0=Y_true[:,0]
    Y_survival=Y_true[:,1]
#            Y_survival_train1=tf.reshape(Y_survival_train, [-1, batch_size,1])
#            Y_survival_train2=Y_survival_train1[-1,:,:] 
    Y_hazard1M=tf.reshape(Y_hazard0, [-1, Y_true.shape[0],1])
    Y_hazard=Y_hazard1M[-1,:,-1]
    
    t, H = unique_set(Y_hazard) # t:unique time. H original index.
    score=score#.numpy()
#    Y_survival=Y_survival.numpy()
#    risk=np.exp(score)
    Y_hazard=Y_hazard#.numpy()
    actual_event_index = np.nonzero(Y_survival)[0]
    H = [list(set(h) & set(actual_event_index)) for h in H]
    n = [len(h) for h in H]
    
    matrix_ones = np.zeros([Y_hazard.shape[0], Y_hazard.shape[0]])
    total = 0.0
    for j in range(len(t)):
#        total_1 = np.sum(np.log(np.absolute(score[H[j]] + epsilon)))
#        total_1 = score[H[j]]
        m = n[j]
        total_2 = 0.0
        for i in range(m):
            matrix_ones[H[j],sum(H[j:],[])]=1
            risk_more_j=np.exp(score[sum(H[j:],[])])
            risk_j=np.exp(score[H[j]])
            
            rec=risk_j-risk_more_j
            rec2=np.maximum(0,1-rec)
            matrix_ones[H[j],sum(H[j:],[])]=rec2[:,-1]
            subtotal = np.sum(matrix_ones[H[j],sum(H[j:],[])])
            
#            subtotal = np.sum(np.exp(score[sum(H[j:],[])]) )
#            subtotal = np.log(np.absolute(subtotal + epsilon))
            total_2 = total_2 + subtotal
        total = total + total_2
    dloss=np.sum(matrix_ones,axis=0)/100
    return np.float32(dloss)# * grad)
############################################################################################
def ordinal_loss_grad(op, grad):
   
   ys_true = op.inputs[0]
   ys_pred= op.inputs[1]
   
#    grad=op.inputs[4]
   tensor1=tf.py_func(ordinal_loss_grad_numpy, [  ys_true ,ys_pred, grad], grad.dtype),\
             tf.zeros(tf.shape(ys_pred)) 
#            tf.zeros(tf.shape( ys_true)), tf.zeros(tf.shape(ys_pred))
   return  tensor1
################################################################################################### 

def py_func(func, inp, Tout, stateful=True, name=None, grad_func=None):

    grad_name = 'PyFuncGrad_' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(grad_name)(grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        func1=tf.py_func(func, inp, Tout, stateful=stateful, name=name)
        return func1
###########################################################################################        

def ordinal_loss_tf(ys_true, ys_pred):
    # use tf.py_func
     loss = py_func(ordinal_loss,
        [ys_true, ys_pred], [tf.float32],
        name = "ordinal_loss",
        grad_func = ordinal_loss_grad)[0]

     return loss    
###########################################################################################        
#@tf.custom_gradient
#def ordinal_loss_tf(ys_true, ys_pred):
#    # use tf.py_func
#    loss=tf.py_function(func=ordinal_loss, inp=[ys_true, ys_pred], Tout=tf.float32) 
##    loss = tf.py_func(mse_numpy, [y, y_predict], tf.float32, name='my_mse')
#
#    def grad(dy):
#        return tf.py_func(func=ordinal_loss_grad_numpy, inp=[ys_true, ys_pred, dy], Tout=tf.float32, name='my_grad')
#
#    return loss, grad    
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
            precision = K.exp(-log_var[i])
            precision= tf.clip_by_value(precision, 0., 1.)
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
#            loss += K.sum(precision * lossA , -1)
            loss += K.sum(precision * lossA + log_var[i], -1)
        
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

c_index10=[]
for ij in range(10):
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
    
    #    Y_hazard1M=tf.reshape(Y_hazard0, [-1, Y_train.shape[0],1])
    #    Y_hazard=Y_hazard1M[-1,:,-1]
        
        t0, H0 = unique_set(Y_hazard0) # t:unique time. H original index.
        
        actual_event_index = np.nonzero(Y_survival)[0]
        H0 = [list(set(h) & set(actual_event_index)) for h in H0]
        ordinal_n = np.asarray([len(h) for h in H0])
        Hj=sum(H0[0:],[])
        
        k_ordinal_H = theano.shared(np.asarray(Hj),borrow=True)
        k_ordinal_t = theano.shared(t0,borrow=True)
        k_ordinal_n = theano.shared(ordinal_n,borrow=True)
     #########################################################################################################################   
     #############################################################################################################################    
    #    input_dim0 =theano.shared(input_dim,borrow=True)
    # Build model structure
        # gene Only
        gene_input = Input(name='gene_input', shape=(1,input_dim))
    #    out1=Bidirectional(LSTM(55,activation='linear',return_sequences=True,kernel_initializer=glorot_uniform(),kernel_regularizer=l2(reg),activity_regularizer=l2(0.001)), merge_mode='concat')(title_input)
        out_gene=Bidirectional(LSTM(100,return_sequences=True), merge_mode='concat')(gene_input)
    #    out2=TimeDistributed(Dense(50, activation='tanh'))(out1)
    #    out_gene=Bidirectional(LSTM(20))(out2)
        
    #    auxiliary_output = Dense(1, activation='linear', name='aux_output')(out_gene)  
    #    GRU(100,  activation='linear', return_sequences=True)(title_input)
         # clinic Only
        clinic_input = Input(name='clinic_input', shape=(1,input_dim))
        
        out_clinic=Bidirectional(LSTM(100,activation='tanh',return_sequences=False), merge_mode='concat')(clinic_input )
        auxiliary_output = Dense(1,activation='tanh', name='aux_output')(out_clinic) #sigmoid
    #    out_clinic=Bidirectional(LSTM(100,return_sequences=False), merge_mode='concat')(clinic_input )
    #    
        out21=TimeDistributed(Dense(50, activation='tanh'))( out_gene)
        out22=Bidirectional(LSTM(20,return_sequences=False))(out21)
    #    model.add(TimeDistributed(Dense(50, activation='tanh')))
    #    model.add(Bidirectional(LSTM(20)))
        # combined with GRU output
    #    input_ = Input(shape=(12,8))
       
    #    com = Concatenate(axis=1)([out_gene, out_clinic])
       
         
        out222=Dense(20, activation='linear')(out22)
    
        
    #    GRU(50, activation='tanh', return_sequences=False)(out1)
        out3=Dropout(0.1)(out22)
        main_output= Dense(1,activation='linear',name='main_output')(out3)
    #    main_output1=main_output[:,-1,:]
    #    auxiliary_output1=auxiliary_output[:,-1,:]
    #    y1_true = Input(shape=(output_dimM,), name='y1_true')
    #    y2_true = Input(shape=(output_dimA,), name='y2_true')
    #    model = Model(inputs=[gene_input,clinic_input],outputs=[main_output, auxiliary_output])
        y1_true = Input(shape=(2,), name='y1_true')
    #    y1_true = Input(shape=(output_dimM,), name='y1_true')
        y2_true = Input(shape=(2,), name='y2_true')
        out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, main_output, auxiliary_output])
        model =Model([gene_input,clinic_input,y1_true, y2_true], out)
        model.summary()
        model.compile(optimizer='adam', loss=None)
        
         #取某一层的输出为输出新建为model，采用函数模型
        dense1_layer_model = Model(inputs=model.input, outputs=[model.get_layer('main_output').output,model.get_layer('aux_output').output])
    #    dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('main_output').output)
        dense1_layer_model.summary()
        
        hist = model.fit([x_train,x_train, y_train,  ystatus2_train], batch_size = n1, epochs =1000)
        
        import pylab
        pylab.plot(hist.history['loss'])
        print([np.exp(K.get_value(log_var[0]))**0.5 for log_var in model.layers[-1].log_vars])
        
        predicted_main, predicted_aux = dense1_layer_model.predict([x_test,x_test,y_test, ystatus2_test],verbose=1)
        prediction =predicted_main+0*predicted_aux
        
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
    np.savetxt("brca_prediction1204_18lstm2222_epoch400_drop01_resnet.csv", ypred, delimiter=",")
    np.savetxt("brca_ytime_test1204_18lstm2222_epoch400_drop01_resnet.csv", time_new, delimiter=",")
    np.savetxt("brca_ystatus_test1204_18lstm2222_epoch400_drop01_resnet.csv", status_new, delimiter=",")
    np.savetxt("brca_ypred_train_median1204_18lstm2222_epoch400_drop01_resnet.csv", ypred_train, delimiter=",")
    
    df = pd.read_csv("brca_prediction1204_18lstm2222_epoch400_drop01_resnet.csv",header=None)    
    month=np.asarray(pd.read_csv("brca_ytime_test1204_18lstm2222_epoch400_drop01_resnet.csv",header=None)) 
    status=np.asarray(pd.read_csv("brca_ystatus_test1204_18lstm2222_epoch400_drop01_resnet.csv",header=None)) 
    
    
    #df=ypred
    #month=time_new
    #status=status_new
    #df1 = pd.read_csv("WAVE_preTest_scores.csv")    
    #dataset_init = np.asarray(df1)    # if only 1 column
    #dataX, dataY = create_interval_dataset(df, 1)    #这里的输入数据来源是csv文件
    
    risk=np.asarray(df)
    c_indices_lstm = c_index3(month, risk,status)
    #np.savetxt("c_indices_nn827.txt", c_indices_mlp, delimiter=",")
    np.save("c_indices",c_indices_lstm) 
    print(c_indices_lstm)
    data_a=np.load('c_indices.npy')
    c_index10.append(c_indices_lstm)
    print(ij)
np.savetxt("c_index10.csv", c_index10, delimiter=",")
print(sum(c_index10)/10)
