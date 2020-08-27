
setwd("C:/2020download/lmQCM-master/tests")
################################################1. ??????R??????????????????
rm(list = ls())
library(lmQCM)
library(Biobase)

library(readxl)
library(cghRA)
library(dplyr)
library(stringr)
library(ggplot2)
library(data.table)
# expr_brca0 = read.csv("BRCA_mRNAseq_RPKM.csv",sep = ',',header=T) 
erged_data= read.csv("BRCA_mRNAseq_RPKM.csv",header = T, fill = T, stringsAsFactors=FALSE)
s1 = nrow(erged_data)
s2 = ncol(erged_data)
# data_matrix=erged_data[1:s1,42:s2]
name=erged_data[,1]

erged_data33=erged_data[,33]
erged_data33[which(erged_data33=="LIVING")] <-0
erged_data33[which(erged_data33=="DECEASED")] <-1
erged_data33<-as.numeric(erged_data33)


erged_data3233=cbind(erged_data[,32],erged_data33)

mydata3233 = data.matrix(erged_data3233)
s1_erged_data = nrow(erged_data)
s2_erged_data = ncol(erged_data)
#brca_data<-matrix(nrow=nrow,ncol=ncol)
data_matrix=erged_data[1:s1_erged_data,42:s2_erged_data]

brca_data1=0
brca_data1=cbind(name,mydata3233,data_matrix)
#brca_data11=filter(brca_data1,brca_data1!="NA")
brca_data1_1=filter(brca_data1,brca_data1[ ,2]!="NA")
brca_data1_11=filter(brca_data1_1,brca_data1_1[ ,3]!="NA")

s2_11_col = ncol(brca_data1_11)

#brca_data2=cbind(name,data_matrix,erged_data[,33])


# brca_data3=cbind(name,data_matrix[,1:16300],erged_data[,33])
# brca_data4=filter(brca_data3,brca_data3[ ,16302]!="NA")
# 
datExpr0 <- brca_data1_11[,4:s2_11_col] ## top 5000 mad genes
# #datExpr00 <- na.omit(datExpr0)
# # 
# # #  removed genes in which  had gene expression values of 0
datExpr1<- transpose( datExpr0, fill=NA, ignore.empty=FALSE)
datExpr11 <- na.omit(datExpr1)
#datExpr2=filter(datExpr1,datExpr1!="NA")
#datExpr3=datExpr11[rowSums(datExpr11==0)==0,]
#is.na(datExpr12) <- !datExpr12
#datExpr12<-datExpr1[-1,]

datExpr11[datExpr11==0]<-NA
datExpr112 <- na.omit(datExpr11)
#datExpr11[which(rowSums(datExpr11==0)==0),]

datExpr4<- transpose( datExpr112, fill=NA, ignore.empty=FALSE)
datExpr <- datExpr4

s12 = nrow(datExpr)
s22 = ncol(datExpr)
#expr_brca1=datExpr
expr_brca1=cbind(brca_data1_11[,1],datExpr)
rownames(expr_brca1) =expr_brca1[,1 ]
expr_brca1=expr_brca1[,-1]
expr_brca2 <- transpose(expr_brca1, fill=NA, ignore.empty=FALSE)
rownames(expr_brca2) = colnames(expr_brca1)
colnames(expr_brca2) = rownames(expr_brca1)
head(expr_brca2)

#expr_brca22<-as.matrix(expr_brca2)

#expr_brca2223<-as.numeric(unlist(expr_brca2))

QCMObject=lmQCM(expr_brca2,minClusterSize=10)

#QCMObject2=localMaximumQCM(expr_brca3)
eigengene_matrix <- transpose(QCMObject@eigengene.matrix, fill=NA, ignore.empty=FALSE)
rownames(eigengene_matrix) = colnames(QCMObject@eigengene.matrix)
data=cbind(brca_data1_11[,2:3],eigengene_matrix)
write.csv(data,file = 'eigengene_matrix_mRNAseq_RPKM_minClusterSize10.csv')

