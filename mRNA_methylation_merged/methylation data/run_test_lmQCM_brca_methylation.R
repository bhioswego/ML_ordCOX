
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

expr_brca0 = read.csv("brca_methylation2020_3233.csv",sep = ',',header=T) 
s1 = nrow(expr_brca0)
s2 = ncol(expr_brca0)
expr_brca1=cbind(expr_brca0[,2],expr_brca0[,5:s2])
rownames(expr_brca1) =expr_brca1[,1 ]
expr_brca1=expr_brca1[,-1]
expr_brca2 <- transpose(expr_brca1, fill=NA, ignore.empty=FALSE)
rownames(expr_brca2) = colnames(expr_brca1)
colnames(expr_brca2) = rownames(expr_brca1)
head(expr_brca2)
expr_brca3=data.matrix(expr_brca2)
write.csv(expr_brca2,file = 'expr_brca_mRNA.csv')

QCMObject=lmQCM(expr_brca2,minClusterSize=20)

#QCMObject2=localMaximumQCM(expr_brca3)
eigengene_matrix <- transpose(QCMObject@eigengene.matrix, fill=NA, ignore.empty=FALSE)
rownames(eigengene_matrix) = colnames(QCMObject@eigengene.matrix)
data=cbind(expr_brca0[,3:4],eigengene_matrix)
write.csv(data,file = 'eigengene_matrix_methylation10.csv')

# data(sample.ExpressionSet)
# data = assayData(sample.ExpressionSet)$exprs
# 
# lmQCM(data)