setwd("C:/2019821topc/mRNA/GBMLGG/20201224")

#install.packages("cghRA")

#install.packages("dplyr")

#install.packages("stringr")

#install.packages("ggplot2")

#install.packages("data.table")
#install.packages("readxl")

library(readxl)
library(cghRA)
library(dplyr)
library(stringr)
library(ggplot2)
library(data.table)

methylation= read.csv("GBMLGG_eigengene_matrix_Methylation_gamma =none_minClusterSize=10_20201224.csv",header = T, fill = T, stringsAsFactors=FALSE)
mRNA= read.csv("GBMLGG_eigengene_matrix_mRNAseq_gamma=none_minClusterSize=10_2020124.csv",header = T, fill = T, stringsAsFactors=FALSE)
clinical= read.csv("GBMLGG_clinical_data_3column_time_dotted_20201127_2.csv",header = T, fill = T, stringsAsFactors=FALSE)
clinical$ID=toupper(clinical$ID)
#colnames(clinical1)[1] <- "ID"
#colnames (clinical)[2] <- c('vital_status') 
#colnames (clinical)[3] <- c('days')

s1_methylation=ncol(methylation)
s2_mRNA=ncol(mRNA)
#methylation1=methylation #[,4:17]

methylation1= cbind(methylation[,2],methylation[,5:s1_methylation])
colnames(methylation1)[1] <- "ID"

#mRNA1= mRNA
mRNA1= mRNA[,2:s2_mRNA]
colnames(mRNA1)[1] <- "ID"
rownames(mRNA1) <- NULL
mRNA1$ID=toupper(mRNA1$ID)
methylation_mRNA1 <- merge(x=mRNA1, y=methylation1,by.x="ID", by.y="ID")
s3=ncol(methylation_mRNA1)
methylation_mRNA2= cbind(methylation_mRNA1[,1],methylation_mRNA1[,4:s3])
colnames(methylation_mRNA2)[1] <- "ID"

methylation_mRNA3 <- merge(x=clinical, y=methylation_mRNA2,by.x="ID", by.y="ID")

write.csv(methylation_mRNA3,file = 'GBMLGG_new_time_methylation_mRNA10_all_lmqcm_gamma=none_minClusterSize=10_202012024_2.csv')
#write.csv(methylation_mRNA[,-1],file = 'GBMLGG_Surv_data_methylation_mRNA_all_lmqcm_no_ID.csv')

# #tableGenes <- read_excel("BRCAMergedWAVE.xlsx")
# tableGenes <- read.table("BRCA.mRNAseq_RPKM.txt", sep="\t",
#                                header = F, fill = T, stringsAsFactors=FALSE,quote = "")
# tableClin <- read.table("brca_tcga_clinical_data.tsv", sep="\t", 
#                          header = T, fill = T, stringsAsFactors=FALSE)
# 
# tableGenesT <- transpose(tableGenes, fill=NA, ignore.empty=FALSE)
# 
# colnames(tableGenesT) = tableGenesT[1, ] # the first row will be the header 第一行提升作为头，
# tableGenesT = tableGenesT[-1, ] #去掉第一行
# 
# tableGenesT22=tableGenesT$HYBRIDIZATION_R
# #tableGenesT$Hybridization_REF<-substring(tableGenesT22,1,15)
# 
# tableClin22=tableClin$Sample.ID
