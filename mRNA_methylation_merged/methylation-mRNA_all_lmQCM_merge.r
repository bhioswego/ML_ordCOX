setwd("C:/2020download/00000000lmQCM-master/tests")

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

methylation= read.csv("eigengene_matrix_methylation.csv",header = T, fill = T, stringsAsFactors=FALSE)
mRNA= read.csv("eigengene_matrix_mRNAseq_RPKM_minClusterSize10.csv",header = T, fill = T, stringsAsFactors=FALSE)
s2_mRNA=ncol(mRNA)
methylation1=methylation #[,4:17]
colnames(methylation1)[1] <- "ID"
mRNA1= cbind(mRNA[,1],mRNA[,4:s2_mRNA])
colnames(mRNA1)[1] <- "ID"

methylation_mRNA <- merge(x=methylation1, y=mRNA1,by.x="ID", by.y="ID")

write.csv(methylation_mRNA,file = 'brca_Surv_data_methylation_mRNA_all_lmqcm.csv')
write.csv(methylation_mRNA[,-1],file = 'brca_Surv_data_methylation_mRNA_all.csv')

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
