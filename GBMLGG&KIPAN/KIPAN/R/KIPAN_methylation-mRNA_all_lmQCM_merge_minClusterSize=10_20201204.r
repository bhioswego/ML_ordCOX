setwd("C:/2019821topc/mRNA/KIPAN")

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

methylation= read.csv("KIPAN_eigengene_matrix_Methylation_only_minClusterSize=10_20201204.csv",header = T, fill = T, stringsAsFactors=FALSE)
mRNA= read.csv("KIPAN_eigengene_matrix_mRNAseq_minClusterSize=10_20201204.csv",header = T, fill = T, stringsAsFactors=FALSE)
s1_methylation=ncol(methylation)
s2_mRNA=ncol(mRNA)
methylation1=methylation #[,4:17]

methylation1= methylation1[,2:s1_methylation]
colnames(methylation1)[1] <- "ID"

mRNA1= cbind(mRNA[,2],mRNA[,5:s2_mRNA])
colnames(mRNA1)[1] <- "ID"

methylation_mRNA <- merge(x=methylation1, y=mRNA1,by.x="ID", by.y="ID")

write.csv(methylation_mRNA,file = 'KIPAN_Surv_data_methylation_mRNA_all_lmqcm_minClusterSize=10_20201204.csv')
write.csv(methylation_mRNA[,-1],file = 'KIPAN_Surv_data_methylation_mRNA_all_lmqcm_minClusterSize=10_20201204_no_ID.csv')

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
