# ML_ordCOX
The source code is an implementation of our method described in the paper "Isabelle Bichindaritz, Guanghui Liu, and Christopher Bartlett. Integrative Survival Analysis of Breast Cancer with Gene Expression and DNA Methylation Data". 

Because github limit that the uploaded file size cannot exceed 100Mb, two original mRNA and methylation datasets fail to upload in github repository. So the whole source code for the proposed method and the datasets have been put online available at https://pan.baidu.com/s/1jQP5e-EBe0FOhvf7s5BIkw (extracted code is: LGH0) for free academic use. You will be appreciated to download from this address.

--Dependence

Before running this code, you need install R and python. In our experiment, R3.62 and python 3.6 or more advanced version are tested. This code is tested in WIN7/win10 64 Bit. It should be able to run in other Linux or Windows systems.

--How to run

step1. In R, in folder: \ML_ordCOX\mRNA_methylation_merged\mRNA data, first,run run_test_lmQCM_mRNA.r. Similarly, in folder: \ML_ordCOX\mRNA_methylation_merged\methylation data, run run_test_lmQCM_brca_methylation.r. If the code runs successfully, the extracted mRNA features and methylation features will be obtained using lmQCM method, respectively. We run methylation-mRNA_all_lmQCM_merge.r, and combine mRNA and methylation features in sequence. Finally, we can obtain a 133-dimensional feature vector which will be viewed as integrated gene feature input.

step2. After obtaining the specific feature representation in step1, copy the integrated features to fold \ML_ordCOX\survival analysis. In Python, run brca_methylation_mRNA_lmqcm.py in fold \ML_ordCOX\survival analysis.

--Output

if the code runs successfully, the results will be placed in \LSTM-COX-CODE\survival analysis\c_indices.npy.
