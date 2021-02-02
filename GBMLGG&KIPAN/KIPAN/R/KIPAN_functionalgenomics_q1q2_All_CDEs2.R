# FUNCTIONAL GENOMICS PROJECT

setwd("C:/2019821topc/mRNA/KIPAN")



# Import clinical data
clinical0 <- read.table('All_CDEs.txt', sep='\t', fill=TRUE, header = TRUE)
clinical0_0 <- read.table('All_CDEs.txt', sep='\t', fill=TRUE, header = FALSE)


# clean up clinical1 and add a couple of interesting variables to Clinical
clinical00<- t(clinical0)
colnames (clinical00) <- clinical00[1,]
clinical1 <- clinical00[-1,]

clinical00_0<- t(clinical0_0)
#colnames (clinical00_0) <- clinical00_0[1,]
clinical1_0 <- clinical00_0[-1,]

days_followup0=clinical1_0[,6] #$days_to_last_followup
days_followup00=t(days_followup0)

days_death0=clinical1_0[,5]
days_death00=t(days_death0)

#clinical1 <- as.data.frame (clinical1)
#days_followup=as.data.frame (clinical1_0[,6]) #$days_to_last_followup
#days_death=as.data.frame (clinical1_0[,5]) #$days_to_death

#rownames(days_followup) <- NULL
days00 <- ifelse(is.na(days_death00),days_followup00,days_death00)
#######################################################################################################
days02=t(days00)
# Cox/Logrank analysis survival data

# library(survival)
# m=nrow(clinical1)
# 
# days = cbind(numeric(m))
# for (i in 1:m) {
#   if(is.na(clinical1$days_to_death[i])){
#     days[i] = as.data.frame(clinical1$days_to_last_followup[i])
#   }
#   else {
#     days[i] =as.data.frame( clinical1$days_to_death[i])
#   }
#days[i] <- ifelse(is.na(clinical1$days_to_death[i]),clinical1$days_to_last_followup[i],clinical1$days_to_death[i])
#}
#days <- ifelse(is.na(clinical1$days_to_death),clinical1$days_to_last_followup,clinical1$days_to_death)

# merging the two columns. It means: if days_to_death is NA, pick days_to_last_followup
# else (days_to_death is not NA), pick it
#status0<-clinical1$vital_status

#clinical1[which(clinical1$vital_status=="alive")][1] <-0
#status0[which(status0=="dead")] <-1
#Clinical22$vital_status<-status0
rowname=rownames(clinical1)

Clinical <- cbind (clinical1[,3], days02)
#colnames (Clinical[1]) <- c('vital_status')
Clinical[which(Clinical[,1]=="alive")] <-0
Clinical[which(Clinical[,1]=="dead")] <-1

Clinical <- cbind (rownames(clinical1),Clinical)
#colnames (Clinical)[1] <- c('ID')


write.table(Clinical, "KIPAN_clinical_data_3column.csv", sep=",", row.names=TRUE, col.names=TRUE)

# days is nbr of days between diagnosis and death (if vital_status == 1 = dead)
# or nbr of days between diagnosis and last followup (if vital_status == 0 = alive)

#Clinical$survival <- with(Clinical, Surv(days, clinical1$vital_status == 2))
# days are noted with a + if right-censored (ie. patient is alive)
# if no +, then patient is dead
# it is a 'Surv' object

#plot (survfit (survival ~ 1, data = Clinical), mark.time = TRUE, mark = 'line')

##################################################################################################


