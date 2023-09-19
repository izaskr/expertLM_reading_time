# This code analyses the effect of background knowledge on reading behavior, specifically on reading measures.
# It reads in the .csv file where each line is a word. 
# Both groups reading BOTH domains

library(tidyr)
library(lme4)
library(psych)
library(ggplot2)
library(lmerTest)
library(stats)
library(car)
library(dplyr)
library(cowplot)


# SET working directory
setwd("/home/skrjanec/finetune_gpt2-de")

# READ IN DATA, PoTeC corpus (https://www.zora.uzh.ch/id/eprint/212186/) preprocessed by Iza Skrjanec to include surprisal estimates
df <- read.csv("data/large/df_by_2Textdomain_word_BOTH.csv")

# code the categorical variables as factors, the rest as numeric
df$WordStr <- as.factor(df$WordStr)
df$TextDomain <- as.factor(df$TextDomain)
df$TextID <- as.factor(df$TextID)
df$SubjectID <- as.factor(df$SubjectID)
df$DomainMajor <- as.factor(df$DomainMajor)
df$Major <- as.factor(df$Major)
df$ExpertStr <- as.factor(df$ExpertStr)
df$ExpertStatus <- as.factor(df$ExpertStatus)
df$TechnicalTerm <- as.factor(df$TechnicalTerm)
df$UniPOS <- as.factor(df$UniPOS)
df$SSTPOS <- as.factor(df$SSTPOS)
df$FixBin <- as.factor(df$FixBin)
df$RR <- as.factor(df$RR)
df$DependencyType <- as.factor(df$DependencyType)

df$NdepLeft <- as.numeric(df$NdepLeft)
df$NdepRight <- as.numeric(df$NdepRight)
df$DistanceToHead <- as.numeric(df$DistanceToHead)
df$TextCompr <- as.numeric(df$TextCompr)
df$BackKnow <- as.numeric(df$BackKnow)
df$WordIndexInText <- as.numeric(df$WordIndexInText)
df$WordIndexInSentence <- as.numeric(df$WordIndexInSentence)
df$SentenceIndex <- as.numeric(df$SentenceIndex)
df$FFD <- as.numeric(df$FFD)
df$FD <- as.numeric(df$FD)
df$FPRT <- as.numeric(df$FPRT)
df$TFT <- as.numeric(df$TFT)
df$RRT <- as.numeric(df$RRT)
df$RPD_exc <- as.numeric(df$RPD_exc)
df$RPD_inc <- as.numeric(df$RPD_inc)

df$TRC_out <- as.numeric(df$TRC_out)
df$TRC_in <- as.numeric(df$TRC_in)
df$WordLen <- as.numeric(df$WordLen)
df$FreqLemma <- as.numeric(df$FreqLemma)
df$FreqDoc <- as.numeric(df$FreqDoc)
df$GPT2 <- as.numeric(df$GPT2)
df$BioGPT <- as.numeric(df$BioGPT)
df$PhysGPT <- as.numeric(df$PhysGPT)

# create a unique itemID for every word: a combination of textID and WordIndexInText. 
df$UniqueWordID <- paste(df$TextID, df$WordIndexInText, sep="-")

# create a new column: a variable that's an interaction between TextDomain and Technical Term
#df$DomainAndTerm <- paste(df$TextDomain, df$TechnicalTerm, sep="-")

####### SUM CODING OF THE MAIN CONDITION EXPLORED: DomainMajor -- create a new variable DomainMajorSum
df$MajorSum= df$DomainMajor # first create a new column, a copy of the current DomainMajor (levels "bio" and "phys")
contrasts(df$MajorSum) = contr.sum(2)
contrasts(df$MajorSum)
#[,1]
#bio     1
#phys   -1


### Create a new variable: Expertise 
df$Expertise <- ifelse(df$DomainMajor=="phys" & df$TextDomain=="physics" | df$DomainMajor=="bio" & df$TextDomain=="biology", "expert", "novice")
df$Expertise <- as.factor(df$Expertise)
# sum code it
contrasts(df$Expertise) <- contr.sum(2)
contrasts(df$Expertise)
#      [,1]
#expert    1
#novice   -1

# if major==phys, take value from PhysGPT, otherwise from BioGPT
df$specSurprisal <- ifelse(df$DomainMajor=="phys", df$PhysGPT, df$BioGPT)

df_x <- df
df_x <- subset(df_x, df_x$SubjectID=="reader0")

####################### DATA CLEANING AND FILTERING
# firstly remove all words that appear at the beginning of the first sentence
# Ideally, this should be done for each slide (first word on the slide, last word on the slide), but we don't have this information in the dataset
nrow(df)  # 142125

ggplot(df, aes(x=TextDomain, y=FPRT)) + geom_boxplot() + ggtitle("Before any cleaning") +
  xlab("Text domain") + ylab("Raw FPRT")
ggplot(df, aes(x=TextDomain, y=TFT)) + geom_boxplot() + ggtitle("Before any cleaning") +
  xlab("Text domain") + ylab("Raw TFT")
ggplot(df, aes(x=TextDomain, y=log(FPRT))) + geom_boxplot() + ggtitle("Before any cleaning") +
  xlab("Text domain") + ylab("Log FPRT")
ggplot(df, aes(x=TextDomain, y=log(TFT))) + geom_boxplot() + ggtitle("Before any cleaning") +
  xlab("Text domain") + ylab("Log TFT")


############################ TODO: Fixation Rate: binary variable as a response; the model below didn't converge for over 5 minutes, so I'll return to this later
# log.model <- glm(Type.new ~ FirstLat, data = hurricanes, family = 'binomial')
#df_for_logreg <- subset(df)
#m_fixbin1 <- glmer(FixBin ~ WordLen + log(FreqLemma+1) + GPT2 + WordIndexInSentence + TechnicalTermSum + DomainMajorSum + (1|SubjectID) + (1|UniqueWordID),
#                data=df_for_logreg, family="binomial")

#summary(m_fixbin1)
###################################################



# remove the first word of each text: common practice in ET data cleaning.
# it would be good to remove also slide-final words
df <- subset(df, df$WordIndexInText != 1)
nrow(df)  # 141225

# Remove datapoints with total fixation time of 0. These datapoints were not fixated, therefore not interesting in our analysis
df <- subset(df, df$TFT != 0)
nrow(df)  # 127113

# also remove words with FPRT = 0 (were not fixated at all in the first pass) 
df <- subset(df, df$FPRT != 0)
nrow(df)  # 102948


################# LOG SCALE FOR CONTINUOUS READING MEASURES
df$LogFPRT <- log(df$FPRT)
df$LogFD <- log(df$FD)
df$LogTFT <- log(df$TFT)
df$LogFFD <- log(df$FFD)
df$LogRPD_inc <- log(df$RPD_inc)

################################################################################### ANALYSIS FOR ALL WORDS ################################################
# Research question: is background knowledge predictive of reading behavior encoded as ET measures?
# 

df_allwords <- subset(df)

#### data cleaning: exclude data points based on FPRT and TFT
d <- 3

mean_fprt <- mean(df_allwords$LogFPRT)
sd_fprt <- sd(df_allwords$LogFPRT)
border1_fprt <- mean_fprt - d * sd_fprt
border2_fprt <- mean_fprt + d * sd_fprt

mean_tft <- mean(df_allwords$LogTFT)
sd_tft <- sd(df_allwords$LogTFT)
border1_tft <- mean_tft - d * sd_tft
border2_tft <- mean_tft + d * sd_tft


#ggplot(df_allwords, aes(x=LogFPRT)) + geom_density() + geom_vline(aes(xintercept=border1_fprt), color="blue", linetype="dashed", size=1) + 
#  geom_vline(aes(xintercept=border2_fprt), color="blue", linetype="dashed", size=1)

#ggplot(df_allwords, aes(x=LogTFT)) + geom_density() + geom_vline(aes(xintercept=border1_tft), color="red", linetype="dashed", size=1) + 
#  geom_vline(aes(xintercept=border2_tft), color="red", linetype="dashed", size=1)


################################################################ remove data points that do not lie within borders

df_allwords_clean <- subset(df_allwords, LogFPRT >= border1_fprt & LogFPRT <= border2_fprt) 
nrow(df_allwords_clean) # nrow 101,893

df_allwords_clean <- subset(df_allwords_clean, LogTFT >= border1_tft & LogTFT <= border2_tft)  
nrow(df_allwords_clean) # nrow 101,702


#### Plotting after cleaning
ggplot(df_allwords_clean, aes(x=TextDomain, y=FPRT)) + geom_boxplot() + ggtitle("After any cleaning, d=2.5") +
  xlab("Text domain") + ylab("Raw FPRT")
ggplot(df_allwords_clean, aes(x=TextDomain, y=TFT)) + geom_boxplot() + ggtitle("After any cleaning, d=2.5") +
  xlab("Text domain") + ylab("Raw TFT")
ggplot(df_allwords_clean, aes(x=TextDomain, y=LogFPRT)) + geom_boxplot() + ggtitle("After any cleaning, d=2.5") +
  xlab("Text domain") + ylab("Log FPRT")
ggplot(df_allwords_clean, aes(x=TextDomain, y=LogTFT)) + geom_boxplot() + ggtitle("After any cleaning, d=2.5") +
  xlab("Text domain") + ylab("Log TFT")

ggplot(df_allwords_clean, aes(y=LogFPRT)) + geom_boxplot() + ggtitle("After any cleaning, d=2.5")  + ylab("Log FPRT")
ggplot(df_allwords_clean, aes(y=LogTFT)) + geom_boxplot() + ggtitle("After any cleaning, d=2.5")  + ylab("Log TFT")

## Q-Q plots
ggplot(df_allwords_clean, aes(sample=LogFPRT)) + stat_qq() + stat_qq_line() + ggtitle("After any cleaning, d=2.5")
ggplot(df_allwords_clean, aes(sample=LogTFT)) + stat_qq() + stat_qq_line() + ggtitle("After any cleaning, d=2.5")

##################################### CENTER THE CONTINUOUS PREDICTORS AROUND THEIR RESPECTIVE MEANS
mean_wordlen <- mean(df_allwords_clean$WordLen)
mean_logfreq <- mean(log(df_allwords_clean$FreqLemma+1))
mean_surprisal_gpt2 <- mean(df_allwords_clean$GPT2)
mean_wordinsentindex <- mean(df_allwords_clean$WordIndexInSentence)

# create new columns - centered predictors
df_allwords_clean$cWordLen <- df_allwords_clean$WordLen - mean_wordlen
df_allwords_clean$cLogFreqLemma <- log(df_allwords_clean$FreqLemma+1) - mean_logfreq
df_allwords_clean$cGPT2 <- df_allwords_clean$GPT2 - mean_surprisal_gpt2
df_allwords_clean$cWordIndexInSentence <- df_allwords_clean$WordIndexInSentence - mean_wordinsentindex


## analysis of number of data points per subject and per text domain
counts <- count(df_allwords_clean, SubjectID)
mean(counts$n)
min(counts$n)
max(counts$n)

counts2<- count(df_allwords_clean, TextDomain)
mean(counts2$n)


############################################################################## CATEGORICAL PREDICTOR CODING ########################################

# df_allwords_clean$MajorSum is already sum coded
#      [,1]
#bio     1
#phys   -1

# TextDomain has to be sum coded

df_allwords_clean$TextDomainSum <- df_allwords_clean$TextDomain
contrasts(df_allwords_clean$TextDomainSum) <- contr.sum(2)
contrasts(df_allwords_clean$TextDomainSum)
#        [,1]
#biology    1
#physics   -1


# Terminology - a binary variable for common (level0) and terminology (level1 and level2) words. Use SUM CODING

df_allwords_clean$TerminologyBinarySum <- ifelse(df_allwords_clean$TechnicalTerm == 0, "2common", "1terminology")
df_allwords_clean$TerminologyBinarySum <- as.factor(df_allwords_clean$TerminologyBinarySum)
contrasts(df_allwords_clean$TerminologyBinarySum) <- contr.sum(2)
contrasts(df_allwords_clean$TerminologyBinarySum)
#            [,1]
#1terminology    1
#2common        -1

# plotting

################################# FPRT

m_bt_bio_t0 <- mean(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="2common"])
m_bt_bio_t12 <- mean(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="1terminology"])

m_bt_phys_t0 <- mean(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="2common"])
m_bt_phys_t12 <- mean(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="1terminology"])


m_pt_bio_t0 <- mean(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="2common"])
m_pt_bio_t12 <- mean(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="1terminology"])

m_pt_phys_t0 <- mean(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="2common"])
m_pt_phys_t12 <- mean(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="1terminology"])

sd_bt_bio_t0 <- sd(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="2common"])
sd_bt_bio_t12 <- sd(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="1terminology"])

sd_bt_phys_t0 <- sd(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="2common"])
sd_bt_phys_t12 <- sd(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="1terminology"])


sd_pt_bio_t0 <- sd(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="2common"])
sd_pt_bio_t12 <- sd(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="1terminology"])

sd_pt_phys_t0 <- sd(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="2common"])
sd_pt_phys_t12 <- sd(df_allwords_clean$FPRT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="1terminology"])

v_textdomain <- c("Biology texts", "Biology texts","Biology texts","Biology texts","Physics texts", "Physics texts","Physics texts","Physics texts")
v_meanfprt <- c(m_bt_bio_t0, m_bt_bio_t12, m_bt_phys_t0, m_bt_phys_t12, m_pt_bio_t0, m_pt_bio_t12, m_pt_phys_t0, m_pt_phys_t12)
v_sdfprt <- c(sd_bt_bio_t0, sd_bt_bio_t12, sd_bt_phys_t0, sd_bt_phys_t12, sd_pt_bio_t0, sd_pt_bio_t12, sd_pt_phys_t0, sd_pt_phys_t12)
v_major = c("Biologist", "Biologist", "Physicist", "Physicist", "Biologist", "Biologist", "Physicist", "Physicist")
v_termin <- c("Common", "Technical", "Common", "Technical","Common", "Technical","Common", "Technical")

plot_df <- data.frame(TextDomain=v_textdomain, Mean_FPRT=v_meanfprt, SD=v_sdfprt, Major=v_major, Terminology=v_termin)

plot1 <- ggplot(plot_df, aes(x=Major, y=Mean_FPRT, fill=Terminology)) + 
  facet_wrap(~ TextDomain) +   geom_bar(position = "dodge", stat="identity", colour="black", show.legend = FALSE) + theme_bw() +
  geom_errorbar(aes(ymin=Mean_FPRT-SD, ymax=Mean_FPRT+SD), width=0.3, position=position_dodge(0.9)) + facet_wrap(~ TextDomain) + 
  labs(y= "Mean FPRT")

####################################### TFT

m_tftbt_bio_t0 <- mean(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="2common"])
m_tftbt_bio_t12 <- mean(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="1terminology"])

m_tftbt_phys_t0 <- mean(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="2common"])
m_tftbt_phys_t12 <- mean(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="1terminology"])


m_tftpt_bio_t0 <- mean(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="2common"])
m_tftpt_bio_t12 <- mean(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="1terminology"])

m_tftpt_phys_t0 <- mean(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="2common"])
m_tftpt_phys_t12 <- mean(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="1terminology"])

sd_bt_bio_t0 <- sd(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="2common"])
sd_bt_bio_t12 <- sd(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="1terminology"])

sd_bt_phys_t0 <- sd(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="2common"])
sd_bt_phys_t12 <- sd(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="biology" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="1terminology"])


sd_pt_bio_t0 <- sd(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="2common"])
sd_pt_bio_t12 <- sd(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="bio" & df_allwords_clean$TerminologyBinarySum=="1terminology"])

sd_pt_phys_t0 <- sd(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="2common"])
sd_pt_phys_t12 <- sd(df_allwords_clean$TFT[df_allwords_clean$TextDomain=="physics" & df_allwords_clean$DomainMajor=="phys" & df_allwords_clean$TerminologyBinarySum=="1terminology"])

v_textdomain <- c("Biology texts", "Biology texts","Biology texts","Biology texts","Physics texts", "Physics texts","Physics texts","Physics texts")
v_meantft <- c(m_tftbt_bio_t0, m_tftbt_bio_t12, m_tftbt_phys_t0, m_tftbt_phys_t12, m_tftpt_bio_t0, m_tftpt_bio_t12, m_tftpt_phys_t0, m_tftpt_phys_t12)
v_sdtft <- c(sd_bt_bio_t0, sd_bt_bio_t12, sd_bt_phys_t0, sd_bt_phys_t12, sd_pt_bio_t0, sd_pt_bio_t12, sd_pt_phys_t0, sd_pt_phys_t12)
v_major = c("Biologist", "Biologist", "Physicist", "Physicist", "Biologist", "Biologist", "Physicist", "Physicist")
v_termin <- c("Common", "Technical", "Common", "Technical","Common", "Technical","Common", "Technical")

plot_df_tft <- data.frame(TextDomain=v_textdomain, Mean_TFT=v_meantft, SD=v_sdtft, Major=v_major, Terminology=v_termin)

plot2 <- ggplot(plot_df_tft, aes(x=Major, y=Mean_TFT, fill=Terminology)) + 
  facet_wrap(~ TextDomain) +   geom_bar(position = "dodge", stat="identity", colour="black") + theme_bw() +
  geom_errorbar(aes(ymin=Mean_TFT-SD, ymax=Mean_TFT+SD), width=0.3, position=position_dodge(0.9)) + facet_wrap(~ TextDomain) + 
  theme(legend.position = c(0.19, 0.885),legend.background = element_rect(fill = "white")) + labs(y= "Mean TRT")

#### put the plots side by side and label them A and B
plot_grid(plot1, plot2, labels="AUTO")


# center BioGPT and PhysGPT, also spec
df_allwords_clean$cBioGPT <- df_allwords_clean$BioGPT - mean(df_allwords_clean$BioGPT)
df_allwords_clean$cPhysGPT <- df_allwords_clean$PhysGPT - mean(df_allwords_clean$PhysGPT)
df_allwords_clean$cspecGPT <- df_allwords_clean$specSurprisal - mean(df_allwords_clean$specSurprisal)
# contrasts
contrasts(df_allwords_clean$MajorSum)
#      [,1]
#bio     1
#phys   -1

##################################################################################################################################################################################
##################################################################################################################################################################################
##################################################################################################################################################################################
##################################################################################################################################################################################

################# Research  question 1 RQ1

# fit the LogFPRT model, random structure: 1|subject, 1 + expertise|wordID
# covariates: length, frequency, general surprisal, word index in sentence
# variable of interest: expertise, terminology

############################# A model according to Merel' suggestion
# both text domains in a single model
# use Expertise - and leave out TextDomain and Major
# Use also cspecGPT instead of BioGPT or PhysGPT

#### FPRT
m_fprt1_new <- lmer(LogFPRT ~ cWordLen + cLogFreqLemma + cWordIndexInSentence + cGPT2 + Expertise*TerminologyBinarySum + (1|SubjectID) + (1 + Expertise|UniqueWordID),
                    data=df_allwords_clean)
summary(m_fprt1_new)


back1 <- step(m_fprt1_new)
final_fprt1 <- get_model(back1)  # everything, but word feequency
summary(final_fprt1)

#### TFT
m_tft1_new <- lmer(LogTFT ~ cWordLen + cLogFreqLemma + cWordIndexInSentence + cGPT2 + Expertise*TerminologyBinarySum + (1|SubjectID) + (1 + Expertise|UniqueWordID),
                   data=df_allwords_clean)
summary(m_tft1_new)
back2 <- step(m_tft1_new)
final_tft1 <- get_model(back2)
summary(final_tft1)

# save the TFT model
saveRDS(m_tft1_new, "/home/skrjanec/finetune_gpt2-de/code_for_emdpa/models_rds/tft_rq1.rds")

# plot the effect sizes
pd <- position_dodge(width = .2)
effects::Effect(c("Expertise"),
                final_tft1, se = T,
                confidence.level = 0.95,
                xlevels = list(Expertise = c(-1,1))
) %>% 
  as.data.frame %>% 
  #code back into categories
  print %>% 
  #mutate(expertise = ifelse(Expertise == -1, "novice", "expert")) %>%
  print %>% 
  ggplot(
    aes(x = Expertise,
        y = fit,
        
        #group = Expertise # add this? Not sure if it will help
    )
  ) +
  geom_errorbar(
    aes(ymin = fit-se,
        ymax= fit+se),
    size = .25,
    width=.3,
    position = pd
  ) +
  geom_point(
    # aes(shape = conn),
    size=2,
    position = pd) +
  #geom_line(
  #  aes(linetype = Expertise), #remove capital letter
  #  size = .5,
  #  position = pd) +
  theme_bw() +
  theme(text = element_text(size = 24)) +
  xlab("Expertise") +
  ylab("Fitted Log RT") +
  scale_linetype_discrete(name="Expertise",
                          labels=c("expert", "novice")
  ) +
  # ylim(550, 700) +
  NULL 



# plot for terminology too
effects::Effect(c("TerminologyBinarySum"),
                final_tft1, se = T,
                confidence.level = 0.95,
                xlevels = list(TerminologyBinarySum = c(-1,1))
) %>% 
  as.data.frame %>% 
  #code back into categories
  print %>% 
  #mutate(expertise = ifelse(Expertise == -1, "novice", "expert")) %>%
  print %>% 
  ggplot(
    aes(x = TerminologyBinarySum,
        y = fit,
        
        #group = Expertise # add this? Not sure if it will help
    )
  ) +
  geom_errorbar(
    aes(ymin = fit-se,
        ymax= fit+se),
    size = .25,
    width=.3,
    position = pd
  ) +
  geom_point(
    # aes(shape = conn),
    size=2,
    position = pd) +
  #geom_line(
  #  aes(linetype = Expertise), #remove capital letter
  #  size = .5,
  #  position = pd) +
  theme_bw() +
  theme(text = element_text(size = 24)) +
  xlab("Terminology") +
  ylab("Fitted Log RT") +
  scale_x_discrete(labels = c("terminology","common")) +
  #scale_linetype_discrete(name="Terminology",
  #                        labels=c("technical", "common")
  #) +
  # ylim(550, 700) +
  NULL 


# plot the interaction: expetise and terminology
effects::Effect(c("Expertise", "TerminologyBinarySum"),
                final_tft1, se = T,
                confidence.level = 0.95,
                xlevels = list(Expertise = c(-1,1),
                               TerminologyBinarySum= c(-1,1))
) %>% 
  as.data.frame %>% 
  #code back into categories
  #mutate(pred = ifelse(predC == -1, "HP", "LP"),
  #       conn = ifelse(connC == -1, "exp", "imp")) %>%
  print %>% 
  ggplot(
    aes(x = TerminologyBinarySum,
        y = fit,
        group = Expertise,
        col=Expertise
    )
  ) +
  geom_errorbar(
    aes(ymin = fit-se,
        ymax= fit+se),
    size = .25,
    width=.3,
    position = pd
  ) +
  geom_point(
    # aes(shape = conn),
    size=2,
    position = pd) +
  geom_line(
    aes(linetype = Expertise),
    size = .5,
    position = pd) +
  theme_bw() +
  theme(text = element_text(size = 24)) +
  xlab("Terminology") +
  ylab("Fitted Log RT") +
  scale_x_discrete(labels = c("terminology","common")) +
  # ylim(550, 700) +
  NULL

effects::Effect(c("cGPT2"),
                final_tft1, se = T,
                confidence.level = 0.95,
                #xlevels = list(TerminologyBinarySum = c(-1,1))
) %>% 
  as.data.frame %>% 
  #code back into categories
  print %>% 
  #mutate(expertise = ifelse(Expertise == -1, "novice", "expert")) %>%
  print %>% 
  ggplot(
    aes(x = cGPT2,
        y = fit,
        
        #group = Expertise # add this? Not sure if it will help
    )
  ) +
  geom_ribbon(aes(ymax=fit+se, ymin=fit-se), alpha=0.1) +
  geom_line(
    # aes(shape = conn),
    #size=2,
    #position = pd)
  ) +
  #geom_line(
  #  aes(linetype = Expertise), #remove capital letter
  #  size = .5,
  #  position = pd) +
  theme_bw() +
  theme(text = element_text(size = 24)) +
  xlab("General surprisal (centered)") +
  ylab("Fitted Log RT") +
  #scale_linetype_discrete(name="Terminology",
  #                        labels=c("technical", "common")
  #) +
  # ylim(550, 700) +
  NULL 


effects::Effect(c("ResidSurprisal"),
                m_tft1_residsur, se = T,
                confidence.level = 0.95,
                #xlevels = list(TerminologyBinarySum = c(-1,1))
) %>% 
  as.data.frame %>% 
  #code back into categories
  print %>% 
  #mutate(expertise = ifelse(Expertise == -1, "novice", "expert")) %>%
  print %>% 
  ggplot(
    aes(x = ResidSurprisal,
        y = fit,
        
        #group = Expertise # add this? Not sure if it will help
    )
  ) +
  geom_ribbon(aes(ymax=fit+se, ymin=fit-se), alpha=0.1) +
  geom_line(
    # aes(shape = conn),
    #size=2,
    #position = pd)
  ) +
  #geom_line(
  #  aes(linetype = Expertise), #remove capital letter
  #  size = .5,
  #  position = pd) +
  theme_bw() +
  theme(text = element_text(size = 24)) +
  xlab("Specialized surprisal (residualized)") +
  ylab("Fitted Log RT") +
  #scale_linetype_discrete(name="Terminology",
  #                        labels=c("technical", "common")
  #) +
  # ylim(550, 700) +
  NULL

# follow up for RQ1, split data by Termin. type
m_fprt1_new_techt <- lmer(LogFPRT ~ cWordLen + cLogFreqLemma + cWordIndexInSentence + Expertise + cGPT2 + (1|SubjectID) + (1 + Expertise|UniqueWordID),
                          data=subset(df_allwords_clean, df_allwords_clean$TerminologyBinarySum=="1terminology"))

summary(m_fprt1_new_techt)

m_fprt1_new_common <- lmer(LogFPRT ~ cWordLen + cLogFreqLemma + cWordIndexInSentence + Expertise + cGPT2 + (1|SubjectID) + (1 + Expertise|UniqueWordID),
                           data=subset(df_allwords_clean, df_allwords_clean$TerminologyBinarySum=="2common"))  # , control = lmerControl(optimizer ="Nelder_Mead")

summary(m_fprt1_new_common)

# above: compare beta for Expertise1: abs(-0.037) > abs(-0.006) the expertise effect is larger for technical terms than common words

# for FPRT: the same model, but without Expertise: I can't do that becuse Expertise is also in my random slope... TODO


##################################################################################################################################################################################
##################################################################################################################################################################################
##################################################################################################################################################################################
##################################################################################################################################################################################


##### Research question 2: Does surprisal from a domain-specific LM better approximate linguistic expectations of high-BK readers? 
#### RQ2 REPHRASED: does surprisal from a domain-specific LM uniquely contribute to reading time prediction?


# RQ2 
m_fprt2_both <- lmer(LogFPRT ~ cWordLen + cLogFreqLemma + cWordIndexInSentence + Expertise*TerminologyBinarySum*cGPT2 + Expertise*TerminologyBinarySum*cspecGPT + (1|SubjectID) + (1+ Expertise|UniqueWordID),
                      data=df_allwords_clean) # , control = lmerControl(optimizer ="Nelder_Mead")

summary(m_fprt2_both)  # word frequency not significant

# no word frequency
m_fprt2_both2 <- lmer(LogFPRT ~ cWordLen + cWordIndexInSentence + Expertise*TerminologyBinarySum*cGPT2 + Expertise*TerminologyBinarySum*cspecGPT + (1|SubjectID) + (1+ Expertise|UniqueWordID),
                     data=df_allwords_clean) # , control = lmerControl(optimizer ="Nelder_Mead")

summary(m_fprt2_both2)


# use residual specialized surprisal
lm1 <- lm(cspecGPT ~ cGPT2, data=df_allwords_clean)
df_allwords_clean$ResidSurprisal <- resid(lm1)

# recalculate model from RQ2 using residual surprisal instead of specialized surprisal
m_fprt2_resid_surp <- lmer(LogFPRT ~ cWordLen + cWordIndexInSentence + Expertise*TerminologyBinarySum*cGPT2 + Expertise*TerminologyBinarySum*ResidSurprisal + (1|SubjectID) + (1+ Expertise|UniqueWordID),
                     data=df_allwords_clean)

summary(m_fprt2_resid_surp)


#### Follow up analysis, RQ2
m_check5_techt <- lmer(LogFPRT ~ cWordLen + cWordIndexInSentence + Expertise*cGPT2 + Expertise*cspecGPT + (1|SubjectID) + (1|UniqueWordID),
                       data=subset(df_allwords_clean, df_allwords_clean$TerminologyBinarySum=="1terminology"))
summary(m_check5_techt)

m_check5_common <- lmer(LogFPRT ~ cWordLen + cWordIndexInSentence + Expertise*cGPT2 + Expertise*cspecGPT + (1|SubjectID) + (1|UniqueWordID),
                        data=subset(df_allwords_clean, df_allwords_clean$TerminologyBinarySum=="2common"))

summary(m_check5_common)



##################################################################################################################################################################################
##################################################################################################################################################################################
#################################################### Fix analysis for RQ2 accounting for the high correlation between generic and specialized surprisal (r=0.97)
##################################################################################################################################################################################
##################################################################################################################################################################################
##################################################################################################################################################################################

# step 1: use model from RQ1, but adding a main effect of residual specialized surprisal from lm1

# not including word frequency
m_fprt1_residsur <- lmer(LogFPRT ~ cWordLen + cWordIndexInSentence + cGPT2 + ResidSurprisal + Expertise*TerminologyBinarySum + (1|SubjectID) + (1 + Expertise|UniqueWordID),
                         data=df_allwords_clean)

summary(m_fprt1_residsur)

# anova comparison of the model without and with residual specialized surpisal
anova(final_fprt1, m_fprt1_residsur)#Data: df_allwords_clean
#Models:
#  final_fprt1: LogFPRT ~ cWordLen + cWordIndexInSentence + cGPT2 + Expertise + TerminologyBinarySum + (1 | SubjectID) + (1 + Expertise | UniqueWordID) + Expertise:TerminologyBinarySum
#m_fprt1_residsur: LogFPRT ~ cWordLen + cWordIndexInSentence + cGPT2 + ResidSurprisal + Expertise * TerminologyBinarySum + (1 | SubjectID) + (1 + Expertise | UniqueWordID)
#          npar    AIC    BIC logLik deviance  Chisq Df Pr(>Chisq)    
#final_fprt1        12 131786 131900 -65881   131762                         
#m_fprt1_residsur   13 131774 131898 -65874   131748 14.456  1  0.0001435 ***
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Observation: the model that includes residual specialized surprisal has a significantly higher log-likelihood (p=0.0001435).
# We expec specialized surprisal to be predictive especially for experts and on technical terms,
# therefore we add a three-way interaction with expertise and terminology in step 2

# step 2: 
m_fprt2_residsurp <- lmer(LogFPRT ~ cWordLen + cWordIndexInSentence + Expertise*TerminologyBinarySum*cGPT2 + Expertise*TerminologyBinarySum*ResidSurprisal + (1|SubjectID) + (1+ Expertise|UniqueWordID),
                             data=df_allwords_clean)

summary(m_fprt2_residsurp)

# we find that both three-way interaction terms are significant: generalSurprisal * expertise * terminology (p=0.001564) 
# and resispecialSurprisal * expertise * terminology (p=0.022267)
# we see that both surprisal types interact with exprtise and terminology. we split the data by terminology and fit the model again to analyze this effect for each terminology type

# step 3: split by terminology type. we omit terminology as a predictor accordingly
m_fprt2_residsurp_common <- lmer(LogFPRT ~ cWordLen + cWordIndexInSentence + Expertise*cGPT2 + Expertise*ResidSurprisal + (1|SubjectID) + (1+ Expertise|UniqueWordID),
                          data=subset(df_allwords_clean, df_allwords_clean$TerminologyBinarySum=="2common"))

summary(m_fprt2_residsurp_common)
# Common words: interaction expertise:general surprisal is significant (p=0.003), while the interaction of residual spec. surprisal with expertise is not

m_fprt2_residsurp_techt <- lmer(LogFPRT ~ cWordLen + cWordIndexInSentence + Expertise*cGPT2 + Expertise*ResidSurprisal + (1|SubjectID) + (1+ Expertise|UniqueWordID),
                                 data=subset(df_allwords_clean, df_allwords_clean$TerminologyBinarySum=="1terminology"))

summary(m_fprt2_residsurp_techt)
# Technical terms: the interaction between expertise and residual spec. surprisal is significant (p=0.003271), while the interation of expertise and general surprisal is not



####################################################
###########################
## Repeat these steps for TFT
###########################
####################################################

# step 1: use model from RQ1, but adding a main effect of residual specialized surprisal from lm1

# including word frequency
m_tft1_residsur <- lmer(LogTFT ~ cWordLen + cLogFreqLemma + cWordIndexInSentence + cGPT2 + ResidSurprisal + Expertise*TerminologyBinarySum + (1|SubjectID) + (1 + Expertise|UniqueWordID),
                        data=df_allwords_clean)

summary(m_tft1_residsur)

# anova comparison of the model without and with residual specialized surpisal
anova(final_tft1, m_tft1_residsur)
#Data: df_allwords_clean
#Models:
#  final_tft1: LogTFT ~ cWordLen + cLogFreqLemma + cWordIndexInSentence + cGPT2 + Expertise * TerminologyBinarySum + (1 | SubjectID) + (1 + Expertise | UniqueWordID)
#m_tft1_residsur: LogTFT ~ cWordLen + cLogFreqLemma + cWordIndexInSentence + cGPT2 + ResidSurprisal + Expertise * TerminologyBinarySum + (1 | SubjectID) + (1 + Expertise | UniqueWordID)
#         npar    AIC    BIC logLik deviance  Chisq Df Pr(>Chisq)    
#final_tft1        13 171463 171587 -85718   171437                         
#m_tft1_residsur   14 171442 171575 -85707   171414 22.999  1  1.621e-06 ***
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Observation: the model that includes residual specialized surprisal has a significantly higher log-likelihood (p=1.621e-06).
# We expect specialized surprisal to be predictive especially for experts and on technical terms,
# therefore we add a three-way interaction with expertise and terminology in step 2

# step 2: 
m_tft2_residsurp <- lmer(LogTFT ~ cWordLen + cLogFreqLemma + cWordIndexInSentence + Expertise*TerminologyBinarySum*cGPT2 + Expertise*TerminologyBinarySum*ResidSurprisal + (1|SubjectID) + (1+ Expertise|UniqueWordID),
                         data=df_allwords_clean)

summary(m_tft2_residsurp)

effects::Effect(c("Expertise", "TerminologyBinarySum", "ResidSurprisal"),
                m_tft2_residsurp, se = T,
                confidence.level = 0.95,
                xlevels = list(Expertise = c(-1,1),
                               TerminologyBinarySum= c(-1,1))
) %>% 
  as.data.frame %>% 
  #code back into categories
  #mutate(pred = ifelse(predC == -1, "HP", "LP"),
  #       conn = ifelse(connC == -1, "exp", "imp")) %>%
  print %>% 
  ggplot(
    aes(x = ResidSurprisal,
        y = fit,
        group = Expertise,
        col = Expertise
    )
  ) +
  geom_errorbar(
    aes(ymin = fit-se,
        ymax= fit+se),
    size = .25,
    width=.3,
    position = pd
  ) +
  #geom_point(
    # aes(shape = conn),
  #  size=2,
  #  position = pd) +
  geom_line(
    aes(linetype = Expertise),
    size = .5,
    position = pd) +
  facet_grid(. ~ TerminologyBinarySum) +
  theme_bw() +
  xlab("Specialized surprisal (residuals)") +
  ylab("Fitted LogRT") +
  #scale_linetype_discrete(name="Terminology",
  #                        labels=c("common", "technical")
  #) +
  # ylim(550, 700) +
  NULL



names_t <- c("terminology", "common")
names(names_t) <- c("1terminology", "2common")

# geom_ribbon(aes(x=num, y=value, ymax=upperLoess, ymin=lowerLoess), alpha=0.2)
effects::Effect(c("Expertise", "TerminologyBinarySum", "ResidSurprisal"),
                m_tft2_residsurp, se = T,
                confidence.level = 0.95,
                xlevels = list(Expertise = c(-1,1),
                               TerminologyBinarySum= c(-1,1))
) %>% 
  as.data.frame %>% 
  #code back into categories
  #mutate(TerminologyBinarySum = factor(ifelse(TerminologyBinarySum == "1terminology", "terminology", "common")), levels=c("terminology", "common"))  %>%
  #       conn = ifelse(connC == -1, "exp", "imp")) %>%
  print %>% 
  ggplot(
    aes(x = ResidSurprisal,
        y = fit,
        group = Expertise
    )
  ) +
  geom_ribbon(aes(ymax=fit+se, ymin=fit-se), alpha=0.1) +
  #geom_point(
  # aes(shape = conn),
  #  size=2,
  #  position = pd) +
  geom_line(
    aes(linetype = Expertise,col = Expertise),
    size = .9,
    position = pd) +
  facet_grid(. ~ TerminologyBinarySum, labeller = labeller(TerminologyBinarySum=names_t)) +
  theme_bw() +
  theme(text = element_text(size = 24)) +
  xlab("Specialized surprisal (residuals)") +
  ylab("Fitted LogRT") +
  #scale_linetype_discrete(name="Terminology",
  #                        labels=c("common", "technical")
  #) +
  # ylim(550, 700) +
  NULL

effects::Effect(c("Expertise", "TerminologyBinarySum", "cGPT2"),
                m_tft2_residsurp, se = T,
                confidence.level = 0.95,
                xlevels = list(Expertise = c(-1,1),
                               TerminologyBinarySum= c(-1,1))
) %>% 
  as.data.frame %>% 
  #code back into categories
  #mutate(TerminologyBinarySum = factor(ifelse(TerminologyBinarySum == "1terminology", "terminology", "common")), levels=c("terminology", "common"))  %>%
  #       conn = ifelse(connC == -1, "exp", "imp")) %>%
  print %>% 
  ggplot(
    aes(x = cGPT2,
        y = fit,
        group = Expertise
    )
  ) +
  geom_ribbon(aes(ymax=fit+se, ymin=fit-se), alpha=0.1) +
  #geom_point(
  # aes(shape = conn),
  #  size=2,
  #  position = pd) +
  geom_line(
    aes(linetype = Expertise,col = Expertise),
    size = .9,
    position = pd) +
  facet_grid(. ~ TerminologyBinarySum, labeller = labeller(TerminologyBinarySum=names_t)) +
  theme_bw() +
  theme(text = element_text(size = 24)) +
  xlab("General surprisal (centered)") +
  ylab("Fitted LogRT") +
  #scale_linetype_discrete(name="Terminology",
  #                        labels=c("common", "technical")
  #) +
  # ylim(550, 700) +
  NULL

# also general surprisal
effects::Effect(c("Expertise", "TerminologyBinarySum", "cGPT2"),
                m_tft2_residsurp, se = T,
                confidence.level = 0.95,
                xlevels = list(Expertise = c(-1,1),
                               TerminologyBinarySum= c(-1,1))
) %>% 
  as.data.frame %>% 
  #code back into categories
  #mutate(pred = ifelse(predC == -1, "HP", "LP"),
  #       conn = ifelse(connC == -1, "exp", "imp")) %>%
  print %>% 
  ggplot(
    aes(x = cGPT2,
        y = fit,
        group = Expertise,
        color = Expertise
    )
  ) +
  geom_errorbar(
    aes(ymin = fit-se,
        ymax= fit+se),
    size = .25,
    width=.3,
    position = pd
  ) +
  #geom_line() +
  geom_line(
    aes(linetype = Expertise),
    size = .5,
    position = pd) +
  facet_grid(. ~ TerminologyBinarySum) +
  theme_bw() +
  xlab("General surprisal (residuals)") +
  ylab("Fitted LogRT") +
  scale_linetype_discrete(name="Terminology",
                         labels=c("common", "technical")
  ) +
  # ylim(550, 700) +
  NULL

# we find that among the 2 three-way interactions, only the tree-way interation of residual spec. surprisal with expertise and terminology is significant (p=0.000372)
# there is a significant two-way interaction between general surprisal and terminology (p=0.047572).
# no other interactions between either of the two surprisal types and other predictors are significant.
# both general and residual spec. surprisal have signficant main effects

# step 3: split by terminology type. we omit terminology as a predictor accordingly
m_tft2_residsurp_common <- lmer(LogTFT ~ cWordLen + cLogFreqLemma + cWordIndexInSentence + Expertise*cGPT2 + Expertise*ResidSurprisal + (1|SubjectID) + (1+ Expertise|UniqueWordID),
                                data=subset(df_allwords_clean, df_allwords_clean$TerminologyBinarySum=="2common"))

summary(m_tft2_residsurp_common)
# Common words: interaction expertise:general surprisal is significant (p=0.0077), and so it the interaction between expertise and residual spec. (p=0.0250)

m_tft2_residsurp_techt <- lmer(LogTFT ~ cWordLen + cLogFreqLemma + cWordIndexInSentence + Expertise*cGPT2 + Expertise*ResidSurprisal + (1|SubjectID) + (1+ Expertise|UniqueWordID),
                               data=subset(df_allwords_clean, df_allwords_clean$TerminologyBinarySum=="1terminology"))

summary(m_tft2_residsurp_techt)
# Technical terms: the interaction between expertise and residual spec. surprisal is significant (p=0.003092), while the interation of expertise and general surprisal is not



##################################################################################################################################################################################
##################################################################################################################################################################################


df_allwords_clean %>% 
  group_by(Expertise,TerminologyBinarySum) %>% 
  summarise(n = n(),
            mean_rt = mean(FPRT),
            sd_rt = sd(FPRT)) %>% 
  mutate( se=sd_rt/sqrt(n)) %>% 
  ggplot(aes(x=Expertise, y = mean_rt, group=TerminologyBinarySum)) +
  geom_line(aes(linetype=TerminologyBinarySum),position=position_dodge(0.1)) +
  geom_errorbar(aes(ymin=mean_rt-se, ymax=mean_rt+se,linetype=TerminologyBinarySum),position=position_dodge(0.1),
                size=.3,    # Thinner lines
                width=.2) +
  geom_point(position=position_dodge(0.1), size=3, shape=21, fill="white") +
  ylab("Reading time (ms)") +  
  theme_bw() 



### Plotting the effects using the effects library
eff_cf <- effect("Expertise*TerminologyBinarySum*cspecGPT", m_fprt2_both)
plot(eff_cf)

################ Analysis of residual error ###### 
# true RT  - predicted RT

df_allwords_clean$DeltaSurprisalGenericMinusSpec <- df_allwords_clean$GPT2 - df_allwords_clean$specSurprisal

df_allwords_clean$ResidualsQ2_m_fprt2_both2 <- resid(m_fprt2_both2)

ggplot(df_allwords_clean) + geom_bar(aes(UniPOS, ResidualsQ2_m_fprt2_both2), position="dodge", stat="summary", fun="mean")

ggplot(df_allwords_clean) + geom_bar(aes(TechnicalTerm, ResidualsQ2_m_fprt2_both2), position="dodge", stat="summary", fun="mean")

ggplot(df_allwords_clean) + geom_bar(aes(TerminologyBinarySum, ResidualsQ2_m_fprt2_both2), position="dodge", stat="summary", fun="mean")

ggplot(df_allwords_clean, aes(x=ResidualsQ2_m_fprt2_both2, color=TechnicalTerm))  + geom_density()

ggplot(df_allwords_clean, aes(x=ResidualsQ2_m_fprt2_both2, color=UniPOS))  + geom_density()

#


ggplot(df_allwords_clean) + geom_bar(aes(SSTPOS, ResidualsQ2_m_fprt2_both2), position="dodge", stat="summary", fun="mean") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

ggplot(df_allwords_clean) + geom_bar(aes(SSTPOS, GPT2), position="dodge", stat="summary", fun="mean") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

ggplot(df_allwords_clean) + geom_bar(aes(SSTPOS, specSurprisal), position="dodge", stat="summary", fun="mean") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


# relationship between surprisal and residuals?

ggplot(df_allwords_clean, aes(y=specSurprisal, x=ResidualsQ2_m_fprt2_both2)) + geom_point()

#### compare reading times between expert and novices: find words with the largest differences
# group by UniqueWordID

# remove words that were fixated by fewer than 30 subjects (out of 75 subjects)
df_allwords_clean2 <- df_allwords_clean
df_allwords_clean2 <- df_allwords_clean2[df_allwords_clean2$UniqueWordID %in% names(which(table(df_allwords_clean$UniqueWordID) > 30)), ]

df2 <- df_allwords_clean2 %>% group_by(UniqueWordID, WordStr, TextDomain, TextID, Expertise, TechnicalTerm, TerminologyBinarySum, GPT2, BioGPT, PhysGPT, DependencyType, NdepLeft, NdepRight, DistanceToHead, WordIndexInText, WordIndexInSentence, WordLen) %>% summarise(across(c(LogFPRT, ResidualsQ2_m_fprt2_both2), mean), .groups = "drop")

# DeltaSurprisalGenericMinusSpec,  ResidualsQ2_m_fprt2_both2

df2_biot <- subset(df2, df2$TextDomain=="biology")
df2_physt <- subset(df2, df2$TextDomain=="physics")

df2_biot$DiffGeneralSpecializedSurp <- df2_biot$GPT2 - df2_biot$BioGPT
df2_physt$DiffGeneralSpecializedSurp <- df2_physt$GPT2 - df2_physt$PhysGPT


df2_biot_biom <- subset(df2_biot, df2_biot$Expertise=="expert")

# order the dataframe by Residual size, take a subset of datapoints with the largest (> 0) and the smallest (< 0) residuals and analyze

df2_biot <- df2_biot[with(df2_biot, order(-ResidualsQ2_m_fprt2_both2)),]

# top 60, bottom 60, our of 1658 words
df2_biot_top <- df2_biot[1:60, ]
nrow(df2_biot) - 60
df2_biot_bottom <- df2_biot[1598:1658, ]



### analyze the top 60 words in biology texts that had the highest positive residuals (true RT - predicted RT)
ggplot(df2_biot_top, aes(x=WordIndexInText, y=ResidualsQ2_m_fprt2_both2))  + geom_point() + 
  ggtitle("Top 60 datapoints with largest mean residuals in biology texts")
ggplot(df2_biot_top, aes(x=WordIndexInSentence, y=ResidualsQ2_m_fprt2_both2))  + geom_point() + 
  ggtitle("Top 60 datapoints with largest mean residuals in biology texts")

ggplot(df2_biot_top, aes(x=GPT2, y=ResidualsQ2_m_fprt2_both2))  + geom_point() + 
  ggtitle("Top 60 datapoints with largest mean residuals in biology texts")

ggplot(df2_biot_top, aes(x=BioGPT, y=ResidualsQ2_m_fprt2_both2))  + geom_point() + 
  ggtitle("Top 60 datapoints with largest mean residuals in biology texts")

ggplot(df2_biot_top, aes(x=TechnicalTerm, y=ResidualsQ2_m_fprt2_both2, fill=Expertise)) + geom_bar(stat="identity", position=position_dodge()) + 
  ggtitle("Top 60 datapoints with largest mean residuals in biology texts")

ggplot(df2_biot_top, aes(x=DependencyType, y=ResidualsQ2_m_fprt2_both2, fill=Expertise)) + geom_bar(stat="identity", position=position_dodge()) + 
  ggtitle("Top 60 datapoints with largest mean residuals in biology texts")

ggplot(df2_biot_top, aes(x=DistanceToHead, y=ResidualsQ2_m_fprt2_both2, fill=Expertise)) + geom_bar(stat="identity", position=position_dodge()) + 
  ggtitle("Top 60 datapoints with largest mean residuals in biology texts")

### bottom 60, most negative residuals
ggplot(df2_biot_bottom, aes(x=WordIndexInText, y=ResidualsQ2_m_fprt2_both2))  + geom_point() + 
  ggtitle("Top 60 datapoints with most negative mean residuals in biology texts")

ggplot(df2_biot_bottom, aes(x=WordIndexInSentence, y=ResidualsQ2_m_fprt2_both2))  + geom_point() + 
  ggtitle("Top 60 datapoints with most negative mean residuals in biology texts")

ggplot(df2_biot_bottom, aes(x=GPT2, y=ResidualsQ2_m_fprt2_both2))  + geom_point() + 
  ggtitle("Top 60 datapoints with most negative mean residuals in biology texts")

ggplot(df2_biot_bottom, aes(x=BioGPT, y=ResidualsQ2_m_fprt2_both2))  + geom_point() + 
  ggtitle("Top 60 datapoints with most negative mean residuals in biology texts")

ggplot(df2_biot_bottom, aes(x=TechnicalTerm, y=ResidualsQ2_m_fprt2_both2, fill=Expertise)) + geom_bar(stat="identity", position=position_dodge()) + 
  ggtitle("Top 60 datapoints with most negative mean residuals in biology texts")

ggplot(df2_biot_bottom, aes(x=DependencyType, y=ResidualsQ2_m_fprt2_both2, fill=Expertise)) + geom_bar(stat="identity", position=position_dodge()) + 
  ggtitle("Top 60 datapoints with most negative mean residuals in biology texts")

ggplot(df2_biot_bottom, aes(x=DistanceToHead, y=ResidualsQ2_m_fprt2_both2, fill=Expertise)) + geom_bar(stat="identity", position=position_dodge()) + 
  ggtitle("Top 60 datapoints with most negative mean residuals in biology texts")







