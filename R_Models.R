library(ggplot2)
library(lme4)
library(lmerTest)
library(effects)
library(ggplot2)
library(tidyr)
library(dplyr)
library(mgcv)
library(sjPlot)
library(segmented)
library(survival)
library(changepoint)
library(car)
library(emmeans)
library(nnet)
library(MASS)

# ==============================================================================
# Read the data
# ==============================================================================
data <- read.csv("C:/Users/zuire/PycharmProjects/GainLoss/data/E2_summary_full.csv")
data$Condition <- factor(data$Condition, levels = c('Baseline', 'Frequency'))
data$TrialType <- factor(data$TrialType, levels = c('AB', 'CD', 'CA', 'CB', 'BD', 'AD'))
data$group_baseline <- factor(data$group_baseline, levels = c(1, 2, 3),
                              labels = c('Good Learner', 'Average Learner', 'Bad Learner'))
data$group_frequency <- factor(data$group_frequency, levels = c(1, 2, 3),
                              labels = c('Good Learner', 'Average Learner', 'Bad Learner'))
unique(data$group_frequency)
data$Gender <- factor(data$Gender, levels = c("Female", "Male", "Other", "Prefer not to answer"))
data$Ethnicity <- factor(data$Ethnicity, levels = c("Hispanic or Latino", "Not Hispanic or Latino","Prefer not to answer"))
data$Race <- factor(data$Race, levels = c("More than one Race", "White", "Asian",
                                          "Prefer not to answer", "Black or African American", 
                                          "American Indian or Alaskan Native",
                                          "Native Hawaiin or Other Pacific Islander"))

individual_data <- data %>%
  dplyr::select(Subnum, group_baseline, prob1_baseline, prob2_baseline, prob3_baseline, 
                group_frequency, prob1_frequency, prob2_frequency, prob3_frequency,
                Gender, Ethnicity, Race, Age, Big5O, Big5C, 
                Big5E, Big5A, Big5N, BISScore, CESDScore, ESIBF_disinhScore, 
                ESIBF_aggreScore, ESIBF_sScore, PSWQScore, STAITScore, STAISScore) %>%
  distinct(Subnum, .keep_all = TRUE)
                       
CA <- data %>%
  filter(TrialType == 'CA')

BD <- data %>%
  filter(TrialType == 'BD')

CB <- data %>%
  filter(TrialType == 'CB')

AD <- data %>%
  filter(TrialType == 'AD')

Group13 <- CA %>%
  filter(group_baseline != 2)

baseline_CA <- CA %>%
  filter(Condition == 'Baseline')

frequency_CA <- CA %>%
  filter(Condition == 'Frequency')

accuracy_wide <- data %>%
  pivot_wider(id_cols = c(Subnum, TrialType, group_baseline, group_frequency, C_diff), 
              names_from = Condition, 
              values_from = c(BestOption, training_accuracy))

accuracy_wide_CA <- accuracy_wide %>%
  filter(TrialType == 'CA')

# ==============================================================================
# Modeling starts here
# ==============================================================================
model <- lmer(BestOption ~ group_baseline * Condition + (1|Subnum), data = data)
summary(model)
plot(allEffects(model))

model <- glm(BestOption ~ Condition * group_baseline + Big5O + Big5C + 
               Big5E + Big5A + Big5N + BISScore + CESDScore + ESIBF_disinhScore 
             + ESIBF_aggreScore + ESIBF_sScore + PSWQScore + STAITScore + STAISScore,
             data = CA)
summary(model)
plot(allEffects(model))


model <- lmer(BestOption ~ training_accuracy * Condition + (1|Subnum), data = CA)
summary(model)
plot(allEffects(model))

model <- lm(BestOption ~ training_accuracy, data = frequency_CA)
summary(model)
plot(allEffects(model))

model <- lmer(BestOption ~ training_accuracy * TrialType * Condition + (1|Subnum), data = data)
summary(model)
plot(allEffects(model))

# Calculate the residual
individual_trend <- ranef(model)$Subnum[["(Intercept)"]]
names(individual_trend) <- rownames(ranef(model)$Subnum)
individual_df <- data.frame(Subnum = names(individual_trend),
                            intercept_me = individual_trend)
personality_merged <- merge(individual_df, individual_data, by = "Subnum")


model <- glm(intercept_me ~ Gender + Ethnicity + Race + Age + Big5O + Big5C + 
               Big5E + Big5A + Big5N + BISScore + CESDScore + ESIBF_disinhScore 
             + ESIBF_aggreScore + ESIBF_sScore + PSWQScore + STAITScore + STAISScore,
             data = personality_merged)
summary(model)
plot(allEffects(model))

#
good_learners_CA <- accuracy_wide_CA %>%
  filter(BestOption_Baseline > .55) %>%
  filter(training_accuracy_Frequency >= .50)

model <- glm(BestOption_Frequency ~ poly(BestOption_Baseline, 3),
             data = accuracy_wide_CA)
summary(model)
plot(allEffects(model))

model <- glm(C_diff ~ BestOption_Baseline,
             data = accuracy_wide_CA)
summary(model)
plot(allEffects(model))

model <- glm(training_accuracy_Baseline ~ training_accuracy_Frequency, 
             data = accuracy_wide_CA)
summary(model)
plot(allEffects(model))

# Wide data with B and F per subject
model <- lm(BestOption_Frequency ~ BestOption_Baseline + training_accuracy_Frequency, data = accuracy_wide_CA)
summary(model)

individual_trend <- resid(model)
# -----------------------------Unused-------------------------------------------
# summary_model <- summary(model)
# 
# model <- polr(group_frequency ~ Gender + Ethnicity + Race + Age + Big5O + Big5C + 
#                 Big5E + Big5A + Big5N + BISScore + CESDScore + ESIBF_disinhScore 
#               + ESIBF_aggreScore + ESIBF_sScore + PSWQScore + STAITScore + STAISScore,
#               data = individual_data, Hess = TRUE)
# summary(model)
# summary_model <- summary(model)
# 
# # Compute z-values and p-values
# coefs <- summary_model$coefficients
# z_values <- coefs[, "Value"] / coefs[, "Std. Error"]
# p_values <- 2 * (1 - pnorm(abs(z_values)))
# 
# # Combine results
# results <- cbind(coefs, "z value" = z_values, "p value" = p_values)
# rounded_results <- round(results, 3)
# 
# print(rounded_results)

# ==============================================================================
# Read the data
# ==============================================================================
data <- read.csv("C:/Users/zuire/PycharmProjects/GainLoss/data/E2_data_testing.csv")
data <- data %>%
  left_join(accuracy_wide_CA %>% dplyr::select(Subnum, BestOption_Baseline),by = "Subnum")
data$Condition <- factor(data$Condition, levels = c('Baseline', 'Frequency'))
data$TrialType <- factor(data$TrialType, levels = c('CA', 'CB', 'BD', 'AD'))
data$group_baseline <- factor(data$group_baseline, levels = c(1, 2, 3),
                              labels = c('Good Learner', 'Average Learner', 'Bad Learner'))
data$group_frequency <- factor(data$group_frequency, levels = c(1, 2, 3),
                               labels = c('Good Learner', 'Average Learner', 'Bad Learner'))

CA <- data %>%
  filter(TrialType == 'CA')

baseline_CA <- CA %>%
  filter(Condition == 'Baseline')

frequency_CA <- CA %>%
  filter(Condition == 'Frequency')


model <- glmer(BestOption ~ training_accuracy * Condition + (1|Subnum), 
               family=binomial, data = CA)
summary(model)
plot(allEffects(model))

model <- glmer(BestOption ~ Condition * BestOption_Baseline + training_accuracy + (1 + Condition|Subnum), 
               family=binomial, data = CA)
summary(model)
plot(allEffects(model))

model <- glmer(BestOption ~ training_accuracy + (1|Subnum), 
               family = binomial(link = "logit"), data = baseline_CA)
summary(model)
plot(allEffects(model))
