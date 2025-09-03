library(ggplot2)
library(lme4)
library(lmerTest)
library(effects)
library(tidyr)
library(dplyr)
library(mgcv)
library(sjPlot)
library(segmented)
library(survival)
library(car)
library(emmeans)
library(nnet)
library(MASS)
library(lavaan)
library(ez)
library(semPlot)
library(purrr)

# ==============================================================================
# Summary Data
# ==============================================================================
data <- read.csv("C:/Users/zuire/PycharmProjects/GainLoss/data/E2_summary_modeled.csv")
data$Condition <- factor(data$Condition, levels = c('Baseline', 'Frequency'))
data$TrialType <- factor(data$TrialType, levels = c('CA', 'AB', 'CD', 'CB', 'BD', 'AD'))
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
                ESIBF_aggreScore, ESIBF_sScore, PSWQScore, STAITScore, STAISScore,
                t, alpha, subj_weight,
                t_All, alpha_All, subj_weight_All) %>%
  distinct(Subnum, .keep_all = TRUE)
                      
training <- data %>%
  filter(TrialType%in%c('AB', 'CD'))

testing <- data %>%
  filter(TrialType%in%c('CA', 'CB', 'BD', 'AD'))

BD <- data %>%
  filter(TrialType == 'BD')

CB <- data %>%
  filter(TrialType == 'CB')

AD <- data %>%
  filter(TrialType == 'AD')

accuracy_wide <- data %>%
  pivot_wider(id_cols = c(Subnum, TrialType, group_baseline, group_frequency, 
                          C_diff, t_All, alpha_All, subj_weight_All), 
              names_from = Condition, 
              values_from = c(BestOption, training_accuracy, t, alpha, subj_weight))

accuracy_wide_CA <- accuracy_wide %>%
  filter(TrialType == 'CA')

accuracy_wide_AB <- accuracy_wide %>%
  filter(TrialType == 'AB')

CA <- data %>%
  filter(TrialType == 'CA')%>%
  left_join(accuracy_wide_CA %>% dplyr::select(Subnum, BestOption_Baseline),by = "Subnum")

baseline_CA <- CA %>%
  filter(Condition == 'Baseline')

frequency_CA <- CA %>%
  filter(Condition == 'Frequency')

# ==============================================================================
# Modeling starts here
# ==============================================================================
model <- lmer(BestOption_Frequency ~ BestOption_Baseline * TrialType +
                (1|Subnum), data = accuracy_wide)
summary(model)
plot(allEffects(model))

model <- lmer(BestOption ~ Condition * TrialType +
                (1|Subnum), data = data)
summary(model)
plot(allEffects(model))

# focus on the interaction BestOption_Baseline:TrialType
rng <- range(frequency_CA$BestOption_Baseline, na.rm = TRUE)
eff <- effect(
  "BestOption_Baseline:TrialType", model,
  xlevels = list(BestOption_Baseline = seq(rng[1], rng[2], length.out = 200))
)
eff_df <- as.data.frame(eff)
write.csv(eff_df, "eff_df.csv", row.names = FALSE)

# Order effects
order_effects_model <- lmer(BestOption ~ Condition + TrialType * order + (1|Subnum), data = data)
summary(order_effects_model)
plot(allEffects(order_effects_model))


accuracy_wide_AB <- accuracy_wide %>%
  filter(TrialType == 'CB')
model <- lm(BestOption_Frequency ~ BestOption_Baseline, data = accuracy_wide_AB)
summary(model)
plot(allEffects(model))


# your model from accuracy_wide
model <- glm(
  BestOption_Frequency ~ BestOption_Baseline * TrialType + 
    training_accuracy_Frequency,
  data = accuracy_wide
)

# Simple slopes of BestOption_Baseline by TrialType
emtr <- emtrends(model, ~ TrialType, var = "BestOption_Baseline")
summary(emtr)     # slope (estimate), SE, t, p for each TrialType vs 0
pairs(emtr)   


model <- glm(BestOption ~ training_accuracy + Condition * (Age + Race + Ethnicity + Big5O + Big5C + 
               Big5E + Big5A + Big5N + BISScore + CESDScore + ESIBF_disinhScore 
             + ESIBF_aggreScore + ESIBF_sScore + PSWQScore + STAITScore + STAISScore),
             data = CA)
summary(model)
plot(allEffects(model))

# Accuracy predicted by training accuracy
model <- lmer(BestOption ~ training_accuracy * Condition + (1|Subnum), data = BD)
summary(model)
plot(allEffects(model))

model <- lm(BestOption ~ training_accuracy, data = frequency_CA)
summary(model)
plot(allEffects(model))

model <- lmer(BestOption ~ training_accuracy * TrialType * Condition + (1|Subnum), data = testing)
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


model <- glm(BestOption_Frequency ~ poly(BestOption_Baseline, 3),
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

model <- glm(subj_weight ~ Gender + Ethnicity + Age + Race + Condition * (Big5O + Big5C + 
               Big5E + Big5A + Big5N + BISScore + CESDScore + ESIBF_disinhScore 
             + ESIBF_aggreScore + ESIBF_sScore + PSWQScore + STAITScore + STAISScore),
             data = CA)
summary(model)
plot(allEffects(model))

# ------------------------------------------------------------------------------
# Model Parameter Analyses
# ------------------------------------------------------------------------------
model <- lmer(BestOption ~ subj_weight * Condition + (1|Subnum), data = CA)
summary(model)
plot(allEffects(model))

model <- lmer(subj_weight ~ BestOption_Baseline * Condition + (1|Subnum), data = CA)
summary(model)
plot(allEffects(model))


model <- glm(subj_weight ~ BestOption_Baseline + training_accuracy, data = frequency_CA)
summary(model)
plot(allEffects(model))

# Whether subj weight in Baseline ~ subj weight in Frequency is moderated
model <- glm(subj_weight_Frequency ~ subj_weight_Baseline + training_accuracy_Frequency + BestOption_Baseline,
             data = accuracy_wide_CA)
summary(model)
plot(allEffects(model))

# ==============================================================================
# Read the data
# ==============================================================================
data <- read.csv("C:/Users/zuire/PycharmProjects/GainLoss/data/E2_data_modeled.csv")
data$Condition <- factor(data$Condition, levels = c('Baseline', 'Frequency'))
data$TrialType <- factor(data$TrialType, levels = c('AB', 'CD', 'CA', 'CB', 'BD', 'AD'))

training_data <- data %>%
  filter(TrialType%in%c('AB', 'CD'))

model <- glmer(BestOption ~ Condition * TrialType * Phase + (1|Subnum), 
               family=binomial, data = training_data)
summary(model)
plot(allEffects(model))

# ==============================================================================
# TrialWise Data
# ==============================================================================
data <- read.csv("C:/Users/zuire/PycharmProjects/GainLoss/data/E2_data_modeled.csv")
data <- data %>%
  left_join(
    accuracy_wide %>% dplyr::select(Subnum, TrialType, BestOption_Baseline),
    by = c("Subnum", "TrialType")
  ) %>%
  left_join(
    model_data %>% dplyr::select(Subnum, alpha_neg, weight),
    by = "Subnum"
  ) %>%
  mutate(
    Condition = factor(Condition, levels = c("Baseline","Frequency")),
    TrialType = factor(TrialType, levels = c("CA","AB","CD","CB","BD","AD"))
  )


frequency <- data %>%
  filter(Condition == 'Frequency')

CA <- data %>%
  filter(TrialType == 'CA')

BD <- data %>%
  filter(TrialType == 'BD')

CB <- data %>%
  filter(TrialType == 'CB')

AD <- data %>%
  filter(TrialType == 'AD')

baseline_CA <- CA %>%
  filter(Condition == 'Baseline')

frequency_CA <- CA %>%
  filter(Condition == 'Frequency')


# Predict p optimal choice by training accuracy
model <- glmer(BestOption ~ training_accuracy * Condition + (1|Subnum), 
               family=binomial, data = CB)
summary(model)
plot(allEffects(model))

model <- glmer(BestOption ~ training_accuracy + (1|Subnum), 
               family=binomial, data = baseline_CA)
summary(model)
plot(allEffects(model))

# Predict p optimal choice by percentage of optimal choice in baseline
model <- glmer(BestOption ~ BestOption_Baseline + TrialType + (1|Subnum), 
               family=binomial, data = frequency)
summary(model)
plot(allEffects(model))

subset <- frequency %>%
  filter(TrialType == 'BD')
model <- glmer(BestOption ~ BestOption_Baseline + (1|Subnum), 
               family=binomial, data = subset)
summary(model)
plot(allEffects(model))

model <- glmer(BestOption ~ Condition + Condition:BestOption_Baseline + training_accuracy + (1 + Condition|Subnum), 
               family=binomial, data = CA)
summary(model)
plot(allEffects(model))

model <- glmer(BestOption ~ BestOption_Baseline + training_accuracy + (1|Subnum), 
               family=binomial, data = frequency_CA)
summary(model)
plot(allEffects(model))

model <- glmer(BestOption ~ BestOption_Baseline + subj_weight + (1|Subnum), 
               family=binomial, data = frequency_CA)
summary(model)
plot(allEffects(model))

# ------------------------------------------------------------------------------
# Model Parameter Analyses
# ------------------------------------------------------------------------------
model <- lmer(BestOption ~ weight * Condition + (1|Subnum), data = CA)
summary(model)
plot(allEffects(model))

model <- lmer(weight ~ BestOption_Baseline + (1|Subnum), data = frequency_CA)
summary(model)
plot(allEffects(model))

# Whether subj weight in Baseline ~ subj weight in Frequency is moderated
model <- glmer(BestOption ~ training_accuracy + best_weight * Condition + (1 + Condition|Subnum),
               family=binomial, data = CA)
summary(model)
plot(allEffects(model))

model <- lmer(subj_weight ~ Condition * training_accuracy + (1|Subnum), data = CA)
summary(model)
plot(allEffects(model))

# ==============================================================================
# Model Summary Data
# ==============================================================================
model_summary <- read.csv("C:/Users/zuire/PycharmProjects/GainLoss/data/model_summary.csv")
model_summary <- model_summary %>%
  left_join(CA %>% dplyr::select(Subnum, Condition, BestOption, BestOption_Baseline, training_accuracy),by = c("Subnum", "Condition"))
model_summary$Condition <- factor(model_summary$Condition, levels = c('Baseline', 'Frequency'))

model_summary_baseline <- model_summary %>%
  filter(Condition == 'Baseline')

model_summary_frequency <- model_summary %>%
  filter(Condition == 'Frequency')

model_summary_long <- model_summary %>%
  pivot_longer(
    cols = -c(Subnum, AIC, BIC, model, Condition, BestOption, BestOption_Baseline, training_accuracy),
    names_to = "parameter",
    values_to = "value"
  ) %>%
  mutate(
    Condition = as.factor(Condition),
    model = as.factor(model),
    Subnum = as.factor(Subnum)
  )

wide_df <- model_summary_long %>%
  dplyr::select(Subnum, Condition, parameter, value, model)%>%
  pivot_wider(names_from = Condition, values_from = value)

model_list <- c('delta', 'delta_PVL', 'delta_asymmetric', 'decay', 'decay_PVL', 
                'decay_win')
model_list <- c('delta_asymmetric_decay_win')
model_list <- c('delta_decay_win', 'delta_decay_PVL_win', 'delta_asymmetric_decay_win')
model_list <- c('delta_decay_win')
model_data <- model_summary_frequency %>%
  filter(model%in%model_list)

model <- lmer(BIC ~ BestOption_Baseline + model + (1|Subnum), data = model_data)
summary(model)
plot(allEffects(model))

model_t <- lmer(BestOption ~ t * Condition + model + (1|Subnum), data = model_summary)
model_alpha <- lmer(BestOption ~ alpha * Condition + model + (1|Subnum), data = model_summary)
model_alpha_neg <- lmer(BestOption ~ alpha_neg * Condition + model + (1|Subnum), data = model_summary)
model_scale <- lmer(BestOption ~ scale * Condition + model + (1|Subnum), data = model_summary)
model_la <- lmer(BestOption ~ la * Condition + model + (1|Subnum), data = model_summary)
model_weight <- lmer(BestOption ~ weight * Condition + (1|Subnum), data = model_summary)

mods <- list(
  t         = model_t,
  alpha     = model_alpha,
  alpha_neg = model_alpha_neg,
  scale     = model_scale,
  la        = model_la,
  weight    = model_weight
)

for (nm in names(mods)) {
  cat("\n====================\n")
  cat("Model:", nm, "\n")
  cat("====================\n")
  print(summary(mods[[nm]]))
}

# Supplement
model_summary_frequency$model <- factor(
  model_summary_frequency$model,
  levels = c('delta_asymmetric_decay_win', 'delta_decay', 'delta_decay_PVL', 'delta_decay_win', 'delta_decay_PVL_win'),
  labels = c('DeltaAsymmetric-DecayWin', 'Delta-Decay', 'DeltaPVL-DecayPVL', 'Delta-DecayWin', 'DeltaPVL-DecayWin')
)

all_hybrid_model <- lmer(weight ~ BestOption_Baseline * model + (1|Subnum), 
                         data = model_summary_frequency)

# 3) Plot the interaction effect (one panel per model)
eff_int <- Effect(c("BestOption_Baseline", "model"), all_hybrid_model)

plt <- plot(
  eff_int,
  multiline = FALSE,                
  ci.style = "bands",
  xlab = "% C Choices in Baseline", 
  ylab = "Delta Learning Weight",        
  main = "Delta Learning Weights By Baseline C Choice Rates",
  layout = c(3, 2)
)

print(plt)

model <- glm(weight ~ BestOption * Condition, data = model_data)
summary(model)
plot(allEffects(model))

model <- lmer(BIC ~ BestOption_Baseline + (1|Subnum), data = model_data)
summary(model)
plot(allEffects(model))

model <- glm(weight ~ BestOption_Baseline + training_accuracy, data = model_data)
summary(model)
plot(allEffects(model))

model <- lmer(Baseline ~ Frequency + parameter + model + (1|Subnum), data = wide_df)
summary(model)
plot(allEffects(model))

# ==============================================================================
# Knowledge Data
# ==============================================================================
knowledge_summary <- read.csv("C:/Users/zuire/PycharmProjects/GainLoss/data/knowledge_summary.csv")
knowledge_summary$Condition <- factor(knowledge_summary$Condition, levels = c('Baseline', 'Frequency'))
knowledge_summary$Option <- factor(knowledge_summary$Option, levels = c('A', 'B', 'C', 'D'))
# knowledge_summary <- knowledge_summary %>%
#   left_join(accuracy_wide_CA %>% dplyr::select(Subnum, BestOption_Baseline,
#                                  BestOption_Frequency, training_accuracy_Baseline,
#                                  training_accuracy_Frequency),by = c("Subnum"))
knowledge_summary$CA_diff_abs <- abs(knowledge_summary$CA_diff)

C_diff_df <- knowledge_summary %>%
  dplyr::select(Subnum, Condition, CA_diff) %>%
  drop_na() %>%
  distinct() %>%
  pivot_wider(names_from = Condition, values_from = CA_diff)

knowledge_wide <- knowledge_summary %>%
  dplyr::select(Subnum, Condition, Phase, Option, abs_err) %>%
  pivot_wider(names_from = Condition, values_from = abs_err)

testing_knowledge <- knowledge_summary %>%
  filter(Phase == 6) %>%
  dplyr::select(Subnum, Condition, CA_diff, BestOption_CA, 
                training_accuracy, CA_diff_abs) %>%
  distinct()

# testing_knowledge <- knowledge_summary %>%
#   filter(Phase == 6) %>%
#   dplyr::select(Subnum, Condition, BestOption.x, BestOption.y, training_accuracy, 
#          CA_diff, CA_diff_normalized) %>%
#   distinct()

frequency <- testing_knowledge %>%
  filter(Condition == 'Frequency')

baseline <- testing_knowledge %>%
  filter(Condition == 'Baseline')

model <- lmer(BestOption ~ Condition + Phase + abs_err + (1|Subnum), data = knowledge_summary)
summary(model)
plot(allEffects(model))

model <- lmer(abs_err ~ Condition * training_accuracy + Phase + (1|Subnum), data = knowledge_summary)
summary(model)
plot(allEffects(model))

model <- lmer(Value ~ BestOption * Option * Condition + Phase + (1|Subnum), data = knowledge_summary)
summary(model)
plot(allEffects(model))

model <- lmer(Confidence ~ Phase + Option * Condition + (1|Subnum), data = knowledge_summary)
summary(model)
plot(allEffects(model))

model <- lmer(Value ~ Condition * Option + Phase + (1|Subnum), data = knowledge_summary)
summary(model)
plot(allEffects(model))

model <- lm(Baseline ~ Frequency, data = C_diff_df)
summary(model)
plot(allEffects(model))

model <- lmer(Value ~ Condition * training_accuracy * Option + (1|Subnum), data = testing_knowledge)
summary(model)
plot(allEffects(model))

model <- lm(abs_err ~ Condition * BestOption_Baseline + (1|Subnum), data = knowledge_summary)
summary(model)
plot(allEffects(model))

model <- glm(CA_diff ~ BestOption_Baseline + training_accuracy_Baseline, data = baseline)
summary(model)
plot(allEffects(model))

model <- lmer(CA_diff ~ training_accuracy * Condition + (1|Subnum), data = testing_knowledge)
summary(model)
plot(allEffects(model))

model <- glm(BestOption_CA ~ Condition * CA_diff, data = testing_knowledge)
summary(model)
plot(allEffects(model))

# ==============================================================================
# SEM
# ==============================================================================
mediation_data <- read.csv("C:/Users/zuire/PycharmProjects/GainLoss/data/mediation_data.csv")
mediation_data$Condition <- factor(mediation_data$Condition, levels = c('Baseline', 'Frequency'))
mediation_data$CA_diff_dev <- mediation_data$CA_diff - 0.1
mediation_data_short <- mediation_data %>%
  dplyr::select(Subnum, Condition, CA_diff, BestOption_CA, 
                training_accuracy, CA_diff_abs, CA_diff_dev) %>%
  distinct()


# Moderated mediation via multi-group: group-specific paths with labels
model_mg <- '
  # a-path (X -> M), group-specific labels
  CA_diff ~ c(a_b, a_f)*training_accuracy

  # b- and c′-paths (M, X -> Y), group-specific labels
  BestOption_CA ~ c(b_b, b_f)*CA_diff +
                  c(cprime_b, cprime_f)*training_accuracy

  # Indirect, total, and (optional) contrasts
  ind_b := a_b*b_b
  ind_f := a_f*b_f
  tot_b := cprime_b + ind_b
  tot_f := cprime_f + ind_f

  # Differences (Frequency minus Baseline)
  diff_a      := a_f - a_b
  diff_b      := b_f - b_b
  diff_cprime := cprime_f - cprime_b
  diff_ind    := ind_f - ind_b
  diff_total  := tot_f - tot_b
'

fit_mg <- sem(model_mg, data = mediation_data, group = "Condition",
              estimator = "MLR", missing = "fiml")  # robust SEs + FIML if needed

summary(fit_mg, fit.measures = TRUE, standardized = TRUE, rsquare = TRUE)
parameterEstimates(fit_mg, standardized = TRUE)[, c("lhs","op","rhs","group","est","se","pvalue","std.all")]

# Plot the path diagram
semPaths(fit,
         what = "std",      # show standardized estimates
         layout = "tree",   # nice tree layout
         edge.label.cex = 1.1,   # size of path coefficient labels
         sizeMan = 8,       # size of observed variable boxes
         sizeLat = 10,      # size of latent variable circles
         residuals = FALSE, # hide residual arrows if not needed
         style = "lisrel")  # cleaner style


# Model with interaction (different slopes)
m_diff_slope <- lm(CA_diff ~ training_accuracy * Condition, data = mediation_data)
summary(m_diff_slope)

# Model with same slope (no interaction) but different intercepts
m_same_slope <- lm(CA_diff ~ training_accuracy + Condition, data = mediation_data)
summary(m_same_slope)

# Compare them
anova(m_same_slope, m_diff_slope)


model <- '
  # a-path
  CA_diff_dev ~ a1*training_accuracy + c1*Condition

  # b-path
  BestOption_CA ~ b1*CA_diff_dev

  # indirect effect by condition
  ind_baseline := a1 * b1
  ind_frequency := (a1) * b1  # slope same, but c1 shifts CA_diff_dev intercept
'

fit <- sem(model, data = mediation_data, meanstructure = TRUE)
summary(fit, fit.measures = TRUE, standardized = TRUE)

model <- lm(BestOption_CA ~ training_accuracy * Condition + CA_diff * Condition, data = mediation_data_short)
summary(model)
plot(allEffects(model))

model <- lm(CA_diff ~ training_accuracy + Condition, data = mediation_data_short)
summary(model)
plot(allEffects(model))

model <- lmer(Value ~ training_accuracy * Option * Condition + (1|Subnum), data = mediation_data)
summary(model)
plot(allEffects(model))

model <- '
  # a-path with intercept shift and interaction
  CA_diff_dev ~ a1*tr_c + a2*Cond01 + a3*trXCond

  # conditional slopes you care about
  a_base := a1
  a_freq := a1 + a3
  diff_slope := a_freq - a_base'
  
fit <- sem(model, data = mediation_data)
summary(fit, fit.measures = TRUE, standardized = TRUE)
# ============================E1 (Unused)=======================================
data <- read.csv("C:/Users/zuire/PycharmProjects/GainLoss/data/E1_summary_modeled.csv")
data$Condition <- factor(data$Condition, levels = c('GainsEF', 'Gains'))
data$TrialType <- factor(data$TrialType, levels = c('AB', 'CD', 'CA', 'CB', 'BD', 'AD'))
data$assignments <- factor(data$assignments, levels = c(1, 2, 3),
                           labels = c('Good Learner', 'Average Learner', 'Bad Learner'))
unique(data$Race)
data$Sex <- factor(data$Sex, levels = c("Female", "Male", "Other", "Prefer not to answer"))
data$Ethnicity <- factor(data$Ethnicity, levels = c("Hispanic or Latino", "Not Hispanic or Latino"))
data$Race <- factor(data$Race, levels = c("White", "More than one Race", "Asian",
                                          "Prefer not to answer", "Black or African American", 
                                          "American Indian or Alaskan Native"))

individual_data <- data %>%
  dplyr::select(Subnum, assignments, prob1, prob2, prob3, Sex, Ethnicity, Race, 
                Big5O, Big5C, Big5E, Big5A, Big5N, Bis11Score, CESD, ESIBF_Disinhibition, 
                ESIBF_SubstanceUse, PSWQ, STAIT, STAIS, NPI, TPM_Boldness,
                TPM_Disinhibition, TPM_Meanness, t, alpha, subj_weight) %>%
  distinct(Subnum, .keep_all = TRUE)

CA <- data %>%
  filter(TrialType == 'CA')

BD <- data %>%
  filter(TrialType == 'BD')

CB <- data %>%
  filter(TrialType == 'CB')

AD <- data %>%
  filter(TrialType == 'AD')

baseline_CA <- CA %>%
  filter(Condition == 'GainsEF')

frequency_CA <- CA %>%
  filter(Condition == 'Gains')

# Modeling starts here
model <- lm(BestOption ~ training_accuracy*assignments, data = frequency_CA)
summary(model)
plot(allEffects(model))

model <- glmer(BestOption ~ training_accuracy * Condition + (1|Subnum), 
               family=binomial, data = CA)
summary(model)
plot(allEffects(model))
