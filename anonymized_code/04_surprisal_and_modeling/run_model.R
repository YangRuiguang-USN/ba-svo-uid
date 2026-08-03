# =============================================================================
# Régression logistique mixte — construction bǎ vs. SVO
# Version 2 : VP_length_log + interaction NP2*VP, sans log_ratio_NP2_VP
# verb_class : activity / mental (2 niveaux)
# =============================================================================

library(lme4)
library(car)
library(MuMIn)
library(broom.mixed)

# ── 1. Chargement ─────────────────────────────────────────────────────────────
df <- read.csv("model_data.csv", stringsAsFactors = FALSE)
cat("Lignes :", nrow(df), "| ba :", sum(df$y), "| svo :", sum(df$y == 0), "\n")

# ── 2. Typage des facteurs avec coding explicite ───────────────────────────────

# verbal_modifier_type : référence = without_verbal_modifier
# Comparaisons : adverbial vs. without, nominal vs. without
df$verbal_modifier_type <- factor(
  df$verbal_modifier_type,
  levels = c("without_verbal_modifier", "adverbial", "nominal")
)

# NP1_pronominality : référence = NP (forme la plus lourde)
# Comparaisons : pronoun vs. NP, null_subject vs. NP
df$NP1_pronominality <- factor(
  df$NP1_pronominality,
  levels = c("NP", "pronoun", "null_subject")
)

# verb_class : référence = activity (classe majoritaire)
# Comparaison : mental vs. activity
df$verb_class <- factor(
  df$verb_class,
  levels = c("activity", "mental")
)

# NP2_givenness binaire
df$NP2_givenness_bin <- as.integer(df$NP2_givenness == "given")

# Centrer et réduire les variables numériques continues
df$NP2_length_log_z <- scale(df$NP2_length_log)[, 1]
df$VP_length_log_z  <- scale(df$VP_length_log)[, 1]

cat("Distribution y :", table(df$y), "\n")
cat("verbal_modifier_type :", table(df$verbal_modifier_type), "\n")
cat("verb_class :", table(df$verb_class), "\n")

# ── 3. Modèle M1 : sans interaction ───────────────────────────────────────────
M1 <- glmer(
  y ~
    verbal_modifier_type +
    NP2_length_log_z +
    VP_length_log_z +
    NP1_pronominality +
    verb_class +
    NP2_givenness_bin +
    (1 | V_core) +
    (1 | source_file),
  data    = df,
  family  = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

cat("\n══ SUMMARY M1 (sans interaction) ══\n")
print(summary(M1))

# ── 4. Modèle M2 : avec interaction NP2 × VP ──────────────────────────────────
M2 <- glmer(
  y ~
    verbal_modifier_type +
    NP2_length_log_z +
    VP_length_log_z +
    NP2_length_log_z:VP_length_log_z +
    NP1_pronominality +
    verb_class +
    NP2_givenness_bin +
    (1 | V_core) +
    (1 | source_file),
  data    = df,
  family  = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

cat("\n══ SUMMARY M2 (avec interaction NP2 × VP) ══\n")
print(summary(M2))

cat("\n══ TEST ANOVA M1 vs M2 (apport de l'interaction) ══\n")
print(anova(M1, M2))

# ── 5. Diagnostics ────────────────────────────────────────────────────────────

# VIF sur M2
model_vif <- lm(
  y ~ verbal_modifier_type + NP2_length_log_z + VP_length_log_z +
      NP2_length_log_z:VP_length_log_z +
      NP1_pronominality + verb_class + NP2_givenness_bin,
  data = df
)
cat("\nVIF (M2) :\n")
print(vif(model_vif))

# Pseudo-R²
cat("\nPseudo-R² M1 :\n")
print(r.squaredGLMM(M1))
cat("\nPseudo-R² M2 :\n")
print(r.squaredGLMM(M2))

# ── 6. Accuracy ───────────────────────────────────────────────────────────────

# Baseline : prédire toujours la classe majoritaire (svo = 0)
baseline_acc <- max(mean(df$y == 0), mean(df$y == 1))
cat(sprintf("\nBaseline accuracy (classe majoritaire) : %.1f%%\n",
            baseline_acc * 100))

# Probabilités prédites par le modèle
df$fitted_M1 <- predict(M1, type = "response")

# Prédiction binaire : > 0.5 → 1 (bǎ), sinon 0 (SVO)
df$cor_pred <- as.numeric(predict(M1, type = "response") > 0.5)

# Accuracy
model_acc <- mean(df$cor_pred == df$y)
cat(sprintf("M1 accuracy : %.1f%%\n", model_acc * 100))

# Table de correctness
df$cor_obs <- df$y
cat("\nTable de correctness M1 :\n")
print(xtabs(~ cor_obs + cor_pred, df))

# ── 7. Export ─────────────────────────────────────────────────────────────────
results_M1 <- tidy(M1, conf.int = TRUE, exponentiate = TRUE)
write.csv(results_M1, "model_results_M1.csv", row.names = FALSE)
cat("\nRésultats exportés → model_results_M1.csv\n")
