# =============================================================================
# Régression logistique mixte — avec UID (delta_variance)
# M1 : baseline | M2 : + interaction NP2×VP | M3 : M2 + delta_variance
# Robustesse : 3 métriques UID
# =============================================================================

library(lme4)
library(car)
library(MuMIn)
library(broom.mixed)

# ── 1. Chargement ─────────────────────────────────────────────────────────────
df <- read.csv("model_data_surprisal.csv", stringsAsFactors = FALSE)
cat("Lignes :", nrow(df), "| ba :", sum(df$y), "| svo :", sum(df$y == 0), "\n")

# ── 2. Typage des facteurs ────────────────────────────────────────────────────
df$verbal_modifier_type <- factor(
  df$verbal_modifier_type,
  levels = c("without_verbal_modifier", "adverbial", "nominal")
)
df$NP1_pronominality <- factor(
  df$NP1_pronominality,
  levels = c("NP", "pronoun", "null_subject")
)
df$verb_class <- factor(
  df$verb_class,
  levels = c("activity", "mental")
)
df$NP2_givenness_bin <- as.integer(df$NP2_givenness == "given")

# ── 3. Standardisation ────────────────────────────────────────────────────────
df$NP2_length_log_z  <- scale(df$NP2_length_log)[, 1]
df$VP_length_log_z   <- scale(df$VP_length_log)[, 1]
df$delta_variance_z  <- scale(df$delta_variance)[, 1]
df$delta_amplitude_z <- scale(df$delta_amplitude)[, 1]
df$delta_max_step_z  <- scale(df$delta_max_step)[, 1]

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

# ── 4. Modèle M1 : baseline sans interaction, sans UID ───────────────────────
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
  data = df, family = binomial(link = "logit"), control = ctrl
)

cat("\n══ SUMMARY M1 (baseline, sans interaction, sans UID) ══\n")
print(summary(M1))

# ── 5. Modèle M2 : + interaction NP2 × VP ────────────────────────────────────
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
  data = df, family = binomial(link = "logit"), control = ctrl
)

cat("\n══ SUMMARY M2 (avec interaction NP2 × VP) ══\n")
print(summary(M2))

cat("\n══ TEST ANOVA M1 vs M2 (apport de l'interaction) ══\n")
print(anova(M1, M2))

# ── 6. Modèle M3 : M2 + delta_variance (UID) ─────────────────────────────────
M3 <- glmer(
  y ~
    verbal_modifier_type +
    NP2_length_log_z +
    VP_length_log_z +
    NP2_length_log_z:VP_length_log_z +
    NP1_pronominality +
    verb_class +
    NP2_givenness_bin +
    delta_variance_z +
    (1 | V_core) +
    (1 | source_file),
  data = df, family = binomial(link = "logit"), control = ctrl
)

cat("\n══ SUMMARY M3 (M2 + delta_variance) ══\n")
print(summary(M3))

cat("\n══ TEST ANOVA M2 vs M3 (apport de delta_variance) ══\n")
print(anova(M2, M3))

# ── 7. Robustesse : trois métriques UID ───────────────────────────────────────

M4 <- glmer(
  y ~ verbal_modifier_type + NP2_length_log_z + VP_length_log_z +
      NP2_length_log_z:VP_length_log_z +
      NP1_pronominality + verb_class + NP2_givenness_bin +
      delta_amplitude_z +
      (1 | V_core) + (1 | source_file),
  data = df, family = binomial, control = ctrl
)

M5 <- glmer(
  y ~ verbal_modifier_type + NP2_length_log_z + VP_length_log_z +
      NP2_length_log_z:VP_length_log_z +
      NP1_pronominality + verb_class + NP2_givenness_bin +
      delta_max_step_z +
      (1 | V_core) + (1 | source_file),
  data = df, family = binomial, control = ctrl
)

extract_uid <- function(model, label, varname) {
  s   <- summary(model)
  fix <- as.data.frame(coef(s))
  uid_row <- fix[varname, ]
  data.frame(
    modele = label,
    beta   = round(uid_row$Estimate, 4),
    SE     = round(uid_row$`Std. Error`, 4),
    z      = round(uid_row$`z value`, 3),
    p      = signif(uid_row$`Pr(>|z|)`, 3),
    AIC    = round(AIC(model), 1)
  )
}

cat("\n══ TABLEAU ROBUSTESSE (3 métriques UID, sur base M2) ══\n")
tableau <- rbind(
  extract_uid(M3, "delta_variance",  "delta_variance_z"),
  extract_uid(M4, "delta_amplitude", "delta_amplitude_z"),
  extract_uid(M5, "delta_max_step",  "delta_max_step_z")
)
print(tableau, row.names = FALSE)

cat("\n══ ANOVA M2 vs M4 (delta_amplitude) ══\n")
print(anova(M2, M4))
cat("\n══ ANOVA M2 vs M5 (delta_max_step) ══\n")
print(anova(M2, M5))

# ── 8. Diagnostics M3 ─────────────────────────────────────────────────────────
model_vif <- lm(
  y ~ verbal_modifier_type + NP2_length_log_z + VP_length_log_z +
      NP2_length_log_z:VP_length_log_z +
      NP1_pronominality + verb_class + NP2_givenness_bin + delta_variance_z,
  data = df
)
cat("\nVIF (M3) :\n")
print(vif(model_vif))

cat("\nPseudo-R² M1 :\n"); print(r.squaredGLMM(M1))
cat("\nPseudo-R² M2 :\n"); print(r.squaredGLMM(M2))
cat("\nPseudo-R² M3 :\n"); print(r.squaredGLMM(M3))

# ── 9. Accuracy ───────────────────────────────────────────────────────────────
baseline_acc <- max(mean(df$y == 0), mean(df$y == 1))
cat(sprintf("\nBaseline accuracy : %.1f%%\n", baseline_acc * 100))

df$cor_pred <- as.numeric(predict(M3, type = "response") > 0.5)
df$cor_obs  <- df$y
cat(sprintf("M3 accuracy : %.1f%%\n", mean(df$cor_pred == df$y) * 100))
cat("\nTable de correctness M3 :\n")
print(xtabs(~ cor_obs + cor_pred, df))

# ── 10. Export ────────────────────────────────────────────────────────────────
results_M3 <- tidy(M3, conf.int = TRUE, exponentiate = TRUE)
write.csv(results_M3, "model_results_M3.csv", row.names = FALSE)
write.csv(tableau,    "robustness_uid.csv",    row.names = FALSE)
cat("\nRésultats exportés → model_results_M3.csv, robustness_uid.csv\n")
