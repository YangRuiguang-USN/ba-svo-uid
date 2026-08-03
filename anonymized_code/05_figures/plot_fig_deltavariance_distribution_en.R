library(ggplot2)
library(dplyr)

df_s <- read.csv("model_data_surprisal.csv")

# English labels
df_s$construction_label <- ifelse(df_s$construction == "ba",
                                   "bǎ construction",
                                   "SVO construction")

p <- ggplot(df_s, aes(x = delta_variance, fill = construction_label,
                       color = construction_label)) +
  geom_density(alpha = 0.35, linewidth = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.5) +
  scale_fill_manual(values = c("bǎ construction" = "#2166ac",
                                "SVO construction"    = "#d6604d")) +
  scale_color_manual(values = c("bǎ construction" = "#2166ac",
                                 "SVO construction"    = "#d6604d")) +
  coord_cartesian(xlim = c(-25, 20)) +
  labs(
    x     = expression(Delta * "variance"),
    y     = "Density",
    fill  = NULL,
    color = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    legend.position   = c(0.82, 0.82),
    legend.background = element_rect(fill = "white", color = NA),
    legend.key.size   = unit(0.45, "cm"),
    axis.title        = element_text(size = 10),
    axis.text         = element_text(size = 9)
  )

ggsave("figures/fig_deltavariance_distribution.pdf",
       plot = p, width = 13, height = 7, units = "cm", device = cairo_pdf)
ggsave("figures/fig_deltavariance_distribution.png",
       plot = p, width = 13, height = 7, units = "cm", dpi = 300)

cat("Exporté -> figures/fig_deltavariance_distribution.pdf/.png\n")
