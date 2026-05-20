library(readr)
library(dplyr)
library(tidyverse)
library(readxl)
library(writexl)
library(ggplot2)
library(e1071)
library(purrr)
library(tibble)

aggdata <- read_excel('total_data_aggregated.xlsx')
aggdata$total_log <- log(aggdata$total)
data <- read_excel('total_data.xlsx')

#Part 1: Visualizations

save_plot <- function(plot_obj, filename, width = 10, height = 6, dpi = 300) {
  ggsave(
    filename = file.path("plots", filename),
    plot     = plot_obj,
    width    = width,
    height   = height,
    dpi      = dpi,
    bg       = "white"
  )
  message("Opgeslagen: plots/", filename)
}
theme_thesis_clean <- function() {
  theme_minimal(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey90"),
      axis.title = element_text(face = "bold"),
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11),
      legend.position = "bottom"
    )
}

# Effect Content diversity on user engagement (A)
p1 <- ggplot(aggdata, aes(x = semantic_diversity, y = watchtime_score)) +
  geom_point(
    size = 2,
    alpha = 0.85,
    color = "#2C3E50"
  ) +
  geom_smooth(
    method = "lm",
    se = FALSE,
    color = "#1F1F1F",
    linewidth = 0.8
  ) +
  labs(
    x = "Semantic diversity (IV)",
    y = "Watchtime score (DV)",
    title = "Relationship between Semantic Diversity and User Engagement",
    subtitle = "Linear regression showing association between IV and DV"
  ) +
  theme_thesis_clean()

save_plot(p1, "01.png")

#Session length VS shannon entropy score (C)
p3 <- ggplot(aggdata, aes(x = total_log, y = semantic_diversity)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(
    title = "Relationship Between Semantic Diversity and Number of videos",
    subtitle = "Semantic diversity plotted against session size (log-transformed)",
    x = "Number of videos(log transformed)",
    y = "Semantic diversity (within-session topic variation)"
  ) +
  scale_x_log10() +
  theme_thesis_clean()
save_plot(p3, "03.png")

# Visuals for moderators
agg_var_seek <- aggdata %>%
  group_by(var_seek) %>%
  summarise(
    mean_sd = mean(semantic_diversity, na.rm = TRUE),
    n = n()
  )

# Combined bars moderators in a visual (D):
agg_var_seek <- aggdata %>%
  group_by(var_seek) %>%
  summarise(mean_sd = mean(semantic_diversity, na.rm = TRUE), n = n()) %>%
  mutate(
    facet_label = "Variety-seeking Behavior",
    x_label     = as.character(var_seek)
  )

agg_engagement <- aggdata %>%
  group_by(engagement_typology) %>%
  summarise(mean_sd = mean(semantic_diversity, na.rm = TRUE), n = n()) %>%
  mutate(
    facet_label = "Engagement Typology",
    x_label     = as.character(engagement_typology)
  )

combined <- bind_rows(agg_var_seek, agg_engagement) %>%
  mutate(
    bar_id = paste(facet_label, x_label),
    facet_label = factor(facet_label,
                         levels = c("Variety-seeking Behavior", "Engagement Typology"))
  )

fill_values <- setNames(
  c("grey20", "grey65", "grey40", "grey85"),
  unique(combined$bar_id)
)

p_combined <- ggplot(combined, aes(x = x_label, y = mean_sd, fill = bar_id)) +
  geom_col(color = "black", linewidth = 0.6, width = 0.6) +
  geom_text(
    aes(label = paste0("n=", n)),
    vjust  = -0.5,
    size   = 3.5,
    color  = "black"
  ) +
  scale_fill_manual(
    values = fill_values,
    guide  = "none"
  ) +
  facet_wrap(
    ~facet_label,
    scales         = "free_x",
    strip.position = "top"
  ) +
  labs(
    title    = "Content Diversity by Moderator Variables",
    subtitle = "Left: Variety-seeking behavior  ·  Right: Engagement typology",
    x        = NULL,
    y        = "Mean semantic diversity score"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", size = 13),
    plot.subtitle      = element_text(size = 10, color = "grey30"),
    strip.background   = element_rect(fill = "grey10"),
    strip.text         = element_text(face = "bold", size = 11, color = "white"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    axis.text          = element_text(color = "black", size = 11),
    axis.title.y       = element_text(size = 11)
  )
dir.create("plots", showWarnings = FALSE)
ggsave(
  "plots/05_06_combined.png",
  plot   = p_combined,
  width  = 10,
  height = 6,
  dpi    = 300,
  bg     = "white"
)

# Average semantic diversity per user (B)
avg_diversity <- data %>%
  group_by(ID) %>%
  summarise(
    mean_diversity = mean(semantic_diversity, na.rm = TRUE)
  )

# Plot
ggplot(avg_diversity, aes(x = factor(ID), y = mean_diversity)) +
  geom_col(fill = "#20B8C5") +
  labs(
    title = "Average Content Diversity per User",
    x = "User ID",
    y = "Mean Semantic Diversity"
  ) +
  theme_minimal()

#Descriptive statistics
  freq_feunitid <- aggdata %>%
    count(feunitid) %>%
    mutate(percentage = 100 * n / sum(n))
  
  freq_feday <- aggdata %>%
    count(feday) %>%
    mutate(percentage = 100 * n / sum(n))
  
  freq_fetime <- aggdata %>%
    count(fetime) %>%
    mutate(percentage = 100 * n / sum(n))

  summary_table <- map_dfr(names(aggdata), function(var) {
    x <- aggdata[[var]]
    tibble(
      Variable = var,
      N = sum(!is.na(x)),
      Mean = if(is.numeric(x)) mean(x, na.rm = TRUE) else NA,
      SD = if(is.numeric(x)) sd(x, na.rm = TRUE) else NA,
      Min = if(is.numeric(x)) min(x, na.rm = TRUE) else NA,
      Max = if(is.numeric(x)) max(x, na.rm = TRUE) else NA,
      Skewness = if(is.numeric(x)) e1071::skewness(x, na.rm = TRUE, type = 2) else NA,
    )
  })
  summary_table
  aggdata$total_log <- log(aggdata$total + 1)

  