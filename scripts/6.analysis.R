library(tidyverse)
library(dplyr)
library(plm)
library(fixest)
library(readxl)

aggdata <- read_excel("total_data_aggregated.xlsx")

model1 <- feols(
  watchtime_score ~ semantic_diversity + var_seek + engagement_typology +
    semantic_diversity:var_seek +
    semantic_diversity:engagement_typology +
    log(total),
  data = aggdata,
  vcov = "hetero"
)
summary(model1)

model2 <- feols(
  watchtime_score ~ semantic_diversity + var_seek + engagement_typology +
    semantic_diversity:var_seek +
    semantic_diversity:engagement_typology +
    log(total) | fetime,
  data = aggdata,
  vcov = "hetero"
)

model3 <- feols(
  watchtime_score ~ semantic_diversity +
    engagement_typology +
    semantic_diversity:var_seek +
    semantic_diversity:engagement_typology +
    log(total) | ID + fetime,
  data = aggdata,
  vcov = ~ID
)
summary(model3)

etable(model1, model2, model3,
       fitstat = c("f", "r2", "n"),
       digits = 3)
