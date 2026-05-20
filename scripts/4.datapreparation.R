######
#Part 1: Data aggregation
######
library(readr)
library(dplyr)
library(tidyverse)
library(readxl)
library(writexl)
library(ggplot2)

data <- read_excel('total_data.xlsx')

data <- data %>%
  mutate(
    datum = ymd_hms(datum, tz = "UTC"),
    dag_van_week = wday(datum, label = TRUE, week_start = 1)
  )

#If you have no ID in dataset
aggregated <- data %>%
  group_by(sessie_nr) %>%
  summarise(
    total =                 n(),
    controltijd =           min(sessietijd, na.rm = TRUE),
    var_seek =              mean(var_seek, na.rm = TRUE), #You have to add this data by hand before you can handle the analysis
    engagement_typology =   mean(engagementtypology, na.rm = TRUE),
    watchtime_score =       mean(kijktijd_score, na.rm = TRUE),
    entropy_bertopic =      mean(entropy_bertopic, na.rm = TRUE),
    semantic_diversity =    mean(semantic_diversity, na.rm = TRUE),
    feday =                 first(dag_van_week)
  ) %>%
  filter(total >= 10) %>%
  mutate(
    fetime = case_when(
      hour(hms(controltijd)) >= 6  & hour(hms(controltijd)) < 12 ~ "ochtend",
      hour(hms(controltijd)) >= 12 & hour(hms(controltijd)) < 18 ~ "middag",
      hour(hms(controltijd)) >= 18 & hour(hms(controltijd)) < 24 ~ "avond",
      TRUE ~ "nacht"
    )
  )

#If you have ID in your dataset
aggregated <- data %>%
  group_by(sessie_nr, ID) %>%
  summarise(
    total =                 n(),
    ID =                    mean(ID, na.rm = TRUE), #Adds only if you
    controltijd =           min(sessietijd, na.rm = TRUE),
    var_seek =              mean(var_seek, na.rm = TRUE),
    engagement_typology =   mean(engagementtypology, na.rm = TRUE),
    watchtime_score =       mean(kijktijd_score, na.rm = TRUE),
    entropy_bertopic =      mean(entropy_bertopic, na.rm = TRUE),
    semantic_diversity =    mean(semantic_diversity, na.rm = TRUE),
    feunitid =              mean(ID, na.rm = TRUE),
    feday =                 first(dag_van_week)
  ) %>%
  filter(total >= 10) %>%
  mutate(
    fetime = case_when(
      hour(hms(controltijd)) >= 6  & hour(hms(controltijd)) < 12 ~ "ochtend",
      hour(hms(controltijd)) >= 12 & hour(hms(controltijd)) < 18 ~ "middag",
      hour(hms(controltijd)) >= 18 & hour(hms(controltijd)) < 24 ~ "avond",
      TRUE ~ "nacht"
    )
  )


write_xlsx(aggregated, "total_data_aggregated.xlsx")
