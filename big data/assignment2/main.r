library(readr)
library(dplyr)

# Load dataset
airline <- read_csv("../datasets/Airline Dataset/Airline Dataset Updated - v2.csv")

# (A) Count male and female passengers
gender_count <- count(airline, Gender)

# (B) Average age of male passengers
male_data <- filter(airline, Gender == "Male")
avg_male_age <- summarise(male_data, mean_age = mean(Age, na.rm = TRUE))

# (C) Average age of female passengers
female_data <- filter(airline, Gender == "Female")
avg_female_age <- summarise(female_data, mean_age = mean(Age, na.rm = TRUE))

# (D) Top 10 nationalities
nationality_count <- count(airline, Nationality, sort = TRUE)
top_nationalities <- head(nationality_count, 10)

# Output results
cat("(A) Male passengers:", gender_count$n[gender_count$Gender == "Male"], ", Female passengers:", gender_count$n[gender_count$Gender == "Female"], "\n")
cat("(B) Average Age of Male Passengers:", round(avg_male_age$mean_age, 2), "\n")
cat("(C) Average Age of Female Passengers:", round(avg_female_age$mean_age, 2), "\n")
cat("(D) Top 10 Nationalities:\n")
print(top_nationalities)
