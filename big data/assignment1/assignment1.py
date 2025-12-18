import pandas as pd

# Write a Python program that takes a number y as an input (70 £ y £ 82), lists all
# the the mpg values of model_year = y, and computes the average fuel efficiency
# using the Pandas package for Python.

mpg_data = pd.read_csv("Automobile.csv")

y = int(input("Input model_year: "))
if 70 <= y <= 82:
    filtered_data = mpg_data[mpg_data["model_year"] == y]
    print(filtered_data["mpg"])
    average_mpg = filtered_data["mpg"].mean()
    print(f"The fuel efficiency of year {y} is {average_mpg} .")
else:
    print("Please enter a valid model year between 70 and 82.")
