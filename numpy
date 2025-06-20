import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("/content/drive/MyDrive/south_africa_population_1974_2023.csv")

# Convert 'Year' to a datetime format and extract just the year and population
df['Year'] = pd.to_datetime(df['Year'], format='%Y').dt.year
df_extracted = df[['Year', 'Population']]

# Calculate the first derivative (rate of change of population)
first_derivative = np.gradient(df_extracted['Population'], df_extracted['Year'])

# Calculate the second derivative (rate of change of the first derivative)
second_derivative = np.gradient(first_derivative, df_extracted['Year'])

# Find inflection points (where second derivative changes sign)
inflection_points = np.where(np.diff(np.sign(second_derivative)))[0]

# Plot the original data and mark the inflection points
plt.figure(figsize=(10,6))
plt.plot(df_extracted['Year'], df_extracted['Population'], '.', label='Population data', color='orange')
plt.scatter(df_extracted['Year'].iloc[inflection_points], df_extracted['Population'].iloc[inflection_points],
            color='red', label='Inflection points', zorder=2)
plt.title('South Africa Population with Inflection Points')
plt.xlabel('Year')
plt.ylabel('Total Population')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Output the inflection points
inflection_years = df_extracted['Year'].iloc[inflection_points]
inflection_populations = df_extracted['Population'].iloc[inflection_points]


for year, pop in zip(inflection_years, inflection_populations):
    print(f"Inflection point: Year {year}, Population {pop}")

