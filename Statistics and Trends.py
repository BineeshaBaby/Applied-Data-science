 # -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 22:03:52 2023

@author: BINEESHA BABY
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind


def read_worldbank_data(filename):
    """
    Reads the World Bank data from a given file and returns two dataframes:
    - data_years: with years as columns
    - data_countries: with countries as columns
    """
    # Read data from file
    df_data = pd.read_csv(filename, skiprows=4)

    # Transpose the data to create a dataframe with years as columns
    data_years = df_data.set_index('Country Name').T

    # Fill any missing values with zeros
    data_years.fillna(0, inplace=True)

    # Transpose the data again to create a dataframe with countries as columns
    data_countries = data_years.T

    # Fill any missing values with zeros
    data_countries.fillna(0, inplace=True)

    return data_years, data_countries

filename = "D:/climate change/climate change data.csv"
data_years, data_countries = read_worldbank_data(filename)



def get_indicator_names(filename):
    """
    Reads the World Bank data from a given file and returns a list of all indicator names.
    """
    # Read data from file
    df_data = pd.read_csv(filename, skiprows=4)

    # Extract unique indicator names
    indicator_names = df_data['Indicator Name'].unique().tolist()

    return indicator_names


def plot_country_data(data_countries, indicator_name, countries, years, title=None):
    """
    Plots the data for a given indicator and list of countries, for the specified years.
    - data_countries: a dataframe with countries as columns
    - indicator_name: the name of the indicator to plot
    - countries: a list of country names to plot
    - years: a list of years to plot
    """
    # Select the data for the indicator you're interested in and the specified countries
    indicator_data = data_countries[data_countries['Indicator Name'] == indicator_name]
    indicator_data = indicator_data.loc[countries]

    # Determine the available years for the selected countries
    available_years = indicator_data.columns[indicator_data.notna().any()].tolist()

    # Filter the years to plot to only include available years
    years_to_plot = list(set(years).intersection(available_years))

    # Select the data for the years to plot
    indicator_data = indicator_data[years_to_plot]

    # Plot the data for each country as a bar chart
    indicator_data.plot(kind='bar', stacked=True)

    # Set the title and labels for the plot
    plt.title(title or indicator_name)
    plt.xlabel('Year')
    plt.ylabel('Metric tons per capita')
    plt.legend(title='Country', loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


# Set the parameters for the function
filename = "D:/climate change/climate change data.csv"
indicator_name = 'CO2 emissions (metric tons per capita)'
countries_rich = ['United States', 'China', 'Japan', 'Germany','India']
countries_poor = ['Burundi', 'Somalia', 'Mozambique', 'Madagascar', 'Sierra Leone']
years = ['1990', '2000', '2010', '2020']


# Read the data from file and plot the data for each group of countries
data_years, data_countries = read_worldbank_data(filename)
plot_country_data(data_countries, indicator_name, countries_rich, years, title='CO2 Emissions in High Income Countries')
plot_country_data(data_countries, indicator_name, countries_poor, years, title='CO2 Emissions in Less Income Countries')


#################################



def heatmap():
    # Read data from CSV file and drop unnecessary columns
    data_years, data_countries = read_worldbank_data(( "D:/climate change/climate change data.csv"))
    data_years, data_countries = data_years, data_countries.drop(columns="Country Name")
    
    # Transpose dataframe and calculate correlation matrix
    
    cor = data_years, data_countries.corr()
    
    # Visualize correlation matrix as heatmap
    sns.heatmap(data=cor, annot=True)
    plt.title("Correlation between indicators in Switzerland", fontsize=15)
    plt.xticks(rotation=90, ha="center", fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("heatmap.png", bbox_inches="tight")
    plt.show()
    
 # Call the heatmap function
heatmap()   
    
####################################################


