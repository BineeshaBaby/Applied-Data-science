 # -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 22:03:52 2023

@author: BINEESHA BABY
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

 
def climate_data(filename):
    """
    Reads the World Bank data from a given file and returns two dataframes:
    Args:
    filename (str): The name of the file to read.

    Returns:
      tuple: Two dataframes.
          data_years: with years as columns.
          data_countries: with countries as columns. 

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


def indicator_names(filename):
    
    """
    Reads the World Bank data from a given file and returns a list of all indicator names.
    Args:
    filename (str): The name of the file to read.

   Returns:
      list: A list of all indicator names.
      
    """

    # Read data from file
    df_indicator_data = pd.read_csv(filename, skiprows=4)

    # Extract unique indicator names
    indicator_names = df_indicator_data['Indicator Name'].unique().tolist()

    return indicator_names


def plot_co2_emissions(file_path):
    
    """Plots CO2 emissions for the top 8 countries.

    Args:
    file_path (str): The path to the CSV file containing CO2 emissions data.

    Returns:
    None.

    The function reads the data from the file at the given path, keeps the necessary columns,
    fills any missing values with zeros, groups data by country, sums CO2 emissions over all years,
    sorts the resulting data in descending order by CO2 emissions, and creates a bar plot of the top
    8 countries with the highest CO2 emissions (in kt).

    Example usage:
    plot_co2_emissions("D:/CO2_emission/CO2_emmission.csv")
    """

    # Read data from file
    df_co2_data = pd.read_csv(file_path, skiprows=4)
    
    #describe
    df_co2_data.describe()
    
    # Keep only necessary columns
    df_co2_data = df_co2_data[['Country Name', '2015', '2016', '2017', '2018', '2019']]
    
    # Fill any missing values with zeros
    df_co2_data.fillna(0, inplace=True)

    # Group data by country and sum CO2 emissions over all years
    df_co2_by_country = df_co2_data.groupby('Country Name').sum()

    # Sort data in descending order by CO2 emissions
    df_co2_by_country = df_co2_by_country.sort_values(by='2019', ascending=False)
  
    # Print summary statistics
    print(df_co2_by_country.describe())
    
    # Create bar plot of top 8 countries
    top_8_countries = df_co2_by_country.head(8)
    top_8_countries.plot(kind='bar', legend=None, width = 0.8)
    
    # Enlarge the outer box
    plt.subplots_adjust(left=0.1, right=1.2, top=0.9, bottom=0.1)
    plt.title('Top 8 Countries with the Highest CO2 Emissions', fontweight='bold')
    plt.xlabel('Country')
    plt.ylabel('CO2 Emissions (kt)')
    plt.legend(loc='upper right')
    
    # add a light color grid
    plt.grid(True, alpha=0.3)
    
    # move the grid to the background
    plt.gca().set_axisbelow(True)
    
    plt.show()
   

def plot_elec_production(data_path, countries):
    """
    Plots a bar chart of electricity production from oil, gas, and coal sources for the specified countries.
    
    Parameters:
    data_path (str): The path to the CSV file containing the electricity production data.
    countries (list of str): The list of countries to include in the chart.
    
    Returns:
    None
    """
    
    # Load electricity production data into a DataFrame
    df_elec_data = pd.read_csv(data_path, skiprows=4)
    
    # Filter electricity production data to include only the specified countries
    df_elec_data = df_elec_data[df_elec_data['Country Name'].isin(countries)]

    # Keep only necessary columns
    df_elec_data = df_elec_data[['Country Name', '1995', '2000', '2005', '2010', '2015']]

    # Fill any missing values with zeros
    df_elec_data.fillna(0, inplace=True)

    # Set the 'Country Name' column as the index
    df_elec_data.set_index('Country Name', inplace=True)
    
    # Define the colors for the bars
    colors = plt.cm.tab20(range(len(countries)))

    # Create a bar graph of electricity production for the specified countries
    ax = df_elec_data.loc[countries].plot(kind='bar', figsize=(10,6), width=0.8, color=colors)
    ax.set_title('Electricity production from oil, gas and coal sources (% of total)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Country', fontsize=14)
    ax.set_ylabel('Electricity Production', fontsize=14)

    plt.xticks(rotation=90, fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14)   
    # add a light color grid
    plt.grid(True, alpha=0.3)
    
    # move the grid to the background
    plt.gca().set_axisbelow(True)
    plt.show()


def heatmap_countries(data_file: str, country_names: list, indicators: list, time_periods: dict):
    """
    Generate heatmaps of correlation between indicators for each country and time period.

    Args:
        data_file (str): The path to the CSV file containing the data.
        country_names (list): A list of country names to include in the analysis.
        indicators (list): A list of indicator names to include in the analysis.
        time_periods (dict): A dictionary of time periods, where the keys are time period names
                             and the values are tuples of start and end years.

    Returns:
        None.
    """

    # Load the data from CSV file
    data = pd.read_csv(data_file, skiprows=4)
    
    # Define the colormap for each country
    cmap_list = ["coolwarm", "viridis", "YlOrBr"]

    # Loop through each country name
    for i, country_name in enumerate(country_names):
        # Filter the data for the given country and drop unnecessary columns
        country_data = data[data["Country Name"] == country_name].drop([
            "Country Name", "Country Code", "Indicator Code"], axis=1)

        # Select only the given indicators
        country_data = country_data[country_data["Indicator Name"].isin(indicators)]

        # Pivot the dataframe to have years as index and indicators as columns
        country_data = country_data.set_index("Indicator Name").T

        # Drop columns that have only one value
        country_data = country_data.loc[:, country_data.nunique() > 1]

        # Loop through each time period
        for time_period_name, time_period_range in time_periods.items():
            start_year, end_year = time_period_range

            # Filter for the desired years
            time_period_data = country_data.loc[start_year:end_year]

            # Print summary statistics
            print(f"{country_name} - {time_period_name}:")
            print(time_period_data.describe())
            
            # Create a heatmap of correlation between indicators
            fig, ax = plt.subplots(figsize=(15, 12))
            sns.heatmap(time_period_data.corr(), center=0, cmap=cmap_list[i],
                                  annot=True, cbar_kws={'orientation': 'vertical'})
            
            # Bold the indicator names
            for tick in ax.get_xticklabels():
              tick.set_fontweight('bold')
            for tick in ax.get_yticklabels():
              tick.set_fontweight('bold')
            ax.set_title(f"{country_name}", fontweight='bold', fontsize=17)

            # Add the legend
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=18)
            cbar.ax.set_ylabel('Correlation', fontsize=18, fontweight='bold', rotation=270, labelpad=18)
            cbar.ax.yaxis.set_label_position('right')
            ax.set_xlabel('Indicator Name', fontsize=18,fontweight='bold')
            ax.set_ylabel('Indicator Name', fontsize=18,fontweight='bold')
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            plt.show()


def plot_gg_countries(data_file: str, countries: list, num_countries: int = 10) -> None:
    """
    Given a CSV data file, a list of countries and the number of countries to include in the plot, this function loads
    the data, filters for the countries with greenhouse gas emissions, calculates the mean greenhouse gas emissions for
    each year, and creates a line plot of greenhouse gas emissions over time for the specified number of top emitting
    countries.

    Parameters:
    - data_file (str): the file path of the CSV data file
    - countries (list): a list of strings containing the names of the countries to filter the data for
    - num_countries (int): the number of top emitting countries to include in the plot (default = 8)

    Returns:
    - None
    """

    # Load the data from a CSV file
    data = pd.read_csv(data_file, skiprows=4)

    # Select the relevant columns for analysis
    relevant_cols = ['Country Name', 'Indicator Name', '2015', '2016', '2017', '2018', '2019']
    climate_data = data.loc[data['Indicator Name'].str.contains
                            ('greenhouse gas emissions', case=False)][relevant_cols]

    climate_data = climate_data[climate_data['Country Name'].isin(countries)]

    # Group the data by country and calculate the mean greenhouse gas emissions for each year
    climate_data = climate_data.groupby(['Country Name'])[relevant_cols[2:]].mean().reset_index()

    # Calculate the total greenhouse gas emissions for each country
    climate_data['total_emissions'] = climate_data.iloc[:, 1:].sum(axis=1)

    # Sort the data by total emissions in descending order
    climate_data = climate_data.sort_values('total_emissions', ascending=False)

    # Print the top countries with the highest greenhouse gas emissions in 2020
    print(f"Top {num_countries} countries with the highest greenhouse gas emissions in 2019:")
    print(climate_data[['Country Name', '2019']].head(num_countries))

    # Set the years to plot
    years = relevant_cols[2:]

    # Loop through each country in the dataframe
    for i in range(num_countries):
        country_data = climate_data.iloc[i]
        country_name = country_data['Country Name']

        # Create a line plot for the country's greenhouse gas emissions over time
        plt.plot(years, country_data[years], label=country_name, linewidth=4)
    
    # Print summary statistics
    print(country_data.describe())
    
    # Set the plot title and axis labels
    plt.title("Greenhouse Gas Emissions Over Time", fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel("Emissions (kt of CO2 equivalent)")

    # Add a legend to the plot
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14)
   
    # add a light color grid
    plt.grid(True, alpha=0.3)
    
    # move the grid to the background
    plt.gca().set_axisbelow(True)
    
    # Display the plot
    plt.show()


def plot_population_data(population_file, countries, years):
    # Read in the population data for the selected countries
    pop_df = pd.read_csv(population_file, skiprows=4)
    pop_df = pop_df[pop_df['Country Name'].isin(countries)][['Country Name'] + years]

    # Remove any rows with missing values
    pop_df.dropna(inplace=True)

    # Sort data in descending order by population
    pop_df = pop_df.sort_values(['2019'], ascending=False)
    
    # Print summary statistics
    print(pop_df.describe())

    # Set x-axis values for each country
    x = np.arange(len(countries)) * 1.2  # Adjust the spacing between each group of bars

    # Set the colors for each year
    colors = ['gray', 'orange', 'blue', 'green', 'red']

    # Plot the population data for each year
    for i, year in enumerate(years):
        plt.bar(x + i/len(years), pop_df[year], width=1/len(years), color=colors[i], label=year)

    # Label the x-axis with the country names
    plt.xticks(x + 0.2, pop_df['Country Name'])
    # Add axis labels and title
    plt.xlabel("Country")
    plt.ylabel("Population")
    plt.title("Top {} Countries by Population, {}-{}".format(len(countries),
                                                    years[0], years[-1]),fontweight="bold")
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)
    
    # Add a legend to the plot
    plt.legend()
    
    # add a light color grid
    plt.grid(True, alpha=0.3)
    
    # move the grid to the background
    plt.gca().set_axisbelow(True)
    # Display the plot
    plt.show()
    

def plot_kurtosis_over_time(file_path, color='blue'):
    # Load CSV file into pandas DataFrame
    df = pd.read_csv(file_path, skiprows=4)

    # Get the indicator columns
    indicator_columns = df.columns[4:]

    # calculate kurtosis for each column from 1960 to 2019
    kurt_values = []
    for col in df.columns[4:]:
        kurt = df[col].kurtosis()
        kurt_values.append(kurt)
        print(f"{col} kurtosis: {kurt:.2f}")

    # slice the y array to have the same length as the x array
    y = kurt_values[:60]

    # plot the kurtosis values
    plt.figure(figsize=(8,6))
    plt.plot(range(1960, 2020), y, color=color, linestyle='-', linewidth=2, marker='o', markersize=9, markerfacecolor=color, markeredgecolor='white')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Kurtosis', fontsize=14)
    plt.title('Kurtosis over time', fontsize=16)
    plt.xticks(rotation=45)
    plt.legend(['Kurtosis'])
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Data extraction and processing
    filename = "D:/climate change/Climate.csv"
    data_years, data_countries = climate_data(filename)
    print(data_years, data_countries.describe())
    
    #call indicator name function
    indicator_names("D:/climate change/Climate.csv")

    # call co2 emission function
    plot_co2_emissions("D:/CO2_emission/CO2_emmission.csv")

    # call plot_elec_production function
    countries = ['Qatar', 'Bahrain', 'Kuwait', 'United Arab Emirates', 'Oman', 'Brunei Darussalam', 'Canada', 'Luxembourg']
    data_path = "D:/Electricity_production/electricity.csv"
    plot_elec_production(data_path, countries)

    # call heatmap_countries function
    data_file = "D:/climate change/Climate.csv"
    country_names = ["Canada", "United Arab Emirates", "Qatar"]
    indicators = ["CO2 emissions (metric tons per capita)",
                  "GDP per capita (constant 2010 US$)",
                  "Access to electricity (% of population)",
                  "Renewable energy consumption (% of total final energy consumption)",
                  "Urban population (% of total population)",
                  "Electricity production from oil, gas and coal sources (% of total)",
                  "Total greenhouse gas emissions (kt of CO2 equivalent)",
                  "Forest area (sq. km)",
                  "Agricultural land (% of land area)",
                  "Population, total"]
    time_periods = {"2019": ["1960", "2015"]}
    heatmap_countries(data_file, country_names, indicators, time_periods)

    # call plot_gg_countries function
    data_file = "D:/greenhouse_gas_emmissions/green_house_gas.csv"
    countries = ['Qatar', 'Bahrain', 'Kuwait', 'United Arab Emirates', 'Oman', 'Brunei Darussalam', 'Canada', 'Luxembourg']
    num_countries = 8
    plot_gg_countries(data_file, countries, num_countries)

    # call plot_population_data function
    population_file = "D:/population/population.csv"
    countries = ['Qatar', 'Bahrain', 'Kuwait', 'United Arab Emirates', 'Oman', 'Brunei Darussalam', 'Canada', 'Luxembourg']
    years = ['2015', '2016', '2017', '2018', '2019']
    plot_population_data(population_file, countries, years)

    # call_kurtosis_over_time function
    plot_kurtosis_over_time("D:/population/population.csv", color='red')








