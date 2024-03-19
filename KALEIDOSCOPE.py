# Imports all required packages
from matplotlib.ticker import MultipleLocator  # Importing specific functionality from matplotlib.ticker module
from matplotlib.widgets import Button  # Importing specific functionality from matplotlib.widgets module
import matplotlib.ticker as ticker  # Importing matplotlib.ticker module with alias 'ticker'
import matplotlib.pyplot as plt  # Importing matplotlib.pyplot module with alias 'plt'
import PySimpleGUI as sg  # Importing PySimpleGUI module with alias 'sg'
import pandas as pd  # Importing pandas module with alias 'pd'
import numpy as np  # Importing numpy module with alias 'np'
import math  # Importing math module
import csv  # Importing csv module

import warnings  # Importing warnings module
warnings.filterwarnings("ignore")  # Ignore warnings during execution

# solar metal model finder
def SOLAR_FINDER(solar_gravity, solar_table, model_metal, model_gravity, model, comp_co, comp_kzz, comp_alpha, min_temp, max_temp):
    """
    Function to find solar metal model based on given parameters.

    Parameters:
        solar_gravity (float): Solar gravity value.
        solar_table (DataFrame): DataFrame containing solar table data.
        model_metal (str): Metal model type.
        model_gravity (str): Gravity model type.
        model (str): Model type.
        comp_co (float): C/O ratio.
        comp_kzz (float): Kzz value.
        comp_alpha (float): Alpha value.
        min_temp (int): Minimum temperature.
        max_temp (int): Maximum temperature.

    Returns:
        DataFrame: DataFrame filtered based on the given parameters.
    """
    df = pd.DataFrame(solar_table)

    if model in ['LOWZ', 'ELF OWL']:
        df = df[(df['C/O'] == comp_co) & 
                (df['log(Kzz)'] == comp_kzz) & 
                (df['log(g)'] == solar_gravity) & 
                (df['[M/H]'] == 0) & 
                (df['teff'] >= min_temp) & 
                (df['teff'] <= max_temp) & 
                (df['teff'] % 100 == 0)]
    elif model == 'SAND':
        df = df[(df['log(g)'] == solar_gravity) & 
                (df['[a/H]'] == 0.05) & 
                (df['teff'] >= min_temp) & 
                (df['teff'] <= max_temp)]

    # Sort the DataFrame by 'teff'
    df.sort_values(by='teff', inplace=True)
    return df
            
# Low metal model finder
def SUBSOLAR_FINDER(comp_metal, comp_gravity, comp_table, model_metal, model_gravity, model, comp_co, comp_kzz, comp_alpha, min_temp, max_temp):  
    """
    Function to find low metal model based on given parameters.

    Parameters:
        comp_metal (float): Metal composition.
        comp_gravity (float): Gravity value.
        comp_table (DataFrame): DataFrame containing composition table data.
        model_metal (str): Metal model type.
        model_gravity (str): Gravity model type.
        model (str): Model type.
        comp_co (float): C/O ratio.
        comp_kzz (float): Kzz value.
        comp_alpha (float): Alpha value.
        min_temp (int): Minimum temperature.
        max_temp (int): Maximum temperature.

    Returns:
        DataFrame: DataFrame filtered based on the given parameters.
    """
    df = pd.DataFrame(comp_table)

    if model in ['LOWZ', 'ELF OWL']:
        df = df[(df['C/O'] == comp_co) & 
                (df['log(Kzz)'] == comp_kzz) & 
                (df['log(g)'] == comp_gravity) & 
                (df['[M/H]'] == comp_metal) & 
                (df['teff'] % 100 == 0) & 
                (df['teff'] >= min_temp) & 
                (df['teff'] <= max_temp)]
    elif model == 'SAND':
        df = df[(df['log(g)'] == comp_gravity) & 
                (df['[a/H]'] == comp_alpha) & 
                (df['teff'] >= min_temp) & 
                (df['teff'] <= max_temp)]

    # Sort the DataFrame by 'teff'
    df.sort_values(by='teff', inplace=True)
    return df

# RMAD model comparison
def RMAD_COMP(phot_list, phot_center, comp_table, solar_table, array, colors, comp_co, comp_kzz, comp_metal, comp_gravity, comp_alpha, total_color_color_list, total_rmad_list,):
    """
    Function to compute RMAD statistics and find all possible color combinations.

    Parameters:
        phot_list (list): List of photometric filters.
        phot_center (str): Center filter for photometric color combinations.
        comp_table (DataFrame): DataFrame containing comparison table data.
        solar_table (DataFrame): DataFrame containing solar table data.
        array (numpy.ndarray): Numpy array to store computed values.
        colors (list): List of color combinations.
        comp_co (float): C/O ratio.
        comp_kzz (float): Kzz value.
        comp_metal (float): Metal composition.
        comp_gravity (float): Gravity value.
        comp_alpha (float): Alpha value.
        total_color_color_list (list): List to store color combinations.
        total_rmad_list (list): List to store RMAD statistics.

    Returns:
        tuple: Top 10 RMAD values, corresponding color combinations, and updated array.
    """
    # Extracting temperature lists
    solar_teff = comp_table['teff'].tolist()
    subsolar_teff = solar_table['teff'].tolist()

    # Find items unique to each list and remove them
    unique_to_solar_teff = set(solar_teff) - set(subsolar_teff)
    unique_to_subsolar_teff = set(subsolar_teff) - set(solar_teff)

    solar_teff = [x for x in solar_teff if x not in unique_to_solar_teff]
    subsolar_teff = [x for x in subsolar_teff if x not in unique_to_subsolar_teff]
    subsolar_teff = list(set(subsolar_teff))
    solar_teff = list(set(solar_teff))

    # Compute RMAD statistics for each color combination
    rmad_list = []
    color_list = []
    for c in range(len(colors)):
        try: 
            # Compute color differences
            subsolar_color = np.subtract((comp_table[str(colors[c][0])]).tolist(), (comp_table[str(colors[c][1])]).tolist())
            solar_color = np.subtract((solar_table[str(colors[c][0])]).tolist(), (solar_table[str(colors[c][1])]).tolist())

            if (str(subsolar_color[0]) == 'nan') or (str(solar_color[0]) == 'nan'): 
                pass
            else:
                # Sort lists in temperature order
                sorted_model = sorted(range(len(subsolar_teff)), key=lambda k: subsolar_teff[k])
                sorted_model_color = [subsolar_color[i] for i in sorted_model]
                
                sorted_solar = sorted(range(len(solar_teff)), key=lambda k: solar_teff[k])
                sorted_solar_color = [solar_color[i] for i in sorted_solar]

                # Compute RMAD
                rmad = np.nanmedian(abs(np.subtract(sorted_solar_color, sorted_model_color)))

                rmad_list.append(rmad)
                color_list.append(colors[c])
        except: 
            pass
    
    # Update array with computed RMAD values
    if array.shape[1] == 32: 
        new_row = np.array([comp_metal, comp_gravity, comp_co, comp_kzz] + rmad_list)  
        array = np.vstack([array, new_row])
    elif array.shape[1] == 30: 
        new_row = np.array([comp_gravity, comp_alpha] + rmad_list)  
        array = np.vstack([array, new_row])
        
    final_rmad_list = []
    final_color_color = []
    for i in range(len(color_list)): 
        temp_color = color_list[i]
        temp_rmad = rmad_list[i]
        for j in range(len(color_list)): 
            if temp_color == color_list[j]:
                pass
            else: 
                temp_final_rmad = np.sqrt(temp_rmad**2 + rmad_list[j]**2)
                temp_color_color = temp_color + color_list[j]
                final_rmad_list.append(temp_final_rmad)
                final_color_color.append(temp_color_color)
    
    combined_lists = list(zip(final_rmad_list, final_color_color))
    
    # Remove duplicate color combinations
    def remove_duplicates_in_second_column(data):
        seen = set()
        output = []
        for item in data:
            value = item[0]
            sublist = item[1]
            sublist_copy = sublist[:]  # Make a copy to avoid modifying the original list
            sublist_copy.sort()  # Sort the sublist to make duplicates adjacent
            sublist_tuple = tuple(sublist_copy)  # Convert the sublist to a tuple for hashing
            if sublist_tuple not in seen:
                output.append((value, sublist))
                seen.add(sublist_tuple)
        return output

    # Remove duplicate color combinations
    result = remove_duplicates_in_second_column(combined_lists)    
    sorted_combined_lists = sorted(result, key=lambda x: x[0], reverse=True)
    
    # Select top 10 RMAD values and corresponding color combinations
    top_10_values = sorted_combined_lists[:8]
    top_10_list1, top_10_list2 = zip(*top_10_values)
    total_color_color_list.append(top_10_list2)
    total_rmad_list.append(top_10_list1)
        
    return top_10_list1, top_10_list2, array
   
# Interactive plot to allow color-color selection
def INTER_PLOT(flattened_rmad, flattened_color, table, model, min_temp, max_temp): 
    """
    Function to create an interactive plot for RMAD analysis.

    Parameters:
        flattened_rmad (list): List of RMAD values.
        flattened_color (list): List of color combinations.
        table (DataFrame): DataFrame containing table data.
        model (str): Model type.
        min_temp (int): Minimum temperature.
        max_temp (int): Maximum temperature.

    Returns:
        list: List of selected color combinations.
    """
    # Function to remove duplicate color combinations
    def remove_duplicates_in_second_column(data):
        seen = set()
        output = []
        for item in data:
            value = item[0]
            sublist = item[1]
            sublist_copy = sublist[:]  # Make a copy to avoid modifying the original list
            sublist_copy.sort()  # Sort the sublist to make duplicates adjacent
            sublist_tuple = tuple(sublist_copy)  # Convert the sublist to a tuple for hashing
            if sublist_tuple not in seen:
                output.append((value, sublist))
                seen.add(sublist_tuple)
        return output
    
    # Combine RMAD values and color combinations
    combined_lists = list(zip(flattened_rmad, flattened_color))
    
    # Remove duplicate color combinations
    result = remove_duplicates_in_second_column(combined_lists)    
    sorted_combined_lists = sorted(result, key=lambda x: x[0], reverse=True)
    temp_rmad_list, temp_color_list = zip(*sorted_combined_lists)
    temp_rmad_list = list(temp_rmad_list)
    temp_color_list = list(temp_color_list)
    
    record = []
    def on_yes(event):
        record.append((event.xdata, event.ydata))
        plt.close()

    def on_no(event):
        record.append((0, 0))
        plt.close()

    name_color_list = ['PANSTARRS_I', 'PANSTARRS_Z', 'PANSTARRS_Y', 'WISE_W1', 
                   'MKO_J', 'MKO_H', 'MKO_K', 'WISE_W2']
    color_name = ['i$_{PS}$', 'z$_{PS}$', 'y$_{PS}$', 'W1', 
                  'J$_{MKO}$', 'H$_{MKO}$', 'K$_{MKO}$', 'W2']

    # INTERACTIVE PLOT
    for i in range(len(temp_color_list)):
        plt.rcParams['font.family'] = 'Times New Roman'
        temp_rmad = temp_rmad_list[i]
        temp_color_combo = temp_color_list[i]
        
        phot_x = [temp_color_combo[0], temp_color_combo[1]]
        phot_y = [temp_color_combo[2], temp_color_combo[3]]
        
        for k in range(len(name_color_list)): 
            if phot_x[0] == name_color_list[k]: 
                temp_color_x1 = color_name[k]
            if phot_x[1] == name_color_list[k]: 
                temp_color_x2 = color_name[k]
            if phot_y[0] == name_color_list[k]: 
                temp_color_y1 = color_name[k]
            if phot_y[1] == name_color_list[k]: 
                temp_color_y2 = color_name[k]
        
        fig, ax = plt.subplots(figsize=[5, 5]) 
        
        ultra_table = pd.read_csv('DATA_MODULE/UltraCool_Dwarfs.csv')
        sub_table = pd.read_csv('DATA_MODULE/UltraCool_Subdwarfs.csv')
        
        ultra_phot_x1 = ultra_table[phot_x[0]].tolist()
        ultra_phot_x2 = ultra_table[phot_x[1]].tolist()
        
        ultra_phot_y1 = ultra_table[phot_y[0]].tolist()
        ultra_phot_y2 = ultra_table[phot_y[1]].tolist()
        
        if phot_x[0] == 'PANSTARRS_Y': 
            ultra_phot_x1 = np.subtract(ultra_phot_x1, 0.634)
        if phot_x[1] == 'PANSTARRS_Y': 
            ultra_phot_x2 = np.subtract(ultra_phot_x2, 0.634)
        if phot_y[0] == 'PANSTARRS_Y': 
            ultra_phot_y1 = np.subtract(ultra_phot_y1, 0.634)
        if phot_y[1] == 'PANSTARRS_Y': 
            ultra_phot_y2 = np.subtract(ultra_phot_y2, 0.634)
        
        ultra_x_list = np.subtract(ultra_phot_x1, ultra_phot_x2)
        ultra_y_list = np.subtract(ultra_phot_y1, ultra_phot_y2)
        
        sub_phot_x1 = sub_table[phot_x[0]].tolist()
        sub_phot_x2 = sub_table[phot_x[1]].tolist()
        
        sub_phot_y1 = sub_table[phot_y[0]].tolist()
        sub_phot_y2 = sub_table[phot_y[1]].tolist()
        
        sub_x_list = np.subtract(sub_phot_x1, sub_phot_x2)
        sub_y_list = np.subtract(sub_phot_y1, sub_phot_y2)
        
        ax.scatter(sub_x_list, sub_y_list, s = 80, c = 'lightpink', alpha = 1, zorder = 2, marker = 'd', edgecolor='k', linewidth=1)
        plt.scatter(ultra_x_list, ultra_y_list, s = 20, c = 'lightgray', alpha = 0.9, zorder = 1)
        
        # Filter the DataFrame based on model conditions
        if model == 'LOWZ':
            metal_list = [0.5, 0, -1, -2]
            color_list = ['tomato', 'lawngreen', 'lightskyblue', 'fuchsia']
            for j in range(len(metal_list)):
                subsolar_df = pd.DataFrame(table)
                subsolar_df = subsolar_df[(subsolar_df['C/O'] == 0.55) & 
                                        (subsolar_df['log(Kzz)'] == 2) & 
                                        (subsolar_df['log(g)'] == 5.0) & 
                                        (subsolar_df['[M/H]'] == metal_list[j]) & 
                                        (subsolar_df['teff'] % 100 == 0) & 
                                        (subsolar_df['teff'] >= min_temp) & 
                                        (subsolar_df['teff'] <= max_temp)]
                subsolar_df.sort_values(by='teff', inplace=True)
                
                subcolor_1 = np.subtract(subsolar_df[str(phot_x[0])], subsolar_df[str(phot_x[1])])
                subcolor_2 = np.subtract(subsolar_df[str(phot_y[0])], subsolar_df[str(phot_y[1])])
                ax.plot(subcolor_1, subcolor_2, c='k', lw=3, alpha=1)
                ax.plot(subcolor_1, subcolor_2, c=color_list[j], lw=2, label=f'[M/H] = {metal_list[j]}', alpha=1)

        elif model == 'ELF OWL':
            # Define metallicity and color lists for ELF OWL model
            metal_list = [0.5, 0, -0.5, -1.0]
            color_list = ['tomato', 'lawngreen', 'lightskyblue', 'fuchsia']
            # Iterate through metallicity list
            for j in range(len(metal_list)):
                # Filter DataFrame based on ELF OWL model conditions
                subsolar_df = pd.DataFrame(table)
                subsolar_df = subsolar_df[(subsolar_df['C/O'] == 0.5) & 
                                        (subsolar_df['log(Kzz)'] == 2) & 
                                        (subsolar_df['log(g)'] == 5.0) & 
                                        (subsolar_df['[M/H]'] == metal_list[j]) & 
                                        (subsolar_df['teff'] % 100 == 0) & 
                                        (subsolar_df['teff'] >= min_temp) & 
                                        (subsolar_df['teff'] <= max_temp)]
                subsolar_df.sort_values(by='teff', inplace=True)
                
                # Calculate color combinations
                subcolor_1 = np.subtract(subsolar_df[str(phot_x[0])], subsolar_df[str(phot_x[1])])
                subcolor_2 = np.subtract(subsolar_df[str(phot_y[0])], subsolar_df[str(phot_y[1])])
                
                # Plot color combinations
                ax.plot(subcolor_1, subcolor_2, c='k', lw=3, alpha=1)
                ax.plot(subcolor_1, subcolor_2, c=color_list[j], lw=2, label=f'[M/H] = {metal_list[j]}', alpha=1)

        elif model == 'SAND':
            # Define metallicity and alpha lists for SAND model
            metal_list = [0.3, 0.1, -1.1, -2.4]
            alpha_list = [0, 0, 0.15, 0.4]
            color_list = ['tomato', 'lawngreen', 'lightskyblue', 'fuchsia']
            # Iterate through metallicity list
            for j in range(len(metal_list)):
                # Filter DataFrame based on SAND model conditions
                subsolar_df = pd.DataFrame(table)
                subsolar_df = subsolar_df[(subsolar_df['log(g)'] == 5.0) & 
                                        (subsolar_df['[M/H]'] == metal_list[j]) & 
                                        (subsolar_df['[a/M]'] == alpha_list[j]) & 
                                        (subsolar_df['teff'] >= min_temp) & 
                                        (subsolar_df['teff'] <= max_temp)]
                subsolar_df.sort_values(by='teff', inplace=True)
                
                # Calculate color combinations
                subcolor_1 = np.subtract(subsolar_df[str(phot_x[0])], subsolar_df[str(phot_x[1])])
                subcolor_2 = np.subtract(subsolar_df[str(phot_y[0])], subsolar_df[str(phot_y[1])])
                
                # Plot color combinations
                ax.plot(subcolor_1, subcolor_2, c='k', lw=3, alpha=1)
                ax.plot(subcolor_1, subcolor_2, c=color_list[j], lw=2, label=f'[M/H] = {metal_list[j]}', alpha=1)


        # Labels and title
        ax.set_xlabel(f'{temp_color_x1} - {temp_color_x2}', fontsize=15)
        ax.set_ylabel(f'{temp_color_y1} - {temp_color_y2}', fontsize=15)
        ax.set_title(f'RMAD: {round(temp_rmad, 2)}', fontsize=20, loc='left', pad=10)  # Adjust padding for title
        
        # Button positions
        ax_yes = plt.axes([0.81, 0.92, 0.1, 0.05])
        ax_no = plt.axes([0.7, 0.92, 0.1, 0.05])
        
        # Buttons
        button_yes = Button(ax_yes, 'Yes', color='lightgreen', hovercolor='limegreen')
        button_no = Button(ax_no, 'No', color='lightcoral', hovercolor='red')
        
        # Button click events
        button_yes.on_clicked(on_yes)
        button_no.on_clicked(on_no)
        
        # Show plot
        ax.grid(True, alpha = 0.2)  # Add grid for better visualization of points
        ax.minorticks_on()  # Enable minor ticks
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, width=1)  # Set ticks direction and appearance
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()
        plt.close()
        
    final_color = []
    for i in range(len(temp_color_list)):
        if str(record[i]) == '(0, 0)': 
            pass
        else: 
            final_color.append(temp_color_list[i])

    # Print recorded points
    return final_color

# Plots RMAD histograms for each metalliticity in the model comparison
def HISTOGRAM(csv_file, output, model):
    """
    Function to create an histograms showing RMAD distribution.

    Parameters:
        csv_file (string): The input RMAD csv file. 
        output (string): Output file name.
        model (string): The selected model. 

    Returns:
        figure: Histogram of RMAD distribution.
    """
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Define colors for each 'M/H' value and label list based on the model
    if model == 'ELF OWL':
        # For ELF OWL model
        colors = ['lightskyblue', 'k']  # Color list for different metallicity values
        label_list = ['[M/H] = $-$0.5', '[M/H] = $-$1.0']  # Labels for legend
        column_num = 4  # Starting column index for data of interest
        unique_values = df['M/H'].unique()  # Unique 'M/H' values in the DataFrame
    
    elif model == 'LOWZ':
        # For LOWZ model
        colors = ['lightskyblue', 'violet', 'slateblue', 'darkmagenta', 'crimson']
        label_list = ['[M/H] = $-$0.5', '[M/H] = $-$1.0', '[M/H] = $-$1.5', '[M/H] = $-$2.0', '[M/H] = $-$2.5']
        column_num = 4
        unique_values = df['M/H'].unique()
    
    elif model == 'SAND':
        # For SAND model
        n = 9
        colors = plt.cm.cool(np.linspace(0,1,n))  # Generate a range of colors
        label_list = ['[α/H] = $-$0.20', '[α/H] = $-$0.35', '[α/H] = $-$0.55', '[α/H] = $-$0.75', '[α/H] = $-$0.95', '[α/H] = $-$1.15', '[α/H] = $-$1.20', '[α/H] = $-$1.65', '[α/H] = $-$2.00']
        column_num = 2
        unique_values = df['a/H'].unique()
    
    # Iterate over each column of interest
    for column in df.columns[column_num:]:
        # Create a new plot for each column
        plt.figure(figsize=(8, 6))
        plt.style.use('seaborn-paper')  # Set plot style
        plt.rcParams['font.family'] = 'Times New Roman'  # Set font style
        # Set title based on column name
        if str(column[0] + column[1]) == 'W1': 
            plt.title(f'Distribution of W1 $-$ W2', fontsize=20)
        else:
            plt.title(f'Distribution of {column[0]} $-$ {column[1:]}', fontsize=20)
        plt.xlabel('RMAD', fontsize=20)  # Set x-axis label
        plt.ylabel('Counts', fontsize=20)  # Set y-axis label
        
        # Determine histogram binning parameters
        min_buck = round(np.min(df[column]), 1)
        max_buck = round(np.max(df[column]), 1)
        bin_number = (max_buck - min_buck) / 16
        bins = np.arange(min_buck, max_buck, bin_number)
        
        # Iterate over unique values of 'M/H' or '[α/H]' and plot histograms
        hist_list = []
        for i, value in enumerate(unique_values):
            # Filter the DataFrame for the current column and metallicity or alpha value
            if model == 'ELF OWL' or model == 'LOWZ':
                filtered_df = df[(df['M/H'] == value)]
            elif model == 'SAND':
                filtered_df = df[(df['a/H'] == value)]
            hist_list.append(filtered_df[column])
            
        # Plot stacked histograms
        plt.hist(hist_list, color=colors, bins=bins, label=label_list, alpha=0.6, align='mid', stacked=True)
        plt.hist(hist_list, edgecolor='black', bins=bins, histtype='step', linewidth=1, stacked=True)
        
        tick_locs = bins[::2]  # Select major tick locations
        plt.xticks(tick_locs, fontsize=10)  # Set x-axis tick font size
        plt.yticks(fontsize=10)  # Set y-axis tick font size
        plt.legend(fontsize=14, frameon=False, bbox_to_anchor=(1, 0.85), loc='upper left')  # Add legend
        plt.grid(True, alpha=0.1)  # Add grid
        plt.minorticks_on()  # Enable minor ticks
        plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))  # Set x-axis tick format
        plt.savefig(f'Output/HISTOGRAM/{output}_{column}_HISTOGRAM.pdf', bbox_inches='tight', pad_inches=0.1, width=11)  # Save plot
        plt.close()  # Close the current figure to release memory

# Creates median RMAD heatmap     
def HEATMAP(output):
    """
    Function to create an heatmap showing photometric RMAD distribution.

    Parameters:
        output (string): Output file name.

    Returns:
        figure: Heatmap of photometric RMAD distribution.
    """
    
    # Load the RMAD and color data from the CSV file
    file = f'Output/RMAD_CSV/{output}_RMAD_COLOR.csv'
    df = pd.read_csv(file)
    
    # Determine unique metallicity or alpha values based on the model
    if model == 'ELF OWL' or model == 'LOWZ':
        unique_values = df['M/H'].unique()
        column_num = 4
    elif model == 'SAND':
        unique_values = df['a/H'].unique()
        column_num = 2
    
    # Iterate over each unique metallicity or alpha value
    for i in range(len(unique_values)):
        rmad_list = []  # List to store median RMAD values
        color_list = []  # List to store color combinations
        # Iterate over each color combination column
        for column in df.columns[column_num:]:
            color_list.append(column)  # Append color combination to the list
            # Filter the DataFrame for the current metallicity or alpha value
            metal_value = unique_values[i]
            if model == 'ELF OWL' or model == 'LOWZ':
                filtered_df = df[(df['M/H'] == metal_value)]
            elif model == 'SAND':
                filtered_df = df[(df['a/H'] == metal_value)]
            # Calculate median RMAD for the current color combination
            rmad_list.append(np.nanmedian(filtered_df[column]))
        
        # Define the location of each color combination on the heatmap grid
        color_location = [[1,2], [1, 3], [1, 7], [1, 8], [1, 4], [1, 5], [1, 6], 
                          [2, 3], [2, 7], [2, 8], [2, 4], [2, 5], [2, 6], 
                          [3, 7], [3, 8], [3, 4], [3, 5], [3, 6], 
                          [7, 8], 
                          [4, 7], [4, 8], [4, 5], [4, 6], 
                          [5, 7], [5, 8], [5, 6], 
                          [6, 7], [6, 8]]

        fig, ax = plt.subplots(figsize=(8, 8))  # Create a single figure with subplots

        x_values, y_values = zip(*color_location)
        x_values = np.subtract(x_values, 0.5)
        y_values = np.subtract(y_values, 0.5)
        plot2 = ax.scatter(x_values, y_values, marker='s', s= 2950, c=rmad_list, cmap='cool')  # Use ax.scatter instead of plt.scatter

        custom_x_ticks = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        ax.set_xticks(custom_x_ticks)  # Use ax.set_xticks to set custom x ticks
        ax.set_xticklabels(['i', 'z', 'y', 'J', 'H', 'K', 'W1', 'W2'], fontsize=25)

        custom_y_ticks = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        ax.set_yticks(custom_y_ticks)  # Use ax.set_yticks to set custom y ticks
        ax.set_yticklabels(['i', 'z', 'y', 'J', 'H', 'K', 'W1', 'W2'], fontsize=25)

        ax.tick_params(which='both', bottom=False, left=False)

        ax.set_xlim(0, 8)  # Use ax.set_xlim to set x-axis limits
        ax.set_ylim(0, 8)  # Use ax.set_ylim to set y-axis limits

        minor_locator = MultipleLocator(1)  # Set minor ticks at every integer
        ax.xaxis.set_minor_locator(minor_locator)
        ax.yaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='minor', linewidth = 1.5, c = 'k')

        cbaxes = fig.add_axes([0.125, 0.89, 0.775, 0.025])
        cbar2 = plt.colorbar(plot2, orientation='horizontal', cax=cbaxes)
        cbar2.ax.tick_params(labelsize=20)
        cbar2.ax.xaxis.set_ticks_position("top")
        cbar2.set_label('RMAD Score', fontsize=25, fontname='Times New Roman')
        cbar2.ax.xaxis.set_label_position('top')
        cbar2.ax.tick_params(labelsize=15)

        plt.savefig(f'Output/HEATMAP/{output}_METAL_{metal_value}_HEATMAP.pdf')

        plt.clf()  # Clear the current figure to release memory

# Creates color-color diagrams based on selected color-color combinations
def COLOR_COLOR(output, color_list, model): 
    """
    Function to create an color-color diagrams.

    Parameters:
        output (string): Output file name.
        color_list (list): List of selected color-color combinations
        model (string): Select model.

    Returns:
        figure: Figure of color-color diagram.
    """
    
    # Loads in all of the data for all four models
    sand_table = pd.read_csv('DATA_MODULE/SAND_SYN_PHOT.csv')
    lowz_table = pd.read_csv('DATA_MODULE/LOWZ_SYN_PHOT.csv')
    elf_table = pd.read_csv('DATA_MODULE/EWLOWL_SYN_PHOT.csv')

    # Loads in the data from the UltraCool sheet
    ultra_table = pd.read_csv('DATA_MODULE/UltraCool_Dwarfs.csv')
    sub_table = pd.read_csv('DATA_MODULE/UltraCool_Subdwarfs.csv')

    for color in color_list: 
        fig, ax = plt.subplots() 
        plt.style.use('seaborn-paper')
        plt.rcParams['font.family'] = 'Times New Roman'
        
        x_phot_1, x_phot_2 = color[0], color[1]
        y_phot_1, y_phot_2 = color[2], color[3]
        
        # Loads in all the columns from the Gerasimov model
        # ---------------------------------------------------------------------
        if model == 'SAND':
            print_metal = ['0.3', '0.1', '$-$1.1', '$-$1.95']
            sand_mh_select = [0.3, 0.1, -1.1, -1.95]
            sand_am_select = [0, 0, 0.15, 0.3]
            color_list = ['tomato', 'lawngreen', 'dodgerblue', 'violet']
            for i in range(len(sand_mh_select)): 
                filtered_df = sand_table[(sand_table['[M/H]'] == sand_mh_select[i]) & (sand_table['log(g)'] == 5.0) & (sand_table['[a/M]'] == sand_am_select[i])]
                teff_list = filtered_df['teff'].tolist()
                phot_x1 = filtered_df[x_phot_1].tolist()
                phot_x2 = filtered_df[x_phot_2].tolist()
                phot_y1 = filtered_df[y_phot_1].tolist()
                phot_y2 = filtered_df[y_phot_2].tolist()
                phot_x = np.subtract(phot_x1, phot_x2)
                phot_y = np.subtract(phot_y1, phot_y2)
                sorted_list = sorted(zip(teff_list, phot_x, phot_y), key=lambda x: x[0], reverse=True)
                sorted_teff, sorted_x, sorted_y = zip(*sorted_list)
                ax.plot(   sorted_x, sorted_y, c='k', lw=4, zorder = 3)
                ax.plot(   sorted_x, sorted_y, c=color_list[i], lw=3, zorder = 3)
                ax.scatter(sorted_x, sorted_y, c=color_list[i], label=f'[M/H] = {print_metal[i]}', s=[200 if teff == 1500 else 80 if (teff % 500 == 0 and teff != 0) else 0 for teff in sorted_teff], edgecolor='k', linewidth=1, zorder = 4)  
            # ---------------------------------------------------------------------
        
        # Loads in all the LOWZ models
        # ---------------------------------------------------------------------
        elif model == 'LOWZ':
            print_metal = ['0.5', '0.0', '$-$1.0', '$-$2.0']
            lowz_mh_select = [0.5, 0, -1, -2]
            color_list = ['tomato', 'lawngreen', 'dodgerblue', 'violet']
            for i in range(len(lowz_mh_select)): 
                filtered_df = lowz_table[(lowz_table['[M/H]'] == lowz_mh_select[i]) & (lowz_table['log(g)'] == 5.0) & (lowz_table['C/O'] == 0.55) & (lowz_table['log(Kzz)'] == 2)]
                teff_list = filtered_df['teff'].tolist()
                phot_x1 = filtered_df[x_phot_1].tolist()
                phot_x2 = filtered_df[x_phot_2].tolist()
                phot_y1 = filtered_df[y_phot_1].tolist()
                phot_y2 = filtered_df[y_phot_2].tolist()
                phot_x = np.subtract(phot_x1, phot_x2)
                phot_y = np.subtract(phot_y1, phot_y2)
                sorted_list = sorted(zip(teff_list, phot_x, phot_y), key=lambda x: x[0], reverse=True)
                sorted_teff, sorted_x, sorted_y = zip(*sorted_list)
                ax.plot(   sorted_x, sorted_y, c='k', lw=4, zorder = 3)
                ax.plot(   sorted_x, sorted_y, c=color_list[i], lw=3, zorder = 3)
                ax.scatter(sorted_x, sorted_y, c=color_list[i], label=f'[M/H] = {print_metal[i]}', s=[200 if teff == 1500 else 80 if (teff % 500 == 0 and teff != 0) else 0 for teff in sorted_teff], edgecolor='k', linewidth=1, zorder = 4)
            # ---------------------------------------------------------------------
        
        # Loads in all the ELF OWL Models
        # ---------------------------------------------------------------------
        elif model == 'ELF OWL':
            print_metal = ['0.5', '0.0', '$-$0.5', '$-$1.0']
            elf_mh_select  = [0.5, 0, -0.5, -1.0]
            color_list = ['tomato', 'lawngreen', 'dodgerblue', 'violet']
            for i in range(len(elf_mh_select)): 
                filtered_df = elf_table[(elf_table['[M/H]'] == elf_mh_select[i]) & (elf_table['log(g)'] == 5.0) & (elf_table['C/O'] == 0.5) & (elf_table['log(Kzz)'] == 2)]
                teff_list = filtered_df['teff'].tolist()
                phot_x1 = filtered_df[x_phot_1].tolist()
                phot_x2 = filtered_df[x_phot_2].tolist()
                phot_y1 = filtered_df[y_phot_1].tolist()
                phot_y2 = filtered_df[y_phot_2].tolist()
                phot_x = np.subtract(phot_x1, phot_x2)
                phot_y = np.subtract(phot_y1, phot_y2)
                sorted_list = sorted(zip(teff_list, phot_x, phot_y), key=lambda x: x[0], reverse=True)
                sorted_teff, sorted_x, sorted_y = zip(*sorted_list)
                ax.plot(   sorted_x, sorted_y, c='k', lw=4, zorder = 3)
                ax.plot(   sorted_x, sorted_y, c=color_list[i], lw=3, zorder = 3)
                ax.scatter(sorted_x, sorted_y, c=color_list[i], label=f'[M/H] = {print_metal[i]}', s=[200 if teff == 1500 else 80 if (teff % 500 == 0 and teff != 0) else 0 for teff in sorted_teff], edgecolor='k', linewidth=1, zorder = 4)   
            # ---------------------------------------------------------------------

        # Loads in the needed photometry from the UltraCool Sheet
        ultra_phot_x1 = ultra_table[x_phot_1].tolist()           # Extracts photometry data for x_phot_1
        ultra_phot_x2 = ultra_table[x_phot_2].tolist()           # Extracts photometry data for x_phot_2
        ultra_phot_x1_e = ultra_table[x_phot_1 + 'e'].tolist()   # Extracts error data for x_phot_1
        ultra_phot_x2_e = ultra_table[x_phot_2 + 'e'].tolist()   # Extracts error data for x_phot_2

        ultra_phot_y1 = ultra_table[y_phot_1].tolist()           # Extracts photometry data for y_phot_1
        ultra_phot_y2 = ultra_table[y_phot_2].tolist()           # Extracts photometry data for y_phot_2
        ultra_phot_y1_e = ultra_table[y_phot_1 + 'e'].tolist()   # Extracts error data for y_phot_1
        ultra_phot_y2_e = ultra_table[y_phot_2 + 'e'].tolist()   # Extracts error data for y_phot_2

        # Adjusts the photometry data if necessary
        if x_phot_1 == 'PANSTARRS_Y': 
            ultra_phot_x1 = np.subtract(ultra_phot_x1, 0.634)
        if x_phot_2 == 'PANSTARRS_Y': 
            ultra_phot_x2 = np.subtract(ultra_phot_x2, 0.634)
        if y_phot_1 == 'PANSTARRS_Y': 
            ultra_phot_y1 = np.subtract(ultra_phot_y1, 0.634)
        if y_phot_2 == 'PANSTARRS_Y': 
            ultra_phot_y2 = np.subtract(ultra_phot_y2, 0.634)

        # Calculates the differences in photometry
        ultra_x_list = np.subtract(ultra_phot_x1, ultra_phot_x2)   # Calculates x-axis differences
        ultra_y_list = np.subtract(ultra_phot_y1, ultra_phot_y2)   # Calculates y-axis differences
        # Calculates error combining errors from x_phot_1 and x_phot_2, and y_phot_1 and y_phot_2
        ultra_x_list_e = np.sqrt(np.square(ultra_phot_x1_e) + np.square(ultra_phot_x2_e))   # Combines errors for x-axis
        ultra_y_list_e = np.sqrt(np.square(ultra_phot_y1_e) + np.square(ultra_phot_y2_e))   # Combines errors for y-axis

        # Extracts photometry data from the subdwarf table
        sub_phot_x1 = sub_table[x_phot_1].tolist()               # Extracts photometry data for x_phot_1
        sub_phot_x2 = sub_table[x_phot_2].tolist()               # Extracts photometry data for x_phot_2
        sub_phot_x1_e = sub_table[x_phot_1 + 'e'].tolist()       # Extracts error data for x_phot_1
        sub_phot_x2_e = sub_table[x_phot_2 + 'e'].tolist()       # Extracts error data for x_phot_2

        sub_phot_y1 = sub_table[y_phot_1].tolist()               # Extracts photometry data for y_phot_1
        sub_phot_y2 = sub_table[y_phot_2].tolist()               # Extracts photometry data for y_phot_2
        sub_phot_y1_e = sub_table[y_phot_1 + 'e'].tolist()       # Extracts error data for y_phot_1
        sub_phot_y2_e = sub_table[y_phot_2 + 'e'].tolist()       # Extracts error data for y_phot_2

        # Calculates the differences in photometry for subdwarf data
        sub_x_list = np.subtract(sub_phot_x1, sub_phot_x2)       # Calculates x-axis differences
        sub_y_list = np.subtract(sub_phot_y1, sub_phot_y2)       # Calculates y-axis differences
        # Calculates error combining errors from x_phot_1 and x_phot_2, and y_phot_1 and y_phot_2
        sub_x_list_e = np.sqrt(np.square(sub_phot_x1_e) + np.square(sub_phot_x2_e))   # Combines errors for x-axis
        sub_y_list_e = np.sqrt(np.square(sub_phot_y1_e) + np.square(sub_phot_y2_e))   # Combines errors for y-axis

        sub_spt_list = sub_table['LIT_SPT_NUM'].tolist()         # Extracts subdwarf spectral type data
        sub_metal_list = sub_table['LIT_METAL_NUM'].tolist()     # Extracts subdwarf metallicity data

        # Separates subdwarf data into metallicity categories
        sd_x_list, esd_x_list, usd_x_list = [], [], []           # Lists for x-axis values
        sd_xe_list, esd_xe_list, usd_xe_list = [], [], []       # Lists for x-axis errors
        sd_y_list, esd_y_list, usd_y_list = [], [], []           # Lists for y-axis values
        sd_ye_list, esd_ye_list, usd_ye_list = [], [], []       # Lists for y-axis errors
        sd_spt_list, esd_spt_list, usd_spt_list = [], [], []     # Lists for spectral type values
        
        # Loop through the subdwarf metallicity list
        for i in range(len(sub_metal_list)): 
            # Categorize subdwarf data based on metallicity
            if sub_metal_list[i] == -1: 
                # Append data for metallicity -1 to respective lists
                sd_x_list.append(sub_x_list[i])     # x-axis values
                sd_y_list.append(sub_y_list[i])     # y-axis values
                sd_xe_list.append(sub_x_list_e[i])  # x-axis errors
                sd_ye_list.append(sub_y_list_e[i])  # y-axis errors
                sd_spt_list.append(sub_spt_list[i]) # spectral type values
            elif sub_metal_list[i] == -2: 
                # Append data for metallicity -2 to respective lists
                esd_x_list.append(sub_x_list[i])    # x-axis values
                esd_y_list.append(sub_y_list[i])    # y-axis values
                esd_xe_list.append(sub_x_list_e[i]) # x-axis errors
                esd_ye_list.append(sub_y_list_e[i]) # y-axis errors
                esd_spt_list.append(sub_spt_list[i])# spectral type values
            elif sub_metal_list[i] == -3: 
                # Append data for metallicity -3 to respective lists
                usd_x_list.append(sub_x_list[i])    # x-axis values
                usd_y_list.append(sub_y_list[i])    # y-axis values
                usd_xe_list.append(sub_x_list_e[i]) # x-axis errors
                usd_ye_list.append(sub_y_list_e[i]) # y-axis errors
                usd_spt_list.append(sub_spt_list[i])# spectral type values

        # Extract ultra_spt data and filter out NaN values from ultra_x_list and ultra_y_list
        ultra_spt = ultra_table['sptnum'].tolist()         # Extracts spectral type data for ultracool objects
        filtered_lists = zip(ultra_spt, ultra_x_list, ultra_y_list)
        filtered_lists = [(a, b, c) for a, b, c in filtered_lists if not (math.isnan(b) or math.isnan(c))]
        list1_filtered, list2_filtered, list3_filtered = zip(*filtered_lists)

        # Plotting
        # Scatter plot for ultracool objects
        plt.errorbar(ultra_x_list, ultra_y_list, xerr=ultra_x_list_e, yerr=ultra_y_list_e, color='lightgray', linestyle='None', linewidth=1, zorder = 1, alpha = 0.4)
        plot1 = ax.scatter(list2_filtered, list3_filtered, s = 20, c = 'lightgray', alpha = 0.9, zorder = 1)

        # Scatter plots for subdwarfs with different metallicities
        plt.errorbar(sd_x_list, sd_y_list, xerr=sd_xe_list, yerr=sd_ye_list, color='black', linestyle='None', linewidth=1, zorder = 2)
        plot2 = ax.scatter(sd_x_list, sd_y_list, s = 100, cmap = 'cool', c = sd_spt_list, alpha = 1, zorder = 2, marker = '^', edgecolor='k', linewidth=1)

        plt.errorbar(esd_x_list, esd_y_list, xerr=esd_xe_list, yerr=esd_ye_list, color='black', linestyle='None', linewidth=1, zorder = 2)
        ax.scatter(esd_x_list, esd_y_list, s = 100, cmap = 'cool', c = esd_spt_list, alpha = 1, zorder = 2, marker = 'd', edgecolor='k', linewidth=1)

        plt.errorbar(usd_x_list, usd_y_list, xerr=usd_xe_list, yerr=usd_ye_list, color='black', linestyle='None', linewidth=1, zorder = 2)
        ax.scatter(usd_x_list, usd_y_list, s = 100, cmap = 'cool', c = usd_spt_list, alpha = 1, zorder = 2, marker = 'p', edgecolor='k', linewidth=1)

        # Dummy scatter plots for legend
        plt.scatter(-100000, -100000, s = 20, c = 'lightgray', label = 'D', alpha = 0.9 )
        plt.scatter(-100000, -100000, s = 80, c = 'k', alpha = 1, marker = '^', label = 'SD')
        plt.scatter(-100000, -100000, s = 80, c = 'k', alpha = 1, marker = 'd', label = 'ESD')
        plt.scatter(-100000, -100000, s = 80, c = 'k', alpha = 1, marker = 'p', label = 'USD')

        # Plot formatting
        plt.minorticks_on()
        ax.set_xlabel(f'{x_phot_1} $-$ {x_phot_2}', fontsize = 20, fontname='Times New Roman')
        ax.set_ylabel(f'{y_phot_1} $-$ {y_phot_2}', fontsize = 20, fontname='Times New Roman')
        plt.tick_params(which='minor', width = 1, length = 3, labelsize = 15)
        plt.tick_params(which='major', width = 2, length = 6, labelsize = 15)
        plt.xticks(fontname='Times New Roman', fontsize = 10)
        plt.yticks(fontname='Times New Roman', fontsize = 10)
        ax.grid(True, alpha = 0.2)

        # Set the limits of the plot
        list2_tuple = tuple(list2_filtered)
        list3_tuple = tuple(list3_filtered)
        ax.set_xlim(np.nanmin(list2_tuple) - 0.5, np.nanmax(list2_tuple) + 0.5)
        ax.set_ylim(np.nanmin(list3_tuple) - 0.5, np.nanmax(list3_tuple) + 0.5)

        # Add legend
        legend = plt.legend(prop={'family': 'Times New Roman', 'size': 15},  edgecolor='black', ncol=1, fontsize=14, frameon=True, bbox_to_anchor=(1, 0.75), loc='upper left')

        # Set figure size
        figure = plt.gcf()
        figure.set_size_inches(9, 8) 
        
        # Initialize an empty list to store spectral type values
        result_list = []

        # Loop through the subdwarf data and filter out NaN values
        for val1, val2, val3 in zip(sub_x_list, sub_y_list, sub_spt_list):
            if val1 is not None and not math.isnan(val1) and val2 is not None and not math.isnan(val2):
                result_list.append(val3)

        # Determine the maximum spectral type and adjust tick positions and labels accordingly
        max_spt = int(max(result_list) / 4)
        if max_spt == 2: 
            tick_positions = [8]
            tick_labels = ['M8']
        elif max_spt == 3: 
            tick_positions = [8, 12]
            tick_labels = ['M8', 'L2']
        elif max_spt == 4: 
            tick_positions = [8, 12, 16]
            tick_labels = ['M8', 'L2', 'L6']
        elif max_spt == 5: 
            tick_positions = [8, 12, 16, 20]
            tick_labels = ['M8', 'L2', 'L6', 'T0']
        elif max_spt == 6: 
            tick_positions = [8, 12, 16, 20, 24]
            tick_labels = ['M8', 'L2', 'L6', 'T0', 'T4']
        elif max_spt == 7:
            tick_positions = [8, 12, 16, 20, 24, 28]
            tick_labels = ['M8', 'L2', 'L6', 'T0', 'T4', 'T8']     

        # Add color bar
        cbaxes = fig.add_axes([0.125, 0.88, 0.775, 0.02])  # Color bar axes position
        cbar2 = plt.colorbar(plot2, orientation='horizontal', cax=cbaxes)  # Create color bar
        cbar2.ax.tick_params(labelsize=10)  # Set tick label size
        cbar2.ax.xaxis.set_ticks_position("top")  # Set tick position
        cbar2.set_label('Subdwarf Spectral Type', fontsize=15, fontname='Times New Roman')  # Set color bar label
        cbar2.ax.xaxis.set_label_position('top')  # Set label position
        cbar2.ax.tick_params(labelsize=8)  # Set tick size
        cbar2.set_ticks(tick_positions)  # Set tick positions
        cbar2.set_ticklabels(tick_labels)  # Set tick labels

        # Save the figure
        plt.savefig(f'Output/COLOR_COLOR/{output}_{x_phot_1}-{x_phot_2}_vs_{y_phot_1}-{y_phot_2}_COLOR_COLOR.pdf', bbox_inches='tight', pad_inches=0.1, width=11)

        # Close the figure and clear all data
        plt.clf()
        plt.close('all')

# The function that the GUI calls to run the rest of the code
def MAIN(model_comp, max_teff_comp, min_teff_comp, output):
    """
    Function to run the rest of the code. 

    Parameters:
        model_comp (string): The comparison model choosen.
        max_teff_comp (int): The minimum temperature. 
        min_teff_comp (int): The maximum temperature.
        output (string): The output file name. 
    """
    
    # Depending on the model comparison type, load the appropriate data table
    if model_comp == 'SAND': 
        table = pd.read_csv('DATA_MODULE/SAND_SYN_PHOT.csv')
        teff_list = table['teff'].tolist()
        metal_list = table['[M/H]'].tolist()
        logg_list = table['log(g)'].tolist()
    elif model_comp == 'LOWZ': 
        table = pd.read_csv('DATA_MODULE/LOWZ_SYN_PHOT.csv')
        teff_list = table['teff'].tolist()
        metal_list = table['[M/H]'].tolist()
        logg_list = table['log(g)'].tolist()
    elif model_comp == 'ELF OWL': 
        table = pd.read_csv('DATA_MODULE/EWLOWL_SYN_PHOT.csv')
        teff_list = table['teff'].tolist()
        metal_list = table['[M/H]'].tolist()
        logg_list = table['log(g)'].tolist()

    # Define photometric bands and their central wavelengths
    phot_list = ['PANSTARRS_I', 'PANSTARRS_Z', 'PANSTARRS_Y', 'WISE_W1', 'MKO_J', 'MKO_H', 'MKO_K', 'WISE_W2']
    phot_center = [0.7563, 0.8690, 0.9644, 3.3526, 1.2528, 1.6422, 2.2132, 4.6028]
    colors = []

    # Generate all possible color combinations
    for i in range(len(phot_list)):
        temp_filt = phot_center[i]
        n_tot = len(phot_center)
        n = 0
        while n < n_tot: 
            if (temp_filt - phot_center[n]) < 0: 
                color_color = [phot_list[i], phot_list[n]]
                colors.append(color_color)
            else: 
                pass
            n += 1
        
    # Perform calculations based on the selected model comparison type
    if model_comp == 'LOWZ':
        # Define input parameters for LOWZ model comparison
        input_metal = [-0.5, -1, -1.5, -2.0, -2.5]
        input_logg = [4.0, 4.5, 5.0]
        input_co = [0.1, 0.55, 0.85]
        input_kzz = [-1, 2, 10]
        total_rmad_list = []
        total_color_color_list = []
        num_columns = 32
        array = np.empty((0, num_columns))
        new_row = np.array(['M/H', 'log(g)', 'C/O', 'Kzz', 'IZ', 'IY', 'IW1', 'IW2', 'IJ', 'IH', 'IK', 'ZY', 'ZW1', 'ZW2', 'ZJ', 'ZH', 'ZK', 
                            'YW1', 'YW2', 'YJ', 'YH', 'YK', 'W1W2', 'JW1', 'JW2', 'JH', 'JK', 'HW1', 'HW2', 'HK', 'KW1', 'KW2'])  
        array = np.vstack([array, new_row])
        comp_alpha = np.nan
        directory = ''
        
        # Iterate over input parameters and perform calculations
        for comp_metal in input_metal: 
            for comp_logg in input_logg: 
                for comp_co in input_co: 
                    for comp_kzz in input_kzz: 
                        solar_comp_list = SOLAR_FINDER(comp_logg, table, metal_list, logg_list, model_comp, comp_co, comp_kzz, comp_alpha, min_teff_comp, max_teff_comp)
                        subsolar_comp_list = SUBSOLAR_FINDER(comp_metal, comp_logg, table, metal_list, logg_list, model_comp, comp_co, comp_kzz, comp_alpha, min_teff_comp, max_teff_comp)
                        if not solar_comp_list.empty:
                            rmad_list, color_list, array = RMAD_COMP(phot_list, phot_center, subsolar_comp_list, solar_comp_list, array, colors, comp_co, comp_kzz, comp_metal, comp_logg, comp_alpha, total_color_color_list, total_rmad_list)
                            file_path = f"Output/RMAD_CSV/{output}_RMAD_COLOR.csv"
                            with open(file_path, 'w', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(array)
        # Process results and generate color-color diagrams
        flattened_rmad = [item for sublist in total_rmad_list for item in sublist]
        flattened_color = [item for sublist in total_color_color_list for item in sublist]
        final_color = INTER_PLOT(flattened_rmad, flattened_color, table, model_comp, min_teff_comp, max_teff_comp)
        csv_file = f"Output/RMAD_CSV/{output}_RMAD_COLOR.csv"
        HISTOGRAM(csv_file, output, model_comp)
        HEATMAP(output)
        if final_color:
            COLOR_COLOR(output, final_color, model_comp)
        print('      FINAL COLOR-COLOR SELECTION INCLUDES:')
        for i in range(len(final_color)):
            print(f'COLOR {i}: {final_color[i]}')
    elif model_comp == 'ELF OWL':
        # Define input parameters for ELF OWL model comparison
        input_metal = [-0.5, -1]  # List of metallicity values
        input_logg = [4.0, 4.5, 5.0, 5.5]  # List of log(g) values
        input_co = [0.5, 1, 1.5, 2]  # List of C/O ratios
        input_kzz = [2, 4, 7, 8, 9]  # List of Kzz values
        total_rmad_list = []  # Initialize list to store RMAD values
        total_color_color_list = []  # Initialize list to store color-color data
        num_columns = 32  # Specify the number of columns for the data array
        array = np.empty((0, num_columns))  # Initialize an empty array for data storage
        new_row = np.array(['M/H', 'log(g)', 'C/O', 'Kzz', 'IZ', 'IY', 'IW1', 'IW2', 'IJ', 'IH', 'IK', 'ZY', 'ZW1', 'ZW2', 'ZJ', 'ZH', 'ZK', 
                            'YW1', 'YW2', 'YJ', 'YH', 'YK', 'W1W2', 'JW1', 'JW2', 'JH', 'JK', 'HW1', 'HW2', 'HK', 'KW1', 'KW2'])  
        array = np.vstack([array, new_row])  # Add column labels to the data array
        comp_alpha = np.nan  # Set alpha value to NaN
        directory = ''  # Define the directory path
        
        # Iterate over input parameters and perform calculations
        for comp_metal in input_metal: 
            for comp_logg in input_logg: 
                for comp_co in input_co: 
                    for comp_kzz in input_kzz: 
                        solar_comp_list = SOLAR_FINDER(comp_logg, table, metal_list, logg_list, model_comp, comp_co, comp_kzz, comp_alpha, min_teff_comp, max_teff_comp)
                        subsolar_comp_list = SUBSOLAR_FINDER(comp_metal, comp_logg, table, metal_list, logg_list, model_comp, comp_co, comp_kzz, comp_alpha, min_teff_comp, max_teff_comp)
                        # Check if solar comparison list is not empty
                        if not solar_comp_list.empty:
                            # Calculate RMAD values, color list, and update the data array
                            rmad_list, color_list, array = RMAD_COMP(phot_list, phot_center, subsolar_comp_list, solar_comp_list, array, colors, comp_co, comp_kzz, comp_metal, comp_logg, comp_alpha, total_color_color_list, total_rmad_list)
                            # Write data array to a CSV file
                            file_path = f"Output/RMAD_CSV/{output}_RMAD_COLOR.csv"
                            with open(file_path, 'w', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(array)
                        else: 
                            pass  # Continue to the next iteration if solar comparison list is empty
        
        # Process results and generate color-color diagrams
        flattened_rmad = [item for sublist in total_rmad_list for item in sublist]
        flattened_color = [item for sublist in total_color_color_list for item in sublist]
        final_color = INTER_PLOT(flattened_rmad, flattened_color, table, model_comp, min_teff_comp, max_teff_comp)
        csv_file = f"Output/RMAD_CSV/{output}_RMAD_COLOR.csv"  # Path to the CSV file
        HISTOGRAM(csv_file, output, model_comp)  # Generate histogram
        HEATMAP(output)  # Generate heatmap
        if final_color:  # If final color data is available
            COLOR_COLOR(output, final_color, model_comp)  # Generate color-color diagrams
        print('      FINAL COLOR-COLOR SELECTION INCLUDES:')
        for i in range(len(final_color)):
            print(f'COLOR {i}: {final_color[i]}')  # Print selected color data
    elif model_comp == 'SAND': 
        total_rmad_list = []  # Initialize list to store RMAD values
        total_color_color_list = []  # Initialize list to store color-color data
        input_alpha = [-0.2, -0.35, -0.55, -0.75, -0.95, -1.15, -1.2, -1.65, -2.0]  # List of alpha values
        input_logg = [4.0, 4.5, 5.0, 5.5, 6.0]  # List of log(g) values
        num_columns = 30  # Specify the number of columns for the data array
        array = np.empty((0, num_columns))  # Initialize an empty array for data storage
        new_row = np.array(['log(g)', 'a/H', 'IZ', 'IY', 'IW1', 'IW2', 'IJ', 'IH', 'IK', 'ZY', 'ZW1', 'ZW2', 'ZJ', 'ZH', 'ZK', 
                            'YW1', 'YW2', 'YJ', 'YH', 'YK', 'W1W2', 'JW1', 'JW2', 'JH', 'JK', 'HW1', 'HW2', 'HK', 'KW1', 'KW2'])  
        array = np.vstack([array, new_row])  # Add column labels to the data array
        directory = ''  # Define the directory path
        print('--------------------------------------------------------')
        print('               STARTED SAND CALCULATION                 ')  # Indicate the start of SAND calculation
        
        # Iterate over alpha and log(g) values and perform calculations
        for comp_alpha in input_alpha: 
            for comp_logg in input_logg: 
                # Find solar comparison list and subsolar comparison list
                solar_comp_list = SOLAR_FINDER(comp_logg, table, metal_list, logg_list, model_comp, np.nan, np.nan, comp_alpha, min_teff_comp, max_teff_comp)
                subsolar_comp_list = SUBSOLAR_FINDER(np.nan, comp_logg, table, metal_list, logg_list, model_comp, np.nan, np.nan, comp_alpha, min_teff_comp, max_teff_comp)
                # Calculate RMAD values, color list, and update the data array
                rmad_list, color_list, array = RMAD_COMP(phot_list, phot_center, subsolar_comp_list, solar_comp_list, array, colors, np.nan, np.nan, np.nan, comp_logg, comp_alpha, total_color_color_list, total_rmad_list)
                # Write data array to a CSV file
                file_path = f"Output/RMAD_CSV/{output}_RMAD_COLOR.csv"
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(array)
                    
        print('        PLEASE SELECT BEST COLOR-COLOR DIAGRAMS          ')  # Prompt to select best color-color diagrams
        
        # Process results and generate color-color diagrams
        flattened_rmad = [item for sublist in total_rmad_list for item in sublist]
        flattened_color = [item for sublist in total_color_color_list for item in sublist]
        final_color = INTER_PLOT(flattened_rmad, flattened_color, table, model_comp, min_teff_comp, max_teff_comp)
        csv_file = f"Output/RMAD_CSV/{output}_RMAD_COLOR.csv"  # Path to the CSV file
        HISTOGRAM(csv_file, output, model_comp)  # Generate histogram
        HEATMAP(output)  # Generate heatmap
        if len(final_color) != 0:  # If final color data is available
            COLOR_COLOR(output, final_color, model_comp)  # Generate color-color diagrams
        print('      FINAL COLOR-COLOR SELECTION INCLUDES:')  # Indicate the final color-color selection
        for i in range(len(final_color)):
            print(f'COLOR {i}: {final_color[i]}')  # Print selected color data
                        
    print('--------------------------------------------------------')  # Indicate the end of SAND calculation

# Set the theme of the GUI
sg.theme('Lightgreen8')

# Define the layout of the GUI window
layout = [
    [sg.Text('KALEIDOSCOPE', justification='center', size=(100, 1), font=('Chalkduster', 45))],
    [
        [sg.Text('SELECT A MODEL:', font=('Rockwell', 20))],
        [sg.Combo(['LOWZ', 'SAND', 'ELF OWL'], size=(40, 1), key='-OPTION-', enable_events=True, font=('Rockwell'))],
        [sg.Text('MIN TEFF:', font=('Rockwell')), sg.InputText(size=(100, 1), key='-MIN_TEFF-', font=('Rockwell'))],
        [sg.Text('MAX TEFF:', font=('Rockwell')), sg.InputText(size=(100, 1), key='-MAX_TEFF-', font=('Rockwell'))],
        [sg.Text('OUTPUT FILE:', font=('Rockwell')), sg.InputText(size=(50, 1), key='-OUTPUT_FILE-', font=('Rockwell'))],
    ],
    [
        sg.Button('RUN', font=('Rockwell'), size=(12), button_color='#95D49B' ),
        sg.Button('HELP', font=('Rockwell'), size=(12), button_color='#F7CC7C'),
        sg.Button('CLOSE', font=('Rockwell'), size=(12), button_color='#E48671')
    ]
]

# Create the window
window = sg.Window('KALEIDOSCOPE', layout, size=(400, 260), grab_anywhere=False, finalize=True, enable_close_attempted_event=True)

# Define lists of temporary photometric bands and their central wavelengths
temp_list = ['PANSTARRS_G', 'PANSTARRS_R', 'PANSTARRS_I', 'PANSTARRS_Z', 'PANSTARRS_Y',
             'GAIA_G', 'GAIA_B', 'GAIA_R','WISE_W1', 
             'MKO_J', 'MKO_H', 'MKO_K', 'WISE_W2']
temp_center = [ 0.49, 0.6241, 0.7563, 0.8690, 0.9644, 
                0.6735, 0.5319, 0.7993,3.3526, 
                1.2528, 1.6422, 2.2132, 4.6028]
  
# Main event loop
while True:
    
    # Read events and values from the window
    event, values = window.read()
    
    # Check for different events
    if event == 'HELP':
        # Provide help information
        print('----------------------------------')
        print('PLEASE EMAIL: hcb98@nau.edu')
        print('----------------------------------')
    elif event == 'CLOSE':
        # Exit the loop and close the window
        break
    elif event == 'RUN': 
        # Execute when the 'RUN' button is clicked
        print('----------------------------------')
        try:
            # Attempt to get the selected model
            model = values['-OPTION-']
        except: 
            # Print error message if model is not selected
            print('PLEASE SELECT A MODEL')
        try: 
            # Attempt to get the minimum temperature
            min_temp = int(values['-MIN_TEFF-'])
        except: 
            # Print error message if minimum temperature is not a number
            print('PLEASE INPUT A NUMBER FOR MIN TEMP')
        try:
            # Attempt to get the maximum temperature
            max_temp = int(values['-MAX_TEFF-'])
        except:
            # Print error message if maximum temperature is not a number
            print('PLEASE INPUT A NUMBER FOR MAX TEMP')
        try:
            # Attempt to get the output file name
            output_file = values['-OUTPUT_FILE-']
        except:
            # Print error message if output file name is not provided
            print('PLEASE INPUT AN OUTPUT FILE NAME')
        
        # Check for validity of inputs
        if model == '':
            print('PLEASE SELECT A MODEL')
        elif type(min_temp) != int:
            print('PLEASE INPUT A NUMBER FOR MIN TEMP')
        elif type(max_temp) != int: 
            print('PLEASE INPUT A NUMBER FOR MAX TEMP')
        elif output_file == '': 
            print('PLEASE INPUT AN OUTPUT FILE NAME')
        else: 
            # Call the MAIN function with provided inputs
            test = MAIN(model, max_temp, min_temp, output_file)
        print('----------------------------------')
    elif event == '-WINDOW CLOSE ATTEMPTED-': 
        # Exit the loop if the window close button is clicked
        break

# Close the window
window.close()