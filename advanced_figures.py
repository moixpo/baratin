#advanced_figures.py
#Version 1.0
#------------------------------------------------------------
#visualisations of a solar system with storage
#Moix P-O
#Albedo Engineering 2025
#MIT license
#------------------------------------------------------------
#This file regroups functions to build advanced figures for the analysis of a solar + storage system
#inputs of function are df and names of columns is linked to what I generally use in various projects with inverters + batteries


#from tkinter import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

from PIL import Image


#settings Csts
FRANCAIS_LANGUAGE = True #sometimes used for labels in french, else in english
I_WANT_TO_SAVE_PNG = False #To export pictures after building them
I_WANT_WATERMARK_ON_FIGURE = True #To add a watermark on the figures

FIGSIZE_WIDTH=12
FIGSIZE_HEIGHT=6
WATERMARK_PICTURE='media/LogoAlbedo_90x380.png'
#WATERMARK_PICTURE='media/watermark_logo2.png'

#colorset
PINK_COLOR='#FFB2C7'
RED_COLOR='#CC0000'
WHITE_COLOR='#FFFFFF'
A_RED_COLOR="#9A031E"
A_YELLOW_COLOR="#F7B53B"
A_BLUE_COLOR="#2E5266"
A_RAISINBLACK_COLOR="#272838"
A_BLUEGREY_COLOR="#7E7F9A"
A_GREY_COLOR_SLATE="#6E8898"
A_GREY_COLOR2_BLUED="#9FB1BC"
A_GREY_COLOR3_LIGHT="#F9F8F8"


#Stud
NX_LIGHT_BLUE="#F0F6F8"
NX_BLUE="#6BA3B8"
NX_BROWN="#A2A569"
NX_LIGHT_BROWN="#E3E4D2"
NX_PINK="#B06B96"
NX_GREEN="#78BE20"



#COLORS CHOICES FROM THE SETS ABOVE:
FIGURE_FACECOLOR=WHITE_COLOR #NX_LIGHT_BROWN
AXE_FACECOLOR=WHITE_COLOR
SOLAR_COLOR=A_YELLOW_COLOR
LOAD_COLOR=A_RED_COLOR
GENSET_COLOR=A_BLUE_COLOR



def build_SOC_heatmap_figure(hours_mean_df):
    """ take the SOC channel from the dataframe given with time index and hours steps
    and build a heatmap figure by hour of the day and day of the year"""

    all_channels_labels = list(hours_mean_df.columns)
    channel_number_SOC = [i for i, elem in enumerate(all_channels_labels) if ('SOC' in elem) ]
        
    #print(all_channels_labels)
    #print(channel_number_SOC)

    # Take the data of SOC channel
    energies_by_hours=hours_mean_df[all_channels_labels[channel_number_SOC[0]]]

    # Extract the values from the dataframe
    consumption_data = energies_by_hours.values
    #print(consumption_data)

    # Determine the shape of the reshaped array
    n_days = len(consumption_data) // (24)
    n_hours = 24
    
    # Reshape the data into a 2D array (days x hours)
    consumption_data = consumption_data.reshape(n_days, n_hours)

    #get the date of each day:
    date_of_consumption_data = energies_by_hours.index.date
    date_of_consumption_data= date_of_consumption_data.reshape(n_days, n_hours) #reshape to get one point each day
    date_only= date_of_consumption_data[:,0] #take the first column
    y_axis = np.arange(0,n_hours)
    
    
    # Create the heatmap
    fig_hours_heatmap, axe_hours_heatmap = plt.subplots(nrows=1, ncols=1, figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    pos = axe_hours_heatmap.pcolormesh(date_only, y_axis, consumption_data.transpose(), shading='auto', cmap='turbo') #cmap='RdBu' cmap='hot' viridis, cmap = 'jet'
      
    #pos = axe_hours_heatmap.imshow(consumption_data.transpose(), cmap='jet', aspect='auto')  
    
    fig_hours_heatmap.suptitle('SOC state by hour [%]', fontweight = 'bold', fontsize = 12)
    #axe_hours_heatmap.set_title("Profile by hour ", fontsize=12, weight="bold")
  
    axe_hours_heatmap.set_xlabel('Day of the Year', fontsize=12)
    axe_hours_heatmap.set_ylabel('Hour of the Day', fontsize=12)
    axe_hours_heatmap.set_ylim(-0.5,23.5)
    axe_hours_heatmap.set_yticks([0, 4, 8, 12, 16, 20])

      
    # Display the colorbar
    fig_hours_heatmap.colorbar(pos, ax=axe_hours_heatmap)
    
    # #remove the frame
    # axe_hours_heatmap.spines['bottom'].set_color('white')
    # axe_hours_heatmap.spines['top'].set_color('white') 
    # axe_hours_heatmap.spines['right'].set_color('white')
    # axe_hours_heatmap.spines['left'].set_color('white')
    # axe_hours_heatmap.grid(True)
    
    if I_WANT_WATERMARK_ON_FIGURE:
        im = Image.open(WATERMARK_PICTURE)
        fig_hours_heatmap.figimage(im, 10, 10, zorder=3, alpha=.2)

    if I_WANT_TO_SAVE_PNG:
        fig_hours_heatmap.savefig("FigureExport/hours_soc_heatmap.png")

   
    return fig_hours_heatmap




def build_production_heatmap_figure(hours_mean_df):
    
    all_channels_labels = list(hours_mean_df.columns)
    channel_number_SOC = [i for i, elem in enumerate(all_channels_labels) if ('Solar power scaled' in elem) ]
        
    #print(all_channels_labels)
    #print(channel_number_SOC)

    # take the data with hourly intervals
    energies_by_hours=hours_mean_df[all_channels_labels[channel_number_SOC[0]]]

    # Extract the values from the dataframe
    consumption_data = energies_by_hours.values
    #get the date of each day:
    date_of_consumption_data = energies_by_hours.index.date

    #print(consumption_data)

    # Determine the shape of the reshaped array
    n_days = len(consumption_data) // (24)
    n_hours = 24
    
    # If the number of data points is not a multiple of 24, truncate to make it so
    if len(consumption_data) % 24 != 0:
        consumption_data = consumption_data[:len(consumption_data) - (len(consumption_data) % 24)]
        date_of_consumption_data = date_of_consumption_data[:len(date_of_consumption_data) - (len(date_of_consumption_data) % 24)]
    
    # Reshape the data into a 2D array (days x hours)
    consumption_data = consumption_data.reshape(n_days, n_hours)       
    date_of_consumption_data= date_of_consumption_data.reshape(n_days, n_hours) #reshape to get one point each day
    date_only= date_of_consumption_data[:,0] #take the first column

    y_axis = np.arange(0,n_hours)
    
    
    # Create the heatmap
    fig_hours_heatmap, axe_hours_heatmap = plt.subplots(nrows=1, ncols=1, figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    pos = axe_hours_heatmap.pcolormesh(date_only, y_axis, consumption_data.transpose(), shading='auto', cmap='hot') #cmap='RdBu' cmap='hot' viridis, cmap = 'jet'
      
    #pos = axe_hours_heatmap.imshow(consumption_data.transpose(), cmap='jet', aspect='auto')  

    
    fig_hours_heatmap.suptitle('Solar production for each hour [kW] -[kWh]', fontweight = 'bold', fontsize = 12)  
    axe_hours_heatmap.set_xlabel('Day of the Year', fontsize=12)
    axe_hours_heatmap.set_ylabel('Hour of the Day', fontsize=12)
    axe_hours_heatmap.set_ylim(-0.5,23.5)
    axe_hours_heatmap.set_yticks([0, 4, 8, 12, 16, 20])

    #axe_hours_heatmap.set_title("Profile by hour ", fontsize=12, weight="bold")
      
    # Display the colorbar
    cbar = fig_hours_heatmap.colorbar(pos, ax=axe_hours_heatmap)
    cbar.set_label("kW  - kWh", rotation=270, labelpad=15)


    if I_WANT_WATERMARK_ON_FIGURE:
        im = Image.open(WATERMARK_PICTURE)
        fig_hours_heatmap.figimage(im, 10, 10, zorder=3, alpha=.2)

    if I_WANT_TO_SAVE_PNG:
        fig_hours_heatmap.savefig("FigureExport/hours_production_heatmap.png")

   
    return fig_hours_heatmap



def build_consumption_heatmap_figure(hours_mean_df):
    
    all_channels_labels = list(hours_mean_df.columns)
    channel_number = [i for i, elem in enumerate(all_channels_labels) if ('Consumption [kW]' in elem) ]
        
    #print(all_channels_labels)
    #print(channel_number_SOC)

    # Take the data to hourly intervals 
    energies_by_hours = hours_mean_df[all_channels_labels[channel_number[0]]]

    # Extract the values from the dataframe
    consumption_data = energies_by_hours.values
    #get the date of each day:
    date_of_consumption_data = energies_by_hours.index.date

    # Determine the shape of the reshaped array
    n_days = len(consumption_data) // (24)
    n_hours = 24
    
       # If the number of data points is not a multiple of 24, truncate to make it so
    if len(consumption_data) % 24 != 0:
        consumption_data = consumption_data[:len(consumption_data) - (len(consumption_data) % 24)]
        date_of_consumption_data = date_of_consumption_data[:len(date_of_consumption_data) - (len(date_of_consumption_data) % 24)]
    
    # Reshape the data into a 2D array (days x hours)
    consumption_data = consumption_data.reshape(n_days, n_hours)
    date_of_consumption_data= date_of_consumption_data.reshape(n_days, n_hours) #reshape to get one point each day
    
    date_only= date_of_consumption_data[:,0] #take the first column
    y_axis = np.arange(0,n_hours)
    
    
    # Create the heatmap
    fig_hours_heatmap, axe_hours_heatmap = plt.subplots(nrows=1, ncols=1, figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    pos = axe_hours_heatmap.pcolormesh(date_only, y_axis, consumption_data.transpose(), shading='auto', cmap='seismic') #cmap= 'plasma' 'RdBu' cmap='hot' viridis, cmap = 'jet' cmap="coolwarm"
      
    #pos = axe_hours_heatmap.imshow(consumption_data.transpose(), cmap='jet', aspect='auto')  
    
    fig_hours_heatmap.suptitle('Consumption for each hour [kW] -[kWh]', fontweight = 'bold', fontsize = 12)
      
   
    axe_hours_heatmap.set_xlabel('Day of the Year', fontsize=12)
    axe_hours_heatmap.set_ylabel('Hour of the Day', fontsize=12)
    axe_hours_heatmap.set_ylim(-0.5,23.5)
    axe_hours_heatmap.set_yticks([0, 4, 8, 12, 16, 20])

    #axe_hours_heatmap.set_title("Profile by hour ", fontsize=12, weight="bold")
      
    # Display the colorbar
    cbar = fig_hours_heatmap.colorbar(pos, ax=axe_hours_heatmap)
    cbar.set_label("kW  - kWh", rotation=270, labelpad=15)

    if I_WANT_WATERMARK_ON_FIGURE:
        im = Image.open(WATERMARK_PICTURE)
        fig_hours_heatmap.figimage(im, 10, 10, zorder=3, alpha=.2)
    
    # fig_hours_heatmap.figimage(im, 10, 10, zorder=3, alpha=.2)
    # fig_hours_heatmap.savefig("FigureExport/hours_soc_heatmap.png")

   
    return fig_hours_heatmap



def build_hours_grid_heatmap_figure(hours_mean_df):
    fig_acsource_hours_heatmap, axe_hours_heatmap_acsource = plt.subplots(nrows=1, ncols=1, figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))

    # Assuming 'data' is your original Pandas timeseries with minute-level data
    # You can replace it with your actual timeseries data
    #data = hours_mean_df
    
    
    
    all_channels_labels = list(hours_mean_df.columns)
    channel_number_Pin_actif_Tot = [i for i, elem in enumerate(all_channels_labels) if "Grid with storage" in elem]        
    energies_by_hours=hours_mean_df[hours_mean_df.columns[channel_number_Pin_actif_Tot]]
    
    # Resample the data to hourly intervals and aggregate using the mean or sum
    #hourly_data = data.resample('H').mean()  # or data.resample('H').sum()
    
    # Extract the values from the Series
    consumption_data = energies_by_hours.values #/1000 already in kW
    
    # Determine the shape of the reshaped array
    n_days = len(consumption_data) // (24)
    n_hours = 24
    
    # Reshape the data into a 2D array (days x hours)
    consumption_data = consumption_data.reshape(n_days, n_hours)

    #get the date of each day:
    date_of_consumption_data = energies_by_hours.index.date
    date_of_consumption_data= date_of_consumption_data.reshape(n_days, n_hours) #reshape to get one point each day
    date_only= date_of_consumption_data[:,0] #take the first column
    y_axis = np.arange(0,n_hours)
    
    pos = axe_hours_heatmap_acsource.pcolormesh(date_only, y_axis, consumption_data.transpose(), shading='auto', cmap='terrain') #gist_ncar gist_stern
      
    
      
        
    # Create the heatmap
    #pos = axe_hours_heatmap_acloads.imshow(consumption_data.transpose(), cmap='jet', aspect='auto')  #cmap='RdBu' cmap='hot' viridis, cmap = 'jet'
    
    fig_acsource_hours_heatmap.suptitle('Energy consumption profile by hour on grid [kW - kWh]', fontweight = 'bold', fontsize = 12)
      
   
    
    axe_hours_heatmap_acsource.set_xlabel('Day of the Year', fontsize=12)
    axe_hours_heatmap_acsource.set_ylabel('Hour of the Day', fontsize=12)
    axe_hours_heatmap_acsource.set_ylim(-0.5,23.5)
    axe_hours_heatmap_acsource.set_yticks([0, 4, 8, 12, 16, 20])

    #axe_hours_heatmap.set_title("Profile by hour ", fontsize=12, weight="bold")
      
    # Display the colorbar
    fig_acsource_hours_heatmap.colorbar(pos, ax=axe_hours_heatmap_acsource)
    
  

    return fig_acsource_hours_heatmap



def build_sunblocked_heatmap_figure(hours_mean_df):
    
    all_channels_labels = list(hours_mean_df.columns)
    channel_number_used = [i for i, elem in enumerate(all_channels_labels) if ('sun_masked' in elem) ]  #consumption_sun_masked sun_masked
        
    #print(all_channels_labels)
    #print(channel_number_SOC)

    # take the data with hourly intervals
    energies_by_hours=hours_mean_df[all_channels_labels[channel_number_used[0]]]

    # Extract the values from the dataframe
    consumption_data = energies_by_hours.values
    #get the date of each day:
    date_of_consumption_data = energies_by_hours.index.date

    #print(consumption_data)

    # Determine the shape of the reshaped array
    n_days = len(consumption_data) // (24)
    n_hours = 24
    
    # If the number of data points is not a multiple of 24, truncate to make it so
    if len(consumption_data) % 24 != 0:
        consumption_data = consumption_data[:len(consumption_data) - (len(consumption_data) % 24)]
        date_of_consumption_data = date_of_consumption_data[:len(date_of_consumption_data) - (len(date_of_consumption_data) % 24)]
    
    # Reshape the data into a 2D array (days x hours)
    consumption_data = consumption_data.reshape(n_days, n_hours)       
    date_of_consumption_data= date_of_consumption_data.reshape(n_days, n_hours) #reshape to get one point each day
    date_only= date_of_consumption_data[:,0] #take the first column

    y_axis = np.arange(0,n_hours)
    
    
    # Create the heatmap
    fig_hours_heatmap, axe_hours_heatmap = plt.subplots(nrows=1, ncols=1, figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    pos = axe_hours_heatmap.pcolormesh(date_only, y_axis, consumption_data.transpose(), shading='auto', cmap='hot') #seismic cmap='RdBu' cmap='hot' viridis, cmap = 'jet'
      
    #pos = axe_hours_heatmap.imshow(consumption_data.transpose(), cmap='jet', aspect='auto')  

    
    fig_hours_heatmap.suptitle('Sun masked [0-1]', fontweight = 'bold', fontsize = 12)  
    axe_hours_heatmap.set_xlabel('Day of the Year', fontsize=12)
    axe_hours_heatmap.set_ylabel('Hour of the Day', fontsize=12)
    axe_hours_heatmap.set_ylim(-0.5,23.5)
    axe_hours_heatmap.set_yticks([0, 4, 8, 12, 16, 20])

    #axe_hours_heatmap.set_title("Profile by hour ", fontsize=12, weight="bold")
      
    # Display the colorbar
    cbar = fig_hours_heatmap.colorbar(pos, ax=axe_hours_heatmap)
    cbar.set_label("0-1", rotation=270, labelpad=15)


    if I_WANT_WATERMARK_ON_FIGURE:
        im = Image.open(WATERMARK_PICTURE)
        fig_hours_heatmap.figimage(im, 10, 10, zorder=3, alpha=.2)

    if I_WANT_TO_SAVE_PNG:
        fig_hours_heatmap.savefig("FigureExport/hours_sunblocked_heatmap.png")

   
    return fig_hours_heatmap



def build_battery_SOC_min_max_analysis_figure(quarters_mean_df):
    #all_channels_labels = list(total_datalog_df.columns)
    quarters_channels_labels=list(quarters_mean_df.columns)
    
    ####
    # channel recorded:
    channels_number_bat_soc = [i for i, elem in enumerate(quarters_channels_labels) if ('SOC' in elem)]
    
    #####
    # resample the day min and the day max
    day_max_df = quarters_mean_df.resample("1d").max()
    day_min_df = quarters_mean_df.resample("1d").min()

    delta_soc_df=day_max_df-day_min_df
    
    fig_batt_soc, axes_batt_soc = plt.subplots(nrows=2, 
                               ncols=1,
                               figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    #axes4_2 = axes4.twinx()
    
    day_max_df.plot(y=day_max_df.columns[channels_number_bat_soc],
                          grid=True,
                          legend="max",
                          ax=axes_batt_soc[0])
    day_min_df.plot(y=day_min_df.columns[channels_number_bat_soc],
                          grid=True,
                          legend="min",
                          ax=axes_batt_soc[0]) 
    
    axes_batt_soc[0].legend(['max', 'min'])
    axes_batt_soc[0].set_ylabel('SOC [%]', fontsize=12)
    axes_batt_soc[0].grid(True) 
    axes_batt_soc[0].set_title('Daily battery SOC min-max analysis', fontsize=12, weight="bold")
   
    
    mean_delta=np.nanmean(delta_soc_df[day_max_df.columns[channels_number_bat_soc]].values) #mean value dropping NaN to avoid error
    
    
    delta_soc_df.plot(y=day_max_df.columns[channels_number_bat_soc],
                      grid=True,
                      legend="delta",
                      ax=axes_batt_soc[1])
    
    plt.axhline(mean_delta, color='r', linestyle='dashed', linewidth=2, alpha=0.5)

    axes_batt_soc[1].grid(True) 
    
    axes_batt_soc[1].set_ylabel('$\Delta$ SOC [%]', fontsize=12)
    axes_batt_soc[1].legend(['$\Delta$ SOC', 'mean'])


    # fig_batt_soc.figimage(im, 10, 10, zorder=3, alpha=.2)
    # fig_batt_soc.savefig("FigureExport/bat_SOC_min_max.png")

    return fig_batt_soc



    return fig_hist

def build_power_histogram_figure(quarters_mean_df):
    all_channels_labels = list(quarters_mean_df.columns)
    channels_number_Pin_actif = [i for i, elem in enumerate(all_channels_labels) if ("Grid with storage" in elem) ]
    channels_number_Pout_actif = [i for i, elem in enumerate(all_channels_labels) if ("Consumption [kW]" in elem) ]


    #take out the 0kW power (when genset/grid is not connected):    
    #chanel_number=channels_number_Pin_actif[0]


    channel_number = channels_number_Pin_actif[0]
    values_for_Pin_hist = quarters_mean_df[all_channels_labels[channel_number]]
    
    channel_number=channels_number_Pout_actif[0]
    values_for_Pout_hist= quarters_mean_df[all_channels_labels[channel_number]]

    fig_hist, axes_hist = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    
    values_for_Pout_hist.hist( bins=80, alpha=0.5, label="Consumption",density=True)
    values_for_Pin_hist.hist( bins=80, alpha=0.5, label="Grid power", density=True)
    plt.axvline(values_for_Pout_hist.mean(), color='b', alpha=0.5, linestyle='dashed', linewidth=2)
    plt.axvline(values_for_Pin_hist.mean(), color='r', alpha=0.5, linestyle='dashed', linewidth=2)

    axes_hist.set_title("Histogram of Powers, consumption and grid exchange", fontsize=12, weight="bold")
    axes_hist.set_xlabel("Power [kW]", fontsize=12)
    axes_hist.set_ylabel("Frequency density", fontsize=12)
    axes_hist.legend(loc='upper right')


    axes_hist.grid(True)
    


    return fig_hist






def build_time_to_go_heatmap_figure(hours_mean_df):
    
    all_channels_labels = list(hours_mean_df.columns)
    channel_number_SOC = [i for i, elem in enumerate(all_channels_labels) if ('Time of backup on battery' in elem) ]
        
    #print(all_channels_labels)
    #print(channel_number_SOC)

    # Resample the data to hourly intervals and aggregate using the mean or sum
    #hourly_data = data.resample('H').mean()  # or data.resample('H').sum()
    energies_by_hours=hours_mean_df[all_channels_labels[channel_number_SOC[0]]]

    # Extract the values from the dataframe to an array:
    consumption_data = energies_by_hours.values
    #print(consumption_data)

    # Determine the shape of the reshaped array
    n_days = len(consumption_data) // (24)
    n_hours = 24
    
    # Reshape the data into a 2D array (days x hours)
    consumption_data = consumption_data.reshape(n_days, n_hours)

    #get the date of each day:
    date_of_consumption_data = energies_by_hours.index.date
    date_of_consumption_data= date_of_consumption_data.reshape(n_days, n_hours) #reshape to get one point each day
    date_only= date_of_consumption_data[:,0] #take the first column
    y_axis = np.arange(0,n_hours)
    
    
    # Create the heatmap
    fig_hours_heatmap, axe_hours_heatmap = plt.subplots(nrows=1, ncols=1, figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    pos = axe_hours_heatmap.pcolormesh(date_only, y_axis, consumption_data.transpose(), shading='auto', cmap='turbo') #cmap='RdBu' cmap='hot' viridis, cmap = 'jet'
      
    #pos = axe_hours_heatmap.imshow(consumption_data.transpose(), cmap='jet', aspect='auto')  
    
    fig_hours_heatmap.suptitle('Time to go by hour [h]', fontweight = 'bold', fontsize = 12)
      
   
    axe_hours_heatmap.set_xlabel('Day of the Year', fontsize=12)
    axe_hours_heatmap.set_ylabel('Hour of the Day', fontsize=12)
    axe_hours_heatmap.set_ylim(-0.5,23.5)
    axe_hours_heatmap.set_yticks([0, 4, 8, 12, 16, 20])

    #axe_hours_heatmap.set_title("Profile by hour ", fontsize=12, weight="bold")
      
    # Display the colorbar
    fig_hours_heatmap.colorbar(pos, ax=axe_hours_heatmap)
    


   
    return fig_hours_heatmap





def build_bat_inout_figure(day_kwh_df, month_kwh_df):

    ##############################
    #CHARGE/DISCHARGE ENERGY OF THE BATTERY AND TRHOUGHPUT
    ################
    
    fig_bat_inout, axes_bat_inout = plt.subplots(nrows=2, ncols=1,figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    
    
    day_kwh_df['Abs Discharge']=-day_kwh_df['Battery discharge power only']
    
    
    
    day_kwh_df[['Battery charge power only','Abs Discharge']].plot(ax=axes_bat_inout[0],
                          kind='line',
                          marker='o',
                          color=['g', 'r'])
    
    axes_bat_inout[0].legend(["CHARGE", "DISCHARGE"])
    axes_bat_inout[0].grid(True)
    #plt.xticks(np.arange(len(list(day_kwh_df.index))), labels=list(day_kwh_df.index.date), rotation=30, ha = 'center')        
    axes_bat_inout[0].set_ylabel("Energy per day [kWh]", fontsize=12)
    axes_bat_inout[0].set_title("How is the battery used? Daily and monthly cycling", fontsize=12, weight="bold")
    
    
    
    #to see both positive on the graph for better comparison:
    dischargeEm=-month_kwh_df['Battery discharge power only']
    chargeEm=month_kwh_df['Battery charge power only']
    ind = np.arange(len(list(month_kwh_df.index.month_name())))
    
    
    width = 0.35  # the width of the bars
    b1=axes_bat_inout[1].bar(ind- width/2, chargeEm.values, width, color='g', label='CHARGE')
    b2=axes_bat_inout[1].bar(ind+ width/2, dischargeEm.values, width, color='r', label='DISCHARGE')
    
    
    
    axes_bat_inout[1].set_ylabel("Energy per month [kWh]", fontsize=12)
    #axes_bat_inout.legend(["CHARGE", "DISCHARGE"])
    axes_bat_inout[1].legend()
    axes_bat_inout[1].grid(True)
    #plt.xticks(ind, labels=list(month_kwh_df.index.month_name()), rotation=30, ha = 'right')        
    labels_month=list(month_kwh_df.index.month_name())
    labels_year=list(month_kwh_df.index.year)
    loc, label= plt.xticks()

    for k,elem in enumerate(labels_month):
        if elem=='January':
            labels_month[k]=str(labels_year[k]) + ' January'
            
    
    #TODO: remove comment: change the ticks first to units
    #loc=[0, 1, 2]
    loc=np.arange(len(labels_month))
    plt.xticks(loc,labels=labels_month,rotation=35)

    #addition of a watermark on the figure
    if I_WANT_WATERMARK_ON_FIGURE:
        im = Image.open(WATERMARK_PICTURE)   
        fig_bat_inout.figimage(im, 0.05*FIGSIZE_WIDTH*150, 0.1*FIGSIZE_HEIGHT*150, zorder=3, alpha=.2)


    return fig_bat_inout




def build_power_profile(quarters_mean_df, label_of_channel):
    #for tests:
    #start_date = dt.date(2018, 7, 1)
    #end_date = dt.date(2018, 8, 30) 
    
    # temp1 = total_datalog_df[total_datalog_df.index.date >= start_date]
    # temp2 = temp1[temp1.index.date <= end_date]

    temp2 = quarters_mean_df

    all_channels_labels = list(quarters_mean_df.columns)
    channel_number = [i for i, elem in enumerate(all_channels_labels) if label_of_channel in elem]
   
    #channel_number=channel_number_Pout_conso_Tot
    time_of_day_in_hours=list(temp2.index.hour+temp2.index.minute/60)
    time_of_day_in_minutes=list(temp2.index.hour*60+temp2.index.minute)
    
    #add a channels to the dataframe with minutes of the day to be able to sort data on it: 
    #Create a new entry in the dataframe:
    temp2['Time of day in minutes']=time_of_day_in_minutes
        
        
    fig_pow_by_min_of_day, axes_pow_by_min_of_day = plt.subplots(nrows=1, ncols=1, figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    
    
    #maybe it is empty if there is no inverter:
    if channel_number:
        
        channel_label=all_channels_labels[channel_number[0]]
        
        axes_pow_by_min_of_day.plot(time_of_day_in_hours,
                          temp2[channel_label].values, 
                          marker='+',
                          alpha=0.25,
                          color='b',
                          linestyle='None')
       
        
    
        #faire la moyenne de tous les points qui sont à la même quart d'heure du jour:
        mean_by_minute=np.zeros(24*4)
        x1=np.array(range(0,24*4))
        for k in x1:
            tem_min_pow1=temp2[temp2['Time of day in minutes'].values == k*15]
            mean_by_minute[k]=np.nanmean(tem_min_pow1[channel_label].values)
            
    
        axes_pow_by_min_of_day.plot(x1/4, mean_by_minute,
                          color='r',
                          linestyle ='-',
                          linewidth=2,
                          drawstyle='steps-post')
    
        #faire la moyenne de tous les points qui sont à la même heure:
        mean_by_hour=np.zeros(24)
        x2=np.array(range(0,24))
        for k in x2:
            tem_min_pow2=temp2[temp2.index.hour == k]
            mean_by_hour[k]=np.nanmean(tem_min_pow2[channel_label].values)
            
    
        axes_pow_by_min_of_day.plot(x2, mean_by_hour,
                          color='g',
                          linestyle ='-',
                          linewidth=2,
                          drawstyle='steps-post')
        
        #mean power:
        #axes_pow_by_min_of_day.axhline(np.nanmean(total_datalog_df[channel_label].values), color='k', linestyle='dashed', linewidth=2)
        axes_pow_by_min_of_day.axhline(mean_by_minute.mean(), color='k', linestyle='dashed', linewidth=2)
        text_to_disp='Mean = ' + str(round(mean_by_minute.mean(), 2)) + ' '
        axes_pow_by_min_of_day.text(0.1,mean_by_minute.mean()+0.1,  text_to_disp, horizontalalignment='left',verticalalignment='bottom')
        axes_pow_by_min_of_day.legend(["All points", "quarter mean profile" ,"hour mean profile"])
        #axes_pow_by_min_of_day.set_ylabel("Power [kW]", fontsize=12)
        axes_pow_by_min_of_day.set_xlabel("Time [h]", fontsize=12)
        axes_pow_by_min_of_day.set_xlim(0,24)
        axes_pow_by_min_of_day.set_title("Profile by hour of the day " + label_of_channel, fontsize=12, weight="bold")
        axes_pow_by_min_of_day.grid(True)
        
    
    else:
        #axes_pow_by_min_of_day.text(0.0, 0.0, "There is no Studer inverter!", horizontalalignment='left',verticalalignment='bottom')
        axes_pow_by_min_of_day.set_title("There is no data with this lable!", fontsize=12, weight="bold")
        
    
    # fig_pow_by_min_of_day.figimage(im, 10, 10, zorder=3, alpha=.2)
    # fig_pow_by_min_of_day.savefig("FigureExport/typical_power_profile_figure.png")

    return fig_pow_by_min_of_day


def build_day_and_month_energy_figure(day_kwh_df,month_kwh_df, 
                                      column_name ="Solar power scaled",
                                      title_start="PV production",
                                      y_axis_label_day="Energy [kWh/day]",
                                      y_axis_label_month="Energy [kWh/month]",
                                      color_day = SOLAR_COLOR):

    all_channels_labels = list(day_kwh_df.columns)
    chanel_number_for_solar = [i for i, elem in enumerate(all_channels_labels) if column_name in elem]
    #day_kwh_df = total_datalog_df.resample("1d").sum() / 60
    #month_kwh_df = total_datalog_df.resample("1M").sum() / 60
    
    
    fig_solar, axes_solar = plt.subplots(nrows=2, ncols=1, figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    
    day_kwh_df[day_kwh_df.columns[chanel_number_for_solar]].plot(ax=axes_solar[0],
              kind='line',
              marker='o',
              color=color_day)
    
    month_kwh_df[month_kwh_df.columns[chanel_number_for_solar[0]]].plot.bar(ax=axes_solar[1],
                          use_index=True)
    
    # if FRANCAIS_LANGUAGE == True:
    # else:
    axes_solar[0].set_ylabel(y_axis_label_day, fontsize=12)
    axes_solar[0].set_title(title_start+" per day and per month", fontsize=12, weight="bold")
    axes_solar[0].legend(["Day " + title_start])
    axes_solar[0].grid(True)
    
    
    axes_solar[1].set_ylabel(y_axis_label_month, fontsize=12)
    #axes_solar[1].set_title("PV production per month", fontsize=12, weight="bold")
    axes_solar[1].legend(["Month " + title_start])
    axes_solar[1].grid(True)
    
    #replace labels with the month name:
    loc, label= plt.xticks()
    #plt.xticks(loc,labels=list(month_kwh_df.index.month_name()), rotation=35, ha = 'right' )
    labels_month=list(month_kwh_df.index.month_name())
    labels_year=list(month_kwh_df.index.year)
    
    for k,elem in enumerate(labels_month):
        if elem=='January':
            labels_month[k]=str(labels_year[k]) + ' January'
    
    loc=np.arange(len(labels_month))
    plt.xticks(loc,labels=labels_month,rotation=35)

    if I_WANT_WATERMARK_ON_FIGURE:
        im = Image.open(WATERMARK_PICTURE)   
        fig_solar.figimage(im, 0.05*FIGSIZE_WIDTH*150, 0.1*FIGSIZE_HEIGHT*150, zorder=3, alpha=.2)

    #fig_solar.figimage(im, 10, 10, zorder=3, alpha=.2)
    #fig_solar.savefig("FigureExport/solar_daily_monthly_production_figure.png")

    return fig_solar



def build_test_figure(array):
    #for tests:

    test_figure, axes_test = plt.subplots(nrows=1, ncols=1, figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    
    
    #maybe it is empty if there is no inverter:            
    axes_test.plot(array)
    
    axes_test.set_title("Test figure", fontsize=12, weight="bold")
    axes_test.grid(True)
        
    return test_figure



def build_daily_indicators_polar_fraction_figure(day_kwh_df):


    
    all_channels_labels=list(day_kwh_df.columns)
    
    #####################################
    # the channels with SYSTEM Power 
    
    channels_number_Psolar_Tot = [i for i, elem in enumerate(all_channels_labels) if 'Solar power scaled' in elem]
    channel_number_Pload_Consumption = [i for i, elem in enumerate(all_channels_labels) if ('Consumption [kW]' in elem) ]

    channel_number_Pin_Consumption_Tot = [i for i, elem in enumerate(all_channels_labels) if "Grid consumption with storage" in elem]
    channel_number_Pin_Injection_Tot = [i for i, elem in enumerate(all_channels_labels) if "Grid injection with storage" in elem]
    
    
    #utilisation directe du label plutot que les indexs des columns: 
    #chanel_label_Pout_actif_tot=all_channels_labels[channel_number_Pout_actif_Tot[0]]
    chanel_label_Pin_Consumption_tot=all_channels_labels[channel_number_Pin_Consumption_Tot[0]]
    chanel_label_Psolar_tot=all_channels_labels[channels_number_Psolar_Tot[0]]
    chanel_label_Pin_Injection=all_channels_labels[channel_number_Pin_Injection_Tot[0]]
    channel_label_Pload_Consumption=all_channels_labels[channel_number_Pload_Consumption[0]]

    fig_indicators_polar, [axes_autarky, axes_selfcon],  = plt.subplots(nrows=1, ncols=2, 
                                                        figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT), 
                                                        subplot_kw={'projection': 'polar'})
    
    
    ############################
    #Daily Energies Fractions:
    #############
    rate_autarky=(1-abs(day_kwh_df[chanel_label_Pin_Consumption_tot])/(day_kwh_df[channel_label_Pload_Consumption] +1e-9))*100 
    rate_autarky = rate_autarky.clip(lower=0, upper=100).fillna(0)

    rate_selfconsumption=(1-abs(day_kwh_df[chanel_label_Pin_Injection])/(day_kwh_df[chanel_label_Psolar_tot]+1e-9))*100
    rate_selfconsumption = rate_selfconsumption.clip(lower=0, upper=100).fillna(0)


    e_grid_conso = day_kwh_df[chanel_label_Pin_Consumption_tot].sum()
    e_grid_inject = day_kwh_df[chanel_label_Pin_Injection].sum()
    e_load_conso = day_kwh_df[channel_label_Pload_Consumption].sum()
    e_solar_prod = day_kwh_df[chanel_label_Psolar_tot].sum()


    rate_autarky_annual = (1-e_grid_conso/(e_load_conso + 1e-9))*100.0
    #print(f'Annual autarky: {rate_autarky_annual : .1f}')
    rate_selfconsumption_annual = (1-abs(e_grid_inject/(e_solar_prod+1e-9)))*100.0

    #to link the points:
    #rate_autarky[-1]=rate_autarky[0]
    #rate_selfconsumption[-1]=rate_selfconsumption[0]

    ind = day_kwh_df.index.dayofyear/365
    theta = 2 * np.pi * ind - 2 * np.pi/len(day_kwh_df) #start first day vertically

    p1=axes_autarky.plot(theta, rate_autarky.values, color=SOLAR_COLOR,  linewidth=3)
    p1_s=axes_autarky.fill_between(theta, rate_autarky.values, color=SOLAR_COLOR,  alpha=0.5)
    
    p2=axes_selfcon.plot(theta, rate_selfconsumption.values, color=GENSET_COLOR,  linewidth=3)
    p2_s=axes_selfcon.fill_between(theta, rate_selfconsumption.values, color=GENSET_COLOR,  alpha=0.5)
    
    axes_autarky.set_ylim([0, 100])
    axes_selfcon.set_ylim([0, 100])

    #plt.xticks(ind, labels=list(day_kwh_df.index.month_name()), rotation=30, ha = 'right')        
    labels_month_full=["January", "\nFebruary",  "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    if FRANCAIS_LANGUAGE:
        labels_month_full=["Janvier", "\nFévrier",  "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]


            
    loc, label = plt.xticks()
    loc = [0, np.pi/2, np.pi, np.pi*3/2]
    loc_full = np.arange(12)*np.pi/6
    axes_autarky.set_xticks(loc_full,labels=labels_month_full) #,rotation=35)
    #axes_selfcon.set_xticks(loc[0:3],labels=labels_month[0:3]) #,rotation=35)
    #axes_selfcon.set_xticks(loc_full[0:9],labels=labels_month_full[0:9]) #,rotation=35)
    axes_selfcon.set_xticks(loc_full,labels=labels_month_full) #,rotation=35)

    loc_y=[20, 40, 60, 80, 100]
    label_y=["20%", "40%", "60%", "80%", "100%"]
    axes_autarky.set_yticks(loc_y,labels=label_y) #,rotation=35)
    axes_selfcon.set_yticks(loc_y,labels=label_y) #,rotation=35)


    axes_autarky.set_theta_zero_location("N")  # theta=0 at the top
    axes_autarky.set_theta_direction(-1)  # theta increasing clockwise
    axes_selfcon.set_theta_zero_location("N")  # theta=0 at the top
    axes_selfcon.set_theta_direction(-1)  # theta increasing clockwise

    #axes_indicator.set_ylabel("Energy fraction [%]", fontsize=12)
    
    # Add a title to the entire figure
        # Add a title to the entire figure
    if FRANCAIS_LANGUAGE:
        fig_indicators_polar.suptitle("Indicateurs solaires \njournaliers", fontsize=14, weight="bold")
        axes_autarky.set_title(f"Taux d'autonomie \n Valeur annuelle: {rate_autarky_annual : .1f} %", fontsize=11, weight="bold")
        axes_selfcon.set_title(f"Taux d'autoconsommation \n Valeur annuelle: {rate_selfconsumption_annual : .1f} %", fontsize=11, weight="bold")

    else:

        fig_indicators_polar.suptitle("Daily solar indicators", fontsize=14, weight="bold")
        axes_autarky.set_title(f"Self-sufficiency \n Annual value: {rate_autarky_annual : .1f} %", fontsize=11, weight="bold")
        axes_selfcon.set_title(f"Self-consumption \n Annual value: {rate_selfconsumption_annual : .1f} %", fontsize=11, weight="bold")

    
    axes_autarky.grid(True)
    axes_selfcon.grid(True)

    #fig_indicators_polar.figimage(im, 10, 10, zorder=3, alpha=.2)
    #fig_indicators_polar.savefig("FigureExport/energy_indicators_polar.png")
    im = Image.open(WATERMARK_PICTURE)   
    fig_indicators_polar.figimage(im, 0.05*FIGSIZE_WIDTH*150, 0.15*FIGSIZE_HEIGHT*150, zorder=3, alpha=.2)

    return fig_indicators_polar



def build_monthly_indicators_polar_figure(day_kwh_df):

    
    all_channels_labels=list(day_kwh_df.columns)
    
    #####################################
    # the channels with SYSTEM Power 
    
    channels_number_Psolar_Tot = [i for i, elem in enumerate(all_channels_labels) if 'Solar power scaled' in elem]
    channel_number_Pload_Consumption = [i for i, elem in enumerate(all_channels_labels) if ('Consumption [kW]' in elem) ]

    channel_number_Pin_Consumption_Tot = [i for i, elem in enumerate(all_channels_labels) if "Grid consumption with storage" in elem]
    channel_number_Pin_Injection_Tot = [i for i, elem in enumerate(all_channels_labels) if "Grid injection with storage" in elem]
    
    
    #utilisation directe du label plutot que les indexs des columns: 
    #chanel_label_Pout_actif_tot=all_channels_labels[channel_number_Pout_actif_Tot[0]]
    chanel_label_Pin_Consumption_tot=all_channels_labels[channel_number_Pin_Consumption_Tot[0]]
    chanel_label_Psolar_tot=all_channels_labels[channels_number_Psolar_Tot[0]]
    chanel_label_Pin_Injection=all_channels_labels[channel_number_Pin_Injection_Tot[0]]
    channel_label_Pload_Consumption=all_channels_labels[channel_number_Pload_Consumption[0]]

    fig_indicators_polar, [axes_autarky, axes_selfcon],  = plt.subplots(nrows=1, ncols=2, 
                                                        figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT), 
                                                        subplot_kw={'projection': 'polar'})
    
    
    ############################
    #Monthly Energies Fractions:
    #############
    rate_autarky=(1-abs(day_kwh_df[chanel_label_Pin_Consumption_tot])/(day_kwh_df[channel_label_Pload_Consumption] +1e-9))*100 
    rate_selfconsumption=(1-abs(day_kwh_df[chanel_label_Pin_Injection])/(day_kwh_df[chanel_label_Psolar_tot]+1e-9))*100
    e_grid_conso = day_kwh_df[chanel_label_Pin_Consumption_tot].sum()
    e_grid_inject = day_kwh_df[chanel_label_Pin_Injection].sum()
    e_load_conso = day_kwh_df[channel_label_Pload_Consumption].sum()
    e_solar_prod = day_kwh_df[chanel_label_Psolar_tot].sum()


    rate_autarky_annual = (1-e_grid_conso/(e_load_conso + 1e-9))*100.0
    #print(f'Annual autarky: {rate_autarky_annual : .1f}')
    rate_selfconsumption_annual = (1-abs(e_grid_inject/(e_solar_prod+1e-9)))*100.0
    #print(f'Annual selfc: {rate_selfconsumption_annual/4 : .1f} % e_grid_inject = {e_grid_inject} and solar ={e_solar_prod}')

    #drop the last one due to the 1 minute of january of the next year: TODO: make it cleaner
    rate_autarky.drop(rate_autarky.index[-1], inplace = True )
    rate_selfconsumption.drop(rate_selfconsumption.index[-1] , inplace = True)


    ind = rate_autarky.index.dayofyear/365
    #ind = np.linspace(0, 2*np.pi, len(rate_autarky))
    theta = 2 * np.pi * ind - 2 * np.pi/len(rate_autarky) #start first day vertically
    #theta = 2 * np.pi * ind - 2 * np.pi/len(rate_autarky) #start first day vertically
    bar_width = 2 * np.pi / (len(theta)+1)   # width for one day

    axes_autarky.bar(theta, rate_autarky.values,
                 width=bar_width,
                 color=SOLAR_COLOR, alpha=0.8)

    
    p2=axes_selfcon.bar(theta, rate_selfconsumption.values, 
                        width=bar_width,
                        color=GENSET_COLOR, alpha=0.8)
    
    axes_autarky.set_ylim([0, 100])
    axes_selfcon.set_ylim([0, 100])

    #plt.xticks(ind, labels=list(day_kwh_df.index.month_name()), rotation=30, ha = 'right')        
    labels_month_full=["January", "\nFebruary",  "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    if FRANCAIS_LANGUAGE:
        labels_month_full=["Janvier", "\nFévrier",  "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]


            
    loc, label = plt.xticks()
    loc = [0, np.pi/2, np.pi, np.pi*3/2]
    loc_full = np.arange(12)*np.pi/6
    axes_autarky.set_xticks(loc_full,labels=labels_month_full) #,rotation=35)
    #axes_selfcon.set_xticks(loc[0:3],labels=labels_month[0:3]) #,rotation=35)
    #axes_selfcon.set_xticks(loc_full[0:9],labels=labels_month_full[0:9]) #,rotation=35)
    axes_selfcon.set_xticks(loc_full,labels=labels_month_full) #,rotation=35)

    loc_y=[20, 40, 60, 80, 100]
    label_y=["20%", "40%", "60%", "80%", "100%"]
    axes_autarky.set_yticks(loc_y,labels=label_y) #,rotation=35)
    axes_selfcon.set_yticks(loc_y,labels=label_y) #,rotation=35)


    axes_autarky.set_theta_zero_location("N")  # theta=0 at the top
    axes_autarky.set_theta_direction(-1)  # theta increasing clockwise
    axes_selfcon.set_theta_zero_location("N")  # theta=0 at the top
    axes_selfcon.set_theta_direction(-1)  # theta increasing clockwise

    #axes_indicator.set_ylabel("Energy fraction [%]", fontsize=12)
    
    # Add a title to the entire figure
    if FRANCAIS_LANGUAGE:
        fig_indicators_polar.suptitle("Indicateurs solaires mensuels", fontsize=14, weight="bold")
        axes_autarky.set_title(f"Taux d'autonomie \n Valeur annuelle: {rate_autarky_annual : .1f} %", fontsize=12, weight="bold")
        axes_selfcon.set_title(f"Taux d'autoconsommation \n Valeur annuelle: {rate_selfconsumption_annual : .1f} %", fontsize=12, weight="bold")

    else:

        fig_indicators_polar.suptitle("Monthly solar indicators", fontsize=14, weight="bold")
        axes_autarky.set_title(f"Self-sufficiency \n Annual value: {rate_autarky_annual : .1f} %", fontsize=12, weight="bold")
        axes_selfcon.set_title(f"Self-consumption \n Annual value: {rate_selfconsumption_annual : .1f} %", fontsize=12, weight="bold")

    axes_autarky.grid(True)
    axes_selfcon.grid(True)

    #fig_indicators_polar.figimage(im, 10, 10, zorder=3, alpha=.2)
    #fig_indicators_polar.savefig("FigureExport/energy_indicators_polar.png")

    #addition of a watermark on the figure
    im = Image.open(WATERMARK_PICTURE)   
    fig_indicators_polar.figimage(im, 0.01*FIGSIZE_WIDTH*150, 0.1*FIGSIZE_HEIGHT*150, zorder=3, alpha=.2)

    return fig_indicators_polar




def build_polar_consumption_profile(total_datalog_df, start_date = datetime.date(2000, 1, 1), end_date = datetime.date(2050, 12, 31)):
    #for tests:
    #start_date = dt.date(2018, 7, 1)
    #end_date = dt.date(2018, 8, 30) 
    
    temp1 = total_datalog_df[total_datalog_df.index.date >= start_date]
    temp2 = temp1[temp1.index.date <= end_date]

    month_name =temp2.index[0].month
    year_name =temp2.index[0].year

    all_channels_labels = list(total_datalog_df.columns)
    channel_number_consumption = [i for i, elem in enumerate(all_channels_labels) if 'Consumption' in elem]
    channels_number_solar = [i for i, elem in enumerate(all_channels_labels) if 'Solar power scaled' in elem]

    #channel_number=channel_number_Pout_conso_Tot
    time_of_day_in_hours=list(temp2.index.hour+temp2.index.minute/60)
    time_of_day_in_minutes=list(temp2.index.hour*60+temp2.index.minute)
    
    #add a channels to the dataframe with minutes of the day to be able to sort data on it: 
    #Create a new entry in the dataframe:
    temp2['Time of day in minutes']=time_of_day_in_minutes
        
    FIGSIZE_WIDTH = 6
    FIGSIZE_HEIGHT = 5     
    fig_pow_by_min_of_day, axes_pow_by_min_of_day = plt.subplots(nrows=1, ncols=1, 
                                                                 figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT), 
                                                                 subplot_kw={'projection': 'polar'})
    
    axes_pow_by_min_of_day.set_theta_zero_location("S")  # theta=0 at the botom
    axes_pow_by_min_of_day.set_theta_direction(-1)  # theta increasing clockwise
    
    #maybe it is empty if there is no inverter:
    if channel_number_consumption and channels_number_solar:
        
        channel_label_consumption=all_channels_labels[channel_number_consumption[0]]
        channel_label_solar=all_channels_labels[channels_number_solar[0]]

        # axes_pow_by_min_of_day.plot(np.array(time_of_day_in_hours)/24*2*np.pi,
        #                   temp2[channel_label_consumption].values, 
        #                   marker='+',
        #                   alpha=0.15,
        #                   color='b',
        #                   linestyle='None')
       
        

        #faire la moyenne de tous les points qui sont à la même 15 minute du jour:
        mean_by_minute = np.zeros(24*4)
        mean_by_minute_sol = np.zeros(24*4)

        x1=np.array(range(0 , 24*4))
        for k in x1:
            tem_min_pow1=temp2[temp2['Time of day in minutes'].values == k*15]
            mean_by_minute[k]=np.nanmean(tem_min_pow1[channel_label_consumption].values)
            
            tem_min_pow_sol1=temp2[temp2['Time of day in minutes'].values == k*15]
            mean_by_minute_sol[k]=np.nanmean(tem_min_pow_sol1[channel_label_solar].values)
   
    
        axes_pow_by_min_of_day.plot(x1/24/4*2*np.pi, mean_by_minute,
                          color=LOAD_COLOR,
                          linestyle ='-',
                          linewidth=2)
        
        axes_pow_by_min_of_day.plot(x1/24/4*2*np.pi, mean_by_minute_sol,
                          color=SOLAR_COLOR,
                          linestyle ='-',
                          linewidth=2)
    

        

        ticks = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        
        axes_pow_by_min_of_day.set_ylim([0, max([mean_by_minute.max(), mean_by_minute_sol.max()])])

        axes_pow_by_min_of_day.set_xticks(ticks)
        axes_pow_by_min_of_day.set_xticklabels(['midnight', '3h', '6h', '9h', '12h', '15h', '18h', '21h'])
        #mean power:
        #axes_pow_by_min_of_day.axhline(np.nanmean(total_datalog_df[channel_label].values), color='k', linestyle='dashed', linewidth=2)
        #axes_pow_by_min_of_day.axhline(mean_by_minute.mean(), color='k', linestyle='dashed', linewidth=2)
        
        #text_to_disp='Mean power= ' + str(round(mean_by_minute.mean(), 2)) + ' kW'
        #axes_pow_by_min_of_day.text(0.1,mean_by_minute.mean()+0.1,  text_to_disp, horizontalalignment='left',verticalalignment='bottom')
        axes_pow_by_min_of_day.legend(["consumption mean", "solar mean",] , loc='lower right', fontsize=10) #["All points", "min mean profile" ,"hour mean profile"] , loc='upper right', fontsize=10)
        #axes_pow_by_min_of_day.legend(["all points","minutes mean" ,"hours mean"] , loc='lower right', fontsize=10) #["All points", "min mean profile" ,"hour mean profile"] , loc='upper right', fontsize=10)
        #axes_pow_by_min_of_day.set_ylabel("Power [kW]", fontsize=12)
        #axes_pow_by_min_of_day.set_xlabel("Time [h]", fontsize=12)
        #axes_pow_by_min_of_day.set_xlim(0,24)
        #axes_pow_by_min_of_day.set_title("Mean consumption profile by time of the day \n in kW", fontsize=12, weight="bold")
        axes_pow_by_min_of_day.set_title(f"Consumption profile around the clock \n Power in kW, average of month  {month_name} {year_name} ", fontsize=12, weight='bold')
        axes_pow_by_min_of_day.set_title(f"Mean consumption vs solar profiles  \n in kW ", fontsize=12, weight='bold')
        axes_pow_by_min_of_day.grid(True)
        
        if I_WANT_WATERMARK_ON_FIGURE:
            im = Image.open(WATERMARK_PICTURE)   
            fig_pow_by_min_of_day.figimage(im, 0.05*FIGSIZE_WIDTH*150, 0.1*FIGSIZE_HEIGHT*150, zorder=3, alpha=.2)

    

    return fig_pow_by_min_of_day





def build_polar_prices_profile(total_datalog_df, start_date = datetime.date(2000, 1, 1), end_date = datetime.date(2050, 12, 31)):
    #for tests:
    #start_date = dt.date(2018, 7, 1)
    #end_date = dt.date(2018, 8, 30) 
    
    temp1 = total_datalog_df[total_datalog_df.index.date >= start_date]
    temp2 = temp1[temp1.index.date <= end_date]

    month_name =temp2.index[0].month
    year_name =temp2.index[0].year

    all_channels_labels = list(total_datalog_df.columns)
    channel_number_buy = [i for i, elem in enumerate(all_channels_labels) if 'price buy' in elem]
    channels_number_sell = [i for i, elem in enumerate(all_channels_labels) if 'price sell PV' in elem]

    #channel_number=channel_number_Pout_conso_Tot
    time_of_day_in_hours=list(temp2.index.hour+temp2.index.minute/60)
    time_of_day_in_minutes=list(temp2.index.hour*60+temp2.index.minute)
    
    #add a channels to the dataframe with minutes of the day to be able to sort data on it: 
    #Create a new entry in the dataframe:
    temp2['Time of day in minutes']=time_of_day_in_minutes
        
    FIGSIZE_WIDTH = 6
    FIGSIZE_HEIGHT = 5     
    fig_pow_by_min_of_day, axes_pow_by_min_of_day = plt.subplots(nrows=1, ncols=1, 
                                                                 figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT), 
                                                                 subplot_kw={'projection': 'polar'})
    
    axes_pow_by_min_of_day.set_theta_zero_location("S")  # theta=0 at the botom
    axes_pow_by_min_of_day.set_theta_direction(-1)  # theta increasing clockwise
    
    #maybe it is empty if there is no inverter:
    if channel_number_buy and channels_number_sell:
        
        channel_label_buy=all_channels_labels[channel_number_buy[0]]
        channel_label_sell=all_channels_labels[channels_number_sell[0]]

        axes_pow_by_min_of_day.plot(np.array(time_of_day_in_hours)/24*2*np.pi,
                          temp2[channel_label_buy].values, 
                          marker='+',
                          alpha=0.15,
                          color=NX_BLUE,
                          linestyle='None')
       
        
        #faire la moyenne de tous les points qui sont à la même 15 minute du jour:
        mean_by_minute = np.zeros(24*4)
        mean_by_minute_sol = np.zeros(24*4)

        x1=np.array(range(0 , 24*4))
        for k in x1:
            tem_min_pow1=temp2[temp2['Time of day in minutes'].values == k*15]
            mean_by_minute[k]=np.nanmean(tem_min_pow1[channel_label_buy].values)
            
            tem_min_pow_sol1=temp2[temp2['Time of day in minutes'].values == k*15]
            mean_by_minute_sol[k]=np.nanmean(tem_min_pow_sol1[channel_label_sell].values)
   
    
        axes_pow_by_min_of_day.plot(x1/24/4*2*np.pi, mean_by_minute,
                          color=NX_PINK,
                          linestyle ='-',
                          linewidth=2)
        
        axes_pow_by_min_of_day.plot(x1/24/4*2*np.pi, mean_by_minute_sol,
                          color=NX_GREEN,
                          linestyle ='-',
                          linewidth=2)
    

        ticks = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        
        #axes_pow_by_min_of_day.set_ylim([0, max([mean_by_minute.max(), mean_by_minute_sol.max()])])
        axes_pow_by_min_of_day.set_xticks(ticks)
        axes_pow_by_min_of_day.set_xticklabels(['midnight', '3h', '6h', '9h', '12h', '15h', '18h', '21h'])
        axes_pow_by_min_of_day.legend(["Buy price all points", "Buy mean price", "Solar sell price",] , loc='lower right', fontsize=10) #["All points", "min mean profile" ,"hour mean profile"] , loc='upper right', fontsize=10)
        axes_pow_by_min_of_day.set_title(f"Prices profiles around the clock \n  in CHF/kWh", fontsize=12, weight='bold')
        axes_pow_by_min_of_day.grid(True)
        
    
  

    return fig_pow_by_min_of_day



def build_consumption_week_analysis(total_datalog_df, day_of_week_wanted = 1, start_date = datetime.date(2000, 1, 1), end_date = datetime.date(2050, 12, 31) ):


    #take only the wanted column:
    all_channels_labels = list(total_datalog_df.columns)
    channel_number = [i for i, elem in enumerate(all_channels_labels) if "Consumption" in elem]
    channel_label=all_channels_labels[channel_number[0]]

    load_df=total_datalog_df[[channel_label]]

    # make the sorting filter with the given dates: 
    temp1 = load_df[load_df.index.date >= start_date]
    temp2 = temp1[temp1.index.date <= end_date]

    time_of_week_in_minutes=list(temp2.index.dayofweek*60*24 + temp2.index.hour*60+temp2.index.minute)
    time_of_week_in_hours=list(temp2.index.dayofweek*24 + temp2.index.hour + temp2.index.minute/60)
    time_of_week_in_days=list(temp2.index.dayofweek + temp2.index.hour/24 + temp2.index.minute/60/24)

    
    #add a channels to the dataframe with minutes of the day to be able to sort data on it: 
    #Create a new entry in the dataframe:
    temp2['Time of week in minutes']=time_of_week_in_minutes
    temp2['Time of week in hours']=time_of_week_in_hours

        
    fig_pow_by_min_of_week, axes_pow_by_min_of_day = plt.subplots(nrows=1, ncols=1, figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    
    
    axes_pow_by_min_of_day.plot(time_of_week_in_days,
                        temp2[channel_label].values, 
                        marker='+',
                        alpha=0.25,
                        color='b',
                        linestyle='None')
    
    

    #faire la moyenne de tous les points qui sont à la même quart d'heure du jour:
    mean_by_minute=np.zeros(24*4*7)
    x1=np.array(range(0,24*4*7))
    for k in x1:
        tem_min_pow1=temp2[temp2['Time of week in minutes'].values == k*15]
        mean_by_minute[k]=np.nanmean(tem_min_pow1[channel_label].values)
        

    axes_pow_by_min_of_day.plot(x1/4/24, mean_by_minute,
                        color='r',
                        linestyle ='-',
                        linewidth=2,
                        drawstyle='steps-post')

    #faire la moyenne de tous les points qui sont à la même heure:
    mean_by_hour=np.zeros(24*7)
    x2=np.array(range(0,24*7))
    for k in x2:
        tem_min_pow2=temp2[temp2['Time of week in hours'].values == k]
        mean_by_hour[k]=np.nanmean(tem_min_pow2[channel_label].values)
        

    axes_pow_by_min_of_day.plot(x2/24, mean_by_hour,
                        color='g',
                        linestyle ='-',
                        linewidth=2,
                        drawstyle='steps-post')
    
    #mean power:
    #axes_pow_by_min_of_day.axhline(np.nanmean(total_datalog_df[channel_label].values), color='k', linestyle='dashed', linewidth=2)
    axes_pow_by_min_of_day.axhline(mean_by_minute.mean(), color='k', linestyle='dashed', linewidth=2)
    text_to_disp='Mean = ' + str(round(mean_by_minute.mean(), 2)) + ' '
    axes_pow_by_min_of_day.text(0.1,mean_by_minute.mean()+0.1,  text_to_disp, horizontalalignment='left',verticalalignment='bottom')
    axes_pow_by_min_of_day.legend(["All points", "quarter mean profile" ,"hour mean profile"])
    axes_pow_by_min_of_day.set_ylabel("Power [kW]", fontsize=12)
    axes_pow_by_min_of_day.set_xlabel("Day of the week", fontsize=12)
    axes_pow_by_min_of_day.set_xlim(0,7)

    xticks_wanted = range(7) #np.arange(0.5,7.5,1)
    axes_pow_by_min_of_day.set_xticks(xticks_wanted)
    DAYS = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat','Sun']
    axes_pow_by_min_of_day.set_xticklabels(DAYS, ha='left')

    axes_pow_by_min_of_day.set_title("Profile for every day of the week" , fontsize=12, weight="bold")
    axes_pow_by_min_of_day.grid(True)
    
    # make the sorting with the day of the week
    #temp3 = temp2[temp2.index.dayofweek ==  day_of_week_wanted]
    #fig_power_profile_of_the_day_of_week = build_power_profile(temp3,channel_label_Pout_actif_Tot)

    return fig_pow_by_min_of_week


def build_daily_energies_heatmap_figure(day_kwh_df):
    
    all_channels_labels = list(day_kwh_df.columns)
    channel_number_Pout_actif_Tot = [i for i, elem in enumerate(all_channels_labels) if "Consumption" in elem]
    channel_label_Pout_actif_Tot=day_kwh_df.columns[channel_number_Pout_actif_Tot]

    
    

    ###############################
    #HEAT MAP OF THE DAY ENERGY
    ###############################
    
    #help and inspiration:
    #https://scipython.com/book/chapter-7-matplotlib/examples/a-heatmap-of-boston-temperatures/
    #https://vietle.info/post/calendarheatmap-python/
    
    
    #select last year of data:
    last_year=day_kwh_df.index.year[-1]
    last_year=day_kwh_df.index.year[0]  #TODO: clean the stuff with the two years...

    temp1=day_kwh_df[day_kwh_df.index.year == last_year]
    energies_of_the_year=temp1[channel_label_Pout_actif_Tot]
    #TODO: put NaN in missing days...
    
    #select the year before
    year_before=last_year-1
    temp2=day_kwh_df[day_kwh_df.index.year == year_before]
    energies_of_the_yearbefore=temp2[channel_label_Pout_actif_Tot]
    
    
    # Define Ticks
    DAYS = ['Sun', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat']
    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    
    if energies_of_the_yearbefore.empty:    
        #then we have only one year of data :
        number_of_graph=1
        cal={str(last_year): energies_of_the_year}
        fig, ax = plt.subplots(number_of_graph, 1, figsize = (15,6))
        fig.suptitle('Energy consumption in kWh/day in the last year', fontweight = 'bold', fontsize = 12) 
        
        val=str(last_year)
        
        start = cal.get(val).index.min()
        end = cal.get(val).index.max()
        start_sun = start - np.timedelta64((start.dayofweek + 1) % 7, 'D')
        end_sun =  end + np.timedelta64(7 - end.dayofweek -1, 'D')
    
        num_weeks = (end_sun - start_sun).days // 7
        heatmap = np.full([7, num_weeks], np.nan)    
        ticks = {}
        y = np.arange(8) - 0.5
        x = np.arange(num_weeks + 1) - 0.5
        for week in range(num_weeks):
            for day in range(7):
                date = start_sun + np.timedelta64(7 * week + day, 'D')
                if date.day == 1:
                    ticks[week] = MONTHS[date.month - 1]
                if date.dayofyear == 1:
                    ticks[week] += f'\n{date.year}'
                if start <= date <= end:
                    heatmap[day, week] = cal.get(val).loc[date, energies_of_the_year.columns[0]]
        mesh = ax.pcolormesh(x, y, heatmap, cmap = 'jet', edgecolors = 'grey')  #cmap = 'jet' cmap = 'inferno'  cmap = 'magma'
    
        ax.invert_yaxis()
    
        # Set the ticks.
        ax.set_xticks(list(ticks.keys()))
        ax.set_xticklabels(list(ticks.values()))
        ax.set_yticks(np.arange(7))
        ax.set_yticklabels(DAYS)
        ax.set_ylim(6.5,-0.5)
        ax.set_aspect('equal')
        ax.set_title(val, fontsize = 15)
    
        # Hatch for out of bound values in a year
        ax.patch.set(hatch='xx', edgecolor='black')
        fig.colorbar(mesh, ax=ax)
        
    
    else:
        #then we have two years of data and we can plot two graphs:
        number_of_graph=2
        cal={str(year_before): energies_of_the_yearbefore, str(last_year): energies_of_the_year}
        fig, ax = plt.subplots(number_of_graph, 1, figsize = (15,6))
        fig.suptitle('Energy consumption in kWh/day in the last 2 years', fontweight = 'bold', fontsize = 12)
      
        
        #for i, val in enumerate(['2018', '2019']):
        for i, val in enumerate(list(cal.keys())):
            
            start = cal.get(val).index.min()
            end = cal.get(val).index.max()
            start_sun = start - np.timedelta64((start.dayofweek + 1) % 7, 'D')
            end_sun =  end + np.timedelta64(7 - end.dayofweek -1, 'D')
        
            num_weeks = (end_sun - start_sun).days // 7
            heatmap = np.full([7, num_weeks], np.nan)    
            ticks = {}
            y = np.arange(8) - 0.5
            x = np.arange(num_weeks + 1) - 0.5
            for week in range(num_weeks):
                for day in range(7):
                    date = start_sun + np.timedelta64(7 * week + day, 'D')
                    if date.day == 1:
                        ticks[week] = MONTHS[date.month - 1]
                    if date.dayofyear == 1:
                        ticks[week] += f'\n{date.year}'
                    if start <= date <= end:
                        heatmap[day, week] = cal.get(val).loc[date, energies_of_the_year.columns[0]]
            mesh = ax[i].pcolormesh(x, y, heatmap, cmap = 'jet', edgecolors = 'grey')  #cmap = 'jet' cmap = 'inferno'  cmap = 'magma'
        
            ax[i].invert_yaxis()
        
            # Set the ticks.
            ax[i].set_xticks(list(ticks.keys()))
            ax[i].set_xticklabels(list(ticks.values()))
            ax[i].set_yticks(np.arange(7))
            ax[i].set_yticklabels(DAYS)
            ax[i].set_ylim(6.5,-0.5)
            ax[i].set_aspect('equal')
            ax[i].set_title(val, fontsize = 15)
        
            # Hatch for out of bound values in a year
            ax[i].patch.set(hatch='xx', edgecolor='black')
            fig.colorbar(mesh, ax=ax[i])
        
        
        # Add color bar at the bottom
        #cbar_ax = fig.add_axes([0.25, 0.10, 0.5, 0.05])
        #fig.colorbar(mesh, orientation="vertical", pad=0.2, cax = cbar_ax)
        #fig.colorbar(mesh, orientation="horizontal", pad=0.2, cax = ax)
        
        
        #colorbar = ax[0].collections[0].colorbar
        #r = colorbar.vmax - colorbar.vmin
        
        fig.subplots_adjust(hspace = 0.5)
    
    
    
    
    
    plt.xlabel('Map of energy consumption in kWh/day', fontsize=12)
    


    
    return fig



def build_day_night_energy_share_figure(total_datalog_df, start_day_hour = 7, stop_day_hour = 19):
    
    all_channels_labels = list(total_datalog_df.columns)
    channel_number_Pout_actif_Tot = [i for i, elem in enumerate(all_channels_labels) if "Consumption" in elem]
    channel_label_Pout_actif_Tot=total_datalog_df.columns[channel_number_Pout_actif_Tot]

    #take only the wanted column:
    col_to_convert=channel_label_Pout_actif_Tot[0]

    #separate the day and night consumption to see if there is a difference in the distribution of consumption values:
    total_datalog_df["hour"] = total_datalog_df.index.hour
    day_consumption = total_datalog_df[total_datalog_df["hour"].between(start_day_hour, stop_day_hour)][col_to_convert]
    night_consumption = total_datalog_df[~total_datalog_df["hour"].between(start_day_hour, stop_day_hour)][col_to_convert]  


    #compute the total energy consumption during the day and during the night:
    total_day_consumption = day_consumption.sum()   
    total_night_consumption = night_consumption.sum()
    print(f"Total day consumption: {total_day_consumption:.2f} kWh")
    print(f"Total night consumption: {total_night_consumption:.2f} kWh")
    #and plot it in a bar chart and in a pie chart to see the fraction of day and night consumption in the total consumption:
    fig_day_night_ratio, axes_day_night = plt.subplots(ncols = 2, figsize=(6, 6))
    axes_day_night[0].bar(["Day", "Night"], [total_day_consumption, total_night_consumption], color=["orange", "blue"])
    axes_day_night[0].set_title("Total consumption during day and night")
    axes_day_night[0].set_ylabel("Total consumption (kWh)")
    axes_day_night[0].grid()  
    axes_day_night[1].pie([total_day_consumption, total_night_consumption], labels=["Day", "Night"], autopct="%1.1f%%", colors=["orange", "blue"])
    axes_day_night[1].set_title(f"Fraction of day ({start_day_hour}-{stop_day_hour}h)\nand night consumption")




    
    # fig, ax = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    
    # ax.bar(['Day', 'Night'], [share_day, share_night], color=[SOLAR_COLOR, 'k'], alpha=0.8)
    
    # ax.set_ylim(0, 100)
    
    # ax.set_ylabel('Energy share [%]', fontsize=12)
    
    # if FRANCAIS_LANGUAGE:
    #     ax.set_title("Part de l'énergie consommée pendant le jour et la nuit", fontsize=14, weight="bold")
    # else:
    #     ax.set_title("Share of energy consumed during day and night", fontsize=14, weight="bold")
    # ax.grid(True, axis='y')

    return fig_day_night_ratio



def build_day_night_energy_share_by_week_figure(total_datalog_df, start_day_hour = 7, stop_day_hour = 19):

    all_channels_labels = list(total_datalog_df.columns)
    channel_number_Pout_actif_Tot = [i for i, elem in enumerate(all_channels_labels) if "Consumption" in elem]
    channel_label_Pout_actif_Tot=total_datalog_df.columns[channel_number_Pout_actif_Tot]

    #take only the wanted column:
    col_to_convert=channel_label_Pout_actif_Tot[0]
    # separate the night consumption and day consumption for each day of the week and plot it in a stacked bar chart
    total_datalog_df["day_of_week"] = total_datalog_df.index.dayofweek  
    week_consumption = total_datalog_df.groupby('day_of_week')[col_to_convert].sum()
    day_by_week = total_datalog_df[total_datalog_df['hour'].between(start_day_hour, stop_day_hour)].groupby('day_of_week')[col_to_convert].sum()
    night_by_week = total_datalog_df[~total_datalog_df['hour'].between(start_day_hour, stop_day_hour)].groupby('day_of_week')[col_to_convert].sum()

    # Ensure days exist for all weekdays (0..6) so bars align correctly
    index = range(7)
    day_by_week = day_by_week.reindex(index, fill_value=0)
    night_by_week = night_by_week.reindex(index, fill_value=0)

    days_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(days_labels, night_by_week.values, label=f"Night ({stop_day_hour }-{start_day_hour}h)", color='navy')
    ax.bar(days_labels, day_by_week.values, bottom=night_by_week.values, label=f"Day ({start_day_hour}-{stop_day_hour}h)", color='orange')
    ax.set_title('Total electricity consumption by weekday: Night + Day')
    ax.set_ylabel(f'Total consumption (kWh)')
    ax.set_xlabel('Day of week')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add numeric labels for day/night segments
    for i, (n, d) in enumerate(zip(night_by_week.values, day_by_week.values)):
        if n > 0:
            ax.text(i, n / 2, f'{n:.1f}', ha='center', va='center', color='white', fontsize=8)
        if d > 0:
            ax.text(i, n + d / 2, f'{d:.1f}', ha='center', va='center', color='black', fontsize=8)

   
    return fig


def build_mean_daily_consumption_by_season_figure(total_datalog_df):

    all_channels_labels = list(total_datalog_df.columns)
    channel_number_Pout_actif_Tot = [i for i, elem in enumerate(all_channels_labels) if "Consumption" in elem]
    channel_label_Pout_actif_Tot = total_datalog_df.columns[channel_number_Pout_actif_Tot]

    # take only the wanted column:
    col_to_convert = channel_label_Pout_actif_Tot[0]

    temp_df = total_datalog_df[[col_to_convert]].copy()

    if len(temp_df.index) < 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(['Winter', 'Spring', 'Summer', 'Autumn'], [0, 0, 0, 0],
               color=['#5DA5DA', '#60BD68', '#F17CB0', '#F15854'])
        ax.set_title('Mean daily electricity consumption by season')
        ax.set_ylabel('Mean daily consumption (kWh/day)')
        ax.set_xlabel('Season')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        return fig

    timestep_hours = temp_df.index.to_series().diff().dropna().median().total_seconds() / 3600.0

    daily_energy = temp_df[col_to_convert].resample('D').sum() * timestep_hours

    def month_to_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        if month in [3, 4, 5]:
            return 'Spring'
        if month in [6, 7, 8]:
            return 'Summer'
        return 'Autumn'

    daily_consumption_df = daily_energy.to_frame(name='daily_consumption')
    daily_consumption_df['season'] = daily_consumption_df.index.month.map(month_to_season)

    seasons_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    seasonal_mean = daily_consumption_df.groupby('season')['daily_consumption'].mean()
    seasonal_mean = seasonal_mean.reindex(seasons_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        seasons_order,
        seasonal_mean.values,
        color=['#5DA5DA', '#60BD68', '#F9A03F', '#F17C67']
    )

    ax.set_title('Mean daily electricity consumption by season')
    ax.set_ylabel('Mean daily consumption (kWh/day)')
    ax.set_xlabel('Season')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, value in zip(bars, seasonal_mean.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{value:.1f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

    return fig


def build_consumption_week_analysis_by_season(total_datalog_df, start_date = datetime.date(2000, 1, 1), end_date = datetime.date(2050, 12, 31)):


    #take only the wanted column:
    all_channels_labels = list(total_datalog_df.columns)
    channel_number = [i for i, elem in enumerate(all_channels_labels) if "Consumption" in elem]
    channel_label = all_channels_labels[channel_number[0]]

    load_df = total_datalog_df[[channel_label]].copy()

    # make the sorting filter with the given dates:
    temp1 = load_df[load_df.index.date >= start_date]
    temp2 = temp1[temp1.index.date <= end_date].copy()

    def month_to_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        if month in [3, 4, 5]:
            return 'Spring'
        if month in [6, 7, 8]:
            return 'Summer'
        return 'Autumn'

    time_of_week_in_minutes = list(temp2.index.dayofweek * 60 * 24 + temp2.index.hour * 60 + temp2.index.minute)
    time_of_week_in_days = list(temp2.index.dayofweek + temp2.index.hour / 24 + temp2.index.minute / 60 / 24)

    # add channels to the dataframe with week position to be able to sort data on it:
    temp2['Time of week in minutes'] = time_of_week_in_minutes
    temp2['Time of week quarter'] = (temp2.index.dayofweek * 24 * 4 + temp2.index.hour * 4 + temp2.index.minute // 15).astype(int)
    temp2['Season'] = temp2.index.month.map(month_to_season)

    fig_pow_by_min_of_week, axes_pow_by_min_of_day = plt.subplots(nrows=1, ncols=1, figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))

    seasons_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    season_colors = {
        'Winter': '#4C78A8',
        'Spring': '#54A24B',
        'Summer': '#ECA82C',
        'Autumn': '#E45756'
    }

    x_quarters = np.array(range(0, 24 * 4 * 7))

    for season in seasons_order:
        temp_season = temp2[temp2['Season'] == season]
        if temp_season.empty:
            continue

        mean_by_minute = np.full(24 * 4 * 7, np.nan)
        grouped_profile = temp_season.groupby('Time of week quarter')[channel_label].mean()
        mean_by_minute[grouped_profile.index.astype(int)] = grouped_profile.values

        axes_pow_by_min_of_day.plot(
            x_quarters / 4 / 24,
            mean_by_minute,
            color=season_colors[season],
            linestyle='-',
            linewidth=2.5,
            drawstyle='steps-post',
            label=season
        )

    axes_pow_by_min_of_day.set_ylabel("Power [kW]", fontsize=12)
    axes_pow_by_min_of_day.set_xlabel("Day of the week", fontsize=12)
    axes_pow_by_min_of_day.set_xlim(0, 7)

    xticks_wanted = range(7)
    axes_pow_by_min_of_day.set_xticks(xticks_wanted)
    DAYS = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
    axes_pow_by_min_of_day.set_xticklabels(DAYS, ha='left')

    axes_pow_by_min_of_day.set_title("Mean weekly consumption profile by season", fontsize=12, weight="bold")
    axes_pow_by_min_of_day.legend()
    axes_pow_by_min_of_day.grid(True)

    return fig_pow_by_min_of_week


def build_polar_consumption_profile_by_season(total_datalog_df, start_date = datetime.date(2000, 1, 1), end_date = datetime.date(2050, 12, 31)):
    # for tests:
    # start_date = dt.date(2018, 7, 1)
    # end_date = dt.date(2018, 8, 30)

    temp1 = total_datalog_df[total_datalog_df.index.date >= start_date]
    temp2 = temp1[temp1.index.date <= end_date].copy()

    all_channels_labels = list(total_datalog_df.columns)
    channel_number_consumption = [i for i, elem in enumerate(all_channels_labels) if 'Consumption' in elem]

    time_of_day_in_minutes = list(temp2.index.hour * 60 + temp2.index.minute)

    # add a channel to the dataframe with minutes of the day to be able to sort data on it:
    temp2['Time of day in minutes'] = time_of_day_in_minutes

    def month_to_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        if month in [3, 4, 5]:
            return 'Spring'
        if month in [6, 7, 8]:
            return 'Summer'
        return 'Autumn'

    temp2['Season'] = temp2.index.month.map(month_to_season)

    FIGSIZE_WIDTH = 6
    FIGSIZE_HEIGHT = 5
    fig_pow_by_min_of_day, axes_pow_by_min_of_day = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT),
        subplot_kw={'projection': 'polar'}
    )

    axes_pow_by_min_of_day.set_theta_zero_location("S")  # theta=0 at the botom
    axes_pow_by_min_of_day.set_theta_direction(-1)  # theta increasing clockwise

    if channel_number_consumption:

        channel_label_consumption = all_channels_labels[channel_number_consumption[0]]
        seasons_order = ['Winter', 'Spring', 'Summer', 'Autumn']
        season_colors = {
            'Winter': '#4C78A8',
            'Spring': '#54A24B',
            'Summer': '#ECA82C',
            'Autumn': '#E45756'
        }

        x1 = np.array(range(0, 24 * 4))
        radial_max = 0

        for season in seasons_order:
            temp_season = temp2[temp2['Season'] == season]
            if temp_season.empty:
                continue

            mean_by_minute = np.full(24 * 4, np.nan)
            grouped_profile = temp_season.groupby((temp_season['Time of day in minutes'] // 15).astype(int))[channel_label_consumption].mean()
            mean_by_minute[grouped_profile.index.astype(int)] = grouped_profile.values

            if not np.all(np.isnan(mean_by_minute)):
                radial_max = max(radial_max, np.nanmax(mean_by_minute))

            axes_pow_by_min_of_day.plot(
                x1 / 24 / 4 * 2 * np.pi,
                mean_by_minute,
                color=season_colors[season],
                linestyle='-',
                linewidth=2.5,
                label=season
            )

        ticks = np.linspace(0, 2 * np.pi, 8, endpoint=False)

        axes_pow_by_min_of_day.set_ylim([0, radial_max if radial_max > 0 else 1])
        axes_pow_by_min_of_day.set_xticks(ticks)
        axes_pow_by_min_of_day.set_xticklabels(['midnight', '3h', '6h', '9h', '12h', '15h', '18h', '21h'])
        axes_pow_by_min_of_day.legend(loc='lower right', fontsize=10)
        axes_pow_by_min_of_day.set_title("Mean consumption profiles by season  \n in kW ", fontsize=12, weight='bold')
        axes_pow_by_min_of_day.grid(True)

        if I_WANT_WATERMARK_ON_FIGURE:
            im = Image.open(WATERMARK_PICTURE)
            fig_pow_by_min_of_day.figimage(im, 0.05 * FIGSIZE_WIDTH * 150, 0.1 * FIGSIZE_HEIGHT * 150, zorder=3, alpha=.2)

    return fig_pow_by_min_of_day


def build_polar_consumption_and_solar_profile_by_season_tiles(total_datalog_df, start_date = datetime.date(2000, 1, 1), end_date = datetime.date(2050, 12, 31)):
    # for tests:
    # start_date = dt.date(2018, 7, 1)
    # end_date = dt.date(2018, 8, 30)

    temp1 = total_datalog_df[total_datalog_df.index.date >= start_date]
    temp2 = temp1[temp1.index.date <= end_date].copy()

    all_channels_labels = list(total_datalog_df.columns)
    channel_number_consumption = [i for i, elem in enumerate(all_channels_labels) if 'Consumption' in elem]
    channels_number_solar = [i for i, elem in enumerate(all_channels_labels) if 'Solar power scaled' in elem]

    time_of_day_in_minutes = list(temp2.index.hour * 60 + temp2.index.minute)
    temp2['Time of day in minutes'] = time_of_day_in_minutes

    def month_to_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        if month in [3, 4, 5]:
            return 'Spring'
        if month in [6, 7, 8]:
            return 'Summer'
        return 'Autumn'

    temp2['Season'] = temp2.index.month.map(month_to_season)

    fig_pow_by_min_of_day, axes_pow_by_min_of_day = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 10),
        subplot_kw={'projection': 'polar'}
    )

    seasons_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    season_colors = {
        'Winter': '#4C78A8',
        'Spring': '#54A24B',
        'Summer': '#ECA82C',
        'Autumn': '#E45756'
    }

    if channel_number_consumption and channels_number_solar:
        channel_label_consumption = all_channels_labels[channel_number_consumption[0]]
        channel_label_solar = all_channels_labels[channels_number_solar[0]]

        x1 = np.array(range(0, 24 * 4))
        ticks = np.linspace(0, 2 * np.pi, 8, endpoint=False)

        for ax, season in zip(axes_pow_by_min_of_day.flatten(), seasons_order):
            ax.set_theta_zero_location("S")
            ax.set_theta_direction(-1)

            temp_season = temp2[temp2['Season'] == season]
            mean_by_minute = np.full(24 * 4, np.nan)
            mean_by_minute_sol = np.full(24 * 4, np.nan)

            if not temp_season.empty:
                grouped_consumption = temp_season.groupby((temp_season['Time of day in minutes'] // 15).astype(int))[channel_label_consumption].mean()
                grouped_solar = temp_season.groupby((temp_season['Time of day in minutes'] // 15).astype(int))[channel_label_solar].mean()

                mean_by_minute[grouped_consumption.index.astype(int)] = grouped_consumption.values
                mean_by_minute_sol[grouped_solar.index.astype(int)] = grouped_solar.values

            radial_max = np.nanmax([np.nanmax(mean_by_minute), np.nanmax(mean_by_minute_sol)])
            if np.isnan(radial_max) or radial_max <= 0:
                radial_max = 1

            ax.plot(
                x1 / 24 / 4 * 2 * np.pi,
                mean_by_minute,
                color=season_colors[season],
                linestyle='-',
                linewidth=2.5,
                label='consumption mean'
            )
            ax.plot(
                x1 / 24 / 4 * 2 * np.pi,
                mean_by_minute_sol,
                color=SOLAR_COLOR,
                linestyle='-',
                linewidth=2.0,
                label='solar mean'
            )

            ax.set_ylim([0, radial_max])
            ax.set_xticks(ticks)
            ax.set_xticklabels(['midnight', '3h', '6h', '9h', '12h', '15h', '18h', '21h'])
            ax.set_title(season, fontsize=12, weight='bold')
            ax.grid(True)
            ax.legend(loc='lower right', fontsize=8)

        fig_pow_by_min_of_day.suptitle("Mean consumption and solar profiles by season", fontsize=12, weight='bold')

        if I_WANT_WATERMARK_ON_FIGURE:
            im = Image.open(WATERMARK_PICTURE)
            fig_pow_by_min_of_day.figimage(im, 0.05 * 12 * 150, 0.1 * 10 * 150, zorder=3, alpha=.2)

    return fig_pow_by_min_of_day


def build_temperature_analysis_figure(total_datalog_df, temperature_column='temp_c', heating_base_temperature=17.0):

    temp_df = total_datalog_df[[temperature_column]].copy()
    temp_df = temp_df.dropna()

    fig_temperature, axes_temperature = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT * 2.0)
    )

    if temp_df.empty:
        axes_temperature[0].set_title("Temperature data", fontsize=12, weight="bold")
        axes_temperature[1].set_title("Mean temperature per day", fontsize=12, weight="bold")
        axes_temperature[2].set_title("Heating degree days", fontsize=12, weight="bold")
        axes_temperature[3].set_title("Heating degree days by season", fontsize=12, weight="bold")
        for ax in axes_temperature:
            ax.grid(True, alpha=0.4)
        return fig_temperature

    daily_mean_temperature = temp_df[temperature_column].resample('D').mean()
    heating_degree_days = (heating_base_temperature - daily_mean_temperature).clip(lower=0)

    def month_to_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        if month in [3, 4, 5]:
            return 'Spring'
        if month in [6, 7, 8]:
            return 'Summer'
        return 'Autumn'

    seasonal_hdd = heating_degree_days.groupby(heating_degree_days.index.month.map(month_to_season)).sum()
    seasons_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    seasonal_hdd = seasonal_hdd.reindex(seasons_order, fill_value=0)

    axes_temperature[0].plot(
        temp_df.index,
        temp_df[temperature_column].values,
        color=A_BLUE_COLOR,
        linewidth=1.0,
        alpha=0.8
    )
    axes_temperature[0].set_ylabel("Temp [degC]", fontsize=11)
    axes_temperature[0].set_title("Temperature data", fontsize=12, weight="bold")
    axes_temperature[0].grid(True, alpha=0.4)

    axes_temperature[1].plot(
        daily_mean_temperature.index,
        daily_mean_temperature.values,
        color=A_RED_COLOR,
        linewidth=2.0
    )
    axes_temperature[1].set_ylabel("Mean [degC]", fontsize=11)
    axes_temperature[1].set_title("Mean temperature per day", fontsize=12, weight="bold")
    axes_temperature[1].grid(True, alpha=0.4)

    axes_temperature[2].bar(
        heating_degree_days.index,
        heating_degree_days.values,
        width=0.8,
        color=A_YELLOW_COLOR,
        edgecolor=A_RED_COLOR,
        linewidth=0.5
    )
    axes_temperature[2].set_ylabel("HDD [degC day]", fontsize=11)
    axes_temperature[2].set_title(f"Heating degree days (base {heating_base_temperature:.1f} degC)", fontsize=12, weight="bold")
    axes_temperature[2].grid(True, alpha=0.4)

    bars = axes_temperature[3].bar(
        seasons_order,
        seasonal_hdd.values,
        color=[A_BLUE_COLOR, NX_GREEN, A_YELLOW_COLOR, A_RED_COLOR]
    )
    axes_temperature[3].set_ylabel("HDD [degC day]", fontsize=11)
    axes_temperature[3].set_title("Total heating degree days by season", fontsize=12, weight="bold")
    axes_temperature[3].set_xlabel("Season", fontsize=12)
    axes_temperature[3].grid(True, axis='y', alpha=0.4)

    for bar, value in zip(bars, seasonal_hdd.values):
        axes_temperature[3].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{value:.1f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

    fig_temperature.tight_layout()

    return fig_temperature


def build_monthly_consumption_vs_hdd_correlation_figure(total_datalog_df, temperature_column='temp_c', heating_base_temperature=17.0):

    all_channels_labels = list(total_datalog_df.columns)
    channel_number_consumption = [i for i, elem in enumerate(all_channels_labels) if 'Consumption' in elem]
    channel_label_consumption = all_channels_labels[channel_number_consumption[0]]

    temp_df = total_datalog_df[[channel_label_consumption, temperature_column]].copy()
    temp_df = temp_df.dropna()

    fig_correlation, axes_correlation = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT * 1.8)
    )

    if temp_df.empty or len(temp_df.index) < 2:
        axes_correlation[0].set_title("Consumption by month", fontsize=12, weight="bold")
        axes_correlation[1].set_title("Heating degree days by month", fontsize=12, weight="bold")
        axes_correlation[2].set_title("Monthly consumption vs heating degree days", fontsize=12, weight="bold")
        for ax in axes_correlation:
            ax.grid(True, alpha=0.4)
        return fig_correlation

    timestep_hours = temp_df.index.to_series().diff().dropna().median().total_seconds() / 3600.0

    monthly_consumption = temp_df[channel_label_consumption].resample('ME').sum() * timestep_hours
    daily_mean_temperature = temp_df[temperature_column].resample('D').mean()
    daily_hdd = (heating_base_temperature - daily_mean_temperature).clip(lower=0)
    monthly_hdd = daily_hdd.resample('ME').sum()

    common_index = monthly_consumption.index.intersection(monthly_hdd.index)
    monthly_consumption = monthly_consumption.loc[common_index]
    monthly_hdd = monthly_hdd.loc[common_index]

    month_labels = [idx.strftime('%Y-%m') for idx in common_index]

    axes_correlation[0].bar(month_labels, monthly_consumption.values, color=A_RED_COLOR, alpha=0.85)
    axes_correlation[0].set_title("Consumption by month", fontsize=12, weight="bold")
    axes_correlation[0].set_ylabel("Consumption [kWh]", fontsize=11)
    axes_correlation[0].tick_params(axis='x', rotation=45)
    axes_correlation[0].grid(True, axis='y', alpha=0.4)

    axes_correlation[1].bar(month_labels, monthly_hdd.values, color=A_BLUE_COLOR, alpha=0.85)
    axes_correlation[1].set_title(f"Heating degree days by month (base {heating_base_temperature:.1f} degC)", fontsize=12, weight="bold")
    axes_correlation[1].set_ylabel("HDD [degC day]", fontsize=11)
    axes_correlation[1].tick_params(axis='x', rotation=45)
    axes_correlation[1].grid(True, axis='y', alpha=0.4)

    axes_correlation[2].scatter(monthly_hdd.values, monthly_consumption.values, color=A_RAISINBLACK_COLOR, alpha=0.85)

    valid_mask = np.isfinite(monthly_hdd.values) & np.isfinite(monthly_consumption.values)
    x_values = monthly_hdd.values[valid_mask]
    y_values = monthly_consumption.values[valid_mask]

    if len(x_values) >= 2:
        fit_coeff = np.polyfit(x_values, y_values, 1)
        fit_x = np.linspace(np.min(x_values), np.max(x_values), 100)
        fit_y = np.polyval(fit_coeff, fit_x)
        correlation_coef = np.corrcoef(x_values, y_values)[0, 1]

        axes_correlation[2].plot(fit_x, fit_y, color=SOLAR_COLOR, linewidth=2.5)
        axes_correlation[2].text(
            0.02,
            0.98,
            f'y = {fit_coeff[0]:.2f}x + {fit_coeff[1]:.1f}\nr = {correlation_coef:.3f} correlation',
            transform=axes_correlation[2].transAxes,
            ha='left',
            va='top',
            fontsize=10
        )

    axes_correlation[2].set_title("Monthly consumption vs heating degree days", fontsize=12, weight="bold")
    axes_correlation[2].set_xlabel("HDD by month [degC day]", fontsize=11)
    axes_correlation[2].set_ylabel("Consumption by month [kWh]", fontsize=11)
    axes_correlation[2].grid(True, alpha=0.4)

    fig_correlation.tight_layout()

    return fig_correlation


def build_daily_consumption_vs_hdd_correlation_figure(total_datalog_df, temperature_column='temp_c', heating_base_temperature=17.0):

    all_channels_labels = list(total_datalog_df.columns)
    channel_number_consumption = [i for i, elem in enumerate(all_channels_labels) if 'Consumption' in elem]
    channel_label_consumption = all_channels_labels[channel_number_consumption[0]]

    temp_df = total_datalog_df[[channel_label_consumption, temperature_column]].copy()
    temp_df = temp_df.dropna()

    fig_correlation, axes_correlation = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT * 1.8)
    )
    daily_base = np.nan

    if temp_df.empty or len(temp_df.index) < 2:
        axes_correlation[0].set_title("Consumption by day", fontsize=12, weight="bold")
        axes_correlation[1].set_title("Heating degree days by day", fontsize=12, weight="bold")
        axes_correlation[2].set_title("Daily consumption vs heating degree days", fontsize=12, weight="bold")
        for ax in axes_correlation:
            ax.grid(True, alpha=0.4)
        return fig_correlation, daily_base

    timestep_hours = temp_df.index.to_series().diff().dropna().median().total_seconds() / 3600.0

    daily_consumption = temp_df[channel_label_consumption].resample('D').sum() * timestep_hours
    daily_mean_temperature = temp_df[temperature_column].resample('D').mean()
    daily_hdd = (heating_base_temperature - daily_mean_temperature).clip(lower=0)

    common_index = daily_consumption.index.intersection(daily_hdd.index)
    daily_consumption = daily_consumption.loc[common_index]
    daily_hdd = daily_hdd.loc[common_index]

    axes_correlation[0].plot(daily_consumption.index, daily_consumption.values, color=A_RED_COLOR, linewidth=1.8)
    axes_correlation[0].set_title("Consumption by day", fontsize=12, weight="bold")
    axes_correlation[0].set_ylabel("Consumption [kWh]", fontsize=11)
    axes_correlation[0].grid(True, alpha=0.4)

    axes_correlation[1].bar(daily_hdd.index, daily_hdd.values, width=0.8, color=A_BLUE_COLOR, alpha=0.85)
    axes_correlation[1].set_title(f"Heating degree days by day (base {heating_base_temperature:.1f} degC)", fontsize=12, weight="bold")
    axes_correlation[1].set_ylabel("HDD [degC day]", fontsize=11)
    axes_correlation[1].grid(True, axis='y', alpha=0.4)

    axes_correlation[2].scatter(daily_hdd.values, daily_consumption.values, color=A_RAISINBLACK_COLOR, alpha=0.55, s=18, label='all points')

    valid_mask = np.isfinite(daily_hdd.values) & np.isfinite(daily_consumption.values)
    x_values = daily_hdd.values[valid_mask]
    y_values = daily_consumption.values[valid_mask]

    if len(x_values) >= 2:
        fit_coeff = np.polyfit(x_values, y_values, 1)
        fit_x = np.linspace(np.min(x_values), np.max(x_values), 100)
        fit_y = np.polyval(fit_coeff, fit_x)
        correlation_coef = np.corrcoef(x_values, y_values)[0, 1]

        axes_correlation[2].plot(fit_x, fit_y, color=SOLAR_COLOR, linewidth=2.5, label='linear fit')

        quad_coeff = np.polyfit(x_values, y_values, 2)
        daily_base = quad_coeff[2]
        quad_y = np.polyval(quad_coeff, fit_x)
        axes_correlation[2].plot(fit_x, quad_y, color='red', linewidth=2.0, label='2nd order fit')

        axes_correlation[2].text(
            0.02,
            0.98,
            f'Linear: y = {fit_coeff[0]:.2f}x + {fit_coeff[1]:.1f}\n2nd order: y = {quad_coeff[0]:.4f}x² + {quad_coeff[1]:.2f}x + {quad_coeff[2]:.1f}\nr = {correlation_coef:.3f} correlation',
            transform=axes_correlation[2].transAxes,
            ha='left',
            va='top',
            fontsize=10
        )

    axes_correlation[2].set_title("Daily consumption vs heating degree days", fontsize=12, weight="bold")
    axes_correlation[2].set_xlabel("HDD by day [degC day]", fontsize=11)
    axes_correlation[2].set_ylabel("Consumption by day [kWh]", fontsize=11)
    axes_correlation[2].legend()
    axes_correlation[2].grid(True, alpha=0.4)

    fig_correlation.tight_layout()

    return fig_correlation, daily_base



def build_dup_figure(total_datalog_df, column_name=None):
    """
    Build a DUP figure from a power time series.

    DUP (duree d'utilisation de la puissance) = energy / peak power
    and is expressed in hours.

    The two subplots are:
      - a load-duration curve with the DUP reference
      - cumulative hours spent below a given power threshold, with the DUP reference
    """

    if total_datalog_df.empty:
        fig_dup, ax_dup = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
        ax_dup.set_title("DUP unavailable: no data", fontsize=12, weight="bold")
        ax_dup.grid(True)
        return fig_dup

    all_channels_labels = list(total_datalog_df.columns)

    if column_name is None:
        channel_number = [i for i, elem in enumerate(all_channels_labels) if "Consumption" in elem]
        if not channel_number:
            fig_dup, ax_dup = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
            ax_dup.set_title("DUP unavailable: no consumption channel found", fontsize=12, weight="bold")
            ax_dup.grid(True)
            return fig_dup
        column_name = all_channels_labels[channel_number[0]]

    temp_df = total_datalog_df[[column_name]].copy().dropna()

    if len(temp_df.index) < 2:
        fig_dup, ax_dup = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
        ax_dup.set_title("DUP unavailable: not enough data", fontsize=12, weight="bold")
        ax_dup.grid(True)
        return fig_dup

    timestep_hours = temp_df.index.to_series().diff().dropna().median().total_seconds() / 3600.0

    power_series = temp_df[column_name].clip(lower=0)
    total_energy = power_series.sum() * timestep_hours
    peak_power = power_series.max()
    global_dup = total_energy / peak_power if peak_power > 0 else 0.0

    fig_dup, ax_dup = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))

    sorted_power_desc = np.sort(power_series.values)[::-1]
    cumulative_hours_desc = np.arange(1, len(sorted_power_desc) + 1) * timestep_hours

    ax_dup.plot(cumulative_hours_desc, sorted_power_desc, color=A_BLUE_COLOR, linewidth=2)
    ax_dup.axvline(global_dup, color=A_YELLOW_COLOR, linestyle="--", linewidth=2)
    ax_dup.hlines(
        peak_power,
        xmin=0,
        xmax=global_dup,
        color=A_YELLOW_COLOR,
        linestyle=":",
        linewidth=2
    )
    ax_dup.fill_between(cumulative_hours_desc, sorted_power_desc, color=A_BLUE_COLOR, alpha=0.15)

    ax_dup.fill_between(
        [0, global_dup],
        [peak_power, peak_power],
        color=A_YELLOW_COLOR,
        alpha=0.12
    )
    ax_dup.set_xlim(left=0)
    ax_dup.set_ylim(bottom=0)
    ax_dup.set_xlabel("Heures cumulees [h]", fontsize=12)
    ax_dup.set_ylabel("Puissance [kW]", fontsize=12)
    ax_dup.set_title("Courbe de duree de charge et equivalent DUP", fontsize=12, weight="bold")
    ax_dup.legend([
        "Puissance triee",
        f"DUP = {global_dup:.1f} h",
        f"Pmax = {peak_power:.1f} kW"
    ])
    ax_dup.grid(True, linestyle="--", alpha=0.4)

    fig_dup.tight_layout()

    if I_WANT_WATERMARK_ON_FIGURE:
        im = Image.open(WATERMARK_PICTURE)
        fig_dup.figimage(im, 0.05 * FIGSIZE_WIDTH * 150, 0.1 * FIGSIZE_HEIGHT * 150, zorder=3, alpha=.2)

    if I_WANT_TO_SAVE_PNG:
        fig_dup.savefig("FigureExport/dup_figure.png")

    return fig_dup


def build_consumption_cdf_figure(total_datalog_df, column_name=None, ribbon_level=10):
    """
    Build the CDF plot of a consumption channel and highlight the ribbon level.

    ribbon_level is expressed as a percentile in %.
    Example: ribbon_level=10 gives the power/energy value present 10% of the sorted CDF,
    which corresponds to the value exceeded 90% of the time in the notebook logic.
    """

    if total_datalog_df.empty:
        fig_cdf, ax_cdf = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
        ax_cdf.set_title("CDF unavailable: no data", fontsize=12, weight="bold")
        ax_cdf.grid(True)
        return fig_cdf, 0.0

    all_channels_labels = list(total_datalog_df.columns)

    if column_name is None:
        channel_number = [i for i, elem in enumerate(all_channels_labels) if "Consumption" in elem]
        if not channel_number:
            fig_cdf, ax_cdf = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
            ax_cdf.set_title("CDF unavailable: no consumption channel found", fontsize=12, weight="bold")
            ax_cdf.grid(True)
            return fig_cdf, 0.0
        column_name = all_channels_labels[channel_number[0]]

    values = pd.to_numeric(total_datalog_df[column_name], errors='coerce').dropna()

    if values.empty:
        fig_cdf, ax_cdf = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
        ax_cdf.set_title("CDF unavailable: no numeric data", fontsize=12, weight="bold")
        ax_cdf.grid(True)
        return fig_cdf, 0.0

    sorted_values = np.sort(values.values)
    cdf_percentage = np.linspace(0, 100, len(sorted_values))
    ribbon_level = float(np.clip(ribbon_level, 0, 100))
    consumption_ribbon = np.percentile(sorted_values, ribbon_level)

    if np.isnan(consumption_ribbon):
        consumption_ribbon = 0.0

    fig_cdf, ax_cdf = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
    ax_cdf.plot(sorted_values, cdf_percentage, color=A_BLUE_COLOR, linewidth=2)
    ax_cdf.axhline(ribbon_level, color=A_RED_COLOR, linestyle="--", linewidth=2, label=f"{ribbon_level:.1f}% of time")
    ax_cdf.scatter([consumption_ribbon], [ribbon_level], color=A_RED_COLOR, zorder=3)

    annotation_x = consumption_ribbon * 1.25 if consumption_ribbon > 0 else values.max() * 0.1
    annotation_y = min(ribbon_level + 8, 98)
    ax_cdf.annotate(
        f"{consumption_ribbon:.2f}",
        xy=(consumption_ribbon, ribbon_level),
        xytext=(annotation_x, annotation_y),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=11
    )

    ax_cdf.set_title("CDF de la consommation electrique", fontsize=12, weight="bold")
    ax_cdf.set_xlabel(column_name, fontsize=12)
    ax_cdf.set_ylabel("Proportion de temps (%)", fontsize=12)
    ax_cdf.set_ylim(0, 100)
    ax_cdf.set_xlim(0, values.max() * 1.1 if values.max() > 0 else 1)
    ax_cdf.grid(True, linestyle="--", alpha=0.4)
    ax_cdf.legend()

    fig_cdf.tight_layout()

    if I_WANT_WATERMARK_ON_FIGURE:
        im = Image.open(WATERMARK_PICTURE)
        fig_cdf.figimage(im, 0.05 * FIGSIZE_WIDTH * 150, 0.1 * FIGSIZE_HEIGHT * 150, zorder=3, alpha=.2)

    if I_WANT_TO_SAVE_PNG:
        fig_cdf.savefig("FigureExport/consumption_cdf_figure.png")

    return fig_cdf, consumption_ribbon


def build_ribbon_fraction_figure(total_datalog_df, column_name=None, ribbon_level=10):
    """
    Build a figure with the consumption profile and a pie chart estimating the
    ribbon share in the total consumption.

    The ribbon estimate follows the notebook logic:
    ribbon_energy = percentile_value * sample_count * (1 - ribbon_level / 100)
    """

    if total_datalog_df.empty:
        fig_ribbon, ax_ribbon = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
        ax_ribbon.set_title("Ribbon fraction unavailable: no data", fontsize=12, weight="bold")
        ax_ribbon.grid(True)
        return fig_ribbon, 0.0, 0.0

    all_channels_labels = list(total_datalog_df.columns)

    if column_name is None:
        channel_number = [i for i, elem in enumerate(all_channels_labels) if "Consumption" in elem]
        if not channel_number:
            fig_ribbon, ax_ribbon = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
            ax_ribbon.set_title("Ribbon fraction unavailable: no consumption channel found", fontsize=12, weight="bold")
            ax_ribbon.grid(True)
            return fig_ribbon, 0.0, 0.0
        column_name = all_channels_labels[channel_number[0]]

    values = pd.to_numeric(total_datalog_df[column_name], errors='coerce').dropna().clip(lower=0)

    if values.empty:
        fig_ribbon, ax_ribbon = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))
        ax_ribbon.set_title("Ribbon fraction unavailable: no numeric data", fontsize=12, weight="bold")
        ax_ribbon.grid(True)
        return fig_ribbon, 0.0, 0.0

    ribbon_level = float(np.clip(ribbon_level, 0, 100))
    consumption_ribbon = np.percentile(values.values, ribbon_level)
    if np.isnan(consumption_ribbon):
        consumption_ribbon = 0.0

    total_consumption = values.sum()
    total_energy_ribbon = consumption_ribbon * len(values) * (1 - ribbon_level / 100)
    total_energy_ribbon = min(total_energy_ribbon, total_consumption)
    remaining_consumption = max(total_consumption - total_energy_ribbon, 0.0)
    ribbon_fraction = (total_energy_ribbon / total_consumption * 100) if total_consumption > 0 else 0.0

    fig_ribbon, axes_ribbon = plt.subplots(ncols=2, figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))

    axes_ribbon[0].plot(values.index, values.values, color=A_BLUE_COLOR, linewidth=1.5)
    axes_ribbon[0].axhline(
        consumption_ribbon,
        color=A_RED_COLOR,
        linestyle="--",
        linewidth=2,
        label=f"Ribbon level: {consumption_ribbon:.2f}"
    )
    axes_ribbon[0].set_title("Consommation electrique", fontsize=12, weight="bold")
    axes_ribbon[0].set_xlabel("Date", fontsize=12)
    axes_ribbon[0].set_ylabel(column_name, fontsize=12)
    axes_ribbon[0].legend()
    axes_ribbon[0].grid(True, linestyle="--", alpha=0.4)

    axes_ribbon[1].pie(
        [total_energy_ribbon, remaining_consumption],
        labels=["Ribbon level", "Rest of consumption"],
        autopct="%1.1f%%",
        colors=[A_RED_COLOR, A_BLUE_COLOR]
    )
    axes_ribbon[1].set_title(
        f"Fraction of the ribbon level in total consumption\n"
        f"Ribbon = {consumption_ribbon:.2f}, share = {ribbon_fraction:.1f}%",
        fontsize=12,
        weight="bold"
    )

    fig_ribbon.tight_layout()

    if I_WANT_WATERMARK_ON_FIGURE:
        im = Image.open(WATERMARK_PICTURE)
        fig_ribbon.figimage(im, 0.05 * FIGSIZE_WIDTH * 150, 0.1 * FIGSIZE_HEIGHT * 150, zorder=3, alpha=.2)

    if I_WANT_TO_SAVE_PNG:
        fig_ribbon.savefig("FigureExport/ribbon_fraction_figure.png")

    return fig_ribbon, consumption_ribbon, ribbon_fraction
