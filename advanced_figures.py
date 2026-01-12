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
    pos = axe_hours_heatmap.pcolormesh(date_only, y_axis, consumption_data.transpose(), shading='auto', cmap='jet') #cmap='RdBu' cmap='hot' viridis, cmap = 'jet'
      
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
    axes_solar[0].set_ylabel("Energy [kWh/day]", fontsize=12)
    axes_solar[0].set_title(title_start+" per day and per month", fontsize=12, weight="bold")
    axes_solar[0].legend(["Day " + title_start])
    axes_solar[0].grid(True)
    
    
    axes_solar[1].set_ylabel("Energy [kWh/month]", fontsize=12)
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
    rate_selfconsumption=(1-abs(day_kwh_df[chanel_label_Pin_Injection])/(day_kwh_df[chanel_label_Psolar_tot]+1e-9))*100
    
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