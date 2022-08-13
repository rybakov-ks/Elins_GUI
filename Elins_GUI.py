import tkinter as tk
from tkinter import filedialog
from tkinter import Menu
from tkinter.filedialog import askopenfilename
from tkinter import filedialog as fd
import os
import re
import matplotlib.pyplot as plt
import pylab
import matplotlib.patches as patches
from itertools import groupby
import pandas as pd
from scipy.signal import argrelextrema
from functools import reduce
from scipy.integrate import simps
from scipy.signal import savgol_filter
import numpy as np
import numpy
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import itertools
import matplotlib.patches as mpatches
import sys
import pyfiglet


def clicked():

    def extract_text(title_name, namef):
        file_name = filedialog.asksaveasfilename(title=title_name, initialfile=namef, filetypes=[('Images','*.jpg')])
        return file_name
        
    def data_import_txt(filename):
        with open(filename, encoding="Windows-1251", errors='ignore') as f:
            ranges_cycles, sweep_speed  = [], []
            data_lines = f.readlines()
            cycles = re.compile("Цикл")
            speeds_CV = re.compile("Скорость развертки")
            for count, line in enumerate(data_lines):
                if cycles.search(line):
                    cycle = line.split()
                    number_lines = count + 13
                    ranges_cycles.append(number_lines)
                if speeds_CV.search(line):
                    speed_CV = line.split()
                    sweep_speed.append(speed_CV[2])
            time, potential, current = [], [], []
            for t, value in enumerate(ranges_cycles):
                time_temp, potential_temp, current_temp = [], [], []
                if t == len(ranges_cycles)-1:
                    [time_temp.append(float(n.split()[0])) for n in data_lines[value-1:]]
                    [potential_temp.append(float(n.split()[1])) for n in data_lines[value-1:]]
                    [current_temp.append(float(n.split()[2])) for n in data_lines[value-1:]]
                else:
                    [time_temp.append(float(n.split()[0])) for n in data_lines[value-1:ranges_cycles[t+1]-15]]
                    [potential_temp.append(float(n.split()[1])) for n in data_lines[value-1:ranges_cycles[t+1]-15]]
                    [current_temp.append(float(n.split()[2])) for n in data_lines[value-1:ranges_cycles[t+1]-15]]
                time.append(time_temp)
                potential.append(potential_temp)
                current.append(current_temp)
        return sweep_speed, current, time, potential
        
    def data_import_edf(filename):
        with open(filename, encoding="Windows-1251", errors='ignore') as f:
            ranges_cycles, sweep_speed = [], []
            data_lines = f.readlines()
            cycles = re.compile("pc")
            speeds_CV = re.compile("cs")
            for count, line in enumerate(data_lines):
                if cycles.search(line):
                    cycle = line.split()
                    number_lines = count + 1
                    ranges_cycles.append(number_lines)
                if speeds_CV.search(line):
                    speed_CV = line.split()
                    sweep_speed.append(speed_CV[1])
            time, potential, current = [], [], []
            for t, value in enumerate(ranges_cycles):
                time_temp, potential_temp, current_temp = [], [], []
                if t == len(ranges_cycles)-1:
                    [time_temp.append(float(n.split()[1])) for n in data_lines[value:-3]]
                    [potential_temp.append(float(n.split()[2])) for n in data_lines[value:-3]]
                    [current_temp.append(float(n.split()[3])) for n in data_lines[value:-3]]
                else:
                    [time_temp.append(float(n.split()[1])) for n in data_lines[value:ranges_cycles[t+1]-20]]
                    [potential_temp.append(float(n.split()[2])) for n in data_lines[value:ranges_cycles[t+1]-20]]
                    [current_temp.append(float(n.split()[3])) for n in data_lines[value:ranges_cycles[t+1]-20]]
                time.append(time_temp)
                potential.append(potential_temp)
                current.append(current_temp)
        return sweep_speed, current, time, potential
        
    def density_current(Mass_active, Theory_сapacity, current):
        density_c = []
        for nn in current:
            density_c.append(round(float(nn[-1])*1000/(Theory_сapacity * Mass_active), 1))
        return density_c
        
    def specific_capacity(Mass_active, current, time):
        specific_capacity = []
        for i, j in zip(time, current):
            capacity = []
            for ii, jj in zip(i, j):
                cap = (abs(float(ii) * float(jj) * 1000 / (3600 * Mass_active)))
                capacity.append(cap)
            specific_capacity.append(capacity)
        capacity_max_anode = [specific_capacity[n][-1] for n in range(0, len(specific_capacity), 2)]
        capacity_max_cathode = [specific_capacity[n][-1] for n in range(1, len(specific_capacity), 2)]
        cycle_capacity = list(range(1, int(len(specific_capacity)/2+1)))
        return capacity_max_anode, capacity_max_cathode, specific_capacity, cycle_capacity
               
    def capacity_cycle(capacity_max_anode, capacity_max_cathode, cycle_capacity, density_current):
        density_current = [density_current[i] for i in range(0, len(density_current), 2)]
        range_c = [0, ]
        for i in range(1, len(density_current)):
            if density_current[i] == density_current[i-1]:
                continue
            else:
                range_c.append(i)
        capacity_max_anode_new, capacity_max_cathode_new, cycle_capacity_new = [], [], []
        for i in range(0, len(range_c)):
            if i == len(range_c)-1:
                capacity_max_anode_new.append(capacity_max_anode[range_c[i]:])
                capacity_max_cathode_new.append(capacity_max_cathode[range_c[i]:])	
                cycle_capacity_new.append(cycle_capacity[range_c[i]:])				
            elif i == 0:
                capacity_max_anode_new.append(capacity_max_anode[:range_c[i+1]])
                capacity_max_cathode_new.append(capacity_max_cathode[:range_c[i+1]])
                cycle_capacity_new.append(cycle_capacity[:range_c[i+1]])				
            else:
                capacity_max_anode_new.append(capacity_max_anode[range_c[i]:range_c[i+1]])
                capacity_max_cathode_new.append(capacity_max_cathode[range_c[i]:range_c[i+1]])
                cycle_capacity_new.append(cycle_capacity[range_c[i]:range_c[i+1]])				
        ax1 = plt.figure(figsize=(15, 6))
        ax1 = plt.subplot(1,2,1)
        plt.suptitle('Cycling capacity', fontsize = 25)
        plt.plot(cycle_capacity, capacity_max_anode, '-', zorder=12, linewidth = 2)
        for i in range(0, len(cycle_capacity_new)):
            plt.plot(cycle_capacity_new[i], capacity_max_anode_new[i], '-o', markersize=8, zorder=12, linewidth = 2, markeredgecolor = 'k', label = str(density_current[i]))
        plt.gca().spines['top'].set_linewidth(3)
        plt.gca().spines['bottom'].set_linewidth(3)
        plt.gca().spines['right'].set_linewidth(3)
        plt.gca().spines['left'].set_linewidth(3)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc = 'lower left', facecolor='white', edgecolor = 'white', fontsize=10)
        plt.tick_params(direction='in', length=6, width=3, grid_alpha=0.3, bottom = True, top = True, left = True, right = True, labelsize = 16)
        plt.ylabel('Specific capacity, mA·h·$\mathregular{g^{-1}}$', fontsize=20)
        plt.xlabel('Cycle number', fontsize=20)
        plt.ylim(0, max(capacity_max_anode)*1.1)
        plt.title('Anode capacity', fontsize=15)
        ax1 = plt.subplot(1,2,2)
        plt.plot(cycle_capacity, capacity_max_cathode, '-', zorder=12, linewidth = 2)
        for i in range(0, len(cycle_capacity_new)):
            plt.plot(cycle_capacity_new[i], capacity_max_cathode_new[i], '-o', markersize=8, zorder=12, linewidth = 2, markeredgecolor = 'k', label = str(density_current[i]))
        plt.gca().spines['top'].set_linewidth(3)
        plt.gca().spines['bottom'].set_linewidth(3)
        plt.gca().spines['right'].set_linewidth(3)
        plt.gca().spines['left'].set_linewidth(3)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc = 'lower left', facecolor='white', edgecolor = 'white', fontsize=10)
        plt.tick_params(direction='in', length=6, width=3, grid_alpha=0.3, bottom = True, top = True, left = True, right = True, labelsize = 16)
        plt.xlabel('Cycle number', fontsize=20)
        plt.ylim(0, max(capacity_max_cathode)*1.1)
        plt.title('Cathode capacity', fontsize=15)
           
    def data_potential(potential):
        potential_avg_anode = [sum(potential[n])/len(potential[n]) for n in range(0, len(potential), 2)]
        potential_avg_cathode = [sum(potential[n])/len(potential[n]) for n in range(1, len(potential), 2)]
        potential_max_anode = [max(potential[n]) for n in range(0, len(potential), 2)]
        potential_min_cathode = [min(potential[n]) for n in range(0, len(potential), 2)]
        potential_difference = [potential_avg_anode[n] - potential_avg_cathode[n] for n in range(0, len(potential_avg_anode), 1)]
        return potential_avg_anode, potential_avg_cathode, potential_max_anode, potential_min_cathode, potential_difference
        
    def arrow_cdc(capacity_max_anode, potential_min_cathode):
        style = "Simple, tail_width=0.5, head_width=8, head_length=8"
        kw = dict(arrowstyle=style, color="m")
        a3 = patches.FancyArrowPatch((capacity_max_anode[0] * 0.4, potential_min_cathode[0]*0.95), (capacity_max_anode[0] * 0.6, potential_min_cathode[0]*0.9), connectionstyle="arc3,rad=-0.3", **kw, zorder=2000)
        plt.gca().add_patch(a3)
        pylab.text((capacity_max_anode[0]) * 0.39, potential_min_cathode[0]*0.915, str("Discharge"), family="Helvetica", fontsize=12, color='b', zorder=2000)
        a3 = patches.FancyArrowPatch((capacity_max_anode[0] * 0.4,  potential_max_anode[0]*0.9), (capacity_max_anode[0] * 0.6,  potential_max_anode[0]*0.95),
                                     connectionstyle="arc3,rad=0.2", **kw, zorder=2000)
        plt.gca().add_patch(a3)
        pylab.text((capacity_max_anode[0]) * 0.4, potential_max_anode[0]*0.92, str("Charge"), family="Helvetica", fontsize=12, color='b', zorder = 2000)    

    def for_diff_curves(potential, specific_capacity):
        potential_new = []
        for yy in potential:
            potential_temp = [el for el, _ in groupby(yy)]
            potential_new.append(potential_temp)
        specific_capacity_new = []
        for rr in range(0, len(specific_capacity), 1):
            specific_capacity_new_temp = []
            for r in range(0, len(specific_capacity[rr]), 1):
                if r > float(len(potential_new[rr])-1):
                    continue
                else:
                    index_potential = potential[rr].index(float(potential_new[rr][r]))
                    specific_capacity_new_temp.append(specific_capacity[rr][index_potential])
            specific_capacity_new.append(specific_capacity_new_temp)
        
        diffQ_new_c, diffP_new_c = [], []
        for i in range(0, len(specific_capacity_new), 2):
            diffQ_temp, diffP_temp = [], []
            for n in range(11, len(specific_capacity_new[i]), 1):
                if float((potential_new[i][n] - potential_new[i][n-11])) == 0:
                    continue
                else:
                    diffQ = (specific_capacity_new[i][n] - specific_capacity_new[i][n-11]) / (2*(potential_new[i][n] - potential_new[i][n-11]))
                    diffP = potential_new[i][n-5]
                    diffP_temp.append(diffP)
                    diffQ_temp.append(diffQ)
            diffP_new_c.append(diffP_temp)   
            diffQ_new_c.append(diffQ_temp)
            
        diffQ_new_dc, diffP_new_dc = [], []
        for i in range(1, len(specific_capacity_new), 2):
            diffQ_temp, diffP_temp = [], []
            for n in range(11, len(specific_capacity_new[i]), 1):
                if (potential_new[i][n] - potential_new[i][n-11]) == 0:
                    continue
                else:
                    diffQ = (specific_capacity_new[i][n] - specific_capacity_new[i][n-11]) / (2*(potential_new[i][n] - potential_new[i][n-11]))
                    diffP = potential_new[i][n-5]
                    diffP_temp.append(diffP)
                    diffQ_temp.append(diffQ)
            diffP_new_dc.append(diffP_temp)   
            diffQ_new_dc.append(diffQ_temp) 
        cmap = plt.get_cmap('inferno')
        N = 4
        ax2 = plt.figure(figsize=(9, 6))
        capacity_df = pd.DataFrame(diffQ_new_c+diffQ_new_dc).T
        potential_df = pd.DataFrame(diffP_new_c+diffP_new_dc).T
        capacity_df_c_inter = diffQ_new_c
        capacity_df_dc_inter = diffQ_new_dc
        capacity_df_c_inter_new = []
        capacity_df_dc_inter_new = []
        for i in range(0, len(capacity_df_c_inter), 1):
            capacity_df_new_c = savgol_filter(capacity_df_c_inter[i], 51, 5) # window size 51, polynomial order 3
            capacity_df_c_inter_new.append(list(capacity_df_new_c))
        for i in range(0, len(capacity_df_dc_inter), 1):   
            capacity_df_new_dc = savgol_filter(capacity_df_dc_inter[i], 51, 5) # window size 51, polynomial order 3
            capacity_df_dc_inter_new.append(list(capacity_df_new_dc))
        capacity_differential = capacity_df_c_inter_new+capacity_df_dc_inter_new
        potentoal_differential = diffP_new_c+diffP_new_dc
        #Создаем список для положения пиков вдоль оси x
        peak_max, peak_min = [], []
        #Создаем список для положения пиков вдоль оси y
        height_max_capacity, height_min_capacity = [], []
        #Находим пики для заданной скорости поляризации
        for m in range(0, len(capacity_differential)):
            #Инвертируем катодный сигнал для алгоритма поиска пиков
            capacity_minus = []
            for rr in capacity_differential[m]:
                capacity_minus.append(rr*-1)
            #Определяем пики
            peaks_max = find_peaks(np.array(capacity_differential[m]), height = max(capacity_differential[m])*0.05, distance = 10)
            peaks_min = find_peaks(np.array(capacity_minus), height = max(capacity_minus)*0.1, distance = 10, plateau_size=1, width=2)
            height_max = peaks_max[1]['peak_heights']
            height_min = peaks_min[1]['peak_heights']
            height_min = [n*-1 for n in height_min]
            peak_pos_max = np.array(potentoal_differential[m])[peaks_max[0]]
            peak_pos_min = np.array(potentoal_differential[m])[peaks_min[0]]
            peak_max.append(peak_pos_max)
            peak_min.append(peak_pos_min)
            height_min_capacity.append(height_min)
            height_max_capacity.append(height_max)
        peak_x_min = list(itertools.chain(*peak_min))
        peak_x_max = list(itertools.chain(*peak_max))
        peak_x = peak_x_min + peak_x_max
        peak_y_min = list(itertools.chain(*height_min_capacity))
        peak_y_max = list(itertools.chain(*height_max_capacity))
        peak_y = peak_y_min + peak_y_max
        #Графиков для пиков со шкалой
        #sc = plt.scatter(peak_x, peak_y, c = peak_y, 
            #linewidth = 1, zorder=100000, edgecolors = 'k', s = 60,
            #vmin=min(peak_y), vmax=max(peak_y), 
            #cmap='brg')
        #plt.colorbar(sc)
        for i in range(0, len(diffQ_new_dc), 1):
            color = cmap(float(i) / len(diffQ_new_dc))
            plt.plot(diffP_new_c[i]+diffP_new_dc[i], capacity_df_c_inter_new[i]+capacity_df_dc_inter_new[i], linewidth=2, zorder = 1000+i*4, color=color, label = str(int(i+1)) + ' cycle')
            plt.gca().spines['top'].set_linewidth(3)
        plt.gca().spines['bottom'].set_linewidth(3)
        plt.gca().spines['right'].set_linewidth(3)
        plt.gca().spines['left'].set_linewidth(3)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title("Differential capacitance curve", fontsize=22)
        plt.tick_params(direction='in', length=6, width=3, grid_alpha=0.3, bottom = True, top = True, left = True, right = True, labelsize = 16)
        plt.xlabel('Potential vs. $\mathregular{Li^{+}}$/Li, V', fontsize=20)
        plt.ylabel('dQ/dE', fontsize=20)
        if len(diffQ_new_dc) < 20:
            plt.legend(loc = 'upper left', facecolor='white', edgecolor = 'white', fontsize=10)
        else:
            patch1 = mpatches.Patch(color=(cmap(float(0) / len(potential))), label=str(round(len(potential)/len(potential)))  + str(' cycle'))
            patch2 = mpatches.Patch(color=(cmap(len(potential) / 4 / len(potential))), label='...')
            patch3 = mpatches.Patch(color=(cmap(float(len(potential)) / len(potential))), label=str(round(len(potential)/2)) + str(' cycle'))
            plt.legend(handles=[patch1, patch2, patch3], loc = 'upper left', bbox_to_anchor=(0.05,0.99))
		
    def for_GS_curves(potential, specific_capacity, density_current, capacity_max_anode, capacity_max_cathode, potential_avg_anode, potential_avg_cathode, potential_max_anode, potential_min_cathode, potential_difference):
        density_current_new = [n for n in density_current if n > 0]
        ax1 = plt.figure(figsize=(9, 6))
        cmap = plt.get_cmap('inferno')
        N = 4
        if len(potential) <= 6:
            for i in range(0, len(potential), 2):
                color = cmap(float(i) / len(potential))
                plt.plot(specific_capacity[i] + [float("nan")] + specific_capacity[i+1], potential[i] + [float("nan")] + potential[i+1], linewidth=2, zorder = 1000-i*4, color=color)       
                pylab.text(capacity_max_anode[0] * 0.5 -(capacity_max_anode[0] * 0.3)*(i/len(density_current)), min(potential_min_cathode) * 1.01, str(str(abs(density_current[i])) + str('C')), fontsize=12, color=color)
            plt.arrow(capacity_max_anode[0] * 0.6, min(potential_min_cathode), -(capacity_max_anode[0] * 0.35), 0, length_includes_head=True,
                      head_width=0.06, head_length=4, color='m')
        else:
            if density_current[0] == abs(density_current[-1]):
                for i in range(0, len(potential), 2):
                    color = cmap(float(i) / len(potential))
                    plt.plot(specific_capacity[i] + [float("nan")] + specific_capacity[i+1], potential[i] + [float("nan")] + potential[i+1], linewidth=2, zorder = 1000-i*4, color=color, label = str(int((i+2)/2)) + ' cycle')
                if len(density_current) < 21:
                    plt.legend(loc = 'lower left', bbox_to_anchor=(0.05,0.01), facecolor='white', edgecolor = 'white', fontsize=10)
                else:
                    patch1 = mpatches.Patch(color=(cmap(float(0) / len(potential))), label=str(round(len(potential)/len(potential)))  + str(' cycle'))
                    patch2 = mpatches.Patch(color=(cmap(len(potential) / 4 / len(potential))), label='...')
                    patch3 = mpatches.Patch(color=(cmap(float(len(potential)) / len(potential))), label=str(round(len(potential)/2)) + str(' cycle'))
                    plt.legend(handles=[patch1, patch2, patch3], loc = 'lower left', bbox_to_anchor=(0.05,0.01))
            elif density_current[0] < density_current[4]:
                for i in range(0, len(potential), 2):
                    color = cmap(float(i) / len(potential))
                    plt.plot(specific_capacity[i] + [float("nan")] + specific_capacity[i+1], potential[i] + [float("nan")] + potential[i+1], linewidth=2, zorder = 1000-i*4, color=color, label = str(int((i+2)/2)) + ' cycle')
                cmap = plt.get_cmap('viridis')
                for i in range(0, len(density_current), 4):
                    color = cmap(float(i) / len(density_current))
                    pylab.text(specific_capacity[i+1][-1]*0.97, potential[1][-1]*0.93, str(str(abs(density_current[i+1])) + str('C')), fontsize=12, color=color, zorder=1500)
            elif density_current[0] == density_current[4]:
                cmap = plt.get_cmap('viridis')
                for i in range(0, len(potential), 2):
                    color = cmap(float(i) / len(potential))
                    plt.plot(specific_capacity[i] + [float("nan")] + specific_capacity[i+1], potential[i] + [float("nan")] + potential[i+1], linewidth=2, zorder = 1000-i*4, color=color, label = str(abs(density_current[i+1])) + 'C')
                pylab.text(specific_capacity[i][-1]*0.5, potential[i][-1]*0.95, str("Charge ") + str(str(abs(density_current[0])) + str('C')), fontsize=12, color='r', zorder=1500)        	
                #plt.legend(loc = 'lower left', bbox_to_anchor=(0.05,0.01), facecolor='white', edgecolor = 'white', fontsize=10, title="Discharge")
        plt.plot([0, capacity_max_anode[0]], [potential_avg_anode[0], potential_avg_anode[0]], color='black', linewidth=2, zorder=50,
                 linestyle='--')
        plt.plot([0, capacity_max_anode[0]], [potential_avg_cathode[0], potential_avg_cathode[0]], color='black', linewidth=2, zorder=50,
                 linestyle='--')
        plt.arrow(capacity_max_anode[0]*0.91, potential_avg_anode[0] - potential_difference[0] / 2, 0, potential_difference[0] * 0.35, length_includes_head=True, head_width=1.0,
                  head_length=0.015, color='black')
        plt.arrow(capacity_max_anode[0]*0.91, potential_avg_anode[0] - potential_difference[0] / 2, 0, -potential_difference[0] * 0.35, length_includes_head=True, head_width=1.0,
                  head_length=0.015, color='black')
        pylab.text(capacity_max_anode[0]*0.92, potential_avg_anode[0] - (potential_difference[0] / 2)*1.3, str(round((potential_difference[0])*1000, 0)) + " mV", family="Helvetica", fontsize=10, color='b')
        arrow_cdc(capacity_max_anode, potential_min_cathode)
        plt.gca().spines['top'].set_linewidth(3)
        plt.gca().spines['bottom'].set_linewidth(3)
        plt.gca().spines['right'].set_linewidth(3)
        plt.gca().spines['left'].set_linewidth(3)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title("Charge-discharge curve", fontsize=22)
        plt.ylim(min(potential_min_cathode)*0.85, max(potential_max_anode)*1.02)
        plt.tick_params(direction='in', length=6, width=3, grid_alpha=0.3, bottom = True, top = True, left = True, right = True, labelsize = 16)
        plt.ylabel('Potential vs. $\mathregular{Li^{+}}$/Li, V', fontsize=20)
        plt.xlabel('Specific capacity, mA·h·$\mathregular{g^{-1}}$', fontsize=20)
       
    def for_CV_curves(current, time, potential, sweep_speed, Mass_active, peak_max, height_max_current, peak_min, height_min_current):
        ax1 = plt.figure(figsize=(10, 7)) 
        current_df = pd.DataFrame(current).T
        potential_df = pd.DataFrame(potential).T
        time_df = pd.DataFrame(time).T
        peak_current = [item for sublist in list(height_min_current + height_max_current) for item in sublist]
        peak_potential = [item for sublist in list(peak_min + peak_max) for item in sublist]
        sc = plt.scatter(peak_potential, peak_current, c = peak_current, 
            linewidth = 1, zorder=10000, edgecolors = 'k', s = 60,
            vmin=min(peak_current), vmax=max(peak_current), 
            cmap='brg')  
        plt.colorbar(sc)
        pylab.text(max(potential[0])*1.01, max(peak_current)*1.13, str("Current, µA"), family="Helvetica", fontsize=16)
        style = "Simple, tail_width=0.5, head_width=8, head_length=8"
        kw = dict(arrowstyle=style, color="m")
        a3 = patches.FancyArrowPatch((min(potential[0]) + max(potential[0])*0.15, max(peak_current)*0.1), (min(potential[0]) + max(potential[0])*0.25, max(peak_current)*0.1),

                                     connectionstyle="arc3,rad=0", **kw)
        plt.gca().add_patch(a3)
        pylab.text(min(potential[0]) + max(potential[0])*0.152,  max(peak_current)*0.13, str("extraction"), family="Helvetica", fontsize=10)
        a3 = patches.FancyArrowPatch((max(potential[0]) + max(potential[0])*0.01, min(peak_current)*0.8), (max(potential[0]) - max(potential[0])*0.12, min(peak_current)*0.8),
                                     connectionstyle="arc3,rad=0", **kw)
        plt.gca().add_patch(a3)
        pylab.text(max(potential[0]) - max(potential[0])*0.095, min(peak_current)*0.77, str("intercalation"), family="Helvetica", fontsize=10)	
        #plt.xlim(min(potential[0])*0.95, max(potential[0])*1)
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tick_params(direction='in', length=6, width=2, grid_alpha=0.3, bottom = True, top = True, left = True, right = True, labelsize = 16)
        plt.title("Current-voltage curve", fontsize=22)
        plt.xlabel('Potential vs. Li$^+$/Li, V', family="Helvetica", fontsize=20)
        plt.ylabel('Current, µA', family="Helvetica", fontsize=20)
        cmap = ['b', 'y', 'g', 'm', 'm']
        for i in range(len(current)):
            color = cmap[i]
            plt.plot(potential_df[i], current_df[i]*1000000, linewidth = 2, zorder=30, label = str(float(sweep_speed[i])*1000) + str(" mV/s"), color=color)
            Capacity1_an = simps([z for z in current_df[i] if z > 0], dx=time[i][120]-time[i][119]) *1000 / (3600 *  Mass_active )
            Capacity1_ca = simps([z for z in current_df[i] if z < 0], dx=time[i][120]-time[i][119]) *1000 / (3600 *  Mass_active )
            plt.fill_between(potential_df[i], current_df[i]*1000000, where = (current_df[i]*1000000 > 0), alpha=0.3, zorder=25-3*i, color=color, label = str(round(Capacity1_an, 1)) + str(" mA·h/g"))
            plt.fill_between(potential_df[i], current_df[i]*1000000, where = (current_df[i]*1000000 < 0), alpha=0.3, zorder=25-3*i, color=color, label = str(round(Capacity1_ca, 1)) + str(" mA·h/g"))
            plt.legend(loc = 'upper left', facecolor='white', edgecolor = 'white', fontsize=10)

    def data_CV(current, time, potential, sweep_speed, Mass_active):
        #Создаем список для положения пиков вдоль оси x
        peak_max, peak_min = [], []
        #Создаем список для положения пиков вдоль оси y
        height_min_current, height_max_current = [], []
        #Находим пики для заданной скорости поляризации
        for m in range(len(sweep_speed)):
            #Определяем пики
            current_mcA = np.array(list(savgol_filter(current[m], 21, 10)))*1000000
            current_mcA_minus = current_mcA * -1
            peaks_max = find_peaks(np.array(current_mcA), height = max(current_mcA)*0.10)
            peaks_min = find_peaks(np.array(current_mcA_minus), height = max(current_mcA_minus)*0.35)
            height_max = peaks_max[1]['peak_heights']
            height_min = peaks_min[1]['peak_heights']
            height_min = [n*-1 for n in height_min]
            peak_pos_max = np.array(potential[m])[peaks_max[0]]
            peak_pos_min = np.array(potential[m])[peaks_min[0]]
            peak_max.append(peak_pos_max)
            peak_min.append(peak_pos_min)
            height_min_current.append(height_min)
            height_max_current.append(height_max)
        return peak_max, height_max_current, peak_min, height_min_current
        
    def data_model_Rendels(peak_max, height_max_current, peak_min, height_min_current):    
        #Формируем данные для применения модели Рэнделса-Шевчика            
        #Сортируем катодные пики
        spisok_peak_min, spisok_peak_min_current = [], []
        for n in range(0, len(peak_min)):
            x = zip(peak_min[n],height_min_current[n])
            xs = sorted(x, key=lambda tup: tup[0], reverse=True)
            a1 = [x[0] for x in xs]
            b1 = [x[1] for x in xs]
            spisok_peak_min.append(a1)
            spisok_peak_min_current.append(b1)
        #Сортируем анодные пики
        spisok_peak_max, spisok_peak_max_current = [], []
        for n in range(0, len(peak_max)):
            x = zip(peak_max[n],height_max_current[n])
            xs = sorted(x, key=lambda tup: tup[0], reverse=True)
            a1 = [x[0] for x in xs]
            b1 = [x[1] for x in xs]
            spisok_peak_max.append(a1)
            spisok_peak_max_current.append(b1)
        #Сопоставляем катодные пики для разных скоростей поляризации
        add_peak_min, add_peak_min_current = [], []
        for i in range(0, len(spisok_peak_min[0])):
            add_peak_min_temp = []
            add_peak_min_temp_current = []
            for n in range(0, len(spisok_peak_min)):
                try:
                    add_peak_min_temp.append(spisok_peak_min[n][i])
                    add_peak_min_temp_current.append(spisok_peak_min_current[n][i])
                except:
                    continue
            add_peak_min.append(add_peak_min_temp)
            add_peak_min_current.append(add_peak_min_temp_current)
        #Сопоставляем анодные пики для разных скоростей поляризации
        add_peak_max, add_peak_max_current = [], []
        for i in range(0, len(spisok_peak_max[0])):
            add_peak_max_temp = []
            add_peak_max_temp_current = []
            for n in range(0, len(spisok_peak_max)):
                try:
                    add_peak_max_temp.append(spisok_peak_max[n][i])
                    add_peak_max_temp_current.append(spisok_peak_max_current[n][i])
                except:
                    continue
            add_peak_max.append(add_peak_max_temp)
            add_peak_max_current.append(add_peak_max_temp_current)  
        return add_peak_max_current, add_peak_min_current
    
    def curve_for_CV(current, time, potential, sweep_speed, Mass_active, peak_max, height_max_current, peak_min, height_min_current):
        current_df = pd.DataFrame(current).T
        potential_df = pd.DataFrame(potential).T
        time_df = pd.DataFrame(time).T
		#указываем размер графика
        fig = plt.figure(figsize=(20, 12))
        plt.suptitle('Current-voltage curve', fontsize = 25)
        cmap = ['b', 'y', 'g', 'm', 'm']
        for m in range(len(sweep_speed)):
            #Положение графика
            ax1 = plt.subplot(2,2, m+1)
            #Переменные
            peak_pos_min = peak_min[m]
            peak_pos_max = peak_max[m]
            height_min = height_min_current[m]
            height_max = height_max_current[m]
            #Графиков для пиков со шкалой
            sc = plt.scatter(list(peak_pos_min)+list(peak_pos_max), list(height_min)+list(height_max), c = list(height_min)+list(height_max), 
                linewidth = 1, zorder=1000, edgecolors = 'k', s = 60,
                vmin=min(list(height_min)+list(height_max)), vmax=max(list(height_min)+list(height_max)), 
                cmap='brg')
            plt.colorbar(sc)
            pylab.text(max(potential[0])*1.01, max(height_max)*1.45, str("Current, µA"), family="Helvetica", fontsize=16)
            #Указываем анатации к пикам
            for l in range(0, len(peak_pos_max)):
                #plt.annotate((round(peak_pos_max[l], 2)), (peak_pos_max[l], height_max[l]), zorder=10000)
                if int(l+1) % 2 == 0:
                    plt.annotate(round(peak_pos_max[l], 2), xy=(peak_pos_max[l], height_max[l]), xytext=(peak_pos_max[l]*1.002, height_max[l]+max(height_max)*0.12), arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=-90,angleB=180,rad=5"), zorder=10000);
                else:
                    if int(l+1) % 3 == 0:
                        plt.annotate(round(peak_pos_max[l], 2), xy=(peak_pos_max[l], height_max[l]), xytext=(peak_pos_max[l]*1.002, height_max[l]+max(height_max)*0.12), arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=-90,angleB=180,rad=5"), zorder=10000);
                    else:
                        plt.annotate(round(peak_pos_max[l], 2), xy=(peak_pos_max[l], height_max[l]), xytext=(peak_pos_max[l]*0.9562, height_max[l]+max(height_max)*0.12), arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=-90,angleB=180,rad=5"), zorder=10000);
            for w in range(0, len(peak_pos_min)):
                #plt.annotate((round(peak_pos_max[l], 2)), (peak_pos_max[l], height_max[l]), zorder=10000)
                if int(w+1) % 2 == 0:
                    plt.annotate(round(peak_pos_min[w], 2), xy=(peak_pos_min[w], height_min[w]), xytext=(peak_pos_min[w]*1.002, height_min[w]+min(height_min)*0.35), arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=-90,angleB=180,rad=5"), zorder=10000);
                else:
                    plt.annotate(round(peak_pos_min[w], 2), xy=(peak_pos_min[w], height_min[w]), xytext=(peak_pos_min[w]*0.9562, height_min[w]+min(height_min)*0.35), arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=-90,angleB=180,rad=5"), zorder=10000);
            #Рисуем стрелочки для направления процессов           
            style = "Simple, tail_width=0.5, head_width=8, head_length=8"
            kw = dict(arrowstyle=style, color="m")
            a3 = patches.FancyArrowPatch((min(potential[0]) + max(potential[0])*0.07, max(height_max)*0.1), (min(potential[0]) + max(potential[0])*0.17, max(height_max)*0.1),
                                     connectionstyle="arc3,rad=0", **kw)
            plt.gca().add_patch(a3)
            pylab.text(min(potential[0]) + max(potential[0])*0.072,  max(height_max)*0.13, str("extraction"), family="Helvetica", fontsize=10)
            a3 = patches.FancyArrowPatch((max(potential[0]) + max(potential[0])*0.01, min(height_min)*0.8), (max(potential[0]) - max(potential[0])*0.12, min(height_min)*0.8),
                                     connectionstyle="arc3,rad=0", **kw)
            plt.gca().add_patch(a3)
            pylab.text(max(potential[0]) - max(potential[0])*0.095, min(height_min)*0.77, str("intercalation"), family="Helvetica", fontsize=10)			
			
            #Оформляем графики
            plt.ylim(min(height_min)*1.5, max(height_max)*1.4)
            plt.gca().spines['top'].set_linewidth(2)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['right'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tick_params(direction='in', length=6, width=2, grid_alpha=0.3, bottom = True, top = True, left = True, right = True, labelsize = 16)
            if m == 0:
                plt.ylabel('Current, µA', family="Helvetica", fontsize=20)
            elif m % 2 == 0:
                plt.ylabel('Current, µA', family="Helvetica", fontsize=20)
                plt.xlabel('Potential vs. Li$^+$/Li, V', family="Helvetica", fontsize=20)
            elif m ==  len(sweep_speed)-1:
                plt.xlabel('Potential vs. Li$^+$/Li, V', family="Helvetica", fontsize=20)
            #Рисуем кривую ЦВА, вычисляем значения ёмкостей и заливаем площадь под кривой цветом
            color = cmap[m]
            plt.plot(potential_df[m], current_df[m]*1000000, linewidth = 2, zorder=30, label = str(float(sweep_speed[m])*1000) + str(" mV/s"), color=color)
            Capacity1_an = simps([z for z in current_df[m] if z > 0], dx=time[m][120]-time[m][119]) *1000 / (3600 *  Mass_active )
            Capacity1_ca = simps([z for z in current_df[m] if z < 0], dx=time[m][120]-time[m][119]) *1000 / (3600 *  Mass_active )
            plt.fill_between(potential_df[m], current_df[m]*1000000, where = (current_df[m]*1000000 > 0), alpha=0.3, zorder=25-3*m, color=color, label = str(round(Capacity1_an, 1)) + str(" mA·h/g"))
            plt.fill_between(potential_df[m], current_df[m]*1000000, where = (current_df[m]*1000000 < 0), alpha=0.3, zorder=25-3*m, color=color, label = str(round(Capacity1_ca, 1)) + str(" mA·h/g"))
            #Легенда
            plt.legend(loc = 'upper left', facecolor='white', edgecolor = 'white', fontsize=10)
            #Отступы для графика
            plt.subplots_adjust(wspace=0.15, hspace=0.3)
        #plt.tight_layout()
		
    def curve_for_Rendels(add_peak_max_current, add_peak_min_current, sweep_speed):
        sweep_speed = [float(n)**0.5 for n in sweep_speed]
        fig = plt.figure(figsize=(9, 6))
        for i in range(0, len(add_peak_max_current)):
            if len(add_peak_max_current[i]) < 3:
                continue
            else:
                coef = np.polyfit(sweep_speed[0:len(add_peak_max_current[i])], add_peak_max_current[i], 1)
                poly1d_fn = np.poly1d(coef)
                correlation_matrix = np.corrcoef(sweep_speed[0:len(add_peak_max_current[i])], add_peak_max_current[i])
                correlation_xy = correlation_matrix[0,1]
                r_squared = correlation_xy**2
                plt.plot(sweep_speed[0:len(add_peak_max_current[i])], poly1d_fn(sweep_speed[0:len(add_peak_max_current[i])]), '--r', linewidth = 2, zorder=1)
                plt.plot(sweep_speed[0:len(add_peak_max_current[i])], add_peak_max_current[i], 'o', markersize=10, label = 'Anodic peak ' + str(i+1) + str(" (R$^{2}$ = ") + str(round(r_squared, 2)) + str(")"), zorder=12, linewidth = 1, markeredgecolor = 'k')
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tick_params(direction='in', length=6, width=2, grid_alpha=0.3, bottom = True, top = True, left = True, right = True, labelsize = 16)
        plt.title('Randles–Sevcik model', fontsize=15)
        plt.ylabel('i, µA·cm$^{-2}$', family="Helvetica", fontsize=16)
        plt.xlabel('ν$^{1/2}$, mV$^{1/2}$·s$^{-1/2}$', family="Helvetica", fontsize=16)
        plt.legend(loc = 'upper left',  bbox_to_anchor=(0.01,0.98), facecolor='white', edgecolor = 'white', fontsize=12, markerscale = 1.2)
        plt.show()
        fig = plt.figure(figsize=(9, 6))
        for i in range(0, len(add_peak_min_current)):
            if len(add_peak_min_current[i]) < 3:
                continue
            else:
                coef = np.polyfit(sweep_speed[0:len(add_peak_min_current[i])], add_peak_min_current[i], 1)
                poly1d_fn = np.poly1d(coef) 
                correlation_matrix = np.corrcoef(sweep_speed[0:len(add_peak_min_current[i])], add_peak_min_current[i])
                correlation_xy = correlation_matrix[0,1]
                r_squared = correlation_xy**2
                plt.plot(sweep_speed[0:len(add_peak_min_current[i])], poly1d_fn(sweep_speed[0:len(add_peak_min_current[i])]), '--r', linewidth = 2, zorder=1)
                plt.plot(sweep_speed[0:len(add_peak_min_current[i])], add_peak_min_current[i], 'o', markersize=10, label = 'Cathodic peak ' + str(i+1) + str(" (R$^{2}$ = ") + str(round(r_squared, 2)) + str(")"), zorder=12, linewidth = 1, markeredgecolor = 'k')
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tick_params(direction='in', length=6, width=2, grid_alpha=0.3, bottom = True, top = True, left = True, right = True, labelsize = 16)
        plt.title('Randles–Sevcik model', fontsize=15)
        plt.ylabel('i, µA·cm$^{-2}$', family="Helvetica", fontsize=16)
        plt.xlabel('ν$^{1/2}$, mV$^{1/2}$·s$^{-1/2}$', family="Helvetica", fontsize=16)
        plt.legend(loc = 'upper right', facecolor='white', edgecolor = 'white', fontsize=12, markerscale = 1.2)
        plt.show()             

    def filter_data_cv(current, time, potential, sweep_speed, Mass_active):
        import matplotlib.pyplot as plt
        from numpy import exp, loadtxt, pi, sqrt
        from lmfit import Model
        import scipy as scipy
        from scipy import optimize
        for i in range(0, len(current)):
            x = potential[i]
            y = list(savgol_filter(current[i], 11, 3))
            y_minus = np.array(y)*-1
            left_endpt=3.9
            right_endpt=4.7         
            #first_index = indices[6]
            left_gauss_bound = 300
            right_gauss_bound = 370
            x_values_1 = list(np.asarray(x[left_gauss_bound:right_gauss_bound]))
            y_values_1 = list(np.asarray(y[left_gauss_bound:right_gauss_bound]))
            amp1=0.001
            cen1=3.61
            sigma1=1
            amp2=0.001
            cen2=3.71
            sigma2=1
            popt_2gauss, pcov_2gauss = scipy.optimize.curve_fit(_2gaussian, x_values_1, y_values_1, p0=[amp1, cen1, sigma1, amp2, cen2, sigma2])
            perr_2gauss = np.sqrt(np.diag(pcov_2gauss))
            pars_1 = popt_2gauss[0:3]
            pars_2 = popt_2gauss[3:6]
            gauss_peak_1 = _1gaussian(x_values_1, *pars_1)
            gauss_peak_2 = _1gaussian(x_values_1, *pars_2)      
            plt.plot(x_values_1, _2gaussian(x_values_1, *popt_2gauss), 'k--')#,\        
            peaks_max = find_peaks(np.array(y), height = max(y)*0.10)
            height_max = peaks_max[1]['peak_heights']
            peak_pos_max = np.array(potential[i])[peaks_max[0]]            
            peaks_min = find_peaks(np.array(y_minus), height = max(y_minus)*0.10)
            height_min = peaks_min[1]['peak_heights']*-1
            peak_pos_min = np.array(potential[i])[peaks_min[0]]          
            sc = plt.scatter(list(peak_pos_min) + list(peak_pos_max), list(height_min) + list(height_max), c = list(height_min) + list(height_max), 
                linewidth = 1, zorder=1000, edgecolors = 'k', s = 60,
                vmin=min(list(height_min) + list(height_max)), vmax=max(list(height_min) + list(height_max)), 
                cmap='brg')
            plt.colorbar(sc)
            plt.plot(x, y, linewidth = 2, zorder=30)

        plt.show()

    def data_import_impedance(filename):
    #Функция для импорта файлов с элинса форматом .txt
        with open(filename, encoding="Windows-1251", errors='ignore') as f:
            ranges_cycles = []
            data_lines = f.readlines()
            cycles = re.compile("f/Hz")
            for count, line in enumerate(data_lines):
                if cycles.search(line):
                    cycle = line.split()
                    number_lines = count + 2
                    ranges_cycles.append(number_lines)
            frequency, z1, z2, time, potential, current = [], [], [], [], [], []
            for t in range(len(ranges_cycles)):
                frequency_temp, z1_temp, z2_temp, time_temp, potential_temp, current_temp = [], [], [], [], [], []
                if t == len(ranges_cycles)-1:
                    [frequency_temp.append(float(n.split()[0])) for n in data_lines[ranges_cycles[t]:]]
                    [z1_temp.append(float(n.split()[1])) for n in data_lines[ranges_cycles[t]:]]
                    [z2_temp.append(float(n.split()[2])) for n in data_lines[ranges_cycles[t]:]]
                    [time_temp.append(float(n.split()[3])) for n in data_lines[ranges_cycles[t]:]]
                    [potential_temp.append(float(n.split()[4])) for n in data_lines[ranges_cycles[t]:]]
                    [current_temp.append(float(n.split()[5])) for n in data_lines[ranges_cycles[t]:]]
                else:
                    [frequency_temp.append(float(n.split()[0])) for n in data_lines[ranges_cycles[t]:ranges_cycles[t+1]-14]]
                    [z1_temp.append(float(n.split()[1])) for n in data_lines[ranges_cycles[t]:ranges_cycles[t+1]-14]]
                    [z2_temp.append(float(n.split()[2])) for n in data_lines[ranges_cycles[t]:ranges_cycles[t+1]-14]]
                    [time_temp.append(float(n.split()[3])) for n in data_lines[ranges_cycles[t]:ranges_cycles[t+1]-14]]
                    [potential_temp.append(float(n.split()[4])) for n in data_lines[ranges_cycles[t]:ranges_cycles[t+1]-14]]
                    [current_temp.append(float(n.split()[5])) for n in data_lines[ranges_cycles[t]:ranges_cycles[t+1]-14]]
                frequency.append(frequency_temp)
                z1.append(z1_temp)
                z2.append(z2_temp)
                time.append(time_temp)
                potential.append(potential_temp)
                current.append(current_temp)
        return frequency, z1, z2, time, potential, current

    def curve_impedance(frequency, z1, z2, time, potential, current):
        cmap = plt.get_cmap('inferno')
        for i in range(0, len(frequency)):
            color = cmap(float(i) / len(frequency))
            ax1 = plt.figure(figsize=(8, 6))
            plt.gca().spines['top'].set_linewidth(2)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['right'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tick_params(direction='in', length=6, width=2, grid_alpha=0.3, bottom = True, top = True, left = True, right = True, labelsize = 16)
            #plt.title("impedance curve", fontsize=22)
            plt.xlabel('${Z^{\'}}$, Ω', family="Helvetica", fontsize=20)
            plt.ylabel('${Z^{\'\'}}$, Ω', family="Helvetica", fontsize=20)
            from matplotlib import colors
            sc = plt.scatter(z1[i], z2[i], c = frequency[i], linewidth = 1, edgecolors = 'k', s = 30, vmin=min(frequency[i]), vmax=max(frequency[i]), cmap='brg', norm=colors.LogNorm())
            pylab.text(max(z1[i])*1.01, max(z2[i])*1.09, str("Frequency, Hz"), family="Helvetica", fontsize=16)
            plt.colorbar(sc)
            #plt.xlim(7, 35)
            #plt.ylim(0, 6)

    
    filename = askopenfilename()
    Mass_active = float(ent_Mass_active.get().replace(',', '.'))/1000
    Theory_сapacity = float(ent_Theory_сapacity.get().replace(',', '.'))
    filename = filename
    filename1, file_extension = os.path.splitext(filename)
    result = pyfiglet.figlet_format("Elins PP", font = "slant"  )
    print(result)
    print("********************************** \n" +
    "Post-processing script for Elins *\n" + 
    "Version: 2.3 (26.03.2022)        *\n" + 
    "Written by Kirill Rybakov        *\n" + 
    "Contacts: rybakov-ks@ya.ru       *\n" +            
    "**********************************" +
    "\n")
    if file_extension == '.txt':
        sweep_speed, current, time, potential = data_import_txt(filename)
    elif file_extension == '.edf':
        sweep_speed, current, time, potential = data_import_edf(filename) 
    elif file_extension == '.P00':
        frequency, z1, z2, time, potential, current = data_import_impedance(filename)
        sweep_speed = [777]
    else:
        print('Неверный формат файла для загрузки!')
        sys.exit(1)
    if float(sweep_speed[0]) == 0:
        print('UPH')
        density_c = density_current(Mass_active, Theory_сapacity, current)
        capacity_max_anode, capacity_max_cathode, specific_capacity, cycle_capacity = specific_capacity(Mass_active, current, time)
        potential_avg_anode, potential_avg_cathode, potential_max_anode, potential_min_cathode, potential_difference = data_potential(potential)
        ax1 = for_GS_curves(potential, specific_capacity, density_c, capacity_max_anode, capacity_max_cathode, potential_avg_anode, potential_avg_cathode, potential_max_anode,potential_min_cathode, potential_difference)
        file_name = extract_text('Сохранить график ГЗР', 'Charge-discharge curve')
        plt.savefig(file_name + str('.jpg'), dpi = 800)
        plt.show()
        for_diff_curves(potential, specific_capacity)
        file_name = extract_text('Сохранить график дифференциальной ёмкости', 'Differential capacitance')
        plt.savefig(file_name + str('.jpg'), dpi = 800)
        plt.show()
        capacity_cycle(capacity_max_anode, capacity_max_cathode, cycle_capacity, density_c)
        file_name = extract_text('Сохранить график циклируемости', 'Cycle capacity')
        plt.savefig(file_name + str('.jpg'), dpi = 800)
        plt.show()
    elif sweep_speed[0] == 777:
        curve_impedance(frequency, z1, z2, time, potential, current)
        file_name = extract_text('Сохранить график EIS', 'EIS')
        plt.savefig(file_name + str('.jpg'), dpi = 800)
        plt.show()
    else:
        print('CV')
        peak_max, height_max_current, peak_min, height_min_current = data_CV(current[1:5], time[1:5], potential[1:5], sweep_speed[1:5], Mass_active)
        curve_for_CV(current[1:5], time[1:5], potential[1:5], sweep_speed[1:5], Mass_active, peak_max, height_max_current, peak_min, height_min_current)
        file_name = extract_text('Сохранить график ЦВА', 'Cyclic voltammetry')
        plt.savefig(file_name + str('.jpg'), dpi = 800)
        plt.show()
        #add_peak_max_current, add_peak_min_current = data_model_Rendels(peak_max, height_max_current, peak_min, height_min_current)
        #curve_for_Rendels(add_peak_max_current, add_peak_min_current, sweep_speed[1:5])
        for_CV_curves(current[1:5], time[1:5], potential[1:5], sweep_speed[1:5], Mass_active, peak_max, height_max_current, peak_min, height_min_current)
        file_name = extract_text('Сохранить график ЦВА', 'Cyclic voltammetry')
        plt.savefig(file_name + str('.jpg'), dpi = 800)
        plt.show()
        #filter_data_cv(current[1:2], time[1:2], potential[1:2], sweep_speed[1:2], Mass_active)
    print("\n" + 
        "**********************************\n" +
        "Script execution completed       *\n" +          
        "**********************************")
    
# Создается новое окно с заголовком 
window = tk.Tk()
window.title("Орбработка результатов измерения на Elins")
 
# Создается новая рамка `frm_form` для ярлыков с текстом и
frm_form = tk.Frame(relief=tk.SUNKEN, borderwidth=3)
# Помещает рамку в окно приложения.
frm_form.pack()
 
# Создает ярлык и текстовок поле для ввода удельной ёмкости.
lbl_Theory_сapacity = tk.Label(master=frm_form, text="Теоретическая удельная ёмкость материала, мА·ч/г:")
ent_Theory_сapacity = tk.Entry(master=frm_form, width=50)
# Использует менеджер геометрии grid для размещения ярлыка и
# однострочного поля для ввода текста в первый и второй столбец
# первой строки сетки.
lbl_Theory_сapacity.grid(row=0, column=0, sticky="e")
ent_Theory_сapacity.grid(row=0, column=1)
 
# Создает ярлык и текстовок поле для массы активного вещества.
lbl_Mass_active = tk.Label(master=frm_form, text="Масса активного компонента, мг:")
ent_Mass_active = tk.Entry(master=frm_form, width=50)
# Размещает виджеты на вторую строку сетки
lbl_Mass_active.grid(row=1, column=0, sticky="e")
ent_Mass_active.grid(row=1, column=1)


# отступами в 5 пикселей горизонтально и вертикально.
frm_buttons = tk.Frame()
frm_buttons.pack(fill=tk.X, ipadx=5, ipady=5)
 
# Создает кнопку "Отправить" и размещает ее
# справа от рамки `frm_buttons`.
btn_submit = tk.Button(master=frm_buttons, text="Выбрать файл", command=clicked)
btn_submit.pack(side=tk.RIGHT, padx=10, ipadx=10)

# Середина экрана.
window.update_idletasks()
s = window.geometry()
s = s.split('+')
s = s[0].split('x')
width_root = int(s[0])
height_root = int(s[1])
 
w = window.winfo_screenwidth()
h = window.winfo_screenheight()
w = w // 2
h = h // 2 
w = w - width_root // 2
h = h - height_root // 2
window.geometry('+{}+{}'.format(w, h))
window.resizable(width=False, height=False)

# Запуск приложения.
window.mainloop()