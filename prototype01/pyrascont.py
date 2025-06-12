import os
import shutil
import psutil
import numpy as np

from math import sin, cos, sqrt
pi = np.pi

#targ_dir = 'sim_copy0'
#basepath = os.path.abspath(os.path.dirname(__file__))
#P_targ = 500000
#T_targ = 323

# %% Function for copy simulation directories
def copysim(dir_orig, dir_prefix, N_copy):
    # dir_org: original directory
    # dir_prefix: prefix for copied directories
    # N_copy: number of copies
    dir_pre = dir_prefix
    #basepath = os.path.dirname(os.path.abspath(__file__))
    basepath = os.getcwd()
    dirlist_copied = []
    for ii in range(N_copy):
        dirnam_tmp = dir_pre + '{0:03d}'.format(ii)
        if os.path.isdir("./"+dirnam_tmp):
            shutil.rmtree(dirnam_tmp)
            print("The following folder is replaced:")
            print(dirnam_tmp)
        shutil.copytree("./"+dir_orig, "./"+dirnam_tmp)
        dirlist_copied.append(dirnam_tmp)
    return dirlist_copied
# Funciton for editing the simulation file within a directory


def editsim(targ_dir, P_targ, T_targ, Sorbent, UnitCells, PrintResult=False):
    basepath = os.getcwd()
    os.chdir(targ_dir)
    f = open('simulation.input','r')
    f_cont = f.readlines()
    f.close()
    n_extP = len('ExternalPressure')
    n_extT = len('ExternalTemperature') 
    n_sorb = len('FrameworkName')
    n_unit = len('UnitCells')
    for line, ii in zip(f_cont, range(len(f_cont))):
        if line[:n_extP] == 'ExternalPressure':
            line_new_P = 'ExternalPressure    ' + str(P_targ) + '\n'
            f_cont[ii] = line_new_P
            
        elif line[:n_extT] == 'ExternalTemperature':
            line_new_T = 'ExternalTemperature    ' + str(T_targ) + '\n'
            f_cont[ii] = line_new_T
        
        elif line[:n_sorb] == 'FrameworkName':
            line_new_F = 'FrameworkName ' + Sorbent + '\n'
            f_cont[ii] = line_new_F
        elif line[:n_unit] == 'UnitCells':
            aa, bb, cc = UnitCells
            line_new_U = 'UnitCells {0:d} {1:d} {2:d}'.format(aa, bb, cc,) + '\n'
            f_cont[ii] = line_new_U
    if PrintResult:
        print('[[' + targ_dir + ']]')
        print()
        for line in f_cont:
            print(line, end ='')

    f = open('simulation.input', 'w')
    for line in f_cont:
        f.write(line)
    f.close()
    os.chdir(basepath)

# %% Function for running simulations
def runsim(targ_dir):
    basepath = os.getcwd()
    os.chdir(targ_dir)
    os.system('nohup sh run simulation.input &')
    os.chdir(basepath)

# %% Find the simulation directories based on its prefix
def findsimdir(targ_prefix):
    dir_list = os.listdir()
    n_pre = len(targ_prefix)
    targ_dir_list = []
    for ff in dir_list:
        if ff[:n_pre] == targ_prefix:
            targ_dir_list.append(ff)
    targ_dir_list.sort()
    return targ_dir_list

# %% Remove simulation directories based on its prefix
def removesim(targ_prefix):
    n_prefix = len(targ_prefix)
    f_list = os.listdir()
    f_list_order = np.sort(f_list)
    for ff in f_list_order:
        if len(ff) < n_prefix:
            continue
        if os.path.isfile(ff):
            continue
        if ff[:n_prefix] == targ_prefix:
            shutil.rmtree(ff)
            print(ff,' is deleted!')

# %% CPU usage check
def cpucheck():
    #load1, load5, load15 = psutil.getloadavg() 
    #cpu_usage_percent = (load15/os.cpu_count()) * 100 
    cpu_use_list = []
    for ii in range(4):
        cpu_use_test = psutil.cpu_percent(0.5)
        cpu_use_list.append(cpu_use_test)
    arg_min = np.argmin(cpu_use_list)
    cpu_dum = cpu_use_list.pop(arg_min)
    cpu_perc_average = np.mean(cpu_dum)
    return cpu_perc_average

# %% Crop the simulation results 
def cropsim(targ_dir):
    dir_targ_nam = targ_dir

    basepath = os.getcwd()
    os.chdir(basepath)
    os.chdir(dir_targ_nam)

    os.chdir('Output/System_0')
    f_nam_list = os.listdir()
    #print(f_nam_list)

    prop_targ = '\tAverage loading absolute [mol/kg frame'
    n_prop_str = len(prop_targ)
    uptake_list = []
    for fn in f_nam_list:
        ff = open(fn)
        uptake_excess = -123.123
        #prop_targ = '\tAverage loading excess [mol/kg frame'
        ff_txt = ff.readlines()
        for ii in range(len(ff_txt)):
            targ_txt = 'Finishing simulation'
            len_txt = len(targ_txt)
            if ff_txt[ii][:len_txt] == targ_txt:
                ff_txt_fin = ff_txt[ii:]
                break
        for txx in ff_txt_fin[::-1]:
            if txx[:n_prop_str] == prop_targ:
                txt_spl = txx.split()
                #print(txt_spl)
                #print(txt_spl[5])
                uptake_excess = float(txt_spl[5])
                uptake_list.append(uptake_excess)
    os.chdir(basepath)
    return uptake_list

# %% Calculate the number of unit cells to satisfy the cutoff value
def cif2Ucell(cif, cutoff, Display = False):
    deg2rad=pi/180.
    
    f_tmp = open(cif + '.cif')
    f_cont = f_tmp.readlines()
    f_tmp.close()
    n_a = len('_cell_length_a')
    n_b = len('_cell_length_b')
    n_c = len('_cell_length_c')
   
    n_alp = len('_cell_angle_alpha')
    n_bet = len('_cell_angle_beta')
    n_gam = len('_cell_angle_gamma')
    
    count_compl = 0
    for ii in range(len(f_cont)):
        if len(f_cont[ii]) > n_a:
            if f_cont[ii][:n_a] == '_cell_length_a':
                txt_tmp = f_cont[ii].split()
                a = float(txt_tmp[1])
                count_compl = count_compl + 1
        if len(f_cont[ii]) > n_b:
            if f_cont[ii][:n_b] == '_cell_length_b':
                txt_tmp = f_cont[ii].split()
                b = float(txt_tmp[1])
                count_compl = count_compl + 1

        if len(f_cont[ii]) > n_c:
            if f_cont[ii][:n_c] == '_cell_length_c':
                txt_tmp = f_cont[ii].split()
                c = float(txt_tmp[1])
                count_compl = count_compl + 1
        
        if len(f_cont[ii]) > n_alp:
            if f_cont[ii][:n_alp] == '_cell_angle_alpha':
                txt_tmp = f_cont[ii].split()
                alpha = float(txt_tmp[1])*deg2rad
                count_compl = count_compl + 1

        if len(f_cont[ii]) > n_bet:
            if f_cont[ii][:n_bet] == '_cell_angle_beta':
                txt_tmp = f_cont[ii].split()
                beta = float(txt_tmp[1])*deg2rad
                count_compl = count_compl + 1
        
        if len(f_cont[ii]) > n_gam:
            if f_cont[ii][:n_gam] == '_cell_angle_gamma':
                txt_tmp = f_cont[ii].split()
                gamma = float(txt_tmp[1])*deg2rad
                count_compl = count_compl + 1

        if count_compl > 5.8:
                break
    if Display:
        print('a = ', a)
        print('b = ', b)
        print('c = ', c)
        print('alpha = ', alpha/deg2rad, 'dgr')
        print('beta = ', beta/deg2rad, 'dgr')
        print('gamma = ', gamma/deg2rad, 'dgr')
# compute cell vectors following https://en.wikipedia.org/wiki/Fractional_coordinates
    v = sqrt(1-cos(alpha)**2-cos(beta)**2- cos(gamma)**2+2*cos(alpha)*cos(beta)*cos(gamma))
    cell=np.zeros((3,3))
    cell[0,:] = [a, 0, 0]
    cell[1,:] = [b*cos(gamma), b*sin(gamma),0]
    cell[2,:] = [c*cos(beta), c*(cos(alpha)-cos(beta)*cos(gamma))/(sin(gamma)),
              c*v /sin(gamma)]
    cell=np.array(cell)

# diagonalize the cell matrix
    diag = np.diag(cell)
# and computing nx, ny and nz
    nx, ny, nz = tuple(int(i) for i in np.ceil(cutoff/diag*2.))

#return nx, ny, nz
    #return "{} {} {}".format(nx, ny, nz)
    return nx, ny, nz


def killall():
    cc = 0
    for proc in psutil.process_iter():
        #print(proc.name())
        #print(proc)
        if proc.name() == 'simulate':
            cc =cc + 1
            proc.kill()
            print(proc.name(), '#', cc,': ', proc.pid, ' is killed !') 

'''

# %% TEST the functions
if __name__ == '__main__':
    import numpy as np
    P_list = np.arange(0.5E5, 10E5+0.2, 0.5E5)
    T_targ =  323
    Adsorbent = 'Cu-BTC'
    n_cell = [1, 1, 1,]
    
    # Copy the sim. dir
    dir_cop = copysim('sim_orig','sim_CuBTC_no', len(P_list))

    # Edit the sim dirs
    for p, dirr in zip(P_list, dir_cop):
        editsim(dirr, p, T_targ, Adsorbent, [1, 1, 1],  True)
    
    # Run based on CPU usage
    import time
    for dirr, pp in zip(dir_cop, P_list):
    # Check the CPU usage 
        for ii in range(300000):
            cpu_usage = cpucheck()
            #cpu_usage = psutil.cpu_percent()
    # Run if cpu_usage is less than the target
            if cpu_usage < 85:
                runsim(dirr)
                print('Simulation for {0:.2f} bar: '.format(pp/1E5) + dirr)
                break
            time.sleep(5) 
            print('{0:.4f}% of CPU is occupied now'.format(cpu_usage))
    print(cpucheck())
'''

