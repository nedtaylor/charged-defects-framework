import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from qdef2d import logging
import re


def get_data_from_file(filename):
    data = {}
    
    with open(filename, 'r') as file:
        content = file.read()
    
    charge_match = re.search(r'charge\s*{\s*posZ\s*=\s*([\d\.]+);', content)
    isolated_match = re.search(r'isolated\s*{\s*fromZ\s*=\s*([\d\.]+);', content)
    
    if charge_match:
        data['charge_posZ'] = float(charge_match.group(1))
    if isolated_match:
        data['isolated_fromZ'] = float(isolated_match.group(1))
    
    return data



def calc(vref,vdef,encut,q,threshold_slope=1e-3,threshold_C=1e-3,max_iter=40,
         vfile='vline-eV.dat',noplots=False,allplots=False,logfile=None,
         slab_edge_lw=None,slab_edge_up=None):
    
    """
    Estimate alignment correction.
    
    vref (str): path to bulk LOCPOT file
    vdef (str): path to defect LOCPOT file
    encut (int): cutoff energy (eV)
    q (int): charge (conventional units)
    [optional] threshold_slope (float): threshold for determining if potential is flat
                                        Default=1e-3.
    [optional] threshold_C (float): threshold for determining if potential is aligned
                                    Default=1e-3.                                        
    [optional] max_iter (int): max. no. of shifts to try. Default=20.
    [optional] vfile (str): vline .dat file. Default='vline-eV.dat'
    [optional] noplots (bool): do not generate plots. Defaule=False.
    [optional] allplots (bool): save all plots. Default=False.
    [optional] logfile (str): logfile to save output to 
    
    """
    
    ## set up logging
    if logfile:
        myLogger = logging.setup_logging(os.path.join(os.getcwd(),logfile))
    else:
        myLogger = logging.setup_logging()
    
    
    ## basic command to run sxdefectalign2d
    command = ['~/sxdefectalign2d', '--vasp',
               '--ecut', str(encut/13.6057), ## convert eV to Ry
               '--vref', vref,
               '--vdef', vdef]
    
    
    ## initialize the range of shift values bracketing the optimal shift
    smin, smax = -np.inf, np.inf
    shift = 0.0
    shifting = 'right'
    done = False
    counter = -1

    ## convert slab_edge_lw and slab_edge_up to bohr from Angstroms
    edge_lw = slab_edge_lw
    edge_up = slab_edge_up
    edge_lw = slab_edge_lw/0.529177 - 9.0
    edge_up = slab_edge_up/0.529177 + 9.0


    time0 = time.time()
    while not done and counter < max_iter:
        counter += 1
        ## run sxdefectalign2d with --shift <shift>
        if logfile:
            command1 = command + ['--shift', str(shift), '--onlyProfile', '>> %s'%logfile]
        else:
            command1 = command + ['--shift', str(shift), '--onlyProfile']
        os.system(' '.join(command1))
        
        ## read in the potential profiles from vline-eV.dat
        ## z  V^{model}  \DeltaV^{DFT}  V^{sr}
        data = np.loadtxt(vfile)
        
        ## plot potential profiles
        if not noplots:
            plt.figure()
            plt.plot(data[:,0],data[:,2],'r',label=r'$V_{def}-V_{bulk}$')
            plt.plot(data[:,0],data[:,1],'g',label=r'$V_{model}$')
            plt.plot(data[:,0],data[:,-1],'b',label=r'$V_{def}-V_{bulk}-V_{model}$')
            plt.xlabel("distance along z axis (bohr)")
            plt.ylabel("potential (eV)")
            plt.xlim(data[0,0],data[-1,0])
            plt.legend() 
            if allplots:
                plt.savefig(os.getcwd()+'/alignment_%d.png'%counter)
            else:
                plt.savefig(os.getcwd()+'/alignment.png')
            plt.close()
        
        ## map slab_edge_lw and slab_edge_up to within the range of data[:,0], i.e. periodic mapping
        edge_lw = edge_lw - np.floor(edge_lw/(data[-1,0]-data[0,0]))*(data[-1,0]-data[0,0])
        edge_up = edge_up - np.floor(edge_up/(data[-1,0]-data[0,0]))*(data[-1,0]-data[0,0])

        # print("edge_lw = %.2f, edge_up = %.2f"%(edge_lw,edge_up))

        ## assumes that the slab is in the center of the cell vertically!
        ## select datapoints corresponding to 2 bohrs at the top and bottom of the supercell 
        ## (i.e. a total of 4 bohrs in the middle of vacuum)
        # z1 = np.min([i for i,z in enumerate(data[:,0]) if z > + 2.])
        # z2 = np.min([i for i,z in enumerate(data[:,0]) if z > (data[-1,0]-2.)])
        z1 = np.min([i for i,z in enumerate(data[:,0]) if z > + edge_up])
        z2 = np.min([i for i,z in enumerate(data[:,0]) if z > edge_lw])
        zmid = int((z1+z2)/2)
        # print("z1 = %d, z2 = %d, zmid = %d"%(z1,z2,zmid))

        ## fit straight lines through each subset of datapoints
        m1,C1 = np.polyfit(data[z1:zmid,0],data[z1:zmid,-1],1)
        m2,C2 = np.polyfit(data[zmid:z2,0],data[zmid:z2,-1],1)
        myLogger.debug("Slopes: %.8f %.8f; Intercepts: %.8f %.8f"%(m1,m2,C1,C2))
        print("Slopes: %.8f %.8f; Intercepts: %.8f %.8f"%(m1,m2,C1,C2))
        m,C = np.polyfit(data[z1:z2,0],data[z1:z2,-1],1)
        
        ## check the slopes and intercepts of the lines
        ## and shift the charge along z until the lines are flat
        if (abs(m1) < threshold_slope and abs(m2) < threshold_slope
            and abs(C1-C2) < threshold_C):
        # if abs(m) < threshold_slope :
        # if (abs(m1) < threshold_slope and abs(m2) < threshold_slope
            # and abs(C1-C2) < threshold_C):
            done = True
            break
        # elif m < 0:
        elif (abs(m1) < threshold_slope and abs(m2) < threshold_slope) and abs(m1 + m2) < threshold_slope:
            done = True
            break
        elif abs(m1 + m2) < threshold_slope:
            done = True
            break
        elif abs(m1 + m2) < threshold_slope * 10:
            shift += (m1 + m2)*np.sign(q) * 20
            myLogger.info("try shift = %.8f"%shift)

        elif m1*m2 < 0:
            myLogger.info("undetermined...make a tiny shift and try again")
            shift += (m1 + m2)*np.sign(q) * 200
            # if shifting == 'right':
            #     shift += 0.01
            # else: 
            #     shift -= 0.01
            myLogger.info("try shift = %.8f"%shift)
        # elif m*np.sign(q) > 0:
        elif (m1+m2)*np.sign(q) > 0:
            smin = shift
            if smax == np.inf:
                shift += 1.0 * abs(m1 + m2) / 2 * 200
            else:
                shift = (smin+smax)/2.0
            shifting = 'right'
            myLogger.debug("optimal shift is in [%.8f, %.8f]"%(smin,smax))
            myLogger.info("shift charge in +z direction; try shift = %.8f"%shift)
        # elif m*np.sign(q) < 0:
        elif (m1+m2)*np.sign(q) < 0:
            smax = shift
            if smin == -np.inf:
                shift -= 1.0 * abs(m1 + m2) / 2 * 200
            else:
                shift = (smin+smax)/2.0
            shifting = 'left'
            myLogger.debug("optimal shift is in [%.8f, %.8f]"%(smin,smax))
            myLogger.info("shift charge in -z direction; try shift = %.8f"%shift)
    
                       
    if done:
        C_ave = C #np.average(data[z1:z2,-1])
        print("Average potential in the middle of vacuum: %.8f"%C_ave)
        myLogger.info("DONE! shift = %.8f & alignment correction = %.8f"%(shift,C_ave))
        ## run sxdefectalign2d with --shift <shift> -C <C_ave> > correction
        command2 = command + ['--shift', str(shift),
                              '-C', str(C_ave),
                              '> correction']
        os.system(' '.join(command2))
    else:
        myLogger.info("Could not find optimal shift after %d tries :("%max_iter)
    
    # read system.sx and get isolated{fromZ,toZ} and charge{posZ}
    data = get_data_from_file('system.sx')
    if data['charge_posZ'] < data['isolated_fromZ'] + shift:
        myLogger.info("Charge outside the isolated region!")
        myLogger.info("Charge posZ = %.8f; isolated fromZ = %.8f"%(data['charge_posZ'],data['isolated_fromZ']))
        


    myLogger.debug("Total time taken (s): %.2f"%(time.time()-time0))
    
    
if __name__ == '__main__':
    

    ## this script can also be run directly from the command line
    parser = argparse.ArgumentParser(description='Estimate alignment correction.')
    parser.add_argument('vref',help='path to bulk LOCPOT file')
    parser.add_argument('vdef',help='path to defect LOCPOT file')
    parser.add_argument('encut',type=int,help='cutoff energy (eV)')
    parser.add_argument('q',type=int,help='charge (conventional units)')
    parser.add_argument('--threshold_slope',type=float,default=1e-3,
                        help='threshold for determining if potential is flat')
    parser.add_argument('--threshold_C',type=float,default=1e-3,
                        help='threshold for determining if potential is aligned')
    parser.add_argument('--max_iter',type=int,default=20,
                        help='max. no. of shifts to try')
    parser.add_argument('--vfile',help='vline .dat file',default='vline-eV.dat')
    parser.add_argument('--noplots',help='do not generate plots',default=False,action='store_true')
    parser.add_argument('--allplots',help='save all plots',default=False,action='store_true')
    parser.add_argument('--logfile',help='logfile to save output to')
       
    ## read in the above arguments from command line
    args = parser.parse_args()
    
    calc(args.vref, args.vdef, args.encut, args.q, 
         args.threshold_slope, args.threshold_C, args.max_iter,
         args.vfile, args.noplots, args.allplots, args.logfile) 
    