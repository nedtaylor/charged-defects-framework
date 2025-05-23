import os
import json
import errno
import argparse
from qdef2d import osutils, logging
from qdef2d.defects.corrections import SPHInX_input_file, alignment_correction_2d
from ase.io import read
import numpy as np


def apply_all(dir_def,dir_ref,eps_slab=None,d_slab=None,dbentry=None,
              functional="GGA",encut=520,soc=False,logfile=None):
    
    """
    Apply sxdefectalign2d correction to all charged defect calulations.
    
    dir_def (str): path to the main defect directory
    dir_ref (str): path to the pristine reference directory
    [optional] eps_slab (float): ave. slab dielectric constant (supply this or dbentry)
    [optional] d_slab (float): slab thickness in Angstroms (supply this or dbentry)
    [optional] dbentry (str): path to the relevant database entry .json file
                              (supply this or eps_slab and d_slab)                          
    [optional] functionl (str): functional used for this set of calculations. Default=GGA.
    [optional] encut (int): cutoff energy (eV). Default=520.
    [optional] soc (bool): whether or not to look in soc(dos) subdirectory. Default=False.
    [optional] logfile (str): logfile to save output to                              

    """
    
    ## set up logging
    if logfile:
        myLogger = logging.setup_logging(logfile)
    else:
        myLogger = logging.setup_logging()

    
    ## if eps_slab and d_slab are directly provided
    if eps_slab and d_slab:
        myLogger.info("eps_slab: %.2f ; d_slab: %.2f"%(eps_slab,d_slab))

    elif dbentry:
        ## extract the eps_slab and d_slab from the relevant dbentry file
        if os.path.exists(dbentry):
            myLogger.info("Using slab properties from " + dbentry)
            with open(dbentry, 'r') as file:
                mater = json.loads(file.read())
                eps_slab = mater[functional]["eps_ave"]
                d_slab = mater[functional]["d_slab"]
                myLogger.info("eps_slab: %.2f ; d_slab: %.2f"%(eps_slab,d_slab))
        else:
            ## if can't find a dbentry file
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dbentry)
            
    else:
        myLogger.info("Insufficient information provided about the slab dielectric profile")


    qs = osutils.listdironly(dir_def)
    for q in [qi for qi in qs if qi != 'charge_0']:      
        print("Charge state: %s" % q)

        for cell in osutils.listdironly(os.path.join(dir_def,q,'')):
            for vac in osutils.listdironly(os.path.join(dir_def,q,cell,'')):
                folder = os.path.join(dir_def,q,cell,vac,'')
                folder_ref = os.path.join(dir_ref,'charge_0',cell,vac,'')

                if soc:  
                    folder = os.path.join(folder,'dos','')
                    folder_ref = os.path.join(folder_ref,'dos','')
                if os.path.exists(folder) and 'restart' in osutils.listdironly(folder):
                    folder = os.path.join(folder,'restart','')

                # read Atoms object from POSCAR
                if not os.path.exists(os.path.join(folder,'POSCAR')):
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                            os.path.join(folder,'POSCAR'))
                atoms = read(os.path.join(folder,'POSCAR'))

                # Get atomic positions
                positions = atoms.positions.copy()

                # Get periodic cell vector along z
                cell_z = atoms.cell[2, 2]  # Third component of the third cell vector

                # Add periodic images above and below
                positions = np.vstack((positions, positions + [0, 0, cell_z])) #, positions - [0, 0, cell_z]))

                # Sort all positions by z
                positions = positions[np.argsort(positions[:, 2])]

                # Remove duplicate positions (tolerance-based filtering)
                filtered_positions = [positions[0]]
                for pos in positions[1:]:
                    if np.linalg.norm(pos - filtered_positions[-1]) > 1e-3:  # Remove near-duplicates
                        filtered_positions.append(pos)
                positions = np.array(filtered_positions)

                # Compute distances between consecutive atoms along z
                distances = np.diff(positions[:, 2])

                # Debugging output
                print("Min distance between atoms: ", np.min(distances))
                print("Max distance between atoms: ", np.max(distances))

                # Find the first large gap (> 60 angstrom)
                lw_idx = np.argmax(distances > 6.0)  # First occurrence of gap > 60 angstrom

                if lw_idx == 0 and distances[lw_idx] <= 6.0:
                    raise ValueError("No vacuum gap larger than 6 angstroms was found!")

                # Print results
                print("Slab is located at %.2f bohr" % positions[lw_idx, 2])
                print(positions[lw_idx : lw_idx + 2])

                slab_edge_up = positions[lw_idx, 2]
                slab_edge_lw = positions[lw_idx + 1, 2]



                
                ## check if defectproperty.json file is present in current directory
                if not os.path.exists(os.path.join(folder,'defectproperty.json')):
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                            os.path.join(folder,'defectproperty.json'))
                    
                else:
                    os.chdir(folder)
                    ## generate SPHInX input file
                    SPHInX_input_file.generate(eps_slab,d_slab, slab_edge_lw, slab_edge_up)
                    
                    if (os.path.exists(os.path.join(folder,'LOCPOT')) and 
                        os.path.exists(os.path.join(folder_ref,'LOCPOT'))):
                        
                        os.chdir(os.path.join(folder,'correction'))
                        myLogger.info("applying correction to calculations in %s"%folder) 
                        
                        ## apply correction
                        alignment_correction_2d.calc(os.path.join(folder_ref,'LOCPOT'),
                                                     os.path.join(folder,'LOCPOT'),
                                                     encut, int(q.split('_')[-1]),
                                                     allplots=True, logfile='getalign.log',
                                                     slab_edge_lw=slab_edge_lw, slab_edge_up=slab_edge_up)
    
                    elif not os.path.exists(os.path.join(folder,'LOCPOT')):
                        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                                os.path.join(folder,'LOCPOT'))

                        
                    elif not os.path.exists(os.path.join(folder_ref,'LOCPOT')):
                        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                                os.path.join(folder_ref,'LOCPOT'))   
                    

if __name__ == '__main__':

    
    ## this script can also be run directly from the command line
    parser = argparse.ArgumentParser(description='Apply sxdefectalign2d correction \
                                     to all charged defect calulations.')
    parser.add_argument('dir_def', help='path to the main defect directory')
    parser.add_argument('dir_ref', help='path to the pristine reference directory')
    parser.add_argument('--eps_slab', type=float,
                        help='average slab dielectric constant (supply this or --dbentry)')
    parser.add_argument('--d_slab', type=float,
                        help='slab thickness in Angstroms (supply this or --dbentry)')
    parser.add_argument('--dbentry', help='path to the relevant database entry .json file \
                        (supply this or --eps_slab and --d_slab)')
    parser.add_argument('--functional',  default='GGA',
                        help='functional that was used for this set of calculations')
    parser.add_argument('--encut', type=int, default=520, help='cutoff energy (eV)')
    parser.add_argument('--soc', default=False,action='store_true',
                        help='whether or not to look in soc(dos) subdirectory')
    parser.add_argument('--logfile', help='logfile to save output to')
       
    ## read in the above arguments from command line
    args = parser.parse_args()
    
    apply_all(args.dir_def, args.dir_ref, args.eps_slab, args.d_slab, args.dbentry,
              args.functional, args.encut, args.soc, args.logfile)

