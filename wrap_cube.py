import numpy as np
from ase.io.cube import read_cube_data, write_cube
from ase.geometry import wrap_positions
from ase.visualize import view
import sys

#the wrap part was modified from wrap_positions in ase.geometry
#this version of wrap only translates atomic positions consistent with the discreet 
#shift in cube data shift.
def wrap_and_center(positions: np.ndarray,           # atomic positions 
					cell: np.ndarray,                # unit cell
					pbc=np.array([True,True,True]),  # periodic boundary condition
					nxnynz=[100,100,100],            # cube grid dimensions
					eps=1e-7):                       # tolerance
	"""
	This wrap function was modified from wrap_positions in ase.geometry. We made changes to only translates atomic positions consistent with the discreet shift in cube data shift.

	Parameters
	----------
		positions: 
			atomic positions 
		cell: 
			unit cell of the cube file
		pbc:
			periodic boundary condition for translating and wrapping atomic positions
		nxnynz:
			cube grid dimensions. The function needs this to translate atoms along this discreet grid
		eps=1e-7
			tolerance for atoms        """
    shift = np.zeros(3) - eps
    # Don't change coordinates when pbc is False
    shift[np.logical_not(pbc)] = 0.0
    #positions in crystal coordinate
    fractional = np.linalg.solve(cell.T,np.asarray(positions).T).T - shift
    #for translation later
    nxnynz_moved=np.zeros(3,dtype=int)
    #for each crystal axis
    for i in range(3):
        if not pbc[i]:
            continue
        # translation increment for both atoms and volume
        increment=1/nxnynz[i]
        
        #wrap part
        #indices for reordering below
        indices = np.argsort(fractional[:, i])
        #reordered fragment x,y,z position
        sp = fractional[indices, i]
        #take the difference between the largest and the smallest
        widths = (np.roll(sp, 1) - sp) % 1.0
        #minus the smallest of the differences
        diff=sp[np.argmin(widths)]
        #total translation once corrected for increment
        diff_inc=(np.floor_divide(diff,increment))*increment
        nxnynz_moved[i]=diff_inc/increment
        #move
        fractional[:, i] -= diff_inc
        #wrap
        fractional[:, i] %= 1.0

        #now move to center
        current_center=(np.max(fractional[:, i])-np.min(fractional[:, i]))/2
        desired_center=0.5
        translate=desired_center-current_center
        translate_inc=(np.floor_divide(translate,increment))*increment
        fractional[:, i] += translate_inc
        nxnynz_moved[i]-=translate_inc/increment
        nxnynz_moved[i]=nxnynz_moved[i] % nxnynz[i]
    return np.dot(fractional, cell), nxnynz_moved

#for reading in geometry and density from a cube file, wraping, centering, and writing 
if __name__ = '__main__':
	#path = './'
	file = sys.argv[1]
    density, atoms = read_cube_data(file)
    translate_directions=np.array([True,True,False])
    pos2, nxnynz_moved=wrap_and_center(atoms.get_positions(),
    				   atoms.get_cell(),
    				   pbc=translate_directions,
    				   nxnynz=np.shape(density))
    atoms.set_positions(pos2)
    #view(atoms) for debugging
    #roll density forward with a negative number 
    density2=np.roll(density,-nxnynz_moved,axis=[0,1,2])
    file2=file.split('.')[0]+'_centered.cube'
    with open(file2,'w') as f:
        write_cube(f,atoms,data=density2)
