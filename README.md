# wrap_cube
This code shifts both the atomic positions and the volume data in a cube file consistently along the voxel grid to center the atoms and writes out a new cube file.

If you, like me, and have performed many periodic electronic structure calculations of low dimensional materials, you have made a density/wf cube file that has atoms and volume wrapped around to the other side of the unit cell as a result of periodic boundary condition. While it is possible to shift the atomic positions and center it, there is no straightforward way to do this with the volume. The volume is represented as weight in a voxel (a volumetric pixel) on a discreet grid sized ($n_x$, $n_y$, $n_z$). This script discretize space inside a unit cell to $n_x$, $n_y$, $n_z$ and gives a translational vector in real space for the atoms and in unit space for the density. Atoms are centered using its center of mass.
