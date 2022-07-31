# A very simple 2D Maxwell Solver using Yee FDTD

This is a simple solver for Maxwell's equations in two dimensions. The application uses the method introduced by Kane Yee in 1966 [1].

The problem is set up with an electric current rotating around the centre of a 1.0m x 1.0m box. By default, the problem is solved on a 4000 x 4000 grid.

At the end of execution, a VTK file is produced for visualisation purposes. This can be loaded into VisIt for analysis.

## Building

The application can be built with the provided Makefile. e.g.

```
$ make
```

This will build a maxwell binary.

## Running

The application can be ran in its default configuration with:

```
$ ./maxwell
```

This will run with 4000 x 4000 grid and will output status every 100 iterations. At the end of execution a VTK file will be produced. 

There are numourous other options available. These can be queried with:

```
$ ./maxwell --help
```

**Beware:** In the default configuration, the VTK file will be in the region of 2 GB for a 4000 x 4000 problem. For visualisation purposes you should decrease the grid resolution. If you would like to visualise the entire execution, you should enable checkpointing and increase the output frequency. You can do this like so:

```
$ mkdir out
$ ./maxwell -x 100 -y 100 -c -f 1 -o out/my_sim -n 1000
```

This will run a 100 x 100 problem (perfect for manageable visualisation), enable checkpointing every timestep, and will output the VTK files to the out directory. It will run for 1000 steps.

## References

[1]: Kane Yee, "Numerical solution of initial boundary value problems involving maxwell's equations in isotropic media," in IEEE Transactions on Antennas and Propagation, vol. 14, no. 3, pp. 302-307, May 1966, doi: 10.1109/TAP.1966.1138693.