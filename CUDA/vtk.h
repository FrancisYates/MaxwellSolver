#ifndef VTK_H
#define VTK_H

void set_default_base();
void set_basename(char *base);
char *get_basename();
int write_checkpoint(int iteration);
int write_result();
int write_vtk(char* filename);

#endif