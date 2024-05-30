# Codimensional Incremental Potential Contact with Cloth

# Most Important Functionality
- calculate Matrix2Xi edges from Matrix3Xi faces

## cloth part:
- modify bending and energy model with nonzero rest dihedral angle
  - triangle_edge_info use this assumption
- strain limiting
- material more than naive stvk
- add dirichlet boudnary condition to test module
- project per element H_elastic to SPD: Robust quasistatic finite elements and flesh simulation

## c-ipc part:
- line search

### build constrained set C

- collision mesh <- rest position, indices, edge_indices(no specific order but unique)

### ACCD based on C

## test part:
- use <cfenv> to fix float rounding issue
- or import to matlab
- or refer to c-ipc toolkit