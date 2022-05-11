#!/bin/bash


## LLFF

# Subdivided
bash x3d/slf/scripts/run_all_llff.sh run_one_subdivided llff_subdivided_affine 4 two_plane 32 "" llff_subdivided
bash x3d/slf/scripts/run_all_llff.sh run_one_subdivided llff_subdivided_half 4 two_plane 32 "" llff_subdivided_half

bash x3d/slf/scripts/run_all_llff.sh run_one_subdivided_teacher_few llff_subdivided_affine 4 two_plane 32 "" llff_subdivided
bash x3d/slf/scripts/run_all_llff.sh run_one_subdivided_teacher_few llff_subdivided_half 4 two_plane 32 "" llff_subdivided_half

bash x3d/slf/scripts/run_all_llff.sh run_one_subdivided_few llff_subdivided_feature 4 two_plane 32 "" llff_subdivided
bash x3d/slf/scripts/run_all_llff.sh run_one_subdivided_few llff_subdivided_no_embed 4 two_plane 32 "" llff_subdivided


## Stanford

bash x3d/slf/scripts/run_all_stanford.sh run_one_lf stanford_affine 4 two_plane 32 "" stanford_lf
bash x3d/slf/scripts/run_all_stanford.sh run_one_lf stanford_half 4 two_plane 32 "" stanford_lf_half

bash x3d/slf/scripts/run_all_stanford.sh run_one_lf stanford_feature 4 two_plane 32 "" stanford_lf
bash x3d/slf/scripts/run_all_stanford.sh run_one_lf stanford_no_embed 4 two_plane 32 "" stanford_lf
