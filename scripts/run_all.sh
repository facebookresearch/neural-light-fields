#!/bin/bash

### LLFF
#
## LF
#bash x3d/slf/scripts/run_all_llff.sh run_one_lf stanford_half 4 two_plane 32 "" llff_lf
#bash x3d/slf/scripts/run_all_llff.sh run_one_lf_reg stanford_half 4 two_plane 32 "" llff_lf
#bash x3d/slf/scripts/run_all_llff.sh run_one_lf_teacher stanford_half 4 two_plane 32 "" llff_lf
#
#
### Shiny
#
## Subdivided
#bash x3d/slf/scripts/run_all_shiny.sh run_one_subdivided llff_subdivided_half 4 two_plane 32 "" shiny_subdivided
#bash x3d/slf/scripts/run_all_shiny.sh run_one_subdivided_reg llff_subdivided_half 4 two_plane 32 "" shiny_subdivided
#
#bash x3d/slf/scripts/run_all_shiny.sh run_one_subdivided llff_subdivided_half 4 two_plane 32 "" shiny_subdivided
#bash x3d/slf/scripts/run_all_shiny.sh run_one_subdivided_reg llff_subdivided_half 4 two_plane 32 "" shiny_subdivided
#
# LF
bash x3d/slf/scripts/run_all_shiny.sh run_one_lf stanford_half 4 two_plane 32 "" shiny_lf
bash x3d/slf/scripts/run_all_shiny.sh run_one_lf_reg stanford_half 4 two_plane 32 "" shiny_lf
#
#
### Stanford
#
## Subdivided
#bash x3d/slf/scripts/run_all_stanford.sh run_one_subdivided_small llff_subdivided_half 4 two_plane 32 "" stanford_subdivided
#bash x3d/slf/scripts/run_all_stanford.sh run_one_subdivided_reg_small llff_subdivided_half 4 two_plane 32 "" stanford_subdivided
