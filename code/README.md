
### Figures for the paper.

- To make fig 1, the four contour plots, run `sassy/four_contours.py`.
- To make fig 2, the domain warpings, run `sassy/generating_functions.py`.
- To make fig 3, the effect of sigma0, run `plot/effect_of_sigma0.py`.
- To make fig 4, steepest descent vs gradient descent, run `SD_vs_GD.py`.
- To make fig 5, the data profile, as well as the data profiles for the 
  appendix, run `tests/cutest_sweep/run_exp.py` to generate the data, 
  `tests/cutest_sweep/post_process_traj.py` to process the data and then
  `tests/cutest_sweep/plot_data_profiles.py` to plot all of the data
  profiles.
- Table 1 of cutest problem dimensions was made manually.
- To make the latex for the table B1, which describes the set of cutest
  problems, run `problems/make_cutest_problem_table.py`
