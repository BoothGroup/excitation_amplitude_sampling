# Data for excitation amplitude sampling paper
Classical shadows and quantum cluster solvers

The directory is structured as follows:
- src contains the core code for using classical shadows (classical_shadow.py) and quantum cluster solvers for embedding (cluster_solver.py)
- data_collection_scripts contains the code which was run to collect the data used in the (upcoming) publication.
  - hydrogen_results_comp_fci.py is for the figure coparing the different approaches of measuring the energy given a quantum circuit - in this case with an FCI wavefcuntion as a stand in
  - hydrogen_results_comp_dmrg.py is the same but using a dmrg wavefunction as a stand in
  - noise_models_hydrogen_circuits.py does the interpolation of the amount of noise and compares the result of using the expectation value and mixed estimator using a quantum simulator
  - the scripts in the real devices folder contain the code to create circuits for the gound sate using ffsim and store in a pickle file and then run on an IBM backend
  - H_properties_comp_all_ci.py contains the code to calculate the RDMs by mapping the ci amplitudes to a CCSD ansatz
  - the callback_BN/NiO.py and NiO_fci_embedding_tests.py contain examples of using the cluster_solver code to do wavefunction embedding
- plots/data conatins the data and plotting scripts to recreate the plots in the publication
  - Figure 1 a/b are done using fci_energy_method_comp_plots.ipynb
  - Figure 2 using noise_model_plotting.ipynb
  - Figure 3 a/b using szsz_plotting.ipynb
  - Figure 4 & 5 use BN_plotting.ipynb and callback_NiO_plotting.ipynb respectively
