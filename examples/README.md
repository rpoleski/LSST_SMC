### Extract epochs

```bash
python extract_all_epochs_1.py ../OpSim_DATABASES/baseline2018a.db out_01
python extract_all_epochs_2.py out_01/out_01.npz baseline2018a_all_epochs.dat
rm -rf out_01/
python extend_epochs.py baseline2018a_all_epochs.dat baseline2018a_all_epochs_extended.dat
```

Similar is run for ```colossus_2664```. 

Then:
```bash
python SMC_Chile_visibility.py
```
produces ```../data/baseline2018a_followup_epochs_v1.dat``` file

To get the 5 sigma limits, I'm extracting all i band epochs from following fields (for baseline2018a; RA and Dec):

* 5.60689 -73.29378
* 9.77726 -70.75769
* 12.96201 -75.51621
* 16.56875 -72.89689
* 19.21674 -70.22777
* 25.15984 -74.83162
* 26.81115 -72.09968

This goes to ```../data/baseline2018a_i_band_near_SMC_5sig.dat```

Now we can prepare epochs of follow-up observations:

```bash
python plan_observations.py ../data/SMC_Chile_visibility_v1.dat baseline2018a_all_epochs_extended.dat ../data/baseline2018a_i_band_near_SMC_5sig.dat > ../data/baseline2018a_followup_epochs_v1.dat
```
__colossus_2664 has to be done now__


