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

Now we can prepare epochs of follow-up observations:

```bash
python plan_observations.py ../data/SMC_Chile_visibility_v1.dat baseline2018a_all_epochs_extended.dat > ../data/baseline2018a_followup_epochs_v1.dat
python plan_observations.py ../data/SMC_Chile_visibility_v1.dat colossus_2664_all_epochs_extended.dat > ../data/colossus_2664_followup_epochs_v1.dat
```

