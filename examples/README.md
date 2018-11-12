### Extract epochs

```
python extract_all_epochs_1.py ../OpSim_DATABASES/baseline2018a.db out_01
python extract_all_epochs_2.py out_01/out_01.npz baseline2018a_all_epochs.dat
rm -rf out_01/
python extend_epochs.py baseline2018a_all_epochs.dat baseline2018a_all_epochs_extended.dat
```

Similar is run for ```colossus_2664```. 


