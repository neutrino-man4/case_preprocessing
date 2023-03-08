### CASE Data Pre-processing

Run using:
```
python3 case_qr.py
```

New signal input files with associated systematic uncertainties and jet energy/mass variations are preprocessed using this script to apply all signal region cuts. Jets are already clustered using the anti-kT algorithm, the PF Candidates per jet are rearranged in the Cambridge-Aachen order by reclustering with the C/A algorithm. 