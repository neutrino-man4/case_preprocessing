import h5py
import numpy as np
from re import sub
import sys
import os
import glob; import pdb; import pathlib
import copy
import case_preprocessing.cut_utils as ctl
import fastjet

#filename = sys.argv[1]
raw_signal_dir='/work/bmaier/CASE/RAWFILES/signals_merged'
file_paths=glob.glob(raw_signal_dir+'/*.h5')
#filename='/work/bmaier/CASE/RAWFILES/signals_merged/WkkToWRadionToWWW_M3000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5'
outfolder='/storage/9/abal/CASE/new_signals'

normalize = 1
deta_jj = 1.3 # CAREFUL: background files have been skimmed with < 1.4
mjj = 1450
jPt = 300

MJJ, DETA, J1PT, J1ETA, J1PHI, J1M, J2PT,J2ETA, J2PHI, J2M = range(10)

SUFFIXES=['_TuneCP2_13TeV-pythia8_TIMBER','_TuneCP5_13TeV_pythia8_TIMBER','_TuneCP5_13TeV-madgraph-pythia8_TIMBER',\
          '_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER']
# Define index positions of jet_kinematic variables
# Define index positions of jet energy variations
pt_JES_up, m_JES_up, pt_JES_down, m_JES_down, pt_JER_up, m_JER_up, pt_JER_down, m_JER_down, m_JMS_up, m_JMS_down, m_JMR_up, m_JMR_down=range(12)

pt_sf_indices=[0,2,4,6]
mass_sf_indices=[1,3,5,7,8,9,10,11]
tags=['JES_up','JES_down','JER_up','JER_down','JMS_up','JMS_down','JMR_up','JMR_down']
#pdb.set_trace()
for f,filename in enumerate(sorted(file_paths)):
    print(f"Signal {f+1}/{len(file_paths)}")
    signal_name=filename.split("/")[-1].replace('.h5','')
    
    for suffix in SUFFIXES:
        signal_name=signal_name.replace(suffix,'') # Get rid of the suffixes for tunes and stuff
    
    if ctl.check_if_exists(signal_name):
        print(f"All output files for {signal_name} exist apparently. Skipping")
        
    print(f'###### SIGNAL = {signal_name} ########')
    with h5py.File(filename, "r") as f:

        
        print("File %s"%filename)

        sig_mask = (f["jet_kinematics"][:,1:2][:,0] < deta_jj) & (f["jet_kinematics"][:,2:3][:,0] > jPt) & (f["jet_kinematics"][:,6:7][:,0] > jPt) & (f["jet_kinematics"][:,0:1][:,0] > mjj) 

        sig_event = np.array(f["event_info"])[sig_mask].astype(np.float32)
        sig_jj = np.array(f["jet_kinematics"])[sig_mask].astype(np.float32)
        sig_j1extra = np.array(f["jet1_extraInfo"])[sig_mask].astype(np.float32)
        sig_j2extra = np.array(f["jet2_extraInfo"])[sig_mask].astype(np.float32)
        sig_truth = np.array(f["truth_label"])[sig_mask].astype(np.float32)
        sig_sysweights = np.array(f["sys_weights"])[sig_mask].astype(np.float32)
        
        j1_JMEVars=np.array(f["jet1_JME_vars"])[sig_mask].astype(np.float32)
        j2_JMEVars=np.array(f["jet2_JME_vars"])[sig_mask].astype(np.float32)
        

        j1_sf_array=np.empty(j1_JMEVars.shape,dtype=np.float32)
        j2_sf_array=np.empty(j2_JMEVars.shape,dtype=np.float32)
        
        for i in pt_sf_indices:
            j1_sf_array[:,i]=np.divide(j1_JMEVars[:,i],sig_jj[:,J1PT],out=np.zeros_like(j1_JMEVars[:,i]), where=sig_jj[:,J1PT]!=0.)
            j2_sf_array[:,i]=np.divide(j2_JMEVars[:,i],sig_jj[:,J2PT],out=np.zeros_like(j2_JMEVars[:,i]), where=sig_jj[:,J2PT]!=0.)
            print(f'{i}, Max j1 SF: {j1_sf_array[:,i].max()}')
            print(f'{i}, Max j2 SF: {j2_sf_array[:,i].max()}')
        
        for i in mass_sf_indices:
            j1_sf_array[:,i]=np.divide(j1_JMEVars[:,i],sig_jj[:,J1M],out=np.zeros_like(j1_JMEVars[:,i]), where=sig_jj[:,J1M]>=1.0e-3)
            j2_sf_array[:,i]=np.divide(j2_JMEVars[:,i],sig_jj[:,J2M],out=np.zeros_like(j2_JMEVars[:,i]), where=sig_jj[:,J2M]!=0.)
            print(f'{i}, Max j1 SF: {j1_sf_array[:,i].max()}')
            print(f'{i}, Max j2 SF: {j2_sf_array[:,i].max()}')
        
        j1_sf_array=np.clip(j1_sf_array,0.,5.)
        j2_sf_array=np.clip(j2_sf_array,0.,5.)

        # Reclustering
        #jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.8)
        
        # 
        for ind,i in enumerate(mass_sf_indices):
            
            # Define output filepath and store 
            pathlib.Path(os.path.join(outfolder,signal_name,tags[ind])).mkdir(parents=True,exist_ok=True)        
            out_filepath=os.path.join(outfolder,signal_name,tags[ind],signal_name+'.h5')
            print('########################')
            print(f"Var: {ind+1}/{len(mass_sf_indices)}")
            
            # Read in Jet PFCands
            sig_pf1 = np.array(f["jet1_PFCands"])[sig_mask].astype(np.float32)
            sig_pf2 = np.array(f["jet2_PFCands"])[sig_mask].astype(np.float32)
            # Perform rescaling
            if ind<4:
                print(f'Now scaling for {tags[ind]}, from indices {i-1},{i}')
                sig_pf1=ctl.perform_jetPF_scaling(sig_pf1,j1_sf_array[:,i-1:i+1])
                sig_pf2=ctl.perform_jetPF_scaling(sig_pf2,j2_sf_array[:,i-1:i+1])
                sig_jj=ctl.perform_jet_scaling(sig_jj,j1_sf_array[:,i-1:i+1],j2_sf_array[:,i-1:i+1])
            else:
                print(f'Now scaling for {tags[ind]}, from index {i}')
                sig_pf1=ctl.perform_jetPF_scaling(sig_pf1,j1_sf_array[:,i])
                sig_pf2=ctl.perform_jetPF_scaling(sig_pf2,j2_sf_array[:,i])
                sig_jj=ctl.perform_jet_scaling(sig_jj,j1_sf_array[:,i],j2_sf_array[:,i])
                
            sig_pf1 = ctl.recluster(sig_pf1,normalize,sig_jj)
            sig_pf2 = ctl.recluster(sig_pf2,normalize,sig_jj,jetindex=1)
            
            
            # Define output filepath and store 
            pathlib.Path(os.path.join(outfolder,signal_name,tags[ind])).mkdir(parents=True,exist_ok=True)        
            out_filepath=os.path.join(outfolder,signal_name,tags[ind],signal_name+'.h5')
            
            sig_hf = h5py.File(out_filepath, 'w')
            sig_hf.create_dataset('jet1_PFCands', data=sig_pf1)
            sig_hf.create_dataset('jet1_PFCands_shape', data=sig_pf1.shape)
            sig_hf.create_dataset('jet2_PFCands', data=sig_pf2)
            sig_hf.create_dataset('jet2_PFCands_shape', data=sig_pf2.shape)
            sig_hf.create_dataset('jet1_extra', data=sig_j1extra)
            sig_hf.create_dataset('jet1_extra_shape', data=sig_j1extra.shape)
            sig_hf.create_dataset('jet2_extra', data=sig_j2extra)
            sig_hf.create_dataset('jet2_extra_shape', data=sig_j2extra.shape)
            sig_hf.create_dataset('jet_kinematics', data=sig_jj)
            sig_hf.create_dataset('jet_kinematics_shape', data=sig_jj.shape)
            sig_hf.create_dataset('event_info', data=sig_event)
            sig_hf.create_dataset('event_info_shape', data=sig_event.shape)
            sig_hf.create_dataset('truth_label', data=sig_truth)
            sig_hf.create_dataset('truth_label_shape', data=sig_truth.shape)
            sig_hf.create_dataset('sys_weights', data=sig_sysweights)
            sig_hf.create_dataset('sys_weights_shape', data=sig_sysweights.shape)
            
            # HERE: Add 


            print(f'Data written to {out_filepath}')
            sig_hf.close()

   # subprocess.call("mv *_sig*.h5 %s"%outfolder,shell=True)

    
