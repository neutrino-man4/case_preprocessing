import h5py
import numpy as np
from re import sub
import sys
import subprocess
import fastjet
import awkward as ak
import os
import glob; import pdb
import copy
from pylorentz import Momentum4
#filename = sys.argv[1]
raw_signal_dir='/work/bmaier/CASE/RAWFILES/signals_merged'
file_paths=glob.glob(raw_signal_dir+'/*.h5')
filename='/work/bmaier/CASE/RAWFILES/signals_merged/WkkToWRadionToWWW_M3000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5'
outfolder='/storage/9/abal/CASE/new_signals'

normalize = 1
deta_jj = 1.3 # CAREFUL: background files have been skimmed with < 1.4
mjj = 1450
jPt = 300

# Define index positions of jet_kinematic variables
MJJ, DETA, J1PT, J1ETA, J1PHI, J1M, J2PT,J2ETA, J2PHI, J2M = range(10)
# Define index positions of jet energy variations
pt_JES_up, m_JES_up, pt_JES_down, m_JES_down, pt_JER_up, m_JER_up, pt_JER_down, m_JER_down, m_JMS_up, m_JMS_down, m_JMR_up, m_JMR_down=range(12)

pt_sf_indices=[0,2,4,6]
mass_sf_indices=[1,3,5,7,8,9,10,11]

tags=['JES_up','JES_down','JER_up','JER_down','JMS_up','JMS_down','JMR_up','JMR_down']

def xyze_to_eppt(constituents,normalize,include_mass=False):
    ''' converts an array [N x 100, 4] of particles
from px, py, pz, E to eta, phi, pt (mass omitted) (8/3/23: Mass added, but values imaginary/NAN)
    '''
    PX, PY, PZ, E = range(4)
    pt = np.sqrt(np.float_power(constituents[:,:,PX], 2) + np.float_power(constituents[:,:,PY], 2), dtype='float32') # numpy.float16 dtype -> float power to avoid overflow
    eta = np.arcsinh(np.divide(constituents[:,:,PZ], pt, out=np.zeros_like(pt), where=pt!=0.), dtype='float32')
    phi = np.arctan2(constituents[:,:,PY], constituents[:,:,PX], dtype='float32')
    
    if normalize == 1:
        print("Hi")
        if include_mass:
            P2 = np.float_power(constituents[:,:,PX], 2) + np.float_power(constituents[:,:,PY], 2)+ np.float_power(constituents[:,:,PZ],2)
            E2=np.float_power(constituents[:,:,E], 2)
            mass= np.sqrt(E2-P2, dtype='float32')
            return np.stack([pt, eta, phi,mass], axis=2) 
        
        return np.stack([pt, eta, phi], axis=2)
    
    if normalize == 0:
        print("Ho")
        return np.stack([constituents[:,:,PX], constituents[:,:,PY], constituents[:,:,PZ]], axis=2)


def epptm_to_xyze(constituents):
    # Does the reverse
    PT,ETA,PHI,M = range(4)
    px = np.sqrt(constituents[:,:,PT]*np.cos(constituents[:,:,PHI]), dtype='float32')
    py = np.sqrt(constituents[:,:,PT]*np.sin(constituents[:,:,PHI]), dtype='float32')
    pz = np.sqrt(constituents[:,:,PT]*np.sinh(constituents[:,:,PHI]))
    E = np.sqrt(np.float_power(pz,2)+np.float_power(py,2)+np.float_power(pz,2)+np.float_power(constituents[:,:,M],2),dtype='float32')
    return np.stack([px, py, pz,E], axis=2) 

def get_m2(constituents):
    PX,PY,PZ,E=range(4)
    P2 = np.float_power(constituents[:,:,PX], 2) + np.float_power(constituents[:,:,PY], 2)+ np.float_power(constituents[:,:,PZ],2)
    E2=np.float_power(constituents[:,:,E], 2)
    return E2-P2,P2

def perform_jetPF_scaling(constituents,scale_factor):
    PT,ETA,PHI,M = range(4); PX, PY, PZ, E = range(4)
    M2,P2=get_m2(constituents)
    m2_sf=np.power(scale_factor[:,-1],2)
    if scale_factor.shape[1]==2:
        constituents[:,:,PX]= constituents[:,:,PX]*scale_factor[:,0,None]
        constituents[:,:,PY]= constituents[:,:,PY]*scale_factor[:,0,None]
        M2_scaled = M2 * m2_sf[:,None]
        constituents[:,:,E]=np.sqrt(M2_scaled+P2)
    else:
        M2_scaled = M2 * m2_sf[:,None]
        constituents[:,:,E]=np.sqrt(M2_scaled+P2)
    return constituents

def perform_jet_scaling(jet_kinematics,j1_scale_factor,j2_scale_factor):
    
    jet_kinematics[:,J1M]=jet_kinematics[:,J1M]*j1_scale_factor[:,-1]
    jet_kinematics[:,J2M]=jet_kinematics[:,J2M]*j2_scale_factor[:,-1]
        
    if j1_scale_factor.shape[1]==2:
        jet_kinematics[:,J1PT]=jet_kinematics[:,J1PT]*j1_scale_factor[:,0]
        jet_kinematics[:,J2PT]=jet_kinematics[:,J2PT]*j2_scale_factor[:,0]
    jet1=Momentum4.m_eta_phi_pt(jet_kinematics[:,J1M],jet_kinematics[:,J1ETA],jet_kinematics[:,J1PHI],jet_kinematics[:,J1PT])
    jet2=Momentum4.m_eta_phi_pt(jet_kinematics[:,J2M],jet_kinematics[:,J2ETA],jet_kinematics[:,J2PHI],jet_kinematics[:,J2PT])
    dijet=jet1+jet2
    jet_kinematics[:,MJJ]=dijet.m
    return jet_kinematics

def recluster(array,normalize,jetkinematics,jetindex=0):
    j1s = np.array(array)
    totlength = len(j1s)
    for i in range(totlength):
        if i % 10000 == 0:
            print(i,' / ',totlength)
        j1_orig = j1s[i]
        #print(j1_orig)
        j1 = j1_orig[j1_orig[:,3]>0]
        cands = ak.zip({
            "px": j1[:,0].astype(np.float16),
            "py": j1[:,1].astype(np.float16),
            "pz": j1[:,2].astype(np.float16),
            "E": j1[:,3].astype(np.float16)
        }, with_name="Momentum4D")
        cluster = fastjet.ClusterSequence(cands, jetdef)
        jets = cluster.inclusive_jets(min_pt=jPt)
        chist = ak.Array(cluster.unique_history_order().to_list())
        chist = chist[chist<len(j1[:,0])]
        #print(chist)
        j1out = j1[chist,...]
        #print(j1out)
        j1final= np.pad(j1out,((0,100-len(j1out[:,0])),(0,0)),'constant')
        #print(j1final)
        j1s[i] = j1final

    j1s = xyze_to_eppt(j1s,normalize=normalize)
    if normalize == 1:
        j1s[:,:,0] = j1s[:,:,0]
        if jetindex == 0:
            j1s[:,:,1] = j1s[:,:,1]#-np.reshape(np.array(jetkinematics[:,3]),(-1,1))*(j1s[:,:,0]>0)
            j1s[:,:,2] = j1s[:,:,2]#-np.reshape(np.array(jetkinematics[:,4]),(-1,1))*(j1s[:,:,0]>0)
        if jetindex == 1:
            j1s[:,:,1] = j1s[:,:,1]#-np.reshape(np.array(jetkinematics[:,7]),(-1,1))*(j1s[:,:,0]>0)
            j1s[:,:,2] = j1s[:,:,2]#-np.reshape(np.array(jetkinematics[:,8]),(-1,1))*(j1s[:,:,0]>0)
        j1s[:,:,2] = np.where((j1s[:,:,2]<np.pi),j1s[:,:,2],j1s[:,:,2]-2*np.pi)
        j1s[:,:,2] = np.where((j1s[:,:,2]>-1*np.pi),j1s[:,:,2],j1s[:,:,2]+2*np.pi)

    return j1s.astype(np.float32)


        
with h5py.File(filename, "r") as f:

    
    print("File %s"%filename)

    #subprocess.call("cp %s ."%filename,shell=True)
    localfile = filename.split("/")[-1]
    #outfolder = filename.replace(localfile,"")
        

    
    #if os.path.exists("%s/%s.lock"%(outfolder,f_sig)):
    #    print("Job already running. Exiting.")
    #    exit(1)

    #if os.path.exists("%s/%s"%(outfolder,f_sig)):
    #    print("File already exists. Exiting.")
    #    exit(1)

    # subprocess.call("touch %s/%s.lock"%(outfolder,f_sig),shell=True)

    # with open("tmplock.txt",'w') as textfile:
    #     textfile.write("%s/%s.lock"%(outfolder,f_sig))
    #     textfile.write("\n")

    #if normalize == 0:
    #    f_sig = sub('_sig', '_unnorm_sig', localfile)

    
    #if int(sys.argv[2]) > 0: 
    #    type_mask = (f["truth_label"][:,0:1][:,0] == int(sys.argv[2]))
    #else:
    #    type_mask = (f["truth_label"][:,0:1][:,0] < 1)


    sig_mask = (f["jet_kinematics"][:,1:2][:,0] < deta_jj) & (f["jet_kinematics"][:,2:3][:,0] > jPt) & (f["jet_kinematics"][:,6:7][:,0] > jPt) & (f["jet_kinematics"][:,0:1][:,0] > mjj) 

    sig_event = np.array(f["event_info"])[sig_mask].astype(np.float32)
    sig_jj = np.array(f["jet_kinematics"])[sig_mask].astype(np.float32)
    sig_j1extra = np.array(f["jet1_extraInfo"])[sig_mask].astype(np.float32)
    sig_j2extra = np.array(f["jet2_extraInfo"])[sig_mask].astype(np.float32)
    sig_truth = np.array(f["truth_label"])[sig_mask].astype(np.float32)
    sig_sysweights = np.array(f["sys_weights"])[sig_mask].astype(np.float32)
    
    j1_JMEVars=np.array(f["jet1_JME_vars"])[sig_mask].astype(np.float32)
    j2_JMEVars=np.array(f["jet2_JME_vars"])[sig_mask].astype(np.float32)
    

    j1_sf_array=j2_sf_array=np.empty(j1_JMEVars.shape,dtype=np.float32)
    for i in pt_sf_indices:
        j1_sf_array[:,i]=np.divide(j1_JMEVars[:,i],sig_jj[:,J1PT],out=np.zeros_like(j1_JMEVars[:,i]), where=sig_jj[:,J1PT]!=0.)
        j2_sf_array[:,i]=np.divide(j2_JMEVars[:,i],sig_jj[:,J2PT],out=np.zeros_like(j1_JMEVars[:,i]), where=sig_jj[:,J2PT]!=0.)
    for i in mass_sf_indices:
        j1_sf_array[:,i]=np.divide(j1_JMEVars[:,i],sig_jj[:,J1M],out=np.zeros_like(j1_JMEVars[:,i]), where=sig_jj[:,J1M]!=0.)
        j2_sf_array[:,i]=np.divide(j2_JMEVars[:,i],sig_jj[:,J2M],out=np.zeros_like(j1_JMEVars[:,i]), where=sig_jj[:,J2M]!=0.)

#### JME VARS: ###
# [pt_JES_up, m_JES_up, pt_JES_down, m_JES_down, pt_JER_up, m_JER_up, pt_JER_down, m_JER_down, m_JMS_up, m_JMS_down, m_JMR_up, m_JMR_down]

    

    # Reclustering
    #jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.8)
    jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8)
    for ind,i in enumerate(mass_sf_indices):
        
        # Read in Jet PFCands
        sig_pf1 = np.array(f["jet1_PFCands"])[sig_mask].astype(np.float32)
        sig_pf2 = np.array(f["jet2_PFCands"])[sig_mask].astype(np.float32)
        # Perform rescaling
        if ind<4:
            print(f'Now scaling for {tags[i]}, from indices {i-1},{i}')
            sig_pf1=perform_jetPF_scaling(sig_pf1,j1_sf_array[:,i-1:i+1])
            sig_pf2=perform_jetPF_scaling(sig_pf2,j2_sf_array[:,i-1:i+1])
            sig_jj=perform_jet_scaling(sig_jj,j1_sf_array[:,i-1:i+1],j2_sf_array[:,i-1:i+1])
        else:
            print(f'Now scaling for {tags[i]}, from index {i}')
            sig_pf1=perform_jetPF_scaling(sig_pf1,j1_sf_array[:,i])
            sig_pf2=perform_jetPF_scaling(sig_pf2,j2_sf_array[:,i])
            sig_jj=perform_jet_scaling(sig_jj,j1_sf_array[:,i],j2_sf_array[:,i])
            
        sig_pf1 = recluster(sig_pf1,normalize,sig_jj)
        sig_pf2 = recluster(sig_pf2,normalize,sig_jj,jetindex=1)

        
        f_sig = sub('\.h5$', f'_sig_{tags[ind]}.h5', localfile)
        sig_hf = h5py.File(os.path.join(outfolder,f_sig), 'w')
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


        
        sig_hf.close()

   # subprocess.call("mv *_sig*.h5 %s"%outfolder,shell=True)

    
