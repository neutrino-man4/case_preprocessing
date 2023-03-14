import numpy as np
from re import sub
import sys
import subprocess
import fastjet
import awkward as ak
import os
import copy
from pylorentz import Momentum4

MJJ, DETA, J1PT, J1ETA, J1PHI, J1M, J2PT,J2ETA, J2PHI, J2M = range(10)


deta_jj = 1.3 # CAREFUL: background files have been skimmed with < 1.4
mjj = 1450
jPt = 300

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
    if len(scale_factor.shape)==2:
        print("pT and m will be scaled")
        m2_sf=np.power(scale_factor[:,1],2)
        constituents[:,:,PX]= constituents[:,:,PX]*scale_factor[:,0,None]
        constituents[:,:,PY]= constituents[:,:,PY]*scale_factor[:,0,None]
        M2_scaled = M2 * m2_sf[:,None]
        constituents[:,:,E]=np.sqrt(M2_scaled+P2)
    else:
        print("Only m will be scaled")
        m2_sf=np.power(scale_factor[:],2)
        M2_scaled = M2 * m2_sf[:,None]
        constituents[:,:,E]=np.sqrt(M2_scaled+P2)
        
    return constituents

def perform_jet_scaling(jet_kinematics,j1_scale_factor,j2_scale_factor):
    
    
    if len(j1_scale_factor.shape)==2:
        print("pT and m will be scaled")
        jet_kinematics[:,J1PT]=jet_kinematics[:,J1PT]*j1_scale_factor[:,0]
        jet_kinematics[:,J2PT]=jet_kinematics[:,J2PT]*j2_scale_factor[:,0]
        jet_kinematics[:,J1M]=jet_kinematics[:,J1M]*j1_scale_factor[:,1]
        jet_kinematics[:,J2M]=jet_kinematics[:,J2M]*j2_scale_factor[:,1]
    else:
        print("Only m will be scaled")
        jet_kinematics[:,J1M]=jet_kinematics[:,J1M]*j1_scale_factor[:]
        jet_kinematics[:,J2M]=jet_kinematics[:,J2M]*j2_scale_factor[:]
    
    jet1=Momentum4.m_eta_phi_pt(jet_kinematics[:,J1M],jet_kinematics[:,J1ETA],jet_kinematics[:,J1PHI],jet_kinematics[:,J1PT])
    jet2=Momentum4.m_eta_phi_pt(jet_kinematics[:,J2M],jet_kinematics[:,J2ETA],jet_kinematics[:,J2PHI],jet_kinematics[:,J2PT])
    dijet=jet1+jet2
    jet_kinematics[:,MJJ]=dijet.m
    return jet_kinematics

def recluster(array,normalize,jetkinematics,jetindex=0):
    jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8)
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

def check_if_exists(signal_name,outfolder='/storage/9/abal/CASE/new_signals',return_list=False):
    mass_sf_indices=[1,3,5,7,8,9,10,11]
    tags=['JES_up','JES_down','JER_up','JER_down','JMS_up','JMS_down','JMR_up','JMR_down']
    exists_ok=[]
    for ind,i in enumerate(mass_sf_indices):
        out_filepath=os.path.join(outfolder,signal_name,tags[ind],signal_name+'.h5')
        if os.path.exists(out_filepath):
            exists_ok.append(True)
        else:
            exists_ok.append(False)
    if return_list:
        return exists_ok
    return all(exists_ok)