#!/usr/bin/env python3
"""Extract Leddy Fig. 4.8 PDF paths and compare with HW_N512 DNS and reconstructed LES fields.
Run with --pdf /path/to/Thesis_JLeddy.pdf. Requires numpy, scipy, matplotlib,
netCDF4 and pymupdf. Existing CSV caches avoid rereading the simulation dumps.
"""
import argparse
import csv
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/leddy_mpl')
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation_U-Net_HW_N512/thesis_comparison'
MODELS = ['Viscosity', 'Hyperviscosity', 'Smagorinsky', 'CENTORI', 'Reference']
COLORS = ['#009eaa', '#5155dd', '#359440', '#db4646', '#777777']

def extract(pdf):
    import pymupdf
    doc = pymupdf.open(pdf)
    paths = [d for d in doc[97].get_drawings() if len(d['items']) > 150]
    assert len(paths) == 10
    expected_colors = [(0., .75, .75), (0., 0., 1.), (0., .5, 0.), (1., 0., 0.), (0., 0., 0.)]
    rows = []
    for i, d in enumerate(paths):
        panel, model = divmod(i, 5)
        assert tuple(d['color']) == expected_colors[model]
        assert all(item[0] == 'l' for item in d['items'])
        points = [d['items'][0][1]] + [item[2] for item in d['items']]
        xy = np.array([(p.x, p.y) for p in points])
        t = (xy[:, 0] - 75.07023620605469) * 600 / (469.9895324707031 - 75.07023620605469)
        top, bottom, lo, hi = ((57.21759033203125, 168.60696411132812, -.1, .6) if panel == 0 else (201.12049865722656, 312.5098571777344, -.05, .25))
        flux = hi - (xy[:, 1] - top) * (hi-lo)/(bottom-top)
        assert np.all(np.diff(t) > 0)
        rows.extend(zip([MODELS[model]]*len(t), ['radial' if panel == 0 else 'poloidal']*len(t), t, flux))
    with (OUT/'leddy_fig4_8_vector_values.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['model','component','time','flux']); w.writerows(rows)
    return rows

def derivative(a, h, axis):
    return (np.roll(a,2,axis)-8*np.roll(a,1,axis)+8*np.roll(a,-1,axis)-np.roll(a,-2,axis))/(12*h)

def local_flux(run):
    cache = OUT / (run+'_flux.csv')
    if cache.exists():
        return np.loadtxt(cache, delimiter=',', skiprows=1)
    folder = ROOT/'BOUT_runs'/run/'data'
    with Dataset(folder/'BOUT.dmp.0.nc') as d:
        t=np.asarray(d['t_array'][:]); ranks=int(d['NXPE'][:]); assert int(d['NYPE'][:])==1
        ngrid=int(d['MXSUB'][:])*ranks
    result=np.zeros((len(t),4))
    for rank in range(ranks):
        with Dataset(folder/f'BOUT.dmp.{rank}.nc') as d:
            assert np.array_equal(t, d['t_array'][:])
            g=int(d['MXG'][:]); assert g>=2
            for start in range(0,len(t),32):
                sl=slice(start,start+32)
                n=np.asarray(d['n'][sl,:,0,:],dtype=float)
                phi=np.asarray(d['phi'][sl,:,0,:],dtype=float)
                px=derivative(phi,50.176/ngrid,1)[:,g:-g,:]
                py=derivative(phi,50.176/ngrid,2)[:,g:-g,:]
                n=n[:,g:-g,:]; fluct=n-n.mean(axis=2,keepdims=True)
                for j,v in enumerate([-n*py,n*px,-fluct*py,fluct*px]):
                    result[sl,j]+=v.sum(axis=(1,2))/ngrid**2
    data=np.column_stack([t,result]); assert np.isfinite(data).all()
    np.savetxt(cache,data,delimiter=',',header='time,radial_full,poloidal_full,radial_fluctuation,poloidal_fluctuation',comments='')
    print(run, 'mean fluxes:',result.mean(axis=0),flush=True)
    return data

def reconstructed_flux(run):
    cache=OUT/(run+'_reconstructed_flux.csv')
    if cache.exists(): return np.loadtxt(cache,delimiter=',',skiprows=1)
    rows=[]
    for path in (ROOT/'BOUT_runs'/run/'results_U-Net/fields').glob('*.npz'):
        with np.load(path) as d:
            n=np.asarray(d['U'],dtype=float); phi=np.asarray(d['V'],dtype=float)
            t=float(d['simtime'])
        px=derivative(phi,50.176/n.shape[0],0)
        py=derivative(phi,50.176/n.shape[1],1)
        fluct=n-n.mean(axis=1,keepdims=True)
        rows.append([t,(-n*py).mean(),(n*px).mean(),(-fluct*py).mean(),(fluct*px).mean()])
    data=np.array(sorted(rows,key=lambda r:r[0]))
    assert np.isfinite(data).all() and np.all(np.diff(data[:,0])>0)
    np.savetxt(cache,data,delimiter=',',header='time,radial_full,poloidal_full,radial_fluctuation,poloidal_fluctuation',comments='')
    print(run,'reconstructed means:',data[:,1:].mean(axis=0),flush=True)
    return data

def plot(rows, local, fluct=False):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for k,ax in enumerate(axes):
        component=['radial','poloidal'][k]
        for model,color in zip(MODELS,COLORS):
            values=np.array([(r[2],r[3]) for r in rows if r[0]==model and r[1]==component])
            ax.plot(values[:,0],values[:,1],color=color,ls='--',lw=1.65,label='Leddy: '+model)
        for (label,data),color in zip(local.items(),['#111111','#0072b2','#e69f00','#a336a3']):
            assert np.allclose(np.diff(data[:,0]),1.0), 'Smoothing assumes unit output spacing'
            y=data[:,1+k+(2 if fluct else 0)]
            ax.plot(data[:,0],savgol_filter(y,51,3),color=color,lw=2,label='HW_N512: '+label)
        ax.axhline(0,color='.65',lw=.6)
        ax.grid(alpha=.18)
        ax.set_title(component.capitalize()+' particle flux',loc='left',fontweight='bold')
        ax.set_ylabel(r'$\Gamma_x$' if k==0 else r'$\Gamma_{\mathrm{pol}}$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=9, loc='upper center', bbox_to_anchor=(.53,.945), frameon=False)
    axes[1].set_xlabel(r'Recorded simulation time $\omega_{ci}t$')
    axes[1].set_xlim(0,1000)
    fig.suptitle('HW_N512 and Leddy (2016), Figure 4.8',fontsize=16,fontweight='bold')
    definition = 'HW_N512 uses density fluctuations; thesis uses full density.' if fluct else 'Full-density flux: radial = −⟨n ∂pol φ⟩; poloidal = ⟨n ∂x φ⟩.'
    fig.text(.08,.084,'Different parameters: thesis κ = 0.1; HW_N512 κ = 1.0 (both α = 1).',fontsize=10,fontweight='bold')
    fig.text(.08,.061,definition+'  Spatial means; no amplitude rescaling.',fontsize=10)
    fig.text(.08,.016,'Dashed: thesis vector extraction (LES 64², reference 512²), already smoothed; ends at t = 600.\nSolid: DNS / reconstructed LES (512² fields), Savitzky–Golay window 51 outputs, degree 3. Time origins are not aligned.',fontsize=9,color='.3')
    fig.tight_layout(rect=(0,.12,1,.865))
    name='particle_flux_vs_leddy_fig4_8'+('_fluctuation' if fluct else '')
    for suffix in ['png','pdf']:
        fig.savefig(OUT/f'{name}.{suffix}',dpi=240)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--pdf',type=Path,required=True); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    rows=extract(args.pdf)
    local={label:(local_flux(run) if run=='HW_DNS512' else reconstructed_flux(run)) for label,run in [('DNS 512²','HW_DNS512'),('LES 256²','HW_DNS512_LES256'),('LES 128²','HW_DNS512_LES128'),('LES 64²','HW_DNS512_LES64')]}
    plot(rows,local); plot(rows,local,fluct=True)
    print(OUT)
if __name__=='__main__': main()
