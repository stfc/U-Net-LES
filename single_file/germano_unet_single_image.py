#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def k2_np(n,lx,ly):
    kx=2*np.pi*np.fft.fftfreq(n,d=lx/n); ky=2*np.pi*np.fft.fftfreq(n,d=ly/n)
    KX,KY=np.meshgrid(kx,ky,indexing='ij'); return KX*KX+KY*KY

def germano_filter(u,a,lx,ly):
    uh=np.fft.fft2(u); return np.fft.ifft2(uh/(1+a*a*k2_np(u.shape[0],lx,ly))).real

def inverse_germano(u,a,lx,ly):
    uh=np.fft.fft2(u); return np.fft.ifft2((1+a*a*k2_np(u.shape[0],lx,ly))*uh).real

def spectral_resize(u,target_n):
    n=u.shape[0]; sh=np.fft.fftshift(np.fft.fft2(u)); th=np.zeros((target_n,target_n),complex)
    c=min(n,target_n); s0=(n-c)//2; t0=(target_n-c)//2
    th[t0:t0+c,t0:t0+c]=sh[s0:s0+c,s0:s0+c]
    th*= (target_n/n)**2
    return np.fft.ifft2(np.fft.ifftshift(th)).real

def k2_torch(n,lx,ly,device,dtype):
    kx=2*math.pi*torch.fft.fftfreq(n,d=lx/n,device=device,dtype=dtype)
    ky=2*math.pi*torch.fft.fftfreq(n,d=ly/n,device=device,dtype=dtype)
    KX,KY=torch.meshgrid(kx,ky,indexing='ij'); return KX.square()+KY.square()

def inverse_germano_torch(u,a,lx,ly):
    n=u.shape[-1]; uh=torch.fft.fft2(u,dim=(-2,-1)); return torch.fft.ifft2((1+a*a*k2_torch(n,lx,ly,u.device,u.dtype))*uh,dim=(-2,-1)).real

class ConvBlock(nn.Module):
    def __init__(self,ci,co):
        super().__init__(); g=min(8,co)
        while co%g: g-=1
        self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1),nn.GroupNorm(g,co),nn.GELU(),nn.Conv2d(co,co,3,padding=1),nn.GroupNorm(g,co),nn.GELU())
    def forward(self,x): return self.net(x)

class UpBlock(nn.Module):
    def __init__(self,ci,cs,co): super().__init__(); self.net=ConvBlock(ci+cs,co)
    def forward(self,x,s):
        x=nn.functional.interpolate(x,size=s.shape[-2:],mode='bilinear',align_corners=False)
        return self.net(torch.cat([x,s],1))

class ResidualUNetSR(nn.Module):
    def __init__(self,c=3,b=24):
        super().__init__(); self.pool=nn.AvgPool2d(2)
        self.e1=ConvBlock(c,b); self.e2=ConvBlock(b,2*b); self.e3=ConvBlock(2*b,4*b); self.e4=ConvBlock(4*b,8*b)
        self.bot=ConvBlock(8*b,8*b); self.u3=UpBlock(8*b,4*b,4*b); self.u2=UpBlock(4*b,2*b,2*b); self.u1=UpBlock(2*b,b,b); self.out=nn.Conv2d(b,c,1)
    def forward(self,x):
        base=nn.functional.interpolate(x,scale_factor=8,mode='bicubic',align_corners=False)
        e1=self.e1(base); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
        d3=self.u3(self.bot(e4),e3); d2=self.u2(d3,e2); d1=self.u1(d2,e1)
        return base+self.out(d1)

def spectrum(u,lx,ly):
    n=u.shape[0]; kx=2*np.pi*np.fft.fftfreq(n,d=lx/n); ky=2*np.pi*np.fft.fftfreq(n,d=ly/n)
    KX,KY=np.meshgrid(kx,ky,indexing='ij'); km=np.sqrt(KX*KX+KY*KY)
    p=np.abs(np.fft.fft2(u))**2/n**4; dk=min(2*np.pi/lx,2*np.pi/ly); sh=np.rint(km/dk).astype(int)
    ps=np.bincount(sh.ravel(),weights=p.ravel()); ct=np.bincount(sh.ravel()); k=np.arange(len(ps))*dk; m=ct>0
    return k[m],ps[m]

def rel_l2(a,b): return float(np.linalg.norm((b-a).ravel())/np.linalg.norm(a.ravel()))

def save_img(u,title,path,lx,ly):
    fig,ax=plt.subplots(figsize=(6.3,5.2)); im=ax.imshow(u.T,origin='lower',extent=(0,lx,0,ly),aspect='equal'); ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y'); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def save_spec(name,dns,ff,fc,uf,rd,coarse,lx,ly,path):
    S={'DNS':spectrum(dns,lx,ly),'Filtered 256':spectrum(ff,lx,ly),'Filtered 32':spectrum(fc,lx,ly),'U-Net filtered 256':spectrum(uf,lx,ly),'Reconstructed DNS':spectrum(rd,lx,ly)}
    fig,ax=plt.subplots(figsize=(7.4,5.6))
    for lab,(k,p) in S.items():
        m=(k>0)&(p>0)
        if lab=='Filtered 32': ax.loglog(k[m],p[m],linestyle='None',marker='o',markersize=4,label=lab)
        elif lab=='U-Net filtered 256': ax.loglog(k[m],p[m],'--',linewidth=1.8,label=lab)
        else: ax.loglog(k[m],p[m],linewidth=1.8,label=lab)
    kn=min(np.pi/(lx/coarse),np.pi/(ly/coarse)); ax.axvline(kn,linestyle=':',linewidth=1.4,label=f'{coarse}-grid Nyquist')
    ax.set_title(f'{name}: isotropic spectrum comparison'); ax.set_xlabel(r'$k$'); ax.set_ylabel(r'Shell power $P(k)$'); ax.set_ylim(bottom=10e-10); ax.grid(True,which='both',alpha=.3); ax.legend(numpoints=1); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input',type=Path); ap.add_argument('--output-dir',type=Path,default=Path('germano_unet_single_output')); ap.add_argument('--fields',nargs=3,default=['U','V','P']); ap.add_argument('--coarse-n',type=int,default=32); ap.add_argument('--lx',type=float,default=2*np.pi); ap.add_argument('--ly',type=float,default=2*np.pi)
    g=ap.add_mutually_exclusive_group(); g.add_argument('--alpha',type=float); g.add_argument('--alpha-factor',type=float,default=1.0)
    ap.add_argument('--epochs',type=int,default=3000); ap.add_argument('--learning-rate',type=float,default=2e-4); ap.add_argument('--base-channels',type=int,default=24); ap.add_argument('--lambda-dns',type=float,default=.05); ap.add_argument('--lambda-spectrum',type=float,default=.02); ap.add_argument('--print-every',type=int,default=100); ap.add_argument('--seed',type=int,default=7)
    a=ap.parse_args(); set_seed(a.seed); a.output_dir.mkdir(parents=True,exist_ok=True)
    z=np.load(a.input); dns=np.stack([np.asarray(z[k],dtype=np.float32) for k in a.fields]); n=dns.shape[-1]; delta=a.lx/n; alpha=a.alpha if a.alpha is not None else a.alpha_factor*delta
    ff=np.stack([germano_filter(x,alpha,a.lx,a.ly) for x in dns]).astype(np.float32); fc=np.stack([spectral_resize(x,a.coarse_n) for x in ff]).astype(np.float32)
    mean=ff.mean((1,2),keepdims=True); std=np.maximum(ff.std((1,2),keepdims=True),1e-8); x=torch.from_numpy(((fc-mean)/std)[None]); y=torch.from_numpy(((ff-mean)/std)[None]); dt=torch.from_numpy(dns[None]); mt=torch.from_numpy(mean[None]); st=torch.from_numpy(std[None])
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); x=x.to(dev); y=y.to(dev); dt=dt.to(dev); mt=mt.to(dev); st=st.to(dev); model=ResidualUNetSR(3,a.base_channels).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-6)
    hist=[]; best=1e99; ck=a.output_dir/'best_unet.pt'; print('Device:',dev); print('alpha=',alpha,' alpha/delta=',alpha/delta)
    for ep in range(1,a.epochs+1):
        opt.zero_grad(set_to_none=True); pn=model(x); pf=pn*st+mt; pd=inverse_germano_torch(pf,alpha,a.lx,a.ly)
        lf=nn.functional.l1_loss(pn,y); ld=nn.functional.l1_loss(pd,dt); ph=torch.fft.rfft2(pn,dim=(-2,-1)); th=torch.fft.rfft2(y,dim=(-2,-1)); ls=nn.functional.l1_loss(torch.log1p(torch.abs(ph)),torch.log1p(torch.abs(th))); loss=lf+a.lambda_dns*ld+a.lambda_spectrum*ls
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); row={'epoch':ep,'total':float(loss.detach()),'filtered_l1':float(lf.detach()),'dns_l1':float(ld.detach()),'spectrum':float(ls.detach())}; hist.append(row)
        if row['total']<best: best=row['total']; torch.save({'model':model.state_dict(),'epoch':ep,'loss':best},ck)
        if ep==1 or ep%a.print_every==0: print(ep,row)
    B=torch.load(ck,map_location=dev); model.load_state_dict(B['model']); model.eval()
    with torch.no_grad(): pn=model(x); pf=(pn*st+mt)[0].cpu().numpy(); pd=inverse_germano_torch(pn*st+mt,alpha,a.lx,a.ly)[0].cpu().numpy()
    np.savez_compressed(a.output_dir/'unet_single_image_results.npz',dns=dns,filtered_256=ff,filtered_32=fc,unet_filtered_256=pf,unet_reconstructed_dns_256=pd,alpha=np.array(alpha),delta=np.array(delta),field_names=np.array(a.fields))
    with (a.output_dir/'training_history.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=hist[0].keys()); w.writeheader(); w.writerows(hist)
    metrics={}
    for i,name in enumerate(a.fields):
        metrics[name]={'filtered_relative_l2':rel_l2(ff[i],pf[i]),'dns_relative_l2':rel_l2(dns[i],pd[i])}
        save_img(pf[i],f'{name}: U-Net reconstructed filtered field',a.output_dir/f'{name}_unet_filtered_256.png',a.lx,a.ly); save_img(pd[i],f'{name}: inverse-filtered U-Net reconstruction',a.output_dir/f'{name}_unet_reconstructed_dns_256.png',a.lx,a.ly); save_img(pd[i]-dns[i],f'{name}: DNS reconstruction error',a.output_dir/f'{name}_unet_dns_error.png',a.lx,a.ly); save_spec(name,dns[i],ff[i],fc[i],pf[i],pd[i],a.coarse_n,a.lx,a.ly,a.output_dir/f'{name}_unet_spectrum_comparison.png')
    meta={'input':str(a.input),'fields':a.fields,'alpha':alpha,'delta':delta,'alpha_over_delta':alpha/delta,'epochs':a.epochs,'best_epoch':int(B['epoch']),'best_loss':float(B['loss']),'metrics':metrics}; (a.output_dir/'metadata.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(metrics,indent=2)); print('Results:',a.output_dir)
if __name__=='__main__': main()
