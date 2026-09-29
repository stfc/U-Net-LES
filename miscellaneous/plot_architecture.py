"""Draw the DirectUNetSR architecture for the archived HW 256/32 training run."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[3]
p = json.loads((ROOT / 'training/training_HW_dns256_les32/training_parameters.json').read_text())
assert (p['model'], p['base_channels'], p['scale_factor']) == ('DirectUNetSR', 24, 8)
fig, ax = plt.subplots(figsize=(15, 7.8))
ax.set(xlim=(-.2, 15), ylim=(-.5, 7.8)); ax.axis('off')
colors = {'input':'#E8EDF2', 'conv':'#D7E8FA', 'up':'#D8EFE9', 'fixed':'#F8E8CC'}
def box(x,y,w,h,title,detail,kind='conv'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.09',facecolor=colors[kind],edgecolor='#425466',linewidth=1.1))
    ax.text(x+w/2,y+h*.69,title,ha='center',va='center',fontsize=10,fontweight='bold')
    ax.text(x+w/2,y+h*.29,detail,ha='center',va='center',fontsize=9,linespacing=1.4)
def arrow(points, color='#425466',style='-'):
    for a,b in zip(points[:-2],points[1:-1]): ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.3,linestyle=style)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=12,color=color,lw=1.3,linestyle=style))
ax.text(0,7.5,'DirectUNetSR | HW DNS 256² / LES 32²',fontsize=17,fontweight='bold',color='#20374D')
ax.text(0,7.05,'Tensor dimensions: channels × height × width   •   1,670,738 trainable parameters   •   deterministic network',fontsize=10,color='#425466')
xs=[0,2.1,4.2,6.3,8.4,10.5,12.6]; w=1.8; y=5.25; h=1.1
nodes=[('LES input','3 × 32 × 32\n(n, φ, ζ)','input'),('Stem','ConvBlock\n96 × 32 × 32','conv'),('Coarse decoder','Concat + ConvBlock\n96 × 32 × 32','up'),('Upsample ×2','ConvBlock\n48 × 64 × 64','up'),('Upsample ×2','ConvBlock\n24 × 128 × 128','up'),('Upsample ×2','ConvBlock\n24 × 256 × 256','up'),('Output head','1 × 1 convolution\n2 × 256 × 256','conv')]
for x,(title,detail,kind) in zip(xs,nodes): box(x,y,w,h,title,detail,kind)
for i in [0,2,3,4,5]: arrow([(xs[i]+w,y+.55),(xs[i+1],y+.55)])
arrow([(3.9,5.95),(4.2,5.95)],color='#A15A16')
ax.text(4.05,6.53,'skip: 96 channels',ha='center',fontsize=9,color='#A15A16')
box(2.1,3.05,1.8,1.15,'Encoder','ConvBlock\n192 × 16 × 16')
box(4.2,3.05,1.8,1.15,'Bottleneck','ConvBlock\n192 × 16 × 16')
arrow([(3,5.25),(3,4.2)])
ax.text(2.83,4.7,'AvgPool\n2 × 2',ha='right',va='center',fontsize=9)
arrow([(3.9,3.625),(4.2,3.625)])
arrow([(5.1,4.2),(5.1,5.25)])
ax.text(5.3,4.7,'Bilinear ×2\n192 + 96 = 288 channels',va='center',fontsize=9)
ax.text(8.6,3.9,'ConvBlock = [3 × 3 circular convolution\n→ GroupNorm → GELU] × 2\n\nAll upsampling uses periodic bilinear interpolation.',fontsize=10,va='center',linespacing=1.5)
ax.text(0,2.25,'Differentiable reconstruction and training objective',fontsize=12,fontweight='bold',color='#20374D')
for args in [(0,.65,3.1,1.15,'Predicted filtered fields','(ñ, φ̃) on the 256² mesh','input'),(3.6,.65,3.1,1.15,'Spectral cutoff','exp[−(|k| / 11.3339)⁸]','fixed'),(7.2,.65,3.1,1.15,'Van Cittert inverse','5 iterations; relaxation = 1\nRemove spatial mean','fixed'),(10.8,.65,3.7,1.15,'DNS reconstruction','(n, φ); ζ = Δₕφ\nSecond-order periodic Laplacian','fixed')]: box(*args)
for x in [3.1,6.7,10.3]: arrow([(x,1.225),(x+.5,1.225)])
arrow([(13.5,5.25),(14.75,5.25),(14.75,2.55),(-.12,2.55),(-.12,1.225),(0,1.225)])
ax.text(7.25,.03,'Loss = L1(n, φ) + 0.1 MSE(n, φ) + 0.1 L1(ζ / sζ) + 0.01 MSE(ζ / sζ)   |   sζ = 1.2094',ha='center',fontsize=11,color='#20374D')
out=ROOT/'papers/CPC26/images'
for ext in ['pdf','png']: fig.savefig(out/f'direct_unet_architecture.{ext}',bbox_inches='tight',dpi=220)
plt.close(fig)
