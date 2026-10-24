"""Generate native editable Context v2 graph and matching PNG from one graph specification."""
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def main():
    out=Path('docs/assets');out.mkdir(parents=True,exist_ok=True)
    nodes=[('input',0,2,'MRI [B,1,128,128,128]'),('e1',2,2,'Context encoder 1\nb channels'),('e2',4,2,'Context encoder 2\n2b channels'),('e3',6,2,'Context encoder 3\n4b channels'),('b',8,2,'Context bottleneck\n8b channels'),('d3',8,4,'Upsample + Conv / concat\nContext decoder 3: 4b'),('d2',6,4,'Upsample + Conv / concat\nContext decoder 2: 2b'),('d1',4,4,'Upsample + Conv / concat\nContext decoder 1: b'),('h3',8,6,'Auxiliary 1x1 logits\n1/4 resolution'),('h2',6,6,'Auxiliary 1x1 logits\n1/2 resolution'),('h1',4,6,'Final 1x1 logits\nFull resolution'),('fusion',2,6,'SUM logits fusion\nUpsample auxiliary logits'),('output',0,6,'6-class output\nTrain + inference')]
    edges=[('input','e1',''),('e1','e2','MaxPool /2'),('e2','e3','MaxPool /2'),('e3','b','MaxPool /2'),('b','d3','x2'),('d3','d2','x2'),('d2','d1','x2'),('e3','d3','skip'),('e2','d2','skip'),('e1','d1','skip'),('d3','h3',''),('d2','h2',''),('d1','h1',''),('h3','fusion','resize'),('h2','fusion','resize'),('h1','fusion',''),('fusion','output','')]
    root=ET.Element('mxfile');diagram=ET.SubElement(root,'diagram',name='Context v2');model=ET.SubElement(diagram,'mxGraphModel');cells=ET.SubElement(model,'root');ET.SubElement(cells,'mxCell',id='0');ET.SubElement(cells,'mxCell',id='1',parent='0')
    positions={}
    fig,ax=plt.subplots(figsize=(16,9));ax.set(xlim=(-0.2,10),ylim=(8,0));ax.axis('off')
    for key,x,y,label in nodes:
        positions[key]=(x,y)
        cell=ET.SubElement(cells,'mxCell',id=key,value=label,vertex='1',parent='1',style='rounded=1;whiteSpace=wrap;html=0;fillColor=#dae8fc;strokeColor=#6c8ebf;')
        ET.SubElement(cell,'mxGeometry',x=str(x*150),y=str(y*100),width='260',height='85',attrib={'as':'geometry'})
        ax.add_patch(FancyBboxPatch((x,y),1.7,.7,boxstyle='round,pad=0.03',facecolor='#dae8fc',edgecolor='#6c8ebf'));ax.text(x+.85,y+.35,label,ha='center',va='center',fontsize=9)
    for i,(a,b,label) in enumerate(edges):
        cell=ET.SubElement(cells,'mxCell',id=f'edge{i}',value=label,edge='1',parent='1',source=a,target=b,style='edgeStyle=orthogonalEdgeStyle;endArrow=block;dashed='+('1' if label=='skip' else '0')+';')
        ET.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
        x,y=positions[a];u,v=positions[b]
        start=(x+.85,y+.7) if v>y else ((x,y+.35) if u<x else (x+1.7,y+.35))
        end=(u+.85,v) if v>y else ((u+1.7,v+.35) if u<x else (u,v+.35))
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='->',mutation_scale=12,linestyle='--' if label=='skip' else '-',color='#44546a',connectionstyle='arc3,rad=0.15' if label in {'skip','resize'} else 'arc3,rad=0'))
        if label and label not in {'resize','skip'}:ax.text((start[0]+end[0])/2,(start[1]+end[1])/2-.1,label,fontsize=7)
    note='Context block: InstanceNorm3d (affine) > LeakyReLU > Conv3d > Dropout3d(0.3) > InstanceNorm3d > LeakyReLU > Conv3d + residual shortcut\nUpsample: trilinear x2 > Conv3d > InstanceNorm3d > LeakyReLU. Dashed arrows: encoder skip concatenation. b=16.\nDeep supervision (training only): separate losses at both auxiliary heads; fused full-resolution logits receive the primary loss.'
    ax.text(0,7.25,note,fontsize=9,va='top')
    cell=ET.SubElement(cells,'mxCell',id='note',value=note,vertex='1',parent='1',style='text;whiteSpace=wrap;html=0;');ET.SubElement(cell,'mxGeometry',x='0',y='750',width='1500',height='120',attrib={'as':'geometry'})
    ax.set_title('ImprovedUNet3D: improved_context_v2 (src/modules.py)',fontsize=16)
    ET.ElementTree(root).write(out/'model_architecture_context_v2.drawio',encoding='utf-8',xml_declaration=True)
    fig.savefig(out/'model_architecture_context_v2.png',dpi=160,bbox_inches='tight');plt.close(fig)
if __name__=='__main__': main()
