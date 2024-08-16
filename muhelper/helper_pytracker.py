from pylab import *
import numpy as np
import scipy as sp
import joblib
# import event
from tqdm import tqdm

from . import util
from . import visualization as vis # This is the original visualization code. still partially useful


class visualization:    
    @staticmethod
    def plot_digi(hits, fig=None, color="red", alpha=0.3, label="digitized"):
        """
        Function to plot the digitization of one event
        Tom Ren, 2023.2
        
        INPUT:
        """
        # Prepare the canvas
        if fig is None:
            fig,axs=subplots(2,2,figsize=(12,6))
            axs=axs.flatten().tolist()
        else:
            axs=fig.axes    
            
        
        # plots hits
        x,y,z,t = [],[],[],[]
        xe,ye,ze,te = [],[],[],[]
        for i in range(len(hits)):
            x.append(hits[i].x)
            y.append(hits[i].y)
            z.append(hits[i].z)
            t.append(hits[i].t)
            xe.append(hits[i].x_err)
            ye.append(hits[i].y_err)
            ze.append(hits[i].z_err)
            te.append(hits[i].t_err)
            
        axs[0].errorbar(z,y,xerr=ze,yerr=ye, fmt=".",capsize=2, color=color, alpha=alpha, label=label)
        axs[1].errorbar(x,y,xerr=xe,yerr=ye, fmt=".",capsize=2, color=color, alpha=alpha, label=label)
        axs[2].errorbar(z,x,xerr=ze,yerr=xe, fmt=".",capsize=2, color=color, alpha=alpha, label=label)
        # axs[3].errorbar(t,y,xerr=te,yerr=ye, fmt=".",capsize=2, color=color, alpha=alpha, label=label)
        
    
        axs[0].set_xlabel("z (beamline) [cm]")
        axs[0].set_ylabel("y (up) [cm]")
        axs[1].set_xlabel("x (other) [cm]")
        axs[1].set_ylabel("y (up) [cm]")
        axs[2].set_xlabel("z (beamline) [cm]")
        axs[2].set_ylabel("x (other) [cm]")
        # axs[3].set_xlabel("t (time) [ns]")
        # axs[3].set_ylabel("y (up) [cm]")    
        # Put legend in the last grid
        # handles, labels = axs[0].get_legend_handles_labels()
        # fig.legend(handles, labels, loc=7,framealpha=1,ncol=1, fontsize=9)
        # fig.subplots_adjust(right=0.75)    # axs[3].axis("off")
        # fig.tight_layout()
        return fig

    @staticmethod
    def plot_track(tracks, fig, linestyle=":", color="k"):
        axs=fig.axes   
        # Plot the pyTracker result
        for itrack,track in enumerate(tracks):
            x,y,z,t = np.array(track.hits_filtered).T
            track_label = f"Track {itrack}"#pdg_name(pid)
            axs[0].plot(z,y,marker=".",linewidth=1,markersize=4,label=track_label, linestyle=linestyle, color=color)
            axs[1].plot(x,y,marker=".",linewidth=1,markersize=4,label=track_label, linestyle=linestyle, color=color)
            axs[2].plot(z,x,marker=".",linewidth=1,markersize=4,label=track_label, linestyle=linestyle, color=color)    
            # axs[3].plot(t,y,marker=".",linewidth=1,markersize=4,label=track_label, linestyle=linestyle, color=color)    
            axs[0].annotate(f"{track.ind}", (z[0], y[0]), color="r", fontsize=9)
            axs[1].annotate(f"{track.ind}", (x[0], y[0]), color="r", fontsize=9)
        return fig

    @staticmethod
    def plot_vertex(vertices, tracks, fig):
        axs=fig.axes   
        for iv, vertex in enumerate(vertices):
            x,y,z,t = np.array(vertex[:4])
            axs[0].plot(z,y,marker="+",linewidth=2,markersize=7,label=f"Vertex {iv}", linestyle=":", color=f"C{iv}", zorder=200)
            axs[1].plot(x,y,marker="+",linewidth=2,markersize=7,label=f"Vertex {iv}", linestyle=":", color=f"C{iv}", zorder=200)
            axs[2].plot(z,x,marker="+",linewidth=2,markersize=7,label=f"Vertex {iv}", linestyle=":", color=f"C{iv}", zorder=200)   
            # axs[3].plot(t,y,marker="+",linewidth=2,markersize=7,label=f"Vertex {iv}", linestyle=":", color=f"C{iv}", zorder=200)   
            for itrack in vertex.tracks:
                track=tracks[itrack]
                xt,yt,zt,tt = np.array(track[:4])
                axs[0].plot([z,zt],[y,yt], color=f"C{iv}",linestyle=":")
                axs[1].plot([x,xt],[y,yt], color=f"C{iv}",linestyle=":")
                axs[2].plot([z,zt],[x,xt], color=f"C{iv}",linestyle=":")
                # axs[3].plot([t,tt],[y,yt], color=f"C{iv}",linestyle=":")
        return fig


    @staticmethod
    def plot_recon(hits, tracks, vertices, fig=None, linestyle=":", color="k", make_legend=True):
        hits_used_inds = [ind for track in tracks for ind in track.hits]
        hits_used=[]
        hits_unused=[]
        for hit in hits:
            if hit.ind in hits_used_inds:
                hits_used.append(hit)
            else:
                hits_unused.append(hit)

        fig = visualization.plot_digi(hits_unused, fig=fig, color="C0", label="Digitized, unused")
        fig = visualization.plot_digi(hits_used, fig=fig, color="red", label="Digitized, used")
        fig = visualization.plot_track(tracks, fig, linestyle=linestyle, color=color)
        fig = visualization.plot_vertex(vertices, tracks, fig)
        axs=fig.axes
        
        if make_legend:
            handles, labels = axs[0].get_legend_handles_labels()
            labels_unique, labels_inds = np.unique(labels, return_index=True)
            handles_unique=[handles[i] for i in labels_inds]
            fig.legend(handles_unique, labels_unique, loc=(0.52,0.1),framealpha=1,ncol=4, fontsize=9)        
        axs[3].axis("off")
        return fig
    

    
       

    @staticmethod
    def drawdet(direction, axis=None, layer_height_vis=0.2, alpha=0.1, set_lim=False, unit="m"):

        # Manually define the corner of all layers
        det_x_width=39
        det_z_width=39
        det_z_height = 16
        det_x_offset = 0
        det_y_offset = 85.47
        det_z_offset = 70
        layer_gap=0.8
        air_gap = 12.6
        wall_height = layer_gap+air_gap
        
        layers_xyz_cms={
            0: [[-det_x_width/2, 0, det_z_offset-layer_gap],  [det_x_width/2,  wall_height, det_z_offset-layer_gap]],
            1: [[-det_x_width/2, 0, det_z_offset],  [det_x_width/2,  wall_height, det_z_offset]],
            2: [[-det_x_width/2, 0, det_z_offset],          [det_x_width/2, 0, det_z_offset+det_x_width]],
            3: [[-det_x_width/2, layer_gap, det_z_offset],[det_x_width/2, layer_gap, det_z_offset+det_x_width]],

            4: [[-det_x_width/2, layer_gap*1+air_gap, det_z_offset],[det_x_width/2, layer_gap*1+air_gap, det_z_offset+det_z_width]],
            5: [[-det_x_width/2, layer_gap*2+air_gap, det_z_offset],[det_x_width/2, layer_gap*2+air_gap, det_z_offset+det_z_width]],
            6: [[-det_x_width/2, layer_gap*3+air_gap, det_z_offset],[det_x_width/2, layer_gap*3+air_gap, det_z_offset+det_z_width]],
            7: [[-det_x_width/2, layer_gap*4+air_gap, det_z_offset],[det_x_width/2, layer_gap*4+air_gap, det_z_offset+det_z_width]],

            8: [[-det_x_width/2, air_gap+layer_gap-9, det_z_offset+det_z_width],  [det_x_width/2,  air_gap+layer_gap, det_z_offset+det_z_width]],
            9: [[-det_x_width/2, air_gap+layer_gap-9, det_z_offset+det_z_width+layer_gap*1],  [det_x_width/2,  air_gap+layer_gap, det_z_offset+det_z_width+layer_gap*1]],
            10: [[-det_x_width/2, air_gap+layer_gap-9, det_z_offset+det_z_width+layer_gap*2],  [det_x_width/2,  air_gap+layer_gap, det_z_offset+det_z_width+layer_gap*2]],
            11: [[-det_x_width/2, air_gap+layer_gap-9, det_z_offset+det_z_width+layer_gap*3],  [det_x_width/2,  air_gap+layer_gap, det_z_offset+det_z_width+layer_gap*3]],

        }
        
        # gaps_xyz_cms={
        #     1: [[-det_x_width/2]],
        # }
        for key in layers_xyz_cms:
            layers_xyz_cms[key][0][1]+=det_y_offset
            layers_xyz_cms[key][1][1]+=det_y_offset    


        if axis is None: axis=plt.gca()

        maps={0:[2,1], 1:[0,1], 2:[2,0]}
        inds= maps[direction]
        verts=[] # vertices of polygons
        map_unit={"m":1, "cm":100, "mm": 1000}

        for i in layers_xyz_cms:
            det_corner1, det_corner2 = layers_xyz_cms[i]
            layerX = np.array([det_corner1[inds[0]],
                      det_corner1[inds[0]],
                      det_corner2[inds[0]],
                      det_corner2[inds[0]]], dtype=float)

            layerY = np.array([det_corner1[inds[1]],
                      det_corner2[inds[1]],
                      det_corner2[inds[1]],
                      det_corner1[inds[1]]], dtype=float)

            if det_corner1[inds[0]]==det_corner2[inds[0]]: layerX+=np.array([0,0, layer_height_vis,layer_height_vis])
            if det_corner1[inds[1]]==det_corner2[inds[1]]: layerY+=np.array([0,layer_height_vis,layer_height_vis,0])
            # print(layerX,layerY)
            if i==0: lim_x,lim_y = [min(layerX),max(layerX)], [min(layerY),max(layerY)]
            lim_x = [min(lim_x[0], min(layerX)), max(lim_x[1], max(layerX))]
            lim_y = [min(lim_y[0], min(layerY)), max(lim_y[1], max(layerY))]
            verts.append(np.transpose([layerX, layerY])*map_unit[unit])         

        col = mpl.collections.PolyCollection(verts, alpha=alpha)
        axis.add_collection(col)
        if set_lim:
            axis.set_xlim((lim_x[0]-0.5)*map_unit[unit], (lim_x[1]+0.5)*map_unit[unit])
            axis.set_ylim((lim_y[0]-0.5)*map_unit[unit], (lim_y[1]+0.5)*map_unit[unit])

    @staticmethod
    def plot_det(fig=None, layer_height_vis=0.2, alpha=0.1, set_lim=False, unit="cm"):
        """
        drawdet_all(set_lim=True)
        """
        if fig is None:
            fig,axs=subplots(2,2,figsize=(12,6))
            axs=axs.flatten().tolist()
        else:
            axs=fig.axes     

        for direction in range(3):
            visualization.drawdet(direction, axis=axs[direction], layer_height_vis=layer_height_vis, alpha=alpha, set_lim=set_lim, unit=unit)
            
        return fig
    
    
    

class PlotEvent:
    import ROOT
    def __init__(self, filename_digis, filename_recons, verbose=False):
        self.filename_digis = filename_digis
        self.filename_recons = filename_recons
        self.truth_file = None
        self.truth_filename = None
        self.recon_file = None
        self.recon_filename = None
        self.verbose=verbose
        
    def get_tree(self, tfile):
        Tree = tfile.Get(tfile.GetListOfKeys()[0].GetName())
        def keys(Tree):
            branch_list = [Tree.GetListOfBranches()[i].GetName() for i in range(len(Tree.GetListOfBranches()))]
            return branch_list
        Tree.keys = types.MethodType(keys,Tree) 
        
        return Tree
        
    def set_event_list(self,runs, entries):
        self.runs=runs
        self.entries=entries
        if len(self.runs)!=self.entries:
            raise("Lenght not equal")
            
    def plot_i(self, i):
        return self.plot(self.runs[i], self.entries[i])



    def plot(self, run, entry, fig=None):
        filename_truth = self.filename_digis[run]
        filename_recon = self.filename_recons[run]    
        if not (filename_truth==self.truth_filename and filename_recon==self.recon_filename \
               and self.recon_file is not None):
            # Open truth file (just a wrapper of ROOT.TFile...)
            self.truth_filename = filename_truth
            self.recon_filename = filename_recon
            tfile = PlotEvent.ROOT.TFile.Open(filename_truth)
            self.truth_file  = tfile.Get(tfile.GetListOfKeys()[0].GetName())
            if self.verbose:
                print("Reading reconstruction file", end="")
            self.recon_file  = joblib.load(filename_recon, mmap_mode="r")
            if self.verbose:
                print("   reconstruction file loaded")            
            
        # Get truth information
        self.truth_file.GetEntry(entry)
        # Get recon information
        hits, tracks, vertices = self.recon_file["hits"][entry], self.recon_file["tracks"][entry], self.recon_file["vertices"][entry]  
        
        if fig is None:
            fig,axs=plt.subplots(2,2,figsize=(14,6))
        fig = vis.plot_truth_new(self.truth_file,fig=fig, disp_det_view=False, disp_filereader_vertex=False, disp_first_hit=False,make_legend=False);
        fig = visualization.plot_recon(hits, tracks, vertices, fig=fig, make_legend=True)
        fig = visualization.plot_det(fig=fig,set_lim=True, unit="cm")
        axs = fig.axes    
        
        return fig, axs    