from pylab import *
import numpy as np
import scipy as sp
import joblib
import event
from tqdm import tqdm

import util


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
            fig,axs=subplots(2,2,figsize=(12,9))
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
            fig.legend(handles_unique, labels_unique, loc=(0.65,0.1),framealpha=1,ncol=4, fontsize=9)        
        axs[3].axis("off")
        return fig
    
class processing:
    @staticmethod
    def process_results(data_recon, Tree_truth, Tree_digi, cut_track_TrackNHitsMin=4):
        cut_track_TrackNHitsMin = 4
        truth = []
        truth_reconstructable = []
        
        data = {}
        data["Entry"]=[]
        data["Event_ndigi"]=[]
        data["Event_ntrack_recon"]=[]
        data["Event_ntrack_reconstructible"]=[]
        data["Event_nvertex_recon"]=[]
        data["Vertex_ntrack_reconstructible"]=[]
        data["Vertex_ntrack_reconstructed"]=[]
        data[f"Vertex_truth"]=[]
        data[f"Vertex_recon"]=[]
        data["Vertex_counts"]=[]
        data["Vertex_unc"] = []
        data["Vertex_cov"] = []
        data["Vertex_chi2"]=[]
        data["Vertex_chi2_prob"]=[]

        # for i in tqdm(range( Tree.GetEntries())):
        entry_range = [0,Tree_truth.GetEntries()]
        for i in tqdm(range(*entry_range)):

            Tree_digi.GetEntry(i)
            Tree_truth.GetEntry(i)
            tracks = data_recon["tracks"][i]
            vertices = data_recon["vertices"][i]
            
            # Calculate reconstrucible tracks/vertices
            ids=np.array(util.c2list(Tree_digi.Digi_track_id))
            ys = np.array(util.c2list(Tree_digi.Digi_y))
            zs = np.array(util.c2list(Tree_digi.Digi_z))
            n_reconstructible=0
            n_reconstructible2=0
            for g4track_id in range(0,200):
                mask = (ids==g4track_id) & (ys>8500) & (zs>7000)
                if len(np.unique(ys[mask]))>=cut_track_TrackNHitsMin:
                    n_reconstructible+=1
            for g4track_id in np.unique(ids):
                if len(np.unique(ys[ids==g4track_id]))>=cut_track_TrackNHitsMin:
                    n_reconstructible2+=1     
                    
            n_vertex_recon=len(vertices)
                    
            data["Entry"].append(i)
            data["Event_ndigi"].append(Tree_digi.Digi_x.size())
            data["Event_ntrack_recon"].append(len(tracks))
            data["Event_ntrack_reconstructible"].append(n_reconstructible2)
            data["Event_nvertex_recon"].append(n_vertex_recon)                
            data["Vertex_ntrack_reconstructible"].append(n_reconstructible)
                
        
            # Vertex========================================================
            if n_vertex_recon==0:
                Vertex_truth = [-99999,-99999,-99999,-99999]
                Vertex_truth_direction = [1,0,0]
                Vertex_recon = [99999,99999,99999,99999]
                Vertex_recon_unc = [-9990,-9990,-9990,-9990]
                Vertex_chi2 = -1
                Vertex_chi2_prob = -1
                Vertex_ntrack = -1
                Vertex_ntrack_reconstructible = -1
                Vertex_tracks_ndigi = [-1]
                Vertex_tracks_purity  = [-1]
                Vertex_tracks_g4ids  = [-1]
                Vertex_tracks_pdgids  = [-1]
                Vertex_tracks_chi2  = [-1]
                Vertex_tracks_chi2_prob  = [-1]
                
            else:
                Vertex_truth = [Tree_truth.GenParticle_y.at(1)*0.1, -Tree_truth.GenParticle_z.at(1)*0.1 + 8547, Tree_truth.GenParticle_x.at(1)*0.1, 0]
                
                # Select 1 vertex
                recon_truth_dist = []
                for iv in range(n_vertex_recon):
                    vrecon = np.array(vertices[iv][:4])
                    recon_truth_dist.append(np.linalg.norm((vrecon - Vertex_truth)[:3]))
                iv1 = int(np.argmin(recon_truth_dist)) # Select the one closest to truth
                # iv2 = int(np.argmax(recon_ntracks))    # Select the one with most tracks
                
                
                # Select the vertex closest to truth
                iv = iv1


                # Get recon info
                Vertex_recon =  np.array(vertices[iv][:4])
                Vertex_ntrack = len(vertices[iv].tracks)
                Vertex_recon_cov = vertices[iv].cov
                Vertex_recon_unc = np.sqrt(np.diag(vertices[iv].cov))
                Vertex_chi2 = vertices[iv].chi2
                Vertex_chi2_prob = 1-sp.stats.chi2.cdf(vertices[iv].chi2, 3*Vertex_ntrack-4)
                
                    
                    
            data["Vertex_ntrack_reconstructed"].append(Vertex_ntrack)
            data["Vertex_truth"].append(Vertex_truth)
            data["Vertex_recon"].append(Vertex_recon)
            # data["Vertex_cov"].append(Vertex_recon_cov)
            data["Vertex_unc"].append(Vertex_recon_unc)
            data["Vertex_counts"].append(n_vertex_recon)
            data["Vertex_chi2"].append(Vertex_chi2)
            data["Vertex_chi2_prob"].append(Vertex_chi2_prob)


        for key in data:
            data[key] = np.array(data[key])

        data["Vertex_residual_xyzt"] = data["Vertex_truth"]-data["Vertex_recon"]
        
        data["Vertex_residual_r"] = np.linalg.norm(data["Vertex_residual_xyzt"][:,:3].tolist(),axis=1) # Total position residual
        Vertices_truth_direction_unit = np.array([np.array(Vertex_truth[:3])/np.linalg.norm(Vertex_truth[:3]) for Vertex_truth in data["Vertex_truth"]])
        data["Vertex_residual_axial"]  = np.sum((data["Vertex_residual_xyzt"][:, :3]*Vertices_truth_direction_unit).tolist(), axis=1)
        data["Vertex_residual_trans"]  = np.sqrt(data["Vertex_residual_r"]**2 - data["Vertex_residual_axial"]**2)
        
        return data