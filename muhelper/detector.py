##################################################################################################
##################################################################################################

#THIS FILE CONTAINS DETECTOR GEOMETRY INFORMATION#

##################################################################################################

import numpy as np
import scipy as sp

class Detector:
    # Use cm as unit
    cm = 1      

    OutBoxLimits =[  [-1950.0, 1950.0],  [8547.0, 8547.0+1754.200],  [7000.0, 7000.0+3900]    ]
    BoxMargin = 50
    BoxLimits   = [  [-1950.0 + BoxMargin, 1950.0 - BoxMargin],  [8547.0 +84.6 + BoxMargin, 8547.0 + 1509.400 - BoxMargin],  [7000.0 + BoxMargin, 7000.0+3900 - BoxMargin]    ]
    WallLimits  = [ BoxLimits[0], [BoxLimits[1][0],BoxLimits[1][0] + 2000.0 ] ] # [x,y] [cm,cm]

    scintillator_height_all = 1.6 # 2cm +0.3*2Al case
    # 2023-04-25 Tom: changing to 6 layers. New numbers pull from simulation

    
# y =  3.000,             8544.000
# y =  84.600,            8462.400
# y =  1346.200,          7200.800
# y =  1427.800,          7119.200
# y =  1509.400,          7037.600
# y =  1591.000,          6956.000
# y =  1672.600,          6874.400
# y =  1754.200,          6792.800

    LayerYLims= [[8547.000,8547.000+scintillator_height_all],
                 [8628.600,8628.600+scintillator_height_all],
                 [9890.200,9890.200+scintillator_height_all],
                 [9971.800,9971.800+scintillator_height_all],
                 [10053.400,10053.400+scintillator_height_all],
                 [10135.000,10135.000+scintillator_height_all]]    
        
    
    
    y_floor = LayerYLims[2][1]
    z_wall = BoxLimits[2][0] + 3 # [cm] add 3 cm to account for wall width

    ModuleXLims = [ [-4950. + 1000.*n, -4050. + 1000*n] for n in range(10) ]
    ModuleZLims = [ [7000.  + 1000.*n,  7900. + 1000*n] for n in range(10) ]

    #               x range              y range             z range

    def __init__(self):
        # print("Detector Constructed")
        self.time_resolution=1e-9 #1ns=1e-9s
        
        self.n_top_layers = 5
        self.x_edge_length = 39.0 # decrease from 99->39 -- Tom
        self.y_edge_length = 39.0 # decrease from 99->39 -- Tom
        self.n_modules = 4

        self.x_displacement = 70.0
        self.y_displacement = -49.5
        self.z_displacement = 6001.5

        self.layer_x_edge_length = 9.0
        self.layer_y_edge_length = 9.0

        self.scint_x_edge_length = 4.5
        self.scint_y_edge_length = 0.045
        self.scintillator_length = self.scint_x_edge_length
        self.scintillator_width = self.scint_y_edge_length
        self.scintillator_height = 0.02
        self.scintillator_casing_thickness = 0.003

        self.steel_height = 0.03

        self.air_gap = 30

        self.layer_spacing = 0.8
        self.layer_count   = 8

        self.module_x_edge_length = 9.0
        self.module_y_edge_length = 9.0
        self.module_case_thickness = 0.02

        self.full_layer_height =self.layer_w_case = self.scintillator_height + 2*self.scintillator_casing_thickness

        self.layer_z_displacement = [                          0.5*self.layer_w_case,
                                        1*self.layer_spacing + 1.5*self.layer_w_case,
                                    5 +                        0.5*self.layer_w_case,
                                    5 + 1*self.layer_spacing + 1.5*self.layer_w_case,
                                    5 + 2*self.layer_spacing + 2.5*self.layer_w_case,
                                    5 + 3*self.layer_spacing + 3.5*self.layer_w_case,
                                    5 + 4*self.layer_spacing + 4.5*self.layer_w_case,
                                    5 + 5*self.layer_spacing + 5.5*self.layer_w_case]

        self.module_x_displacement = [i*(self.module_x_edge_length + 1.0) -0.5 * self.x_edge_length + 0.5*self.module_x_edge_length for i in range(self.n_modules)]
        self.module_y_displacement = [i*(self.module_y_edge_length + 1.0) -0.5 * self.y_edge_length + 0.5*self.module_y_edge_length for i in range(self.n_modules)]
        pass
    def xLims(self):
        return self.BoxLimits[0]

    def yLims(self):
        return self.BoxLimits[1]

    def zLims(self):
        return self.BoxLimits[2]

    def LayerY(self, n):
        return self.LayerYLims[n]

    def LayerYMid(self, n):
        return (self.LayerYLims[n][0] + self.LayerYLims[n][1])/2.

    def numLayers(self):
        return len(self.LayerYLims)

    def DrawColor(self):
        return "tab:gray"

    def inLayer(self, yVal):
        for layerN, layerLims in enumerate(self.LayerYLims):
            if yVal > layerLims[0] and yVal < layerLims[1]:
                return layerN
        return -1

    def nextLayer(self, yVal):
        for n in range(len(self.LayerYLims)-1):
            if yVal > self.LayerYLims[n][1] and yVal < self.LayerYLims[n+1][0]:
                return n+1
        return 999


    def inLayer_w_Error(self, yVal, yErr):
        for layerN, layerLims in enumerate(self.LayerYLims):

            lower = yVal - yErr
            upper = yVal + yErr

            if lower < layerLims[0] and upper > layerLims[1]:
                return layerN

            if lower < layerLims[0] and upper > layerLims[0]:
                return layerN

            if lower < layerLims[1] and upper > layerLims[1]:
                return layerN


        return -1

    def inBox(self, x, y, z):
        if x > self.xLims()[0] and x < self.xLims()[1]:
            if y > self.yLims()[0] and y < self.yLims()[1]:
                if z > self.zLims()[0] and z < self.zLims()[1]:
                    return True
        return False


    #determine number of layers a track goes through
    def nLayers(self, x0, y0, z0, vx, vy, vz):
        count = 0
        for n in range(len(self.LayerYLims)):
            layerY = self.LayerYMid(n)
            if (layerY-y0)/vy < 0:
                continue
            else:
                dt = (layerY - y0)/vy

                x1 = x0 + dt*vx
                z1 = y0 + dt*vz

                if inBox(x1, layerY, z1):
                    count += 1

        return count

    #determine number of SENSITIVE layers a track goes through
    def nSensitiveLayers(self, x0, y0, z0, vx, vy, vz):
        count = 0
        for n in range(len(self.LayerYLims)):
            layerY = self.LayerYMid(n)
            if (layerY-y0)/vy < 0:
                continue
            else:
                dt = (layerY - y0)/vy

                x1 = x0 + dt*vx
                z1 = y0 + dt*vz

                if self.inSensitiveElement(x1, layerY, z1):
                    count += 1

        return count


    ##get points inside the detector for reconstructed track
    def RecoTrackPoints(self, x0, y0, z0, vx, vy, vz):
        x, y, z = [], [], []
        _x, _y, _z = x0, y0, z0

        time_spacing = 0.1 #ns

        while self.inBox(_x, _y, _z):
            x.append(_x)
            y.append(_y)
            z.append(_z)

            _x += vx*time_spacing
            _y += vy*time_spacing
            _z += vz*time_spacing


        return x, y, z


    def FindIntercept(self, x0, y0, z0, vx, vy, vz):
        #x, y, z = [], [], []
        #_x, _y, _z = x0, y0, z0


        pos = np.array([x0, y0, z0])
        v = np.array([vx, vy, vz])
        t = []

        #print("v is {}".format(v))

        for i in range(len(pos)):
            if v[i] > 0:
                t.append((self.BoxLimits[i][1] - pos[i]) / v[i])

            else:
                t.append((self.BoxLimits[i][0] - pos[i]) / v[i])

        t_intercept = np.amin(t)
        #print("the ts are {}".format(t))
        #print("t intercept is {}".format(t_intercept))

        return (pos + t_intercept * v)


    def inModuleX(self, xVal):
        for moduleN, moduleLims in enumerate(self.ModuleXLims):
            if xVal > moduleLims[0] and xVal < moduleLims[1]:
                return moduleN
        return -1

    def inModuleZ(self, zVal):
        for moduleN, moduleLims in enumerate(self.ModuleZLims):
            if zVal > moduleLims[0] and zVal < moduleLims[1]:
                return moduleN
        return -1

    def inSensitiveElement(self, x, y, z):
        if self.inLayer(y) >= 0:
            if self.inModuleX(x) >= 0:
                if self.inModuleZ(z) >= 0:
                    return True
        return False

    
    
class Layer():
    """
    Layer index: 0-9. floor:0-1, mid track: 2-3, top layers: 4-9
    layer 2 is more accurate along the beamline.
    
    """
    detector=Detector()
    optic_fiber_n = 1.58
    
    @staticmethod
    def width(index): 
        detector=Layer.detector
        if index%2==0:
            return detector.scintillator_width, detector.scintillator_length 
        else:
            return detector.scintillator_length, detector.scintillator_width
    @staticmethod
    def uncertainty(index):
        detector=Layer.detector
        if index%2==1:
            return  [detector.scintillator_width/np.sqrt(12.),\
                    detector.scintillator_height/np.sqrt(12.),\
                    detector.time_resolution*(sp.constants.c/Layer.optic_fiber_n)/np.sqrt(2)]
        else:
            return  [detector.time_resolution*(sp.constants.c/Layer.optic_fiber_n)/np.sqrt(2), \
                    detector.scintillator_height/np.sqrt(12.), \
                    detector.scintillator_width/np.sqrt(12.) ]    