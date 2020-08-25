# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:22:18 2020

@author: Campbell
"""


### NOTE: this code uses the internal geo kernel

import numpy as np
import gmsh
import time
import os
import csv
from scipy import interpolate

gmsh.initialize()


class deLavalNozzleGenerator:
    """
    This class is used for generating computational meshes of de Laval nozzles via
    the Gmsh Python API (https://gmsh.info/) for use within OpenFOAM CFD simulations
    (https://openfoam.org/).
    
    Functions Overview:
        - generateCrossSection: Generates the 2D cross-section geometry of the 
            domain.
        - nozzleWallFunction: Specifies the function describing desired nozzle shape.
        - importNozzleFile: Specifies parameters for importing the nozzle profile
            via wall coordinates provided in an external .csv file
        - generate2DMesh: Generates a 2D structured mesh within the 2D cross-section
        - generateReservoir: Generates a rectangular reservoir at the nozzle inlet
        - generateWedge: Generates a 3D axi-symmetric wedge section
        - generateChannel: Generates a 3D channel nozzle
        - declarePhysicalGroups: Groups and names the mesh boundaries
        - saveMesh(): Saves the generated mesh as well as documents the 
            input parameters used.
       
    Suggested order for running the class functions:
        1. generateCrossSection()
        2. generate2DMesh()
        3. (Optional) generateReservoir()
        4. generateWedge() OR generateChannel()
        5. saveMesh()
        
    Installation Instructions:
        - Gmsh: https://gmsh.info/
        - Gmsh Python API: https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/tutorial/python/README.txt
    
    Potential future additions:
        error handling
        add support for full 3D cylindrical nozzle mesh
        more generalized physical groups
    
    """
    
    
    def __init__(self):
        
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("test")
        
        # Set variables for knowing what functions were run
        self.GEOMETRY   = 0
        self.MESH2D     = 0
        self.WEDGE      = 0
        self.CHANNEL    = 0
        self.RESERVOIR  = 0
        
        # Set class variables
        self.wallPlotMethod = 'file'        # Specify nozzle shape via .csv file
        #self.wallPlotMethod = 'function'   # Specify nozzle shape via piecewise function
        
        
    def generateCrossSection(self,nozzleLength, cavityLength, cavityHeight, numWallPoints=1, lc=1):
        """
        Function to generate the 2D cross-section geometry of the de Laval nozzle.
        
        An external cavity is attached to the nozzle output to simulate the nozzle
        output flow properties. The geometry consists of three sections, the nozzle
        and the upper and lower sections of the attached cavity.  The nozzle is 
        oriented such that the nozzle axis is parallel with the z-axis. The nozzle
        wall shape can be defined within self.nozzleWallFunction or within an
        external .csv file imported in self.importNozzleFile.

        Parameters
        ----------
        nozzleLength : TYPE float
            Length of the de Laval nozzle along the nozzle axis. Must be > 0.
        cavityLength : TYPE float
            Length of the attached external cavity along the nozzle axis. Must
            be > 0.
        cavityHeight : TYPE float
            Height of the attached external cavity.  
            NOTE: This must be greater than the nozzle exit radius
        numWallPoints : TYPE int, optional
            Number of discrete points used to approximate the nozzle wall shape. 
            The default is 1.
            NOTE: this value also represents the number of mesh cells generated
                along the nozzle axis.
        lc : TYPE float, optional
            Curently does nothing. The default is 1.

        Returns
        -------
        None.

        """
        
        
        # Set class variables
        self.nozzleLength = nozzleLength
        self.cavityLength = cavityLength
        self.cavityHeight = cavityHeight
        self.numWallPoints = numWallPoints
        self.lc = lc
        
        # marker that this function has run
        self.GEOMETRY = 1
        
        # generate points and lines for nozzle wall
        pList, lList = self.plotWallShape(nozzleLength, numWallPoints, lc)
        
        # Set axis points for nozzle
        p1 = gmsh.model.geo.addPoint(0,                 0, 0,               lc)
        p2 = gmsh.model.geo.addPoint(0,                 0, nozzleLength,    lc)
        
        # Set points for lower cavity
        p5 = gmsh.model.geo.addPoint(0,                 0, nozzleLength + cavityLength, lc)
        p6 = gmsh.model.geo.addPoint(self.outletRadius, 0, nozzleLength + cavityLength, lc)
        
        # Set points for upper cavity
        p7 = gmsh.model.geo.addPoint(cavityHeight,      0, nozzleLength + cavityLength,   lc)
        p8 = gmsh.model.geo.addPoint(cavityHeight,      0, nozzleLength,                  lc)
        
        
        # Set straight line sections of nozzle
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, pList[-1])
        l4 = gmsh.model.geo.addLine(pList[0], p1)
        
        # Set lines for lower cavity
        l5 = gmsh.model.geo.addLine(p2, p5)
        l6 = gmsh.model.geo.addLine(p5, p6)
        l7 = gmsh.model.geo.addLine(p6, pList[-1])
        
        # Set lines for upper cavity
        l8  = gmsh.model.geo.addLine(p6, p7)
        l9  = gmsh.model.geo.addLine(p7, p8)
        l10 = gmsh.model.geo.addLine(p8, pList[-1])
              
        
        # Create curve loops and surfaces
        # Surface loop for nozzle
        temp = np.append(np.flip(lList),[l4, l1, l2]).tolist()
        gmsh.model.geo.addCurveLoop(temp, 11)
        gmsh.model.geo.addPlaneSurface([11],1)
        
        # Surface loop for lower cavity
        gmsh.model.geo.addCurveLoop([l5, l6, l7, -l2], 12)
        gmsh.model.geo.addPlaneSurface([12], 2)
        
        # Surface loop for upper cavity
        gmsh.model.geo.addCurveLoop([l8, l9, l10, -l7], 13)
        gmsh.model.geo.addPlaneSurface([13], 3)
        
        # set additional class variables
        self.p1 = p1
        self.p2 = p2
        self.p5 = p5
        self.p6 = p6
        self.p7 = p7
        self.p8 = p8
        
        self.lList = lList
        self.pList = pList
        
        self.l1 = l1
        self.l2 = l2
        self.l4 = l4
        self.l5 = l5
        self.l6 = l6
        self.l7 = l7
        self.l8 = l8
        self.l9 = l9
        self.l10 = l10

   
        gmsh.model.geo.synchronize()
        # Not really sure what this does but you need it for things to work
        
    
    
    def nozzleWallFunction(self, z):
        """
        Function which defines the shape of the nozzle.

        Parameters
        ----------
        z : TYPE list
            Coordinates along nozzle (z) axis

        Returns
        -------
        radius : TYPE list
            Nozzle radius at each corresponding point along the nozzle axis

        """
               
        # Robinson Paper nozzle shape
        # Robinson C., "Flow Through a de Laval Nozzle", https://pdfslide.net/documents/flow-through-a-de-laval-nozzle-boston-2-1-overview-of-the-case-in-this-project.html
        # temp = z / 1e-3      
        # #radius = 1 - (0.868 * (-z + 2)**2) + (0.432 * (-z + 2)**3) 
        # radius = 1 - (0.868 * (-temp + 2)**2) + (0.432 * (-temp + 2)**3) 
        # radius *= 1e-3
                

        # CLPU experiment nozzle shape
        radius = np.zeros(len(z))
        
        for i in range(0, len(z)):
            if z[i] <= 2e-3:
                radius[i] = 0.4e-3
            else:
                radius[i] = (2.1/10.5) * z[i]

        
        return(radius)    
    
    
    
    def importNozzleFile(self, z):
        """
        Function for importing the nozzle profile as a series of coordinates 
        given in an external .csv file.
        
        No special attention is needed for the coordinates given (i.e. equal
        spacing, number of points, etc.) as the function only uses the provided
        points as a reference and then performs interpolation to calculate the
        nozzle radius at the required points.  Scaling can be optionally
        conducted on the nozzle dimensions using the multiplier_* variables.
        NOTE: In the .csv file, the first column should contain the axial (z) 
            coordinates while the second column should contain the radial (x)
            coordinates.

        Parameters
        ----------
        z : TYPE list
            Coordinates along nozzle (z) axis

        Returns
        -------
        radius : TYPE list
            Nozzle radius at each corresponding point along the nozzle axis

        """
        
        # Set function variables
        #fileDelimiter       = ' '
        fileDelimiter       = ','
        interpolationMethod = 'cubic'
        # Lorenz et al; Matter Radiat. Extremes 4, 015401 (2019); doi: 10.1063/1.5081509
        filePath            = "ELI_nozzle_profile_scaled.csv"
        multiplier_global   = 1e-6
        multiplier_radial   = 1
        multiplier_axial    = 1.08006
        
        # Import .csv with nozzle wall coordinates
        with open(filePath) as nozzleFile:
            givenWallPoints = np.asarray(list(csv.reader(nozzleFile, delimiter=fileDelimiter)))
        givenWallPoints = givenWallPoints.astype(np.float)
        
        # Perform scaling of nozzle shape
        givenWallPoints      *= multiplier_global                              # convert values to meters
        givenWallPoints[:,0] *= multiplier_axial                               # scale nozzle length
        givenWallPoints[:,1] *= multiplier_radial                              # scale nozzle radius
        
        # Interpolate the given wall coordinates to get desired mesh density
        interpolatedFunction = interpolate.interp1d(givenWallPoints[:,0], givenWallPoints[:,1], kind=interpolationMethod)
        radius = interpolatedFunction(z)
        
        return(radius)
    
    
    
        
    def plotWallShape(self, nozzleLength, numWallPoints, lc):
        """
        Function for setting geometry points along the nozzle wall and connecting 
        these points together as lines.

        Parameters
        ----------
        nozzleLength : TYPE float
            Length of the de Laval nozzle along the nozzle axis. Must be > 0.
        numWallPoints : TYPE int
            Number of discrete points used to approximate the nozzle wall shape. 
            NOTE: this value also represents the number of mesh cells generated
                along the nozzle axis.
        lc : TYPE float
            Curently does nothing.

        Returns
        -------
        List of tags for the points and lines along the nozzle wall

        """
        
        pList = np.zeros(numWallPoints,dtype=np.int)                            # List to hold the tags of the points located on the nozzle wall
        lList = np.zeros(numWallPoints-1,dtype=np.int)                          # List to hold the tags of the lines located on the nozzle wall
        
        z = np.linspace(0,nozzleLength,numWallPoints)
        
        # calculate the nozzle radius at each point along axis
        if(self.wallPlotMethod == 'function'):          # Specify nozzle shape via piecewise function
            radius = self.nozzleWallFunction(z)                                 
        elif(self.wallPlotMethod == 'file'):            # Specify nozzle shape via external .csv file
            radius = self.importNozzleFile(z)
        else:
            temp = 0  

        # Uncomment these lines to add a sawtooth shape to the nozzle wall
        #radius[0::4] -= 50e-6   
        #radius[2::4] += 50e-6                             
        
        # set constants for the inlet and outlet radii
        self.inletRadius  = radius[0]
        self.outletRadius = radius[-1]
        
        # add points to model, connect adjacent points into lines along nozzle wall
        for i in range(0, numWallPoints):
            if i == 0:
                pList[i] = gmsh.model.geo.addPoint(radius[i], 0, z[i], lc)
            else:
                pList[i] = gmsh.model.geo.addPoint(radius[i], 0, z[i], lc)
                lList[i-1] = gmsh.model.geo.addLine(pList[i], pList[i-1])
        
        # return list of point and line tags along nozzle wall
        return(pList, lList)
        
        
        
    def generate2DMesh(self, nozzleMeshDensityX, cavityMeshDensityZ=1, uCavityMeshDensityX=1, meshType='Structured', grading='False'):
        """
        Function to generate a 2D structured mesh within the nozzle cross-section.
        
        NOTE: The number of mesh cells within the nozzle along the radial direction
            is the same as the number of mesh cells within the lower external cavity
            section along the radial direction.

        Parameters
        ----------
        nozzleMeshDensityX : TYPE int
            Number of mesh cells to be generated within the nozzle in the transverse 
            direction. Must be >= 1.
        cavityMeshDensityZ : TYPE int, optional
            Number of mesh cells to be generated within the external cavity along
            the nozzle axis. Must be >= 1. The default is 1.
        uCavityMeshDensityX : TYPE int, optional
            Number of mesh cells to be generated within the upper section of the 
            external cavity along the transverse direction. Must be >= 1. The 
            default is 1.
        meshType : TYPE str, optional
             Selects whether to generate a structured or an unstructured mesh.
             Currently the only available option is 'Structured'.
        grading : TYPE str, optional
            Selects the type of mesh grading within the external cavity.
            The available options are:
                - 'False': Generates a uniform mesh cell size throughout the cavity
                    according to the number of cells specified under cavityMeshDensityZ
                    and uCaivtyMeshDensityX.
                - 'True': Grades the mesh within the external cavity such that
                    there is a smooth cell size transition between the nozzle and
                    the cavity.
                    NOTE: Grading may not be performed if the number of mesh cells
                    within the cavity is too high.
                - 'Uniform': Calculates the number of mesh cells to generate within
                    the external cavity such that the cell size is consistent with
                    the mesh inside of the nozzle. Mesh cell size is uniform.
                    NOTE: with this option, the values of cavityMeshDensityZ and
                    uCavityMeshDensityX are ignored.
            The default is 'False'.

        Returns
        -------
        None.

        """
        
        # prevents structured mesh lines from curving when grading
        gmsh.option.setNumber("Mesh.Smoothing", 0)
        
        
        # set class variables
        self.nozzleMeshDensityX     = nozzleMeshDensityX
        self.cavityMeshDensityZ     = cavityMeshDensityZ
        self.uCavityMeshDensityX    = uCavityMeshDensityX
        self.meshType               = meshType
        self.grading                = grading
        
        # marker that this function has run
        self.MESH2D = 1
               
                
        if grading == 'True':
            # calculate progression required for smooth cell size transition
            print("Grading cavity z-axis...")
            adjacentSize   = self.nozzleLength / (self.numWallPoints+1)
            cavityGradingZ = self.findRoots(cavityMeshDensityZ, self.cavityLength, adjacentSize)
            
            print("Grading upper cavity x-axis...")
            adjacentSize    = self.outletRadius / (self.nozzleMeshDensityX)
            uCavityGradingX = self.findRoots(uCavityMeshDensityX, self.cavityHeight - self.outletRadius, adjacentSize)
        
        elif grading == 'False':
            cavityGradingZ  = 1
            uCavityGradingX = 1
        
        elif grading == 'Uniform':          
            cavityGradingZ  = 1
            uCavityGradingX = 1
            
            # calculate the nozzle mesh cell z-dimension
            adjacentSize = self.nozzleLength / (self.numWallPoints+1)
            # calculate number of mesh cells in cavity required to achieve similar size
            cavityMeshDensityZ = round(self.cavityLength / adjacentSize)
            cavityMeshDensityZ = int(cavityMeshDensityZ)
            
            # calculate the nozzle mesh cell x-dimension
            adjacentSize = self.outletRadius / (self.nozzleMeshDensityX)
            # calculate number of mesh cells in cavity required to achieve similar size
            uCavityMeshDensityX = round((self.cavityHeight - self.outletRadius) / adjacentSize)
            uCavityMeshDensityX = int(uCavityMeshDensityX)
                                  
        else:
            cavityGradingZ = 1
            uCavityGradingX = 1
            
            
        # Set transfinite curves (for structured mesh), sets the number of mesh cells in each direction
        # nozzle wall
        for i in self.lList:
            gmsh.model.geo.mesh.setTransfiniteCurve(i, 2)
        
        # nozzle axis
        gmsh.model.geo.mesh.setTransfiniteCurve(self.l1,  self.numWallPoints,     meshType="Progression", coef=1)
            
        # nozzle inlet/outlet, lower cavity x-axis
        gmsh.model.geo.mesh.setTransfiniteCurve(self.l4,  nozzleMeshDensityX + 1,     meshType="Progression", coef=1)
        gmsh.model.geo.mesh.setTransfiniteCurve(self.l2,  nozzleMeshDensityX + 1,     meshType="Progression", coef=1)
        gmsh.model.geo.mesh.setTransfiniteCurve(self.l6,  nozzleMeshDensityX + 1,     meshType="Progression", coef=1)
        
        # upper/lower cavity z-axis
        gmsh.model.geo.mesh.setTransfiniteCurve(self.l5,  cavityMeshDensityZ + 1,     meshType="Progression", coef=cavityGradingZ)
        gmsh.model.geo.mesh.setTransfiniteCurve(self.l7,  cavityMeshDensityZ + 1,     meshType="Progression", coef=-1*cavityGradingZ)
        gmsh.model.geo.mesh.setTransfiniteCurve(self.l9,  cavityMeshDensityZ + 1,     meshType="Progression", coef=-1*cavityGradingZ)
        
        # upper cavity x-axis
        gmsh.model.geo.mesh.setTransfiniteCurve(self.l10, uCavityMeshDensityX + 1,    meshType="Progression", coef=-1*uCavityGradingX)
        gmsh.model.geo.mesh.setTransfiniteCurve(self.l8,  uCavityMeshDensityX + 1,    meshType="Progression", coef=uCavityGradingX)
        
        # Set transfinite surfaces
        gmsh.model.geo.mesh.setTransfiniteSurface(1, cornerTags=[self.p1,           self.p2, self.pList[-1],    self.pList[0]])
        gmsh.model.geo.mesh.setTransfiniteSurface(2, cornerTags=[self.p2,           self.p5, self.p6,           self.pList[-1]])
        gmsh.model.geo.mesh.setTransfiniteSurface(3, cornerTags=[self.pList[-1],    self.p6, self.p7,           self.p8])
                 
        
        gmsh.model.geo.synchronize()
              
        
        ### Mesh not generated here so that rotations, etc. can be performed in other 
        ### processes without destroying the mesh structure
            
        
        
    def generateReservoir(self, reservoirLength, reservoirHeight, reservoirMeshDensityZ, uReservoirMeshDensityX, grading='False'):
        """
        (Optional) Function for generating a rectangular reservoir at the nozzle inlet

        Parameters
        ----------
        reservoirLength : TYPE float
            Length of the reservoir along the nozzle (z) axis
        reservoirHeight : TYPE float
            Height of the reservoir along the transverse (x) direction
        reservoirMeshDensityZ : TYPE int
            Number of mesh cells to be generated within the reservoir along the 
            nozzle (z) axis.
        uReservoirMeshDensityX : TYPE int
            Number of mesh cells to be generated within the reservoir along the 
            upper section of the x-axis
        grading : TYPE str, optional
            Selects the type of mesh grading within the reservoir.
            The available options are:
                - 'False': Generates a uniform mesh cell size throughout the reservoir
                    according to the number of cells specified under reservoirMeshDensityZ
                    and uReservoirMeshDensityX.
                - 'True': Grades the mesh within the reservoir such that
                    there is a smooth cell size transition between the nozzle and
                    the reservoir.
                    NOTE: Grading may not be performed if the number of mesh cells
                    within the reservoir is too high.
                - 'Uniform': Calculates the number of mesh cells to generate within
                    the reservoir such that the cell size is consistent with
                    the mesh inside of the nozzle. Mesh cell size is uniform.
                    NOTE: with this option, the values of reservoirMeshDensityZ and
                    uReservoirMeshDensityX are ignored.
            The default is 'False'.

        Returns
        -------
        None.

        """
        
        # set class variables
        self.reservoirLength        = reservoirLength
        self.reservoirHeight        = reservoirHeight
        self.reservoirMeshDensityZ  = reservoirMeshDensityZ
        self.reservoirMeshDensityX  = uReservoirMeshDensityX
        self.reservoirGradingOpt    = grading
        
        # marker that this function has run
        self.RESERVOIR = 1
        
       
        # ------------------------------------------------ #
        #                Generate Geometry                 #
        # ------------------------------------------------ #
          
        # generate points for reservoir
        rp1 = gmsh.model.geo.addPoint(reservoirHeight,  0, 0)
        rp2 = gmsh.model.geo.addPoint(reservoirHeight,  0, -reservoirLength)
        rp3 = gmsh.model.geo.addPoint(self.inletRadius, 0, -reservoirLength)
        rp4 = gmsh.model.geo.addPoint(0,                0, -reservoirLength)
        
        rl1 = gmsh.model.geo.addLine(self.pList[0], rp1)
        rl2 = gmsh.model.geo.addLine(rp1,           rp2)
        rl3 = gmsh.model.geo.addLine(rp2,           rp3)
        rl4 = gmsh.model.geo.addLine(rp3,           rp4)
        rl5 = gmsh.model.geo.addLine(rp4,           self.p1)
        
        # set reservoir surface
        temp = [-self.l4, rl1, rl2, rl3, rl4, rl5]
        rcl1 = gmsh.model.geo.addCurveLoop(temp)
        self.rs1 = gmsh.model.geo.addPlaneSurface([rcl1])
        
        # Set additional class variables
        self.rp1 = rp1
        self.rp2 = rp2
        self.rp3 = rp3
        
        self.rl1 = rl1
        self.rl2 = rl2
        self.rl3 = rl3
        self.rl4 = rl4
           
        gmsh.model.geo.synchronize()
        
          
        
        # ------------------------------------------------ #
        #            Generate Structured Mesh              #
        # -------------------------------------------------#
           
        # Select desired mesh grading method
        if grading == 'True':
            # calculate progression required for smooth cell size transition
            print("Grading reservoir z-axis...")
            adjacentSize = self.nozzleLength / (self.numWallPoints+1)
            reservoirGradingZ = self.findRoots(reservoirMeshDensityZ, reservoirLength, adjacentSize)
            
            print("Grading upper reservoir x-axis...")
            adjacentSize = self.inletRadius / (self.nozzleMeshDensityX)
            uReservoirGradingX = self.findRoots(uReservoirMeshDensityX, self.reservoirHeight - self.inletRadius, adjacentSize)
            
        elif grading == 'False':
            reservoirGradingZ = 1
            uReservoirGradingX = 1
            
        elif grading == 'Uniform':
            reservoirGradingZ = 1
            uReservoirGradingX = 1
            
            # calculate the nozzle mesh cell z-dimension
            adjacentSize = self.nozzleLength / (self.numWallPoints+1)
            # calculate number of mesh cells in reservoir required to achieve similar size
            reservoirMeshDensityZ = round(self.reservoirLength / adjacentSize)
            reservoirMeshDensityZ = int(reservoirMeshDensityZ)
            
            # calculate the nozzle mesh cell x-dimension
            adjacentSize = self.inletRadius / (self.nozzleMeshDensityX)
            # calculate number of mesh cells in cavity required to achieve similar size
            uReservoirMeshDensityX = round((self.reservoirHeight - self.inletRadius) / adjacentSize)
            uReservoirMeshDensityX = int(uReservoirMeshDensityX)
            
        else:
            reservoirGradingZ = 1
            uReservoirGradingX = 1
        
        
        # Set transfinite curves in reservoir for generating structured mesh
        # upper reservoir x-axis
        gmsh.model.geo.mesh.setTransfiniteCurve(rl1, uReservoirMeshDensityX + 1, meshType='Progression', coef=uReservoirGradingX)
        gmsh.model.geo.mesh.setTransfiniteCurve(rl3, uReservoirMeshDensityX + 1, meshType='Progression', coef=-1*uReservoirGradingX)
        
        # reservoir z-axis
        gmsh.model.geo.mesh.setTransfiniteCurve(rl2, reservoirMeshDensityZ + 1, meshType='Progression', coef=reservoirGradingZ)
        gmsh.model.geo.mesh.setTransfiniteCurve(rl5, reservoirMeshDensityZ + 1, meshType='Progression', coef=-1*reservoirGradingZ)
        
        # lower reservoir x-axis
        gmsh.model.geo.mesh.setTransfiniteCurve(rl4, self.nozzleMeshDensityX + 1, meshType='Progression', coef=1)
        
        # Set transfinite surface
        gmsh.model.geo.mesh.setTransfiniteSurface(4, cornerTags=[rp1, rp2, rp4, self.p1])
        
        gmsh.model.geo.synchronize()
        
      
        
    def generateWedge(self, wedgeAngle, extrudeLayers=1, recombineMesh=True):
        """
        Function for generating a 3D  axisymmetric wedge section of the nozzle 
        geometry.
        
        The mesh is generated through rotational extrusion. This function assumes
        axial symmetry in the nozzle flow properties. The wedge is rotated such
        that it is symmetric across the x-z plane

        Parameters
        ----------
        wedgeAngle : TYPE float
            Angle of the generated wedge section in the theta direction. Must be
            < 2*pi. Should be kept small for wedge simulations.
        extrudeLayers : TYPE, optional
            Number of mesh layers to generate along the theta direction. The default
            is 1.
        recombineMesh : TYPE bool, optional
            If True, recombines mesh cells in the theta direction to form hexahedral
            cells. The default is True.

        Returns
        -------
        None.

        """
              
        
        # structured mesh is preserved under transformations
        gmsh.option.setNumber("Geometry.CopyMeshingMethod", 1)   
        
        
        # set class variables
        self.wedgeAngle = wedgeAngle
        self.extrudeLayers = extrudeLayers
        self.recombineMesh = recombineMesh
        
        # marker that this function has run
        self.WEDGE = 1
    
        
        # rotationally extrude the cross-section to produce the 3D geometry
        # for full 3D cylindrical nozzle
        if (wedgeAngle == 2 * np.pi):
            # needs more work. mesh doesn't fully connect, physical groups not declared
            volume1 = gmsh.model.geo.revolve([(2, 1)], 0, 0, 0, 0, 0, 1, np.pi, numElements = [extrudeLayers], recombine=recombineMesh)
            volume2 = gmsh.model.geo.revolve([(2, 2)], 0, 0, 0, 0, 0, 1, np.pi, numElements = [extrudeLayers], recombine=recombineMesh)
            volume3 = gmsh.model.geo.revolve([(2, 3)], 0, 0, 0, 0, 0, 1, np.pi, numElements = [extrudeLayers], recombine=recombineMesh)
            
            volume4 = gmsh.model.geo.revolve([(2, 1)], 0, 0, 0, 0, 0, -1, np.pi, numElements = [extrudeLayers], recombine=recombineMesh)
            volume5 = gmsh.model.geo.revolve([(2, 2)], 0, 0, 0, 0, 0, -1, np.pi, numElements = [extrudeLayers], recombine=recombineMesh)
            volume6 = gmsh.model.geo.revolve([(2, 3)], 0, 0, 0, 0, 0, -1, np.pi, numElements = [extrudeLayers], recombine=recombineMesh)
        
        # For axisymmetric wedge section
        else: 
            self.volume1 = gmsh.model.geo.revolve([(2, 1)], 0, 0, 0, 0, 0, 1, wedgeAngle, numElements = [extrudeLayers], recombine=recombineMesh)
            self.volume2 = gmsh.model.geo.revolve([(2, 2)], 0, 0, 0, 0, 0, 1, wedgeAngle, numElements = [extrudeLayers], recombine=recombineMesh)
            self.volume3 = gmsh.model.geo.revolve([(2, 3)], 0, 0, 0, 0, 0, 1, wedgeAngle, numElements = [extrudeLayers], recombine=recombineMesh)
            
            # extrude reservoir is present
            if(self.RESERVOIR == 1):              
                self.volume4 = gmsh.model.geo.revolve([(2, 4)], 0, 0, 0, 0, 0, 1, wedgeAngle, numElements = [extrudeLayers], recombine=recombineMesh)
            
            gmsh.model.geo.synchronize()
            
            self.declarePhysicalGroups(nozzleType='Wedge')
            
            gmsh.model.geo.synchronize()
            
            # rotate wedge back so it's symmetric across the xz-plane
            entityTags = gmsh.model.getEntities(dim=-1)
            gmsh.model.geo.rotate(entityTags, 0, 0, 0, 0, 0, -1, wedgeAngle/2)
            
            gmsh.model.geo.synchronize()
                        
        
        # generate the 3D mesh
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.recombine()        
        gmsh.model.mesh.generate(3)
        
        
          
    def generateChannel(self, extrudeHeight, extrudeLayers=1, recombineMesh=True):
        """
        Function to generate a 3D channel nozzle

        Parameters
        ----------
        extrudeHeight : TYPE float
            Height of the nozzle channel along the y-axis (direction of extrusion).
        extrudeLayers : TYPE int, optional
            Number of mesh cells along the y-axis (direction of extrusion). The 
            default is 1.
        recombineMesh : TYPE, optional
            If True, recombines mesh cells in the extrusion direction to form hexahedral
            cells. The default is True.

        Returns
        -------
        None.

        """
        
        # set class variables
        self.extrudeHeight = extrudeHeight
        self.extrudeLayers = extrudeLayers
        self.recombineMesh = recombineMesh
        
        # marker that this function has run
        self.CHANNEL = 1
        
        
        # structured mesh is preserved under transformations
        gmsh.option.setNumber("Geometry.CopyMeshingMethod", 1)
        # prevents structured mesh lines from curving when grading
        gmsh.option.setNumber("Mesh.Smoothing", 0)
        
        gmsh.model.geo.synchronize()
        #gmsh.model.mesh.clear()               # possible solution for rotation problem?
        
        # reflect nozzle geometry across the axis
        # reflect nozzle surface
        s1_reflect = gmsh.model.geo.copy((2, 1))       
        gmsh.model.geo.symmetrize(s1_reflect, 1, 0, 0, 0)
        
        # reflect lower cavity surface
        s2_reflect = gmsh.model.geo.copy((2, 2))
        gmsh.model.geo.symmetrize(s2_reflect, 1, 0, 0, 0)
        
        # reflect upper cavity surface
        s3_reflect = gmsh.model.geo.copy((2, 3))
        gmsh.model.geo.symmetrize(s3_reflect, 1, 0, 0, 0)
        
        gmsh.model.geo.synchronize()
        
        # get corner point tags of reflected nozzle
        temp1 = gmsh.model.getBoundary(s1_reflect)
        temp2 = gmsh.model.getBoundary(dimTags=[temp1[-1], temp1[-2]])
        temp3 = self.findSides(temp2, 0,-1)
        
        # set reflected nozzle surface as a transfinte sureface
        gmsh.model.geo.mesh.setTransfiniteSurface(s1_reflect[0][1], cornerTags=temp3)
        
        gmsh.model.geo.synchronize()
        
        # Extrude nozzle cross section
        self.volume1 = gmsh.model.geo.extrude((2, 1),        0, extrudeHeight, 0, numElements=[extrudeLayers], heights=[], recombine=recombineMesh) 
        self.volume2 = gmsh.model.geo.extrude((2, 2),        0, extrudeHeight, 0, numElements=[extrudeLayers], heights=[], recombine=recombineMesh) 
        self.volume3 = gmsh.model.geo.extrude((2, 3),        0, extrudeHeight, 0, numElements=[extrudeLayers], heights=[], recombine=recombineMesh) 
        self.volume4 = gmsh.model.geo.extrude(s1_reflect,    0, extrudeHeight, 0, numElements=[extrudeLayers], heights=[], recombine=recombineMesh)
        self.volume5 = gmsh.model.geo.extrude(s2_reflect,    0, extrudeHeight, 0, numElements=[extrudeLayers], heights=[], recombine=recombineMesh)
        self.volume6 = gmsh.model.geo.extrude(s3_reflect,    0, extrudeHeight, 0, numElements=[extrudeLayers], heights=[], recombine=recombineMesh)
        
        self.s1_reflect = s1_reflect
        self.s2_reflect = s2_reflect
        self.s3_reflect = s3_reflect     
        
        
        if(self.RESERVOIR == 1):  
            # copy and reflect reservoir across yz-plane
            s4_reflect = gmsh.model.geo.copy((2, self.rs1))
            gmsh.model.geo.symmetrize(s4_reflect, 1, 0, 0, 0)
            
            gmsh.model.geo.synchronize()
            
            # To create structured mesh in reflected surface
            # find tags of corner points
            temp1 = gmsh.model.getBoundary(s4_reflect)
            temp2 = gmsh.model.getBoundary(dimTags=[temp1[0], temp1[3]])
            temp3 = self.findSides(temp2, start=0, stop=-1)           
            gmsh.model.geo.mesh.setTransfiniteSurface(s4_reflect[0][1], cornerTags=temp3)
             
            # extrude reservoir sections
            self.volume7 = gmsh.model.geo.extrude((2, self.rs1),    0, extrudeHeight, 0, numElements=[extrudeLayers], heights=[], recombine=recombineMesh)
            self.volume8 = gmsh.model.geo.extrude(s4_reflect,       0, extrudeHeight, 0, numElements=[extrudeLayers], heights=[], recombine=recombineMesh)
            self.s4_reflect = s4_reflect    
          
        
        # Group and name boundary patches
        self.declarePhysicalGroups(nozzleType='Channel')        
        
        gmsh.model.geo.synchronize()
             
        # generate the 3D mesh
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.recombine()        
        gmsh.model.mesh.generate(3)
       
       
       
    def declarePhysicalGroups(self, nozzleType):
        """
        Function for grouping and naming boundary patches.  Used for setting
        boundary conditions in OpenFOAM.

        Parameters
        ----------
        nozzleType : TYPE str
            Shape of the nozzle being generated.
            The available options are:
                - 'Wedge': For axisymmetric wedge-shapped nozzles.
                - 'Cylinder': Not currently in use.
                - 'Channel': For channel nozzle shapes

        Returns
        -------
        None.

        """
        
        ### potentially add 'arrangment' variable to set where wall extends to
        
        # Set names of boundary patches
        internalMeshName    = "internal"
        inletPatchName      = "Input"
        outletPatchName     = "Output"
        wallPatchName       = "Outerwall"
        rightPatchName      = "Right"
        leftPatchName       = "Left"
        
        
        if(nozzleType == 'Wedge'):
            
            # import tags from extruded surfaces
            volume1 = self.volume1
            volume2 = self.volume2
            volume3 = self.volume3
            
            
            # Group boundary patch tags
            wallTags = self.findSides(tagList=volume1, start=2, stop=-3)
            #wallTags.extend([volume3[-3][1], volume3[-2][1]])                 # sets top of cavity as wall
            wallTags.extend([volume3[-2][1]])                                  # sets top of cavity as outlet
            
            print(volume3)
            
            leftTags        = [1, 2, 3]
            rightTags       = [volume1[0][1], volume2[0][1], volume3[0][1]]
            #outletTags      = [volume2[2][1], volume3[2][1]]                  # sets top of cavity as wall
            outletTags      = [volume2[2][1], volume3[2][1], volume3[-3][1]]   # sets top of cavity as outlet
            internalTags    = [volume1[1][1], volume2[1][1], volume3[1][1]]
            
            # Add reservoir boundary patches to appropriate groups
            if(self.RESERVOIR == 1):
                volume4 = self.volume4
                
                wallTags.extend([volume4[3][1], volume4[4][1]])
                leftTags.extend([4])
                rightTags.extend([volume4[0][1]])
                internalTags.extend([volume4[1][1]])
                inletTags = [volume4[-2][1], volume4[-1][1]]
            
            # if there is no reservoir present                
            else:
                inletTags = [volume1[-2][1]]
            
            
            # Group for nozzle wall patches
            gmsh.model.addPhysicalGroup(dim=2, tags=wallTags,       tag=1)
            gmsh.model.setPhysicalName( dim=2, tag=1, name=wallPatchName)
            
            # Group for left wedge side patches
            gmsh.model.addPhysicalGroup(dim=2, tags=leftTags,       tag=2)
            gmsh.model.setPhysicalName( dim=2, tag=2, name=leftPatchName)
            
            # Group for right wedge side patches
            gmsh.model.addPhysicalGroup(dim=2, tags=rightTags,      tag=3)
            gmsh.model.setPhysicalName( dim=2, tag=3, name=rightPatchName)
            
            # Group for inlet patches
            gmsh.model.addPhysicalGroup(dim=2, tags=inletTags,      tag=5)
            gmsh.model.setPhysicalName( dim=2, tag=5, name=inletPatchName)
            
            # Group for outlet patches
            gmsh.model.addPhysicalGroup(dim=2, tags=outletTags,     tag=6)
            gmsh.model.setPhysicalName( dim=2, tag=6, name=outletPatchName)
            
            # Group for internal mesh
            gmsh.model.addPhysicalGroup(dim=3, tags=internalTags,   tag=7)
            gmsh.model.setPhysicalName( dim=3, tag=7, name=internalMeshName)
            
            
        elif(nozzleType == 'Cylinder'):
            ### Not currently in use
            temp = 0
            
        elif(nozzleType == 'Channel'):
            
            # import tags from extruded surfaces
            volume1 = self.volume1
            volume2 = self.volume2
            volume3 = self.volume3
            volume4 = self.volume4
            volume5 = self.volume5
            volume6 = self.volume6
            
            s1_reflect = self.s1_reflect
            s2_reflect = self.s2_reflect
            s3_reflect = self.s3_reflect
            
            # Isolate tags for surfaces on the nozzle wall
            wallTags = self.findSides(tagList=volume1, start=2, stop=-5)
            wallTags.extend(self.findSides(tagList=volume4, start=2, stop=-4))
            wallTags.extend([volume3[4][1], volume6[4][1]])
            
            # Group tags for the remaining boundary patches
            leftTags        = [1,                   2,                  3, 
                               s1_reflect[0][1],    s2_reflect[0][1],   s3_reflect[0][1]]
            rightTags       = [volume1[0][1],       volume2[0][1],      volume3[0][1],
                               volume4[0][1],       volume5[0][1],      volume6[0][1]]
            outletTags      = [volume2[3][1],       volume3[2][1],      volume3[3][1],
                               volume5[3][1],       volume6[2][1],      volume6[3][1]]
            internalTags    = [volume1[1][1],       volume2[1][1],      volume3[1][1],
                               volume4[1][1],       volume5[1][1],      volume6[1][1]]
            
            # If the reservoir is present at the nozzle inlet
            if(self.RESERVOIR == 1):
                # import tags for extruded surfaces
                volume7     = self.volume7
                volume8     = self.volume8           
                s4_reflect  = self.s4_reflect
                
                
                wallTags.extend([       volume7[3][1], volume7[4][1],
                                        volume8[3][1], volume8[4][1]])
                leftTags.extend([       4,
                                        s4_reflect[0][1]])
                rightTags.extend([      volume7[0][1],
                                        volume8[0][1]])
                internalTags.extend([   volume7[1][1],
                                        volume8[1][1]])
                inletTags = [           volume7[5][1], volume7[6][1],
                                        volume8[5][1], volume8[6][1]]
             
            # if the reservoir is not present at the nozzle inlet
            else:
                inletTags = [volume1[-4][1],    volume4[-3][1]]
                
                
            # Group for nozzle wall patches
            gmsh.model.addPhysicalGroup(dim=2, tags=wallTags,       tag=1)
            gmsh.model.setPhysicalName( dim=2, tag=1, name=wallPatchName)
            
            # Group for left wedge side patches
            gmsh.model.addPhysicalGroup(dim=2, tags=leftTags,       tag=2)
            gmsh.model.setPhysicalName( dim=2, tag=2, name=leftPatchName)
            
            # Group for right wedge side patches
            gmsh.model.addPhysicalGroup(dim=2, tags=rightTags,      tag=3)
            gmsh.model.setPhysicalName( dim=2, tag=3, name=rightPatchName)
            
            # Group for inlet patches
            gmsh.model.addPhysicalGroup(dim=2, tags=inletTags,      tag=5)
            gmsh.model.setPhysicalName( dim=2, tag=5, name=inletPatchName)
            
            # Group for outlet patches
            gmsh.model.addPhysicalGroup(dim=2, tags=outletTags,     tag=6)
            gmsh.model.setPhysicalName( dim=2, tag=6, name=outletPatchName)
            
            # Group for internal mesh
            gmsh.model.addPhysicalGroup(dim=3, tags=internalTags,   tag=7)
            gmsh.model.setPhysicalName( dim=3, tag=7, name=internalMeshName)
            
               
        return()
       
    
        
    def saveMesh(self, directory=None):
        """
        Function for saving the generated mesh as well as a text file containing
        the geometry and mesh parameters.

        Parameters
        ----------
        directory : TYPE str, optional
            Directory of the desired location to save the mesh. If set to None,
            the mesh is saved to the working directory.  The default is None.

        Returns
        -------
        None.

        """
        
        # Generate timestamp string for current date and time
        timeStamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        folderName = 'nozzleMesh_%s' %(timeStamp)
        if directory is not None:
            folderName = directory + folderName
        # make folder to hold mesh and mesh specifications file
        os.makedirs(folderName)
        
        # Save the mesh file to the created folder
        # NOTE: it must be formatted as version 2 for gmshToFoam utility to work
        gmsh.write(folderName + '/nozzleMesh.msh2')
        
        # Generate text file to contain mesh specifications
        self.documentationWriter(folderName, timeStamp)
        
        
        
    def documentationWriter(self, folderName, timeStamp):
        """
        Function for saving the input geometry and meshing parameters to a text
        file.

        Parameters
        ----------
        folderName : TYPE str
            Directory of the desired location to save the mesh.
        timeStamp : TYPE str
            String containing information on the current date and time.

        Returns
        -------
        None.

        """        
        
        # Write documentation containing all meshing parameters
        docFile = open(folderName + '/meshSpecs.txt', 'w')
        docFile.write("GENERAL\n")
        docFile.write("Date generated: %s\n" %(timeStamp))
        docFile.write("\n")
        
        # If the function for generating the 2D cross-section geometry has been run
        if self.GEOMETRY == 1:
            docFile.write("GEOMETRY\n")
            docFile.write("nozzleLength:            %f\n"   %(self.nozzleLength))
            docFile.write("inletRadius:             %f\n"   %(self.inletRadius))
            docFile.write("outletRadius:            %f\n"   %(self.outletRadius))
            docFile.write("cavityLength:            %f\n"   %(self.cavityLength))
            docFile.write("cavityHeight:            %f\n"   %(self.cavityHeight))
            docFile.write("numWallPoints:           %f\n"   %(self.numWallPoints))
            docFile.write("lc:                      %f\n"   %(self.lc))
            docFile.write("\n")
        
        # if the function for generating the 2D cross-section mesh has been run
        if self.MESH2D == 1:
            docFile.write("MESH\n")
            docFile.write("nozzleMeshDensityX:      %d\n"    %(self.nozzleMeshDensityX))
            docFile.write("cavityMeshDensityZ:      %d\n"    %(self.cavityMeshDensityZ))
            docFile.write("uCavityMeshDensityX:     %d\n"    %(self.uCavityMeshDensityX))
            docFile.write("meshType:                %s\n"    %(self.meshType))
            docFile.write("grading:                 %s\n"    %(self.grading))
            docFile.write("\n")
         
        # if the function for generating a reservoir at the nozzle inlet has been run
        if self.RESERVOIR == 1:
            docFile.write("RESERVOIR\n")
            docFile.write("reservoirLength:         %f\n"       %(self.reservoirLength))
            docFile.write("reservoirHeight:         %f\n"       %(self.reservoirHeight))
            docFile.write("reservoirMeshDensityZ:   %d\n"       %(self.reservoirMeshDensityZ))
            docFile.write("reservoirMeshDensityX:   %d\n"       %(self.reservoirMeshDensityX))
            docFile.write("grading:                 %s\n"       %(self.reservoirGradingOpt))
            docFile.write("\n")
        
        # if the function for generating a 3D wedge geometry has been run
        if self.WEDGE == 1:
            docFile.write("WEDGE\n")
            docFile.write("wedgeAngle:              %f\n"     %(self.wedgeAngle))
            docFile.write("extrudeLayers:           %d\n"     %(self.extrudeLayers))
            docFile.write("recombineMesh:           %s\n"     %(self.recombineMesh))
            docFile.write("\n")
         
        # if the function for generating a 2D channel nozzle has been run
        if self.CHANNEL == 1:
            docFile.write("CHANNEL\n")
            docFile.write("extrudeHeight:           %f\n"     %(self.extrudeHeight))
            docFile.write("extrudeLayers:           %d\n"     %(self.extrudeLayers))
            docFile.write("recombineMesh:           %s\n"     %(self.recombineMesh))
            docFile.write("\n")
            
        docFile.close()
        
        
    def findRoots(self, numCells, length, adjacentCellSize):
        """
        Function used for finding the roots of the equation relating the length,
        the number of mesh cells along the length, and the size of the smallest 
        mesh cell to the size ratio between adjacent cells via Newton's method.
        Used to determine the amount of grading in the cavity to ensure a smooth
        mesh cell size transition from the nozzle to the cavity.
        
            dx_s = l * (r - 1) / (r^n - 1)
            
        where:
            dx_s = size of the smallest cell
            n = number of cells in region
            r = ratio between one cell and the next
            l = length of the region
        https://cfd.direct/openfoam/user-guide/v7-cavity/#x5-270002.1.6

        Parameters
        ----------
        numCells : TYPE int
            Number of mesh cells in the direction to be graded
        length : TYPE float
            Length of the dimension to be graded
        adjacentCellSize : TYPE float
            Length of the mesh cells in the adjacent section along the direction
            to be graded. Used for creating a smooth cell size transition between
            sections.

        Returns
        -------
        None.

        """
        
        # Rootfinding parameters 
        functResidual  = 1e-5         # maximum value of the function when calculated at the current root value
        initialGuess   = 1.1          # initial guess for the value of the root
        guessIncrement = 1            # amount to increment initial guess to find next root
        maxIterations  = 1000         # maximum number of iterations to find the appropriate root
        numIterations  = 0
        
        # if the initial, uniform mesh density in the region is too high for grading
        uniformSize = length / numCells
        if uniformSize <= adjacentCellSize:
            print("Grading could not be performed")
            return(1)
        
        r     = initialGuess
        funct = 1000          
        
        # Specify how close the solution can approach 1
        # used to distinguish between finding 1 as the root vs a root very close to 1
        tolerance = ((uniformSize / adjacentCellSize) - 1) / 1000
        
        while(True):
            # iterate until the root falls within the desired accuracy
            while (abs(funct) > functResidual):
                # use Newton's method for finding the root
                funct = (adjacentCellSize * r**numCells) - (length * r) - adjacentCellSize + length
                functPrime = (adjacentCellSize * numCells * r**(numCells-1)) - length
                r -= (funct / functPrime)
                
            # check if the root found is 1
            if (r < 1 + tolerance):
                if (numIterations >= maxIterations):
                    # If maximum number of iterations has been exceeded
                    print("Maximum number of iterations exceeded")
                    return(1)
                else:
                    # increment the initial guess to find the region of convergence of the next root
                    funct = 1000
                    initialGuess += guessIncrement
                    r = initialGuess * 1
                    numIterations += 1
            else:
                print("Root found: %f" %(r))
                return(r)
            
            
    def findSides(self, tagList, start, stop):
        """
        Function used for isolating the element tag number from the element dimension
        in the nested lists produced by Gmsh

        Parameters
        ----------
        tagList : TYPE list
            Nested list produced by Gmsh giving the tag number and dimension of
            various elements within the model.
        start : TYPE int
            The starting index of the nested list from the section you wish to
            isolate.
        stop : TYPE int
            The ending index of the nested list from the section you wish to isolate.

        Returns
        -------
        sides : TYPE list
            List of isolated element tags with the corresponding dimension number
            removed.

        """
        
        sides = []
        
        # if indexing from the end of the list
        if stop < 0:
            stop = len(tagList) +  stop
        if start < 0:
            start = len(tagList) + start
        
        # extract tag of entity
        for i in range(start, stop + 1):
            sides.append(tagList[i][1])
            
        return (sides)
  


test = deLavalNozzleGenerator()
test.generateCrossSection(nozzleLength=15.63e-3, cavityLength=75e-3, cavityHeight=45e-3, numWallPoints=200)
test.generate2DMesh(nozzleMeshDensityX=40, cavityMeshDensityZ=200, uCavityMeshDensityX=100, grading='True')
test.generateReservoir(reservoirLength=30e-3, reservoirHeight=30e-3, uReservoirMeshDensityX=60, reservoirMeshDensityZ=40, grading='True')
#test.generateWedge(wedgeAngle=0.01, extrudeLayers=1, recombineMesh=True)
test.generateChannel(extrudeHeight=1e-3, extrudeLayers=1, recombineMesh=True)

"""
test = deLavalNozzleGenerator()
test.generateCrossSection(nozzleLength=12.5e-3, cavityLength=50e-3, cavityHeight=25e-3, numWallPoints=80)
test.generate2DMesh(nozzleMeshDensityX=40, cavityMeshDensityZ=200, uCavityMeshDensityX=100, grading='True')
test.generateReservoir(reservoirLength=20e-3, reservoirHeight=20e-3, uReservoirMeshDensityX=60, reservoirMeshDensityZ=40, grading='True')
test.generateWedge(wedgeAngle=0.01, extrudeLayers=1, recombineMesh=True)
#test.generateChannel(extrudeHeight=1e-3, extrudeLayers=1, recombineMesh=True)
"""

#test.saveMesh()      # Save the mesh
gmsh.fltk.run()      # Open Gmsh to view mesh
gmsh.finalize()
















