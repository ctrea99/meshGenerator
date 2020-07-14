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

gmsh.initialize()


class deLavalNozzleGenerator:
    """
    This class is used for generating computational meshes of de Laval nozzles via
    the Gmsh Python API (https://gmsh.info/) for use within OpenFOAM CFD simulations
    (https://openfoam.org/).
    
    Functions Overview:
        - generateCrossSection: Generates the 2D cross-section geometry of the 
            domain.
        - nozzleWallFunction: Specifies the desired nozzle shape.
        - generate2DMesh: Generates a 2D structured mesh within the 2D cross-section
        - generateWedge: Generates a 3D axi-symmetric wedge section
        - generateChannel: Generates a 3D channel nozzle
        - saveMesh(): Saves the generated mesh as well as documents the 
            input parameters used.
       
    Suggested order for running the class functions:
        1. generateCrossSection()
        2. generate2DMesh()
        3. generateWedge() OR generateChannel()
        4. saveMesh()
        
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
        
        
    def generateCrossSection(self,nozzleLength, cavityLength, cavityHeight, numWallPoints=1, lc=1):
        """
        Function to generate the 2D cross-section geometry of the de Laval nozzle.
        
        An external cavity is attached to the nozzle output to simulate the nozzle
        output flow properties. The geometry consists of three sections, the nozzle
        and the upper and lower sections of the attached cavity.  The nozzle is 
        oriented such that the nozzle axis is parallel with the z-axis. The nozzle
        wall shape is defined within self.nozzleWallFunction.

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
        
        #multiplier = 1e-3
        
        #radius = 1 - (0.868 * (-z + 2)**2) + (0.432 * (-z + 2)**3)  
        radius = np.zeros(len(z))
        
        for i in range(0, len(z)):
            if z[i] <= 2e-3:
                radius[i] = 0.4e-3
            else:
                radius[i] = (2.1/10.5) * z[i]
          
        
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
        radius = self.nozzleWallFunction(z)                                     # calculate the nozzle radius at each point along axis
        
        # set constants for the inlet and outlet radii
        self.inletRadius  = radius[0]
        self.outletRadius = radius[-1]
        
        # add points to model, connect adjacent points into lines along nozzle wall
        for i in range(0, numWallPoints):
            if i == 0:
                pList[i] = gmsh.model.geo.addPoint(radius[i], 0, z[i], lc)
            else:
                pList[i] = gmsh.model.geo.addPoint(radius[i], 0, z[i], lc)
                lList[i-1] = gmsh.model.geo.addLine(pList[i],pList[i-1])
        
        # return list of point and line tags along nozzle wall
        return(pList, lList)
        
        
        
    def generate2DMesh(self, nozzleMeshDensityX, cavityMeshDensityZ=1, uCavityMeshDensityX=1, meshType='Structured', grading='False'):
        """
        Function to generate a 2D structured mesh within the nozzle cross-section.
        
        NOTE: The number of mesh cells within the nozzle along the transverse direction
            is the same as the number of mesh cells within the lower external cavity
            section along the transverse direction.

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
            adjacentSize = self.nozzleLength / (self.numWallPoints+1)
            cavityGradingZ = self.findRoots(cavityMeshDensityZ, self.cavityLength, adjacentSize)
            
            print("Grading upper cavity x-axis...")
            adjacentSize = self.outletRadius / (self.nozzleMeshDensityX)
            uCavityGradingX = self.findRoots(uCavityMeshDensityX, self.cavityHeight - self.outletRadius, adjacentSize)
        
        elif grading == 'False':
            cavityGradingZ = 1
            uCavityGradingX = 1
        
        elif grading == 'Uniform':          
            cavityGradingZ = 1
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
        if (wedgeAngle == 2 * np.pi):
            # needs more work. mesh doesn't fully connect, physical groups not declared
            volume1 = gmsh.model.geo.revolve([(2, 1)], 0, 0, 0, 0, 0, 1, np.pi, numElements = [extrudeLayers], recombine=recombineMesh)
            volume2 = gmsh.model.geo.revolve([(2, 2)], 0, 0, 0, 0, 0, 1, np.pi, numElements = [extrudeLayers], recombine=recombineMesh)
            volume3 = gmsh.model.geo.revolve([(2, 3)], 0, 0, 0, 0, 0, 1, np.pi, numElements = [extrudeLayers], recombine=recombineMesh)
            
            volume4 = gmsh.model.geo.revolve([(2, 1)], 0, 0, 0, 0, 0, -1, np.pi, numElements = [extrudeLayers], recombine=recombineMesh)
            volume5 = gmsh.model.geo.revolve([(2, 2)], 0, 0, 0, 0, 0, -1, np.pi, numElements = [extrudeLayers], recombine=recombineMesh)
            volume6 = gmsh.model.geo.revolve([(2, 3)], 0, 0, 0, 0, 0, -1, np.pi, numElements = [extrudeLayers], recombine=recombineMesh)
        else: 
            volume1 = gmsh.model.geo.revolve([(2, 1)], 0, 0, 0, 0, 0, 1, wedgeAngle, numElements = [extrudeLayers], recombine=recombineMesh)
            volume2 = gmsh.model.geo.revolve([(2, 2)], 0, 0, 0, 0, 0, 1, wedgeAngle, numElements = [extrudeLayers], recombine=recombineMesh)
            volume3 = gmsh.model.geo.revolve([(2, 3)], 0, 0, 0, 0, 0, 1, wedgeAngle, numElements = [extrudeLayers], recombine=recombineMesh)
                        
            
        gmsh.model.geo.synchronize()
          
        
        ### move declaration of physical groups to new function?
        # Set physical groups
        # Group for nozzle wall patches
        temp = self.findSides(tagList=volume1, start=2, stop=-3)                # isolate tags for desired sides from extrusion
        temp.extend(self.findSides(tagList=volume3, start=-3, stop=-2))                            # ^
        gmsh.model.addPhysicalGroup(dim=2, tags=temp, tag=1)
        gmsh.model.setPhysicalName( dim=2, tag=1, name="Outerwall")
        
        # Group for left wedge side patches
        gmsh.model.addPhysicalGroup(dim=2, tags=[1,2,3], tag=2)
        gmsh.model.setPhysicalName( dim=2, tag=2, name="Left")
        
        # Group for wedge right side patches
        gmsh.model.addPhysicalGroup(dim=2, tags=[volume1[0][1], volume2[0][1], volume3[0][1]], tag=3)
        gmsh.model.setPhysicalName( dim=2, tag=3, name="Right")
        
        # Group for wedge axis patch
        gmsh.model.addPhysicalGroup(dim=1, tags=[self.l1, self.l5], tag=4)
        gmsh.model.setPhysicalName( dim=1, tag=4, name="Axis")
        
        # Group for wedge inlet patches
        gmsh.model.addPhysicalGroup(dim=2, tags=[volume1[-2][1]], tag=5)
        gmsh.model.setPhysicalName( dim=2, tag=5, name="Input")
        
        # Group for wedge outlet patches
        gmsh.model.addPhysicalGroup(dim=2, tags=[volume2[2][1], volume3[2][1]], tag=6)
        gmsh.model.setPhysicalName( dim=2, tag=6, name="Output")
        
        # Group for internal mesh
        gmsh.model.addPhysicalGroup(dim=3, tags=[volume1[1][1], volume2[1][1], volume3[1][1]], tag=7)
        gmsh.model.setPhysicalName( dim=3, tag=7, name="internal")
             
        
        gmsh.model.geo.synchronize()
        
        # rotate wedge back so it is symmetric across the x-axis
        entityTags= gmsh.model.getEntities(dim=-1)
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
        volume1 = gmsh.model.geo.extrude((2, 1),        0, extrudeHeight, 0, numElements=[extrudeLayers], heights=[], recombine=recombineMesh) 
        volume2 = gmsh.model.geo.extrude((2, 2),        0, extrudeHeight, 0, numElements=[extrudeLayers], heights=[], recombine=recombineMesh) 
        volume3 = gmsh.model.geo.extrude((2, 3),        0, extrudeHeight, 0, numElements=[extrudeLayers], heights=[], recombine=recombineMesh) 
        volume4 = gmsh.model.geo.extrude(s1_reflect,    0, extrudeHeight, 0, numElements=[extrudeLayers], heights=[], recombine=recombineMesh)
        volume5 = gmsh.model.geo.extrude(s2_reflect,    0, extrudeHeight, 0, numElements=[extrudeLayers], heights=[], recombine=recombineMesh)
        volume6 = gmsh.model.geo.extrude(s3_reflect,    0, extrudeHeight, 0, numElements=[extrudeLayers], heights=[], recombine=recombineMesh)
        
        
        # Set physical groups
        # Group for nozzle wall patches
        temp = self.findSides(tagList=volume1, start=2, stop=-5)
        temp.extend(self.findSides(tagList=volume4, start=2, stop=-4))      
        temp.extend([volume3[4][1], volume6[4][1]])
        gmsh.model.addPhysicalGroup(dim=2, tags=temp, tag=1)
        gmsh.model.setPhysicalName(dim=2, tag=1, name="Outerwall")
           
        # Group for the side patches
        temp = [1,                  2,                  3, 
                s1_reflect[0][1],   s2_reflect[0][1],   s3_reflect[0][1],
                volume1[0][1],      volume2[0][1],      volume3[0][1],
                volume4[0][1],      volume5[0][1],      volume6[0][1]]
        gmsh.model.addPhysicalGroup(dim=2, tags=temp, tag=2)
        gmsh.model.setPhysicalName(dim=2, tag=2, name="Sides")
        
        # Group for the nozzle inlet patches
        temp = [volume1[-4][1], volume4[-3][1]]
        gmsh.model.addPhysicalGroup(dim=2, tags=temp, tag=3)
        gmsh.model.setPhysicalName(dim=2, tag=3, name='Input')
        
        # Group for the nozzle outlet patches
        temp = [volume2[3][1], volume3[2][1], volume3[3][1], 
                volume5[3][1], volume6[2][1], volume6[3][1]]
        gmsh.model.addPhysicalGroup(dim=2, tags=temp, tag=4)
        gmsh.model.setPhysicalName(dim=2, tag=4, name="Output")
        
        # Group for the internal mesh
        temp = [volume1[1][1], volume2[1][1], volume3[1][1],
                volume4[1][1], volume5[1][1], volume6[1][1]]
        gmsh.model.addPhysicalGroup(dim=3, tags=temp, tag=5)
        gmsh.model.setPhysicalName(dim=3, tag=5, name="internal")
        
        
        gmsh.model.geo.synchronize()
        
        
        # generate the 3D mesh
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.recombine()        
        gmsh.model.mesh.generate(3)
       
       
    
        
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
            docFile.write("nozzleLength:    %.2f\n"   %(self.nozzleLength))
            docFile.write("inletRadius:     %.2f\n"   %(self.inletRadius))
            docFile.write("outletRadius:    %.2f\n"   %(self.outletRadius))
            docFile.write("cavityLength:    %.2f\n"   %(self.cavityLength))
            docFile.write("cavityHeight:    %.2f\n"   %(self.cavityHeight))
            docFile.write("numWallPoints:   %.2f\n"   %(self.numWallPoints))
            docFile.write("lc:              %.2f\n"   %(self.lc))
            docFile.write("\n")
        
        # if the function for generating the 2D cross-section mesh has been run
        if self.MESH2D == 1:
            docFile.write("MESH\n")
            docFile.write("nozzleMeshDensityX:  %d\n"    %(self.nozzleMeshDensityX))
            docFile.write("cavityMeshDensityZ:  %d\n"    %(self.cavityMeshDensityZ))
            docFile.write("uCavityMeshDensityX: %d\n"    %(self.uCavityMeshDensityX))
            docFile.write("meshType:            %s\n"    %(self.meshType))
            docFile.write("grading:             %s\n"    %(self.grading))
            docFile.write("\n")
        
        # if the function for generating a 3D wedge geometry has been run
        if self.WEDGE == 1:
            docFile.write("WEDGE\n")
            docFile.write("wedgeAngle:      %.2f\n"   %(self.wedgeAngle))
            docFile.write("extrudeLayers:   %d\n"     %(self.extrudeLayers))
            docFile.write("recombineMesh:   %s\n"     %(self.recombineMesh))
            docFile.write("\n")
            
        if self.CHANNEL == 1:
            docFile.write("CHANNEL\n")
            docFile.write("extrudeHeight:   %.2f\n"   %(self.extrudeHeight))
            docFile.write("extrudeLayers:   %d\n"     %(self.extrudeLayers))
            docFile.write("recombineMesh:   %s\n"     %(self.recombineMesh))
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
        functResidual = 1e-5        # maximum value of the function when calculated at the current root value
        initialGuess = 1.1          # initial guess for the value of the root
        guessIncrement = 1          # amount to increment initial guess to find next root
        
        # if the initial, uniform mesh density in the region is too high for grading
        uniformSize = length / numCells
        if uniformSize <= adjacentCellSize:
            print("Grading could not be performed")
            return(1)
        
        r = initialGuess
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
                # increment the initial guess to find the region of convergence of the next root
                funct = 1000
                initialGuess += guessIncrement
                r = initialGuess * 1
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
test.generateCrossSection(nozzleLength=12.5e-3, cavityLength=50e-3, cavityHeight=25e-3, numWallPoints=80)
test.generate2DMesh(nozzleMeshDensityX=40, cavityMeshDensityZ=60, uCavityMeshDensityX=60, grading='Uniform')
#test.generateWedge(2 * np.pi, 20)
#test.generateWedge(0.05, 1)
test.generateChannel(extrudeHeight=0.5e-3, extrudeLayers=1)
test.saveMesh()
gmsh.fltk.run()     # opens Gmsh to view generated mesh
gmsh.finalize()

















