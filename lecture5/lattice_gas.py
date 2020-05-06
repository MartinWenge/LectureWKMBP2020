'''
modifed BFM Simulator for noninteracting particles
* no lattice occupation check -> no excluded volume
* no neighbor interactions -> model of ideal gas
* no connections -> no bond partner list, bond vector checks
''' 
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

#  ------------------------------------------------  #####  --------------------------------------------------  #
#  ------------------------------------------------  #####  --------------------------------------------------  #
class monomer:
    ''' monomer class with unique index, containing the coordinates and attributes of a single monomer '''
    def __init__(self, idx_, coords_, attributes_):
        ''' setting properties of monomer:
        idx: unique index (int),
        coords: d-dimensional coordinates (np.array),
        attributes: dict of properties (python dict),
        '''
        self.idx = idx_
        self.coords = coords_
        self.attributes = attributes_

#  ------------------------------------------------  #####  --------------------------------------------------  #
#  ------------------------------------------------  #####  --------------------------------------------------  #
class LatticeGasSimulator:
    ''' class providing utilities for 3D lattice gas simulations:
    monomer container, move and apply function '''
    def __init__(self, box_, periodicity_, delta_):
        ''' setting up simulation box:
        box = [boxX, boxY, boxZ] (python list of int),
        periodicity = [pX, pY, pZ] (python list of bool), True = is periodic, False = wall
        delta (float, interaction energy)
        ... and setup:
        empty molecules as empty list,
        calculate probability factor from delta (float),
        list of moves (python list)'''
        self.boxX, self.boxY, self.boxZ = box_
        self.pX, self.pY, self.pZ = periodicity_
        self.delta = delta_
        self.probabilityMultiplicator = np.exp(-self.delta)
        self.molecules = []
        self.moves = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        
    def addMonomer(self, coords, attributes):
        ''' add new monomer at the end of molecules '''
        newIdx = len(self.molecules)
        self.molecules.append(monomer(newIdx ,coords, attributes))

    def plotConfig(self):
        ''' plot all monomers using scatter and bonds using plot with the box as axis boundaries '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colorList = ["green","red"]
        
        # here, fold back is simple because there are no bonds
        myX = np.array([((x.coords[0]%self.boxX)+self.boxX)%self.boxX for x in self.molecules])
        myY = np.array([((x.coords[1]%self.boxY)+self.boxY)%self.boxY for x in self.molecules])
        myZ = np.array([((x.coords[2]%self.boxZ)+self.boxZ)%self.boxZ for x in self.molecules])
        
        myFixed = [(x.coords[2]==0 or x.coords[2]== (self.boxX-2) ) for x in self.molecules]
        myColors = [colorList[int(c)] for c in myFixed ]
        ax.scatter(myX,myY,myZ, c=myColors)
        
        ax.set_xlim3d(0, self.boxX)
        ax.set_ylim3d(0, self.boxY)
        ax.set_zlim3d(0, self.boxZ)
        
        fig.show()
    
    def checkMove(self, idx, direction):
        ''' apply the move checks for monomer with index idx:
        takes monomer id and key of move direction [0,self.Nmoves)
        check boundaries and adsorption energy'''

        # get new position
        oldPos = self.molecules[idx].coords
        oldX, oldY, oldZ = oldPos
        newPos = self.molecules[idx].coords + self.moves[direction]
        x, y, z = newPos
        totalProb = 1.0
        
        # check boundaries and adsorption energy: 
        if not self.pX:
            if x == -1 or x == (self.boxX-1):
                return False
            
            # monomer approaches the wall: 
            if (x == 0 and oldX == 1) or (x == (self.boxX-2) and oldX == (self.boxX-3)):
                # increase move probability if monomer wants to attach the wall
                totalProb /= self.probabilityMultiplicator
                #print("wall attach in x dir: ", x, oldPos, self.boxX, 1/self.probabilityMultiplicator, totalProb)
            # monomer detaches from the wall 
            elif (x == 1 and oldX == 0) or (x == (self.boxX-3) and oldX == (self.boxX-2)):
                # reduce move probability if monomer wants to leave the wall
                totalProb *= self.probabilityMultiplicator
                #print("wall detach in x dir: ", x, oldPos, self.boxX, self.probabilityMultiplicator, totalProb)
                
        if not self.pY:
            if y == -1 or y == (self.boxY-1):
                return False
            
            # monomer approaches the wall: 
            if (y == 0 and oldY == 1) or (y == (self.boxY-2) and oldY == (self.boxY-3)):
                # increase move probability if monomer wants to attach the wall
                totalProb /= self.probabilityMultiplicator
                #print("wall attach in y dir: ", y, oldPos, self.boxY, 1/self.probabilityMultiplicator, totalProb)
            # monomer detaches from the wall 
            elif (y == 1 and oldY == 0) or (y == (self.boxY-3) and oldY == (self.boxY-2)):
                # reduce move probability if monomer wants to leave the wall
                totalProb *= self.probabilityMultiplicator
                #print("wall detach in y dir: ", y, oldPos, self.boxY, self.probabilityMultiplicator, totalProb)
                
        if not self.pZ:
            if z == -1 or z == (self.boxZ-1):
                return False
            
            # monomer approaches the wall: 
            if (z == 0 and oldZ == 1) or (z == (self.boxZ-2) and oldZ == (self.boxZ-3)):
                # increase move probability if monomer wants to attach the wall
                totalProb /= self.probabilityMultiplicator
                #print("wall attach in z dir: ", z, oldPos, self.boxZ, 1/self.probabilityMultiplicator, totalProb)
            # monomer detaches from the wall 
            elif (z == 1 and oldZ == 0) or (z == (self.boxZ-3) and oldZ == (self.boxZ-2)):
                # reduce move probability if monomer wants to leave the wall
                totalProb *= self.probabilityMultiplicator
                #print("wall detach in z dir: ", z, oldPos, self.boxZ, self.probabilityMultiplicator, totalProb)
                
        # perform metropolis algorithm
        if ( totalProb < 1.0 ):
            if ( np.random.random() > totalProb ):
                return False
        
        # if still here, all checks have been passed
        return True
        
    def applyMove(self,idx,direction):
        ''' apply a move: set new coordinate in molecules '''
        self.molecules[idx].coords = (self.molecules[idx].coords + self.moves[direction])

    def performMCS(self,time):
        ''' apply the lattice gas algorithm on a given system for 'time' Monte Carlo sweeps '''
        counter = 0
        mol_size = len(self.molecules)
        num_steps = len(self.moves)
        for t in range(time):
            for n in range(mol_size):
                randomIdx = np.random.randint(mol_size)
                randomDir = np.random.randint(num_steps)
                if self.checkMove(randomIdx,randomDir):
                    self.applyMove(randomIdx,randomDir)
                    counter += 1
        if counter == 0:
            print( "nothing moved... ")
        #print("applied moves / attempted moves:\n{} / {} = {}".format(counter, time*mol_size, counter/(time*mol_size))) 

#  ------------------------------------------------  #####  --------------------------------------------------  #
#  ------------------------------------------------  #####  --------------------------------------------------  #
def calculateWallContacts(simulator):
    ''' calcualte the number of monomers at nonperiodic walls and normalize it by the total number of monomers'''
    nContacts = 0
    for mono in simulator.molecules:
        if not simulator.pX:
            if ( mono.coords[0] == 0 or mono.coords[0] == (simulator.boxX-2) ):
                nContacts+=1
        if not simulator.pY:
            if ( mono.coords[1] == 0 or mono.coords[1] == (simulator.boxY-2) ):
                nContacts+=1
        if not simulator.pZ:
            if ( mono.coords[2] == 0 or mono.coords[2] == (simulator.boxZ-2) ):
                nContacts+=1
    return nContacts/len(simulator.molecules)

#  ------------------------------------------------  #####  --------------------------------------------------  #
#  ------------------------------------------------  #####  --------------------------------------------------  #
def adsorptIsothermTheo(d,V):
    ''' returns the theoretical prediction of the adsorption isotherm of ideal gas from Boltzmann statistics'''
    return np.exp(d)/(np.exp(d)+V)
