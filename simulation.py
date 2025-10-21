# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 15:36:23 2025

@author: pdalc
"""
 
import numpy as np
# import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
from cc3d.core.PySteppables import SteppableBasePy
import random
from datetime import datetime
import os
import parameter as p
import time

sim_dir = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(sim_dir, 'output.txt')
# Set this directory as the working directory
os.chdir(sim_dir)

fluctuation_amplitude = p.fluctuation_amplitude
target_volume = p.target_volume
target_surface = p.target_surface
lambda_volume = p.lambda_volume
lambda_surface = p.lambda_surface
Medium_cell_CE = p.Medium_cell_CE
face_size = 4*np.sqrt(target_volume)


SKIP_TIME = 100
SIM_TIME = 1000 + SKIP_TIME #10000

from cc3d.core.PyCoreSpecs import PottsCore, CenterOfMassPlugin, PixelTrackerPlugin

specs = [PottsCore(dim_x=100, dim_y=100, neighbor_order=2, boundary_x='Periodic', boundary_y='Periodic', fluctuation_amplitude=fluctuation_amplitude), 
         CenterOfMassPlugin(),
         PixelTrackerPlugin()]

from cc3d.core.PyCoreSpecs import CellTypePlugin
cell_types = ["cell"]
cell_type_specs = CellTypePlugin(*cell_types)
specs.append(cell_type_specs)

from cc3d.core.PyCoreSpecs import VolumePlugin
volume_specs = VolumePlugin()
volume_specs.param_new("cell", target_volume=target_volume, lambda_volume=lambda_volume)
specs.append(volume_specs)

from cc3d.core.PyCoreSpecs import ContactPlugin
contact_specs = ContactPlugin(neighbor_order=4)
contact_specs.param_new(type_1="Medium", type_2="cell", energy=Medium_cell_CE)
specs.append(contact_specs)

from cc3d.core.PyCoreSpecs import SurfacePlugin
surface_specs = SurfacePlugin()
surface_specs.param_new("cell", target_surface=target_surface, lambda_surface=lambda_surface)
specs.append(surface_specs)


from cc3d.core.PyCoreSpecs import UniformInitializer
unif_init_specs = UniformInitializer()
unif_init_specs.region_new(width=face_size, pt_min=(5, 5, 0), pt_max=(5+int(face_size), 5+int(face_size), 1),
                           cell_types=["cell"])
specs.append(unif_init_specs)


#from cc3d.core.simservice import service_cc3d, service_function

class DataCollection(SteppableBasePy):
     
    #def __init__(self, random_seed): 
        #self.random_seed = random_seed
        

    def start(self):
        #service_function(self.lattice)
        for cell in self.cell_list:
            cell.dict["vol_list"] = []
            cell.dict["sur_list"] = []

    def step(self, mcs):
        
        #if not mcs % 10:
        #    self.lattice()
        if mcs > SKIP_TIME:
            for cell in self.cell_list:
                #print(cell.volume, cell.surface)
                cell.dict["vol_list"].append(cell.volume)
                cell.dict["sur_list"].append(cell.surface)
                
        if mcs == SIM_TIME-1:
            for cell in self.cell_list:
                average_vol = np.mean(cell.dict["vol_list"])
                std_vol = np.std(cell.dict["vol_list"])
                average_sur = np.mean(cell.dict["sur_list"])
                std_sur = np.std(cell.dict["sur_list"])
                # print(average_vol, std_vol, average_sur, std_sur)
                with open(output_file_path, 'a') as f:
                    f.write(f'{average_vol} {std_vol} {average_sur} {std_sur}\n')
            # create an output file with these results after the parameters, such as
            # 10 10 10 10 10 10 07 14
        
    # def lattice(self):
    #     #if mcs % 10 == 0:
    #     #print(mcs)
    #     lat = np.zeros(shape=(self.dim.x, self.dim.y))
    #     for x, y, z in self.every_pixel():
    #         cell = self.cell_field[x, y, z]
    #         if cell: lat[y,x] = cell.type
    #     fig, axs = plt.subplots(1, 1, figsize=(2, 2))
    #     axs.imshow(lat, cmap='gray')
    #     axs.set_xticks([])
    #     axs.set_yticks([])
    #     axs.set_xticklabels([])
    #     axs.set_yticklabels([])
    #     axs.set_title("gray scale")
    #     return(lat)
            
from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService

if __name__ == '__main__':

    
    # sim = CC3DSimService()
    # sim.register_specs(specs)
    # sim.register_steppable(steppable=DataCollection, frequency=1)
    # sim.run()
    # sim.init()
    # sim.start()
        
    
    # while sim.current_step < SIM_TIME:
    #     sim.step()
            
    ###################
    #start_time = time.time()
    
    N=10
    sim_list = []
    
    for n in range(N):
        sim = CC3DSimService()
        sim.register_specs(specs)
        sim.register_steppable(steppable=DataCollection, frequency=1)
        sim.run()
        sim.init()
        sim.start()
        sim_list.append(sim)
        
    for n in range(N):
        #print(n)
        while sim_list[n].current_step < SIM_TIME:
            sim_list[n].step()
            
    #end_time = time.time()
    
    #elapsed_time = end_time - start_time
    #print(f"Elapsed time: {elapsed_time:.4f} seconds")