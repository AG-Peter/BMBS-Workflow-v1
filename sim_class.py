import numpy as np
from MDAnalysis import Universe
import MDAnalysis.analysis.distances
from MDAnalysis.analysis import align
import MDAnalysis.transformations as trans
import os
import running_rabbit as rr
import shutil
from utility_functions import *
import backward
import time
import pickle
import re
import encodermap as em

#used as template to initialize sims
class Peptides:
    peptides = []
    def __init__(self, charge_state, peptide_name, cluster_name):
        Peptides.peptides.append(self)
        self.peptide_name = peptide_name
        self.cluster_name = cluster_name 
        self.charge_state = charge_state


class Cluster_sim:
    is_class_object = True
    base_directory = "/home/nicolas.schneider/nicolas/master_thesis/"
    CG_sim_directory = base_directory + "CG_sim/martini3/"
    evaluation_data_base_directory = base_directory + "evaluation_data/"
    evaluation_data_spec_directory = ""
    sel_charges = None
    sel_peptides = None
    descriptor_selection = None
    high_d_data_concat = np.array([[]])
    frame_dict_concat = {}
    cg_sim_list = []
    failed_runs = []
    contact_count = False
    cluster_ids = False
    high_d_data_file = "high_d_data.npy"
    frame_dict_file = "frame_dict.pickle"
    e_map = None
    e_map_path = False
    pending_sims = False
    
    def __init__(self, sim_type_folder, charge_folder, peptide_folder, cluster_folder, frame_number = None, run_folder = None, sim_directory = None):
        self.is_class_object = False
        self.sim_type_folder = sim_type_folder 
        self.charge_folder = charge_folder
        self.peptide_folder = peptide_folder
        self.cluster_folder = cluster_folder
        #self.contact_count = False      
        #self.ndx_path = None
        self.sim_setup_path = None
        
        #if statement to create specialized versions of the class(differentiate cg from aa)
        if "CG_sim" in self.sim_type_folder:
            self.cluster_directory = os.path.join(Cluster_sim.base_directory, self.sim_type_folder, self.charge_folder, self.peptide_folder, self.cluster_folder, "plain/")
            self.sim_directory = os.path.join(self.cluster_directory, "4-md/")
            self.em_runs_path = os.path.join(self.cluster_directory, "emruns")
            self.topology = os.path.join(self.sim_directory, "topol.tpr")
            self.trajectory = os.path.join(self.sim_directory, "traj_comp_center.xtc")
            self.aa_top_path = os.path.join(Cluster_sim.base_directory, self.sim_type_folder, "init_atomistic/", peptide_folder, peptide_folder + "_" + charge_folder.split("_")[2] + ".top")
            self.child_sims = []

        if "AA_sim" in self.sim_type_folder:
            self.run_folder = run_folder
            self.charge_folder = charge_folder
            self.sim_name = "-"+frame_number+"-"+charge_folder+"-"+peptide_folder+"-"+cluster_folder+"-"+sim_type_folder+"-"
            self.frame_number = frame_number
            self.sim_directory = sim_directory #os.path.join(Cluster_sim.base_directory, self.sim_type_folder, "backward_atomistic", self.charge_folder, self.peptide_folder, self.cluster_folder, "runs/", run_folder, "aa_structures/", self.sim_name)
            self.topology = os.path.join(self.sim_directory, "production.tpr")
            self.trajectory = os.path.join(self.sim_directory, "production.xtc")
            self.setup_path = os.path.join(self.sim_directory, "setup.sh")
            self.init_top_path = os.path.join(self.sim_directory, "init.top")
            self.init_gro_path = os.path.join(self.sim_directory, "init.gro")
            self.coarsegrained_path = os.path.join(self.sim_directory, "coarsegrained")
            self.sim_status = "pending"
                    

        
    @property
    def universe(self):
        #creates a universe when calling the universe attribute, if it doesn't exist
        if not hasattr(self, "universep"):
            self.universep = Universe(self.topology, self.trajectory)
        return self.universep
    
    def save_state(path):
        #saves class and all object states
        attr_dict = {}

        for attribute in Cluster_sim.__dict__.keys():
                if attribute[:2] != '__':
                    value = getattr(Cluster_sim, attribute)
                    if not callable(value) and not isinstance(value, property) and type(value) != em.encodermap.EncoderMap:
                        attr_dict[attribute] = value                    

        with open(path, "wb") as file:
            pickle.dump(attr_dict, file)

    def load_state(path):
        #loads a cluster sim state from a pickle, compatible with encodermaps
        dict_copy = list(Cluster_sim.__dict__.keys()).copy()
        for attribute in dict_copy:
            if attribute[:2] != '__':
                    value = getattr(Cluster_sim, attribute)
                    if not callable(value):
                        delattr(Cluster_sim, attribute)
        with open(path, "rb") as file:
            coll_attrs = pickle.load(file)
            for attr, value in coll_attrs.items():
                setattr(Cluster_sim, attr, value)
                
        if Cluster_sim.e_map_path:
            loaded_parameters = em.Parameters.load(os.path.join(Cluster_sim.e_map_path, "parameters.json"))
            
            cwd = os.getcwd()
            os.chdir(os.path.join(Cluster_sim.e_map_path, "checkpoints/"))
            steps = glob.glob("step*.ckpt*")
            os.chdir(cwd)
            
            stepnums = []
            for step in steps:
                stepnums.append(int(re.search("step([0-9]+)", step).group(1)))
            
            checkpoint_path = os.path.join(Cluster_sim.e_map_path, "checkpoints/step{}.ckpt".format(str(max(stepnums))))

            Cluster_sim.e_map = em.EncoderMap(loaded_parameters, checkpoint_path=checkpoint_path, n_inputs=len(Cluster_sim.high_d_data_concat[0]))
        
        
    def gen_end2end_dist_ts(self, selection = "name BB"):
        #calculates an end to end distance timeseries, adapted for cg asp/glu 
        start_term = self.universe.select_atoms(selection)[0]
        end_term = self.universe.select_atoms(selection)[-1]
        self.sim_time = []
        self.e2e_dist = []
        
        for ts in self.universe.trajectory:
            r = start_term.position - end_term.position  # end-to-end vector from atom positions
            self.sim_time.append(ts.time)
            self.e2e_dist.append(np.linalg.norm(r))
            
        return self.e2e_dist
            
    
    def gen_pair_dist_ts(self, selection, sample_frames=None, frame_offset=0, utype = "def"):
        #generates a pair distance matrix for self, selection: MDA selection string; sample_frames:amount of frames to sample; utype is for recoarsegrained aa_sims  
        self.idx_list = []
        self.idx_dict = {}      #generate idx dictionary to assign cluster_ids to time steps, can be exported and saved with pandas
        
        if utype == "def":
            universe = self.universe
        elif utype == "cg":
            universe = self.cg_universe

            
        #this part is for selecting frames with a gromacs ndx file from clustering
        if hasattr(self, "ndx_path"):
            print(self.ndx_path)
            if os.path.isfile(self.ndx_path):
                ndx = gromacs.fileformats.NDX()
                ndx.read(self.ndx_path)

                for name in ndx.groups:
                    for frame in ndx.get(name):
                        self.idx_list.append([(frame-1)*100, int(name.replace("Cluster_", ""))])
                self.sample_frames = len(self.idx_list)

                for i in range(len(self.idx_list)):                  #save dict once for all clusters(?)
                    self.idx_dict[self.idx_list[i][0]] = self.idx_list[i][1]
        
        #this part is for selecting frames with a gromacs ndx file from clustering
        if sample_frames == None:
            self.nth_step = None
            self.cluster_ids = np.array([])
            self.frame_dict = {}     #assigns the simulation frame number to each time step

            sel = universe.select_atoms(selection)
            self.high_d_data = np.zeros((int(len(self.idx_list)), int(sel.n_atoms*(sel.n_atoms-1)/2)))

            frame_idx = 0
            for frame in np.array(self.idx_list)[:,0]:

                self.cluster_ids = np.append(self.cluster_ids, self.idx_dict[frame])           

                universe.trajectory[frame]     # https://www.mdanalysis.org/MDAnalysisTutorial/analysismodule.html doesnt work? using workaround, new selection for each frame
                sel = universe.select_atoms(selection) 
                MDAnalysis.analysis.distances.self_distance_array(sel.positions, box = universe.trajectory.ts.dimensions, backend='OpenMP', result = self.high_d_data[frame_idx])

                self.frame_dict[frame_idx] = frame
                frame_idx += 1
        
        
        else:
            self.sample_frames = sample_frames
            self.frame_offset = frame_offset
            self.nth_step =(len(universe.trajectory)-1)/(sample_frames-1) #every n'th step is appended to the pair distance time step array
            sel = universe.select_atoms(selection)
            columns = int(sel.n_atoms*(sel.n_atoms-1)/2)
            self.high_d_data = np.zeros((sample_frames, columns))
            ts = 0 + frame_offset
            self.cluster_ids = np.array([]) 
            self.frame_dict = {}     #assigns the simulation frame number to each time step   
                                        
            for i in range(self.sample_frames):
                rts = int(round(ts,0))

                if rts in self.idx_dict:
                    self.cluster_ids = np.append(self.cluster_ids, self.idx_dict[rts])  #if the frame has a cluster id, the id is added to the list, else 0 is added to the list         

                else:
                    self.cluster_ids = np.append(self.cluster_ids, 0)


                universe.trajectory[rts]     # https://www.mdanalysis.org/MDAnalysisTutorial/analysismodule.html doesnt work? using workaround, new selection for each frame
                sel = universe.select_atoms(selection) 
                MDAnalysis.analysis.distances.self_distance_array(sel.positions, box = universe.trajectory.ts.dimensions, backend='OpenMP', result = self.high_d_data[i])

                self.frame_dict[i] = rts       

                ts += self.nth_step

        return self.high_d_data

    def write_frames(self, frames, path, selection_str = "name BB or name SC1", universe = False, sims = False, pdb = True, gro = True):
        #writes the selected frames and selected atoms as pdb and gro, with the protein centered in the box, frames has to be a list of frame identifiers(the values of the frame dict)
        if self.is_class_object == False:
            if universe:
                mobile = universe
            else:
                mobile = Universe(self.topology, self.trajectory)

            protein_mob = mobile.select_atoms('protein')
            not_protein_mob = mobile.select_atoms('not protein')
            
            #this centers the protein in the box
            workflow_mob = [trans.unwrap(protein_mob),
                        trans.center_in_box(protein_mob, wrap=False),
                        trans.wrap(not_protein_mob)]
            mobile.trajectory.add_transformations(*workflow_mob)

            selection = mobile.select_atoms(selection_str)

            self.pdb_path_list = []
            self.gro_path_list = []
            
            if not os.path.exists(os.path.dirname(os.path.join(path, "traj_frames"))):
                os.makedirs(os.path.join(path, "traj_frames"))
            
            for frame in frames:
                mobile.trajectory[frame[0]]

                #creates a file name from frame identifier
                frame_name = "-"+str(frame[0])+"-"+frame[1]+"-"+frame[2]+"-"+frame[3]+"-"+frame[4]+"-"

                selection.write(path + "/traj_frames" + "/selected_traj_{frame_nr}.pdb".format(frame_nr=frame_name))
                selection.write(path + "/traj_frames" + "/selected_traj_{frame_nr}.gro".format(frame_nr=frame_name))
                
                self.pdb_path_list.append(path + "/traj_frames" + "/selected_traj_{frame_nr}.pdb".format(frame_nr=frame_name))
                self.gro_path_list.append(path + "/traj_frames" + "/selected_traj_{frame_nr}.gro".format(frame_nr=frame_name))
            
            return self.pdb_path_list, self.gro_path_list
    
        #second option when writing pdbs for multiple sims, you can also just call this method for each sim instead
        if self.is_class_object == True:
            for sim in sims:
                
                mobile = Universe(sim.topology, sim.trajectory)

                protein_mob = mobile.select_atoms('protein')
                not_protein_mob = mobile.select_atoms('not protein')

                workflow_mob = [trans.unwrap(protein_mob),
                            trans.center_in_box(protein_mob, wrap=False),
                            trans.wrap(not_protein_mob)]
                mobile.trajectory.add_transformations(*workflow_mob)

                selection = mobile.select_atoms("name BB or name SC1")

                sim.pdb_path_list = []
                sim.gro_path_list = []

                for frame in frames:
                    mobile.trajectory[frame[0]]

                    frame_name = "-"+str(frame[0])+"-"+frame[1]+"-"+frame[2]+"-"+frame[3]+"-"+frame[4]+"-"
                    
                    selection.write(path + "/traj_frames" + "/selected_traj_{frame_nr}.pdb".format(frame_nr=frame_name))
                    selection.write(path + "/traj_frames" + "/selected_traj_{frame_nr}.gro".format(frame_nr=frame_name))

                    sim.pdb_path_list.append(path + "/traj_frames" + "/selected_traj_{frame_nr}.pdb".format(frame_nr=frame_name))
                    sim.gro_path_list.append(path + "/traj_frames" + "/selected_traj_{frame_nr}.gro".format(frame_nr=frame_name))

                
    
    def backward_frames(self, em_run_path, rr_template, sims = False, frame_paths = False, skip_em0 = False):
        #runs backward on selected frames, does an energy minimization and sets up the atomistic simulation, em_run_path: Cluster_sim.run_folder - this method is set up for the specific folder structure used in this project, rr_template: running rabbit template name to set up atomistic simulations, frame_paths: paths to the frames that should be backmapped
        if self.is_class_object == True:
        
            if not sims:
                sims = Cluster_sim.sim_list
                        
            for sim in sims:
                
                print(sim.cluster_folder)
                
                #this part generates an atomistic topology file
                if not os.path.exists(sim.aa_top_path):

                    cwd = os.getcwd()

                    folder = os.path.join(Cluster_sim.base_directory,sim.sim_type_folder,"init_atomistic/",sim.peptide_folder)
                    os.chdir(folder)
                    if not os.path.exists(folder + "/charmm36-jul2021.ff"):
                        os.symlink("/home/soft/gromacs/forcefields/charmm36-jul2021.ff", folder + "/charmm36-jul2021.ff")
                    input_pdb = os.path.basename(glob.glob(folder + "/*.pdb")[0])
                    
                    
                    if "10" in sim.peptide_folder:
                        output_top = os.path.basename(folder) + "_neutral.top"
                        os.system("printf '1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1' | gmx pdb2gmx -f {input_pdb} -p {output_top} -ff charmm36-jul2021 -asp -glu -ter -ignh".format(input_pdb=input_pdb, output_top=output_top))


                        output_top = os.path.basename(folder) + "_charged.top"
                        os.system("printf '1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0' | gmx pdb2gmx -f {input_pdb} -p {output_top} -ff charmm36-jul2021 -asp -glu -ter -ignh".format(input_pdb=input_pdb, output_top=output_top))

                    elif "20" in sim.peptide_folder:

                        output_top = os.path.basename(folder) + "_neutral.top"
                        os.system("printf '1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1' | gmx pdb2gmx -f {input_pdb} -p {output_top} -ff charmm36-jul2021 -asp -glu -ter -ignh".format(input_pdb=input_pdb, output_top=output_top))


                        output_top = os.path.basename(folder) + "_charged.top"
                        os.system("printf '1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0' | gmx pdb2gmx -f {input_pdb} -p {output_top} -ff charmm36-jul2021 -asp -glu -ter -ignh".format(input_pdb=input_pdb, output_top=output_top))
                        
                    os.chdir(cwd)
                #looks for frame paths if none are given
                if hasattr(sim, 'pdb_path_list') and not frame_paths:
                    frame_paths = sim.pdb_path_list
                    
                if frame_paths:

                    for frame_path in frame_paths:
                        #folder setup
                        frame_number, charge_folder, peptide_folder, cluster_folder, sim_type_folder = os.path.basename(frame_path).split("-")[1:-1]
                        run_folder = os.path.basename(os.path.normpath(em_run_path))
                        active_aa_sim_dir = os.path.join(em_run_path.replace("CG_sim", "AA_sim/backward_atomistic").replace("evaluation_data/", ""), "aa_structures")
                        active_aa_sim_dir += "/" + os.path.basename(frame_path)[:-4]
                        #creates a new Cluster_sim object
                        aa_sim = Cluster_sim("AA_sim", charge_folder, peptide_folder, cluster_folder, frame_number = frame_number, run_folder = run_folder, sim_directory = active_aa_sim_dir)
                        sim.child_sims.append(aa_sim)



                        print("active_aa_sim_dir: ", active_aa_sim_dir)
                        if not os.path.exists(active_aa_sim_dir):
                            os.makedirs(active_aa_sim_dir)
                        #linking required files
                        cwd = os.getcwd()
                        os.chdir(active_aa_sim_dir)
                        cg_structure = os.path.join(active_aa_sim_dir, os.path.basename(frame_path))
                        aa_topology = os.path.join(active_aa_sim_dir, os.path.basename(sim.aa_top_path))
                        if not os.path.exists(cg_structure):
                            os.symlink(frame_path, cg_structure)
                        if not os.path.exists(aa_topology):
                            os.symlink(sim.aa_top_path, aa_topology)
                        if not os.path.exists(os.path.join(active_aa_sim_dir, "charmm36-jul2021.ff")):
                            os.symlink("/home/soft/gromacs/forcefields/charmm36-jul2021.ff", os.path.join(active_aa_sim_dir, "charmm36-jul2021.ff"))    
                        
                        aa_structure_raw = "init.gro"
                        aa_topology_raw = "init.top"
                        input_gro = aa_structure_raw
                        
                        #skip_em0 = True to skip the initial charmm energy minimization to repair broken structures
                        if not skip_em0:
                            aa_structure_raw = "init0.gro"
                            aa_topology_raw = "init0.top"
                        

                        args = '-f {} -p {} -o {} -po {} -to charmm36'.format(os.path.basename(frame_path), os.path.basename(sim.aa_top_path), aa_structure_raw, aa_topology_raw)  # -kick 0.05
                        args_list = args.split()

                        backward.workflow(args_list)
                        
                        if not skip_em0:
                        
                            #sets up a position restraints file with variable position restraints: POSRES_FC_BB - C-alpha, POSRES_FC_H - hydrogen, POSRES_FC_SC - all other protein atoms
                            os.system("printf '0' | gmx genrestr -f {} -o posre.itp".format(aa_structure_raw))
                            universe = MDAnalysis.Universe(os.path.join(active_aa_sim_dir, aa_structure_raw))
                            selC = universe.select_atoms("name CA")
                            selH = universe.select_atoms("name H*")

                            with open('posre.itp', 'r+') as file:
                                data = file.read()
                                for i in selC.indices:
                                    i += 1
                                    data = re.sub("\s" + str(i) + r"([ ]*)1([ ]*)1000([ ]*)1000([ ]*)1000" , " " + str(i) + r"\g<1>1\2POSRES_FC_BB\3POSRES_FC_BB\4POSRES_FC_BB", data)

                                for i in selH.indices:
                                    i += 1
                                    data = re.sub("\s" + str(i) + r"([ ]*)1([ ]*)1000([ ]*)1000([ ]*)1000" , " " + str(i) + r"\g<1>1\2POSRES_FC_H\3POSRES_FC_H\4POSRES_FC_H", data)

                                data = re.sub("1000", "POSRES_FC_SC", data)

                                file.seek(0)
                                file.write(data)
                                file.truncate()

                            ## make all necessary run files like .mdp
                            rabbit = rr.Rabbit(ff="charmm36", template_name="backmapping_em", script_name = "setup0.sh", initial_name = "init0")                       
                            rabbit.run()
                            
                            #run em0
                            os.system("bash setup0.sh")
                            
                            input_gro = "em0.gro"
                        
                        output_gro = "init.gro" #aa_structure_raw.replace("charmm36", "gromos54a7")
                        output_top = "init.top"

                        #change to gromos ff and set charges
                        if "10" in sim.peptide_folder:
                        
                            if "charge" in sim.charge_folder:
                                os.system("printf '1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0' | gmx pdb2gmx -f {input_gro} -p {output_top} -o {output_gro} -ff gromos54a7 -asp -glu -ter -ignh".format(input_gro=input_gro, output_top=output_top, output_gro=output_gro))
                                print("Peptide set to charged: " + sim.charge_folder)
                                #!gmx editconf -box Boxgröße auslesen und korrigieren 

                            if "neutral" in sim.charge_folder:
                                os.system("printf '1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1' | gmx pdb2gmx -f {input_gro} -p {output_top} -o {output_gro} -ff gromos54a7 -asp -glu -ter -ignh".format(input_gro=input_gro, output_top=output_top, output_gro=output_gro))
                                print("Peptide set to neutral: " + sim.charge_folder)
                                
                        elif "20" in sim.peptide_folder:
                            if "charge" in sim.charge_folder:
                                os.system("printf '1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0' | gmx pdb2gmx -f {input_gro} -p {output_top} -o {output_gro} -ff gromos54a7 -asp -glu -ter -ignh".format(input_gro=input_gro, output_top=output_top, output_gro=output_gro))
                                print("Peptide set to charged: " + sim.charge_folder)
                                #!gmx editconf -box Boxgröße auslesen und korrigieren 

                            if "neutral" in sim.charge_folder:
                                os.system("printf '1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1' | gmx pdb2gmx -f {input_gro} -p {output_top} -o {output_gro} -ff gromos54a7 -asp -glu -ter -ignh".format(input_gro=input_gro, output_top=output_top, output_gro=output_gro))
                                print("Peptide set to neutral: " + sim.charge_folder)

                        #path_em = active_aa_sim_dir + "/1_em"
                        
                        rabbit              = rr.Rabbit(ff="gromos54a7", template_name=rr_template)
                        rabbit.structure    = output_gro
                        rabbit.topology     = output_top
                        #rabbit.destination  = path_em
                        ## make all necessary run files like .mdp
                        rabbit.run()

                        #fixes running rabbits not setting correct names for steps
                        with open('setup.sh', 'r+') as file:
                            data = file.read().replace('name="solvate"\nname_pre="eq1"', 'name="solvate"\nname_pre="eq5"')
                            data = data.replace("-r ${name_pre}.gro", "-r init.gro")
                            data = data.replace("-p ${name_pre}.top -o ${name}.tpr -po ${name}_out.mdp -maxwarn 2 -r init.gro -maxwarn 4", "-p ${name_pre}.top -o ${name}.tpr -po ${name}_out.mdp -maxwarn 4 -r init.gro")
                            file.seek(0)
                            file.write(data)
                            file.truncate()
                        
                        #generate posre file
                        os.system("printf '0' | gmx genrestr -f {} -o posre.itp".format(output_gro))
                        
                        universe = MDAnalysis.Universe(os.path.join(active_aa_sim_dir, output_gro))
                        selC = universe.select_atoms("name CA")
                        selH = universe.select_atoms("name H*")
                        
                        with open('posre.itp', 'r+') as file:
                            data = file.read()
                            for i in selC.indices:
                                i += 1
                                data = re.sub("\s" + str(i) + r"([ ]*)1([ ]*)1000([ ]*)1000([ ]*)1000" , " " + str(i) + r"\g<1>1\2POSRES_FC_BB\3POSRES_FC_BB\4POSRES_FC_BB", data)
                                
                            for i in selH.indices:
                                i += 1
                                data = re.sub("\s" + str(i) + r"([ ]*)1([ ]*)1000([ ]*)1000([ ]*)1000" , " " + str(i) + r"\g<1>1\2POSRES_FC_H\3POSRES_FC_H\4POSRES_FC_H", data)
                                
                            data = re.sub("1000", "POSRES_FC_SC", data)
                            
                            file.seek(0)
                            file.write(data)
                            file.truncate()
                        
                        os.remove(cg_structure)
                        os.remove(aa_topology)
                        os.remove(os.path.join(active_aa_sim_dir, "charmm36-jul2021.ff"))
                        os.chdir(cwd)

                    print("\n\n\n\n\n\n\n\n\nend of loop\n\n\n\n\n\n")
            
    def run_sim(sim_list, location = "MLS", MLS_working_directory = "/home/kn/kn_kn/kn_pop511058/sim", client = False, ssh = False, force_retry = False, submission_script = "submission_script.sh", local_gmx = None):
        # runs the sims in sim_list set up with backward frames
        Cluster_sim.failed_runs = []
        
        #for running on local hardware
        if location == "local":
            
            pending_sims = sim_list.copy()
            progress(len(sim_list) - len(pending_sims), len(sim_list))
            
            for aa_sim in sim_list:

                cwd = os.getcwd()
                os.chdir(aa_sim.sim_directory)
                #os.system("export OMP_NUM_THREADS=4")
                #workaround to get te right gromacs version
                with open('setup.sh', 'r+') as file:
                    data = file.read()
                    if local_gmx == "gromacs-2021.5-cuda-shared":
                        data = re.sub("# This script was auto generated.", "# This script was auto generated.\nunset OMP_NUM_THREADS\nubuntu_version=$(lsb_release -r | awk '{ print $2 }');\nsource /home/soft/gromacs/gromacs-2021.5/inst/cuda_shared_${ubuntu_version}/bin/GMXRC.bash", data)
                        
                    else:
                        data = re.sub("# This script was auto generated.", "# This script was auto generated.\nexport OMP_NUM_THREADS=4", data)
                    print(data)
                    file.seek(0)
                    file.write(data)
                    file.truncate()

                #ensuring the right posre file is used. might be redundant    
                os.system("printf '0' | gmx genrestr -f {} -o posre.itp".format("init.gro"))

                universe = MDAnalysis.Universe(os.path.join(aa_sim.sim_directory, "init.gro"))
                selC = universe.select_atoms("name CA")
                selH = universe.select_atoms("name H*")

                with open('posre.itp', 'r+') as file:
                    data = file.read()
                    for i in selC.indices:
                        i += 1
                        data = re.sub("\s" + str(i) + r"([ ]*)1([ ]*)1000([ ]*)1000([ ]*)1000" , " " + str(i) + r"\g<1>1\2POSRES_FC_BB\3POSRES_FC_BB\4POSRES_FC_BB", data)

                    for i in selH.indices:
                        i += 1
                        data = re.sub("\s" + str(i) + r"([ ]*)1([ ]*)1000([ ]*)1000([ ]*)1000" , " " + str(i) + r"\g<1>1\2POSRES_FC_H\3POSRES_FC_H\4POSRES_FC_H", data)

                    data = re.sub("1000", "POSRES_FC_SC", data)

                    file.seek(0)
                    file.write(data)
                    file.truncate()
                
                os.system("bash setup.sh")
                os.chdir(cwd)

                while not os.path.exists(aa_sim.topology) and not os.path.exists(aa_sim.trajectory):
                    time.sleep(2)
                    
                pending_sims.pop(pending_sims.index(aa_sim))
                progress(len(sim_list) - len(pending_sims), len(sim_list))
        
        #for connecting with a slurm computing cluster, this requires to pass the paramiko client and ssh as keyword arguments
        if location == "MLS":
            if not client:
                client = connect_MLS()
                ssh = establish_sftp(client)
            
            if not ssh:
                ssh = establish_sftp(client)
            
            for aa_sim in sim_list:

                ###make aa_sim folder on cluster
                if aa_sim.sim_status == "suspended":
                    continue
                if aa_sim.sim_status == "pending" or aa_sim.sim_status == "failed":
                    aa_sim.MLS_working_directory = MLS_working_directory

                    aa_sim.MLS_sim_path = os.path.join(MLS_working_directory, os.path.relpath(aa_sim.sim_directory, start = Cluster_sim.base_directory))

                    aa_sim.MLS_topology = os.path.join(aa_sim.MLS_sim_path, "production.top")
                    aa_sim.MLS_trajectory = os.path.join(aa_sim.MLS_sim_path, "production.xtc")

                    command_return(client, "mkdir -p " + aa_sim.MLS_sim_path)

                    ###upload setup, structure, topology, mdp

                    put_folder(client, ssh, aa_sim.sim_directory, aa_sim.MLS_sim_path)
                    aa_sim.sim_status = "uploaded"
                
                ###copy submission_script, give job name, e-mail confirmation on last queue entry
                if aa_sim.sim_status == "uploaded":
                    aa_sim.MLS_setup_path = os.path.join(aa_sim.MLS_sim_path, "setup.sh")
                    submission_script_path = os.path.join(MLS_working_directory, submission_script)
                    aa_sim.MLS_submission_script_path = os.path.join(aa_sim.MLS_sim_path, submission_script)

                    command_return(client, "cp {} {}".format(submission_script_path, aa_sim.MLS_submission_script_path))

                    command_return(client, "sed -i 's;runcommand;/usr/bin/bash {};' {}".format(aa_sim.MLS_setup_path, aa_sim.MLS_submission_script_path))

                    command_return(client, "sed -i 's/jobname/{}/' {}".format(aa_sim.sim_name[1:-1] ,aa_sim.MLS_submission_script_path))

                    if aa_sim == sim_list[-1]:
                        command_return(client, "sed -i 's/--mail-type=FAIL/--mail-type=FAIL,END/' {}".format(aa_sim.MLS_submission_script_path))
                    
                    aa_sim.sim_status = "prepared"
                
                ###run sim
                if aa_sim.sim_status == "prepared":                
                    response = command_return(client, "cd " + aa_sim.MLS_sim_path + ";" + "sbatch "+ submission_script)[0]

                    if "Submitted batch job" in response:
                        aa_sim.jobname = re.search("([0-9])\w+", response).group()
                        aa_sim.sim_status = "queued"
                    else:
                        Cluster_sim.failed_runs.append(aa_sim)
                        aa_sim.sim_status = "failed"

                    print("cd " + aa_sim.MLS_sim_path + ";" + "sbatch "+ submission_script)
                               
                #sinfo_t_idle um freie nodes anzuzeigen                
                
                
            ###get progress of simulations and download finished simulations #squeue to show queue

            print("simulations started, waiting for completion")

            Cluster_sim.pending_sims = sim_list.copy()
            
            for aa_sim in Cluster_sim.pending_sims:
                aa_sim.prev_status = aa_sim.sim_status

            time.sleep(5)
            
            while len(Cluster_sim.pending_sims) > 0:
                
                for aa_sim in Cluster_sim.pending_sims:
                    status = get_slurm_job_status(client, aa_sim.jobname)
                    
                    if aa_sim.sim_status == "failed":
                        Cluster_sim.pending_sims.pop(Cluster_sim.pending_sims.index(aa_sim))
                        continue
                    
                    if aa_sim.sim_status == "downloaded":
                        Cluster_sim.pending_sims.pop(Cluster_sim.pending_sims.index(aa_sim)) 
                    
                    elif "COMPLETED" in status and remote_file_exists(client, ssh, aa_sim.MLS_topology) and remote_file_exists(client, ssh, aa_sim.MLS_trajectory) and aa_sim.sim_status != "finished" and aa_sim.sim_status != "downloaded":
                        aa_sim.sim_status = "finished"
                        
                    elif "PENDING" in status:
                        aa_sim.sim_status = "queued"
                        
                    elif "RUNNING" in status:
                        aa_sim.sim_status = "running"
                        
                    elif "COMPLETED" in status and not remote_file_exists(client, ssh, aa_sim.MLS_topology) and not remote_file_exists(client, ssh, aa_sim.MLS_trajectory):
                        aa_sim.sim_status = "failed"
                        Cluster_sim.failed_runs.append(aa_sim)
                        Cluster_sim.pending_sims.pop(Cluster_sim.pending_sims.index(aa_sim))
                        
                    elif "FAILED" in status:
                        aa_sim.sim_status = "failed"
                        Cluster_sim.failed_runs.append(aa_sim)
                        Cluster_sim.pending_sims.pop(Cluster_sim.pending_sims.index(aa_sim))
                        
                    elif "TIMEOUT" in status:
                        aa_sim.sim_status = "failed"
                        Cluster_sim.failed_runs.append(aa_sim)
                        Cluster_sim.pending_sims.pop(Cluster_sim.pending_sims.index(aa_sim))
                        
                    elif "CANCELLED" in status:
                        aa_sim.sim_status = "failed"
                        Cluster_sim.failed_runs.append(aa_sim)
                        Cluster_sim.pending_sims.pop(Cluster_sim.pending_sims.index(aa_sim))
                    
                    if aa_sim.sim_status == "finished":
                        get_folder(client, ssh, aa_sim.MLS_sim_path, aa_sim.sim_directory)
                        Cluster_sim.pending_sims.pop(Cluster_sim.pending_sims.index(aa_sim))
                        aa_sim.sim_status = "downloaded"
                        
                    if force_retry and aa_sim.sim_status == "failed":
                        Cluster_sim.pending_sims.append(aa_sim)
                        response = command_return(client, "cd " + aa_sim.MLS_sim_path + ";" + "sbatch "+ submission_script)[0]

                        if "Submitted batch job" in response:
                            aa_sim.jobname = re.search("([0-9])\w+", response).group()
                            aa_sim.sim_status = "queued"
                        else:
                            Cluster_sim.failed_runs.append(aa_sim)
                            aa_sim.sim_status = "failed"
                    
                    #else:
                    #    Cluster_sim.failed_runs.append(aa_sim)
                    #    Cluster_sim.pending_sims.pop(Cluster_sim.pending_sims.index(aa_sim))
                    
                    if aa_sim.prev_status != aa_sim.sim_status:
                        print(aa_sim.sim_name + "  --------  " + aa_sim.sim_status)
                    aa_sim.prev_status = aa_sim.sim_status
                        
                progress(len(sim_list) - len(Cluster_sim.pending_sims), len(sim_list))
            
                if not len(Cluster_sim.pending_sims) == 0:
                    time.sleep(20)
            
            #+client.close()
            
        if len(Cluster_sim.failed_runs) > 0:
            failstring = "The following simulations have failed:\n"
            for aa_sim in Cluster_sim.failed_runs:
                failstring += aa_sim.sim_name + "\n"
                    
            raise Exception(failstring)
            
    def coarsegrain(aa_sim_list, file=False):
        #coarse-grains given aa_sims whole trajectory. if a filename inside the aa sim folder is given, it will be coarsegrained instead
        if file:        
            cg_universe_dict = {}
            for aa_sim in aa_sim_list:

                frame_name = file
                filename, file_extension = os.path.splitext(file)
                frame_name_charged = filename + "_charged" + file_extension
                frame_name_cg = filename + "_cg" + ".pdb"
                working_dir = aa_sim.sim_directory

                if not os.path.exists(os.path.join(working_dir, frame_name_cg)):

                    cwd = os.getcwd()
                    os.chdir(working_dir)
                    
                    if "neutral" in aa_sim.charge_folder:
                        
                        if not os.path.exists(os.path.join(working_dir, "charmm36-jul2021.ff")):
                            os.symlink("/home/soft/gromacs/forcefields/charmm36-jul2021.ff", os.path.join(working_dir, "charmm36-jul2021.ff"))                            
                        os.system("printf '1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0' | gmx pdb2gmx -f {frame_name} -o {output_gro} -ff charmm36-jul2021 -asp -glu -ter -ignh".format(frame_name=frame_name, output_gro=frame_name_charged))

                        os.system("martinize2 -f {frame_name} -x {frame_name_cg} -p backbone -pf 10000 -ff martini3001 -nt".format(frame_name=frame_name_charged, frame_name_cg=frame_name_cg))

                    else:
                        os.system("martinize2 -f {frame_name} -x {frame_name_cg} -p backbone -pf 10000 -ff martini3001".format(frame_name=frame_name, frame_name_cg=frame_name_cg))

                    os.chdir(cwd)

                cg_universe = Universe(os.path.join(working_dir, frame_name_cg))
                cg_universe_dict[aa_sim] = cg_universe

            return cg_universe_dict
            
        for aa_sim in aa_sim_list:
            frame_list = []
            
            for frame in range(len(aa_sim.universe.trajectory)):
                frame_list.append([frame, aa_sim.charge_folder, aa_sim.peptide_folder, aa_sim.cluster_folder, aa_sim.sim_type_folder])

            if not os.path.exists(os.path.join(aa_sim.sim_directory, "traj_frames")):
                os.makedirs(os.path.join(aa_sim.sim_directory, "traj_frames"))
            aa_sim.write_frames(frame_list, aa_sim.sim_directory, selection_str = "protein")

            output_pdb_list = []

            for frame in frame_list:

                frame_name = "selected_traj_{frame_nr}.pdb".format(frame_nr="-"+str(frame[0])+"-"+frame[1]+"-"+frame[2]+"-"+frame[3]+"-"+frame[4]+"-")
                frame_name_charged = "selected_traj_{frame_nr}.pdb".format(frame_nr="-"+str(frame[0])+"-"+frame[1]+"-"+frame[2]+"-"+frame[3]+"-"+frame[4]+"-charged-")
                frame_name_cg = "selected_traj_{frame_nr}_cg.pdb".format(frame_nr="-"+str(frame[0])+"-"+frame[1]+"-"+frame[2]+"-"+frame[3]+"-"+frame[4]+"-")
                working_dir = os.path.join(aa_sim.sim_directory, "traj_frames")

                if not os.path.exists(os.path.join(working_dir, frame_name_cg)):

                    cwd = os.getcwd()
                    os.chdir(working_dir)
                    
                    if "neutral" in aa_sim.charge_folder:
                    
                        with open(frame_name, 'r+') as file:
                            data = file.read()

                            data = data.replace("ASPHX", "ASP")

                            file.seek(0)
                            file.write(data)
                            file.truncate()
                            
                        os.system("printf '1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0' | gmx pdb2gmx -f {frame_name} -o {output_gro} -ff gromos54a7 -asp -glu -ter -ignh".format(frame_name=frame_name, output_gro=frame_name_charged))

                        os.system("martinize2 -f {frame_name} -x {frame_name_cg} -p backbone -pf 10000 -ff martini3001 -nt".format(frame_name=frame_name_charged, frame_name_cg=frame_name_cg))

                    else:
                        os.system("martinize2 -f {frame_name} -x {frame_name_cg} -p backbone -pf 10000 -ff martini3001".format(frame_name=frame_name, frame_name_cg=frame_name_cg))
                        
                    os.chdir(cwd)

                output_pdb_list.append(os.path.join(working_dir, frame_name_cg))

            aa_sim.cg_universe = Universe(output_pdb_list[0], *output_pdb_list)
            
    
    
    def concatenate_high_d_data(sim_list):
        #concatenates the high d data of given aa sims and their frame dictionaries
        
        high_d_data_list = []
        cluster_ids_list = []
        tccpf_list = []
        caccpf_list = []
        clccpf_list = []

        for sim in sim_list:
            high_d_data_list.append(sim.high_d_data)
            #if sim.cluster_ids.any():
            #    sim.cluster_pep_ids = []
            #    for id in sim.cluster_ids:
            #        sim.cluster_pep_ids.append([sim.peptide_folder, sim.cluster_folder, sim.sim_folder, str(id)])
            #    cluster_ids_list.append(sim.cluster_pep_ids)
        
        Cluster_sim.high_d_data_concat = np.concatenate(high_d_data_list, axis=0)
         
        Cluster_sim.frame_dict_concat = {}    
        frame_idx = 0         
        for sim in sim_list:
            if sim.sim_type_folder == "AA_sim":
                for frame in sim.frame_dict:
                    Cluster_sim.frame_dict_concat[frame_idx] = [sim.frame_dict[frame], sim.charge_folder, sim.peptide_folder, sim.cluster_folder, sim.sim_type_folder, sim.frame_number, sim]
                    frame_idx += 1
                
            else:
                for frame in sim.frame_dict:
                    Cluster_sim.frame_dict_concat[frame_idx] = [sim.frame_dict[frame], sim.charge_folder, sim.peptide_folder, sim.cluster_folder, sim.sim_type_folder, sim]
                    frame_idx += 1
        
        return Cluster_sim.high_d_data_concat, Cluster_sim.frame_dict_concat
        
    def norm_high_d_data(*high_d_data_list):
        #normalizes high d data along the time axis
        high_d_data_concat = np.array([])
        split_list = []
        index = 0
        
        for high_d_data in high_d_data_list:
            
            if high_d_data_concat.size == 0:
                high_d_data_concat = high_d_data
            else:
                high_d_data_concat = np.concatenate([high_d_data_concat, high_d_data], axis = 0)

            mindex = index
            index += len(high_d_data)
            split_list.append((mindex, index))

           
        
        if len(high_d_data_concat) == 1:
            avg = np.average(high_d_data_concat)   #normalizes the generated distances: (value-avg)/stdev along each column
            sub = np.subtract(high_d_data_concat, avg)                  
            stdev = np.std(high_d_data_concat)
            high_d_data_norm_concat = np.divide(sub, stdev)
        
        elif len(high_d_data_concat) > 1:
            avg = np.average(high_d_data_concat, axis=0)   #normalizes the generated distances: (value-avg)/stdev along each column
            sub = np.subtract(high_d_data_concat, avg)                  
            stdev = np.std(high_d_data_concat, axis = 0)
            high_d_data_norm_concat = np.divide(sub, stdev)
            
        rlist = []
        for interv in split_list:
            rlist.append(high_d_data_norm_concat[interv[0]:interv[1]])

        return rlist
                   
           
def concatenate_high_d_data(high_d_data_list, frame_dict_list):
    #also concatenates high d data and frame dicts, but takes them as input directly
    Cluster_sim.high_d_data_concat = np.concatenate(high_d_data_list, axis=0)
    Cluster_sim.frame_dict_concat = {}    
    frame_idx = 0         
    for frame_dict in frame_dict_list: 
        for frame in frame_dict:
            Cluster_sim.frame_dict_concat[frame_idx] = frame_dict[frame]
            frame_idx += 1
    return Cluster_sim.high_d_data_concat, Cluster_sim.frame_dict_concat