import paramiko
import os
import sys
import time
import glob
import shutil

def connect_MLS(host = "bwfor.cluster.uni-mannheim.de", username = "kn_pop******", password = "***", two_factor = True, host_key_path = "/home/nicolas.schneider/.ssh/known_hosts"):
    #connects to a computing cluster via ssh
    def connect_loop(repeats = 5, r_num = 0):
        last_try = False
        try:
            client.connect(host, username = username, password = password, two_factor = two_factor)
        except:
            if r_num < repeats-1:
                time.sleep(2)
                r_num += 1
                connect_loop(r_num = r_num)
                
            if r_num >= repeats-1:
                last_try = True
                
        if last_try == True:
            client.connect(host, username = username, password = password, two_factor = two_factor)
    
    client = paramiko.SSHClient()

    client.load_host_keys(host_key_path)
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_loop()
    
    return client

def establish_sftp(client):
    #required for file transfer
    t = client.get_transport()
    t.set_log_channel("log_channel")
    ssh = client.open_sftp()
    
    return ssh
    
    
def command_return(client, command):
    #runs a shell command on paramiko client and returns the return
    stdin, stdout, stderr = client.exec_command(command)
    results = []
    for line in iter(stdout.readline, ""):
        results.append(line)
    return results

def recursive_oldname(path, it = 1, og_path = None): 
    #renames file if it already exists
    if os.path.isfile(path):
        if og_path == None:
            og_path = path
        
        directory = os.path.dirname(path)
        basename = os.path.basename(path)
        
        if basename[0] == "#" and basename[-len(str(it-1))-1] == "#":
            newold_name = basename[:-len(str(it-1))] + str(it)            
        else:
            newold_name = "#" + basename + "#" + str(it)
        newold_path = os.path.join(directory, newold_name)
        it += 1
        
        recursive_oldname(newold_path, it = it, og_path = og_path)
        
    elif og_path != None:
        os.rename(og_path, path)
        
def put_folder(client, ssh, origin_path, destination_path):
    #uploads folder contents to server
    all_file_paths = []
    
    for root, dirs, files in os.walk(origin_path):
        for file in files:
            all_file_paths.append(os.path.join(root,file))
            
    for file_path in all_file_paths:
        file_path = os.path.normpath(file_path)
        dest_path = os.path.join(destination_path, os.path.basename(file_path))
        #remote_recursive_oldname(dest_path)
        ssh.put(file_path, dest_path)
        print("uploaded: ", os.path.basename(file_path))    

def get_folder(client, ssh, origin_path, destination_path):
    #downloads folder contents to server
    def try_loop(ssh, file_path, dest_path, repeats = 5, r_num = 0):
        try:
            ssh.get(file_path, dest_path)
        except:
            if r_num < repeats:
                r_num += 1
                time.sleep(5)
                try_loop(ssh, file_path, dest_path, r_num = r_num)
                
            else:
                print("failed download on: ", file_path)
        
    all_file_paths = command_return(client,"find {} -type f".format(origin_path))
    for file_path in all_file_paths:
        file_path = file_path[:-1]
        dest_path = os.path.join(destination_path, os.path.basename(file_path))
        recursive_oldname(dest_path)
        print("file_path: ", file_path)
        print("dest_path: ", dest_path)
        try_loop(ssh, file_path, dest_path)
            
        print("downloaded: ", os.path.basename(file_path))
        
def remote_file_exists(client, ssh, file_path, search_depth = 1, base_path = None):
    #checks if the file exists on the server
    if base_path == None:
        base_path = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
        
    all_file_paths = command_return(client,"find {} -name {} -type f -maxdepth {}".format(base_path, file_name, search_depth))

    for found_file_path in all_file_paths:
        found_file_path = found_file_path[:-1]
        found_file_name = os.path.basename(found_file_path)
        if found_file_name == file_name:
            return True
        
    return False

def get_slurm_job_status(client, jobid):
    status = command_return(client, "sacct --format State -j " + str(jobid))
    try:
        status = status[2]
    except:
        print(status)
    return status

def make_run_folder(path):
    #makes the next iteration of the run folder
    if not os.path.exists(path):
        os.makedirs(path)
    cwd = os.getcwd()
    os.chdir(path)
    results = glob.glob("run*")
    os.chdir(cwd)
    if len(results) > 0:
        for i, result in enumerate(results):
            if "run" in result:
                results[i] = int(result[3:])     

        newrun = "run" + str(max(results)+1)
    else:
        newrun = "run1"
    newrun = os.path.join(path, newrun)
    os.makedirs(newrun)
    return newrun

def delete_folder_contents(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                delete_folder_contents(file_path)
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
        