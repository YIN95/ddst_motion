import pandas as pd
import os
from libs.parsers import BVHParser
from libs.preprocessing import *
from libs.writers import *
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib as jl

def inv_standardize(data, scaler):      
    shape = data.shape
    flat = data.reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled        

def fit_and_standardize(data):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler

def standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.transform(flat).reshape(shape)
    return scaled

def select_bvh(root_dir="./MotionData/100STYLE/", 
                   content_name = ["FW"]):
    frame_cuts_path = os.path.join(root_dir, "Frame_Cuts.csv")
    frame_cuts = pd.read_csv(frame_cuts_path)
    n_styles = len(frame_cuts.STYLE_NAME)
    style_name = [frame_cuts.STYLE_NAME[i] for i in range(n_styles)]

    def extractSeqRange(start,end):
        start = start.astype('Int64')
        end = end.astype('Int64')

        return [[(start[i]),(end[i])] for i in range(len(start))]

    content_range = {name:extractSeqRange(frame_cuts[name+"_START"],frame_cuts[name+"_STOP"]) 
                        for name in content_name}

    all_file = []
    all_range = []
    for i in range(n_styles):
        anim_style = {}
        folder = os.path.join(root_dir, style_name[i])
        for content in content_name:
            ran = content_range[content][i]
            if(type(content_range[content][i][0])!=type(pd.NA)):
                file = os.path.join(folder, style_name[i]+"_"+content+".bvh")

                all_file.append(file)
                all_range.append(ran)
    
    return all_file, all_range, style_name

def get_filename_and_extension(file_path):
    filename = os.path.basename(file_path)  
    file_extension = os.path.splitext(filename)[1]
    file_name_without_extension = os.path.splitext(filename)[0] 
    return file_name_without_extension, file_extension

def extract_feature(all_bvh, all_range, destpath, fps=30):
    p = BVHParser()

    data_all = list()
    print("Importing data...")

    for f in all_bvh:
        ff = os.path.join(f)
        print(ff)
        data_all.append(p.parse(ff))
        
    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=fps)),
        ('mir', Mirror(axis='X', append=True)),
        # ('rev', ReverseTime(append=True)),
        ('jtsel', JointSelector(['LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe', 'RightHip', 
                                 'RightKnee', 'RightAnkle', 'RightToe', 'Chest', 'Chest2', 'Chest3', 
                                 'Chest4', 'Neck', 'Head', 'LeftCollar', 'LeftShoulder', 'LeftElbow', 
                                 'LeftWrist', 'RightCollar', 'RightShoulder', 'RightElbow', 'RightWrist'], include_root=True)),
        ('root', RootTransformer('pos_rot_deltas', position_smoothing=4, rotation_smoothing=4)),
        ('exp', MocapParameterizer('expmap')), 
        ('cnst', ConstantsRemover()),
        ('npf', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    jl.dump(data_pipe, os.path.join(destpath, 'data_pipe.sav'))

    fi=0
    for f, ran in zip(all_bvh, all_range):
        f_name, _ = get_filename_and_extension(f)

        ff = os.path.join(destpath, f_name)
        np.savez(ff + ".npz", clips=out_data[fi][ran[0]:ran[1]])
        np.savez(ff + "_mirrored.npz", clips=out_data[len(all_bvh)+fi][ran[0]:ran[1]])

def slice_data(data, window_size, overlap):

    nframes = data.shape[0]
    overlap_frames = (int)(overlap*window_size)
    
    n_sequences = (nframes-overlap_frames)//(window_size-overlap_frames)
    sliced = np.zeros((n_sequences, window_size, data.shape[1])).astype(np.float32)
    
    if n_sequences>0:

        # extract sequences from the data
        for i in range(0,n_sequences):
            frameIdx = (window_size-overlap_frames) * i
            sliced[i,:,:] = data[frameIdx:frameIdx+window_size,:].copy()
    else:
        print("WARNING: data too small for window")
                    
    return sliced
            
def import_and_slice(files, all_style, motion_path, slice_window, slice_overlap, start=0, end=None):

    all_motion, all_control, all_label = [], [], []
    for file in files:
        f_name, _ =get_filename_and_extension(file)
        motion_data = np.load(os.path.join(motion_path, f_name + '.npz'))['clips'].astype(np.float32)
        motion_data_mirror = np.load(os.path.join(motion_path, f_name + '_mirrored.npz'))['clips'].astype(np.float32)
        if motion_data.shape[0]<slice_window:
            #import pdb;pdb.set_trace()
            print("Too few frames in file: " + str(file))
            continue
        
        sliced = slice_data(motion_data, slice_window, slice_overlap)
        sliced_mirror = slice_data(motion_data_mirror, slice_window, slice_overlap)

        
        out_data = sliced
        out_data = np.concatenate((out_data, sliced_mirror), axis=0)
        index = all_style.index(f_name.split('_')[0])
        motion, control = out_data[:,:,:-3], out_data[:,:,-3:]
        labels = np.full((out_data.shape[0], 1), index)

        all_motion.append(motion)
        all_control.append(control)
        all_label.append(labels)
    
    return all_motion, all_control, all_label


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--preprocess", action="store_true")
    args = parser.parse_args()

    window_secs = 5
    window_overlap = 0.95
    fps = 30

    root_dir = "./MotionData/100STYLE/"
    content_name = ["FW"]

    processed_dir = './data/loco_processed/'
    processed_scaled_dir = './data/loco_processed_scaled/'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    if not os.path.exists(processed_scaled_dir):
        os.makedirs(processed_scaled_dir)

    all_bvh, all_range, all_style = select_bvh(root_dir, content_name)

    if(args.preprocess==True):
        print("Processing...")
        extract_feature(all_bvh, all_range, processed_dir, fps=fps)

    print("Preparing datasets...")

    files = []    
    # r=root, d=directories, f = files
    for r, d, f in os.walk(processed_dir):
        for file in f:
            if '.npz' in file:
                ff=os.path.join(r, file)
                basename = os.path.splitext(os.path.basename(ff))[0]
                files.append(basename)
    
    slice_win = window_secs*fps
    
    motion, ctrl, label = import_and_slice(all_bvh, all_style, processed_dir, slice_win, window_overlap)
    motion = np.concatenate(motion, axis=0)
    ctrl = np.concatenate(ctrl, axis=0)
    label = np.concatenate(label, axis=0)

    motion, m_scaler = fit_and_standardize(motion)
    np.savez(os.path.join(processed_scaled_dir,'output_scaler.npz'), stds=m_scaler.scale_, means=m_scaler.mean_)
    ctrl, c_scaler = fit_and_standardize(ctrl)
    np.savez(os.path.join(processed_scaled_dir,'input_scaler.npz'), stds=c_scaler.scale_, means=c_scaler.mean_)
    
    jl.dump(c_scaler, os.path.join(processed_scaled_dir,f'input_scaler.sav'))         
    jl.dump(m_scaler, os.path.join(processed_scaled_dir,f'output_scaler.sav'))

    np.savez(os.path.join(processed_scaled_dir,f'train_output_{fps}fps.npz'), clips = motion)
    np.savez(os.path.join(processed_scaled_dir,f'train_input_{fps}fps.npz'), clips = ctrl)
    np.savez(os.path.join(processed_scaled_dir,f'train_label_{fps}fps.npz'), clips = label)
    np.savez(os.path.join(processed_scaled_dir,f'train_style_{fps}fps.npz'), clips = np.array(all_style))

    # train model
    # transfer
    # test visualize

    output_scaler = jl.load('data/loco_processed_scaled/output_scaler.sav')
    input_scaler = jl.load('data/loco_processed_scaled/input_scaler.sav')
    data_pipe = jl.load('data/loco_processed_scaled/data_pipe.sav')
    motion = np.load('data/loco_processed_scaled/train_output_30fps.npz')['clips']

    def save_animation(control_data, motion_data, filename):
        #import pdb;pdb.set_trace()
        anim_data = np.concatenate((inv_standardize(motion_data, output_scaler), 
                                    inv_standardize(control_data, input_scaler)), axis=2)
        np.savez(filename + ".npz", clips=anim_data)  
        write_bvh(anim_data, filename)
    
    

    def write_bvh(anim_clips, filename):
        print('inverse_transform...')
        inv_data=data_pipe.inverse_transform(anim_clips)
        writer = BVHWriter()
        for i in range(0,anim_clips.shape[0]):
            filename_ = f'{filename}_{str(i)}.bvh'
            print('writing:' + filename_)
            with open(filename_,'w') as f:
                writer.write(inv_data[i], f, framerate=fps)
    
    # save_animation(ctrl[:2], motion[:2], 'text')
