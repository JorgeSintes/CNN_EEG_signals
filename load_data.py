import pyedflib
import numpy as np


def get_label(annotation, run_name):
    if run_name in ["R04", "R08", "R12"]:
        if annotation == "T1":
            return "L"
        elif annotation == "T2":
            return "R"
        else:
            return "error"

    elif run_name in ["R06", "R10", "R14"]:
        if annotation == "T1":
            return "LR"
        elif annotation == "T2":
            return "F"
        else:
            return "error"
        
    elif run_name == "R01":
        if annotation == "T0":
            return "0"
        else:
            return "error"


def load_data(path="./data/raw_data/",
              no_of_subjects=109,
              Fs=160,
              no_channels=64,
              t=6,
              max_chunks=19):

    # t is the time at which the signal will be cut. Most of them last around 4.1-4.2 s.
    # but some of them are shorter and have to be padded with zeros so that all of them
    # have the same shape.

    # max_chunks is the actual maximum number of useful chunks a run has (not resting).
    # Keep in mind most of them will have less and therefore X_separated, y_separated
    # will have lots of empty occurences. Use X_final and y_final for the actual datapoints

    runs = ["R01", "R04", "R06", "R08", "R10", "R12", "R14"]

    # The "-3" comes from the fact that the subjects 87, 91 and 99 are excluded
    # 87, 91 and 99 (in python indexing) are excluded because the chop times do not match the length of the signals
    # NB !: subject 88 (in python indexing) has the annotations T0 and T1 with 2 chop times and length for rest 
    # period, supposed to be "T0" only. Maybe it should be excluded.
    X = np.zeros((no_of_subjects-3, len(runs), max_chunks, no_channels, t * Fs))
    targets = np.zeros((no_of_subjects-3, len(runs), max_chunks), dtype="U2")
    electrodes = None

    subjects = [x for x in range(no_of_subjects) if x not in [87,91,99]]

    for sub_nb, subject in enumerate(subjects):
        for run in range(len(runs)):
            # Open file
            subject_name = f"S{(subject+1):03d}"
            run_name = runs[run]
            file = pyedflib.EdfReader("C:/Users/cleml/Documents/00000 - Special Course Bayesian Machine Learning/CNN_EEG_signals" + path + subject_name + "/" +
                                      subject_name + run_name + ".edf")
        
            # Needed parameters
            annotations = file.readAnnotations()[2]             # Loading tasks annotations
            
            if run_name == "R01":   #Treating special case of 'R01' (run 1) which only contains taks T0 (rest)
                
                sig_len = file.getNSamples()[0]
                
                chunk_len = t*Fs
                
                win_inds = (sig_len - chunk_len)/(21*chunk_len)*np.linspace(0,21*chunk_len,21)
                
                # Get 2d matrix of signals
                signal_2d = np.zeros((file.signals_in_file, sig_len))
                for channel in range(file.signals_in_file):
                    signal_2d[channel, :] = file.readSignal(channel)
                
                for i in range(21):
                    targets[sub_nb, run, i] = get_label("T0", run_name)
                    
                    chop_time = int(win_inds[i])
                    
                    X[sub_nb, run, i, :, :] = signal_2d[:, chop_time:chop_time + chunk_len]
                
            else:
                
                len_chunks = file.readAnnotations()[1] * Fs         # Loading length of tasks
                chop_times = file.readAnnotations()[0] * Fs         # loading starting times of tasks
                chunks = min(len(annotations) // 2, max_chunks)  ### (only take those chunks with T1 or T2)
            
                electrodes = file.getSignalLabels()
                
                # Get 2d matrix of signals
                signal_2d = np.zeros((file.signals_in_file, file.getNSamples()[0]))
                for channel in range(file.signals_in_file):
                    signal_2d[channel, :] = file.readSignal(channel)
            
                # Get labels
                for i in range(chunks-1):              # Here -1 comes from the fact that for the last task we cannot take 1s of rest at the end.
                    targets[sub_nb, run, i] = get_label(annotations[2 * i + 1],
                                                         run_name)
                    chop_time = int(chop_times[2 * i + 1])-Fs       # subtracting 1s to take 1s rest prior to task
                    len_chunk = int(len_chunks[2 * 1 + 1])
                    next_chop_time = chop_time + t*Fs
                    # This long function is just in case the signal_2d is shorter than t*Fs, we append 0 until it reaches the size
                    X[sub_nb, run, i, :, :] = signal_2d[:, chop_time:next_chop_time]
                    #X[subject, run, i, :, :] = np.append(signal_2d[:, chop_time:next_chop_time],np.zeros((no_channels, max(t* Fs - signal_2d[:, chop_time:next_chop_time].shape[1],0))),axis=1)
            
        
            file.close()

    X_separated = np.zeros((no_of_subjects - 4, 105, no_channels, t * Fs))
    targets_separated = np.zeros((no_of_subjects - 4, 105), dtype="U2")

    # Only one subject does not fullfill the below requirement, if filtered out further up this could be removed.
    subject_105 = 0
    for subject in range(no_of_subjects - 3):
        X_temp = X[subject, :, :, :, :].reshape(
            (len(runs) * max_chunks, no_channels, t * Fs))
        t_temp = targets[subject, :, :].reshape((len(runs) * max_chunks))
        keep = np.argwhere(t_temp != '').flatten()
        
        if X_temp[keep].shape[0] == 105:
            X_separated[subject_105,:,:,:] = X_temp[keep]
            targets_separated[subject_105,:] = t_temp[keep]
            subject_105 += 1

    X_ordered = np.zeros((X_separated.shape[0], 105, no_channels, t*Fs))

    for i, X_slice in enumerate(X_separated):
        #if np.sum(targets_separated[i] == 'L') != 21 or np.sum(targets_separated[i] == 'R') != 21 or np.sum(targets_separated[i] == 'LR') != 21 or np.sum(targets_separated[i] == 'F') != 21:
        #    print(f"problem with {i}")
        
        X_ordered[i,:21,:,:] = X_slice[targets_separated[i] == 'L',:,:][:21, :, :]
        X_ordered[i,21:42,:,:] = X_slice[targets_separated[i] == 'R',:,:][:21, :, :]
        X_ordered[i,42:63,:,:] = X_slice[targets_separated[i] == 'LR',:,:][:21, :, :]
        X_ordered[i,63:84,:,:] = X_slice[targets_separated[i] == 'F',:,:][:21, :, :]
        X_ordered[i,84:105,:,:] = X_slice[targets_separated[i] == '0',:,:][:21, :, :]

    targets_ordered = np.repeat(np.repeat(['L','R','LR','F','0'], 21).reshape(1,-1),X_separated.shape[0], axis=0)
    electrodes_filtered = [el.replace('.', '') for el in electrodes]

    # np.save("./data/filtered_data/signals_separated", X_separated)
    # np.save("./data/filtered_data/targets_separated", targets_separated)
    np.save("./data/filtered_data/signals_ordered_6s_all_tags", X_ordered)
    np.save("./data/filtered_data/targets_ordered_6s_all_tags", targets_ordered)
    # np.save("./data/filtered_data/electrodes", electrodes_filtered)


if __name__ == "__main__":
    load_data()
