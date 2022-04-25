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

    runs = ["R04", "R06", "R08", "R10", "R12", "R14"]

    # The "-3" comes from the fact that the subjects 87, 91 and 99 are excluded
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
            
            len_chunks = file.readAnnotations()[1] * Fs         # Loading length of tasks
            chop_times = file.readAnnotations()[0] * Fs         # loading starting times of tasks
            chunks = min(len(annotations) // 2, max_chunks)  ### (only take those chunks with T1 or T2)
        
            electrodes = file.getSignalLabels()
            
            # if chop_times[-1] > file.getNSamples()[0] or chop_times[-1] > len(file.readSignal(1)):
            #     print(f"Subject {subject_name} run {run_name}")
            
            # print(f"Subject {subject_name} run {run_name}: getNSamples: {file.getNSamples()[0]}")
            # print(f"Subject {subject_name}run {run_name}: last chop: {chop_times[-1]}")
            # print(f"Subject {subject_name} run {run_name}: lenSamples: {len(file.readSignal(1))}")
            
            # Get 2d matrix of signals
            signal_2d = np.zeros((file.signals_in_file, file.getNSamples()[0]))
            for channel in range(file.signals_in_file):
                signal_2d[channel, :] = file.readSignal(channel)
        
            # Get labels
            for i in range(chunks-1):                   # Here -1 comes from the fact that for the last task we cannot take 1s of rest at the end.
                targets[sub_nb, run, i] = get_label(annotations[2 * i + 1],
                                                     run_name)
                chop_time = int(chop_times[2 * i + 1])-Fs       # subtracting 1s to take 1s rest prior to task
                len_chunk = int(len_chunks[2 * 1 + 1])
                next_chop_time = chop_time + t*Fs
                # This long function is just in case the signal_2d is shorter than t*Fs, we append 0 until it reaches the size
                X[sub_nb, run, i, :, :] = signal_2d[:, chop_time:next_chop_time]
                #X[subject, run, i, :, :] = np.append(signal_2d[:, chop_time:next_chop_time],np.zeros((no_channels, max(t* Fs - signal_2d[:, chop_time:next_chop_time].shape[1],0))),axis=1)
        
        
            file.close()

    X_mixed = X.reshape(((no_of_subjects - 3) * len(runs) * max_chunks, no_channels, t * Fs))
    targets_mixed = targets.reshape(((no_of_subjects - 3) * len(runs) * max_chunks))

    # There is some empty records given that not all the signals had the same value of chunks.
    keep = np.argwhere(targets_mixed != "").flatten()

    X_final = X_mixed[keep]
    targets_final = targets_mixed[keep]

    X_separated = np.zeros((no_of_subjects - 4, 84, no_channels, t * Fs))
    targets_separated = np.zeros((no_of_subjects - 4, 84), dtype="U2")

    subject_84 = 0
    for subject in range(no_of_subjects - 3):
        X_temp = X[subject, :, :, :, :].reshape(
            (len(runs) * max_chunks, no_channels, t * Fs))
        t_temp = targets[subject, :, :].reshape((len(runs) * max_chunks))
        keep = np.argwhere(t_temp != '').flatten()
        print(X_temp[keep].shape[0])
        if X_temp[keep].shape[0] == 84:
            X_separated[subject_84,:,:,:] = X_temp[keep]
            targets_separated[subject_84,:] = t_temp[keep]
            subject_84 += 1

    X_ordered = np.zeros((X_separated.shape[0], 84, no_channels, t*Fs))

    for i, X_slice in enumerate(X_separated):
        #if np.sum(targets_separated[i] == 'L') != 21 or np.sum(targets_separated[i] == 'R') != 21 or np.sum(targets_separated[i] == 'LR') != 21 or np.sum(targets_separated[i] == 'F') != 21:
        #    print(f"problem with {i}")
        
        X_ordered[i,:21,:,:] = X_slice[targets_separated[i] == 'L',:,:][:21, :, :]
        X_ordered[i,21:42,:,:] = X_slice[targets_separated[i] == 'R',:,:][:21, :, :]
        X_ordered[i,42:63,:,:] = X_slice[targets_separated[i] == 'LR',:,:][:21, :, :]
        X_ordered[i,63:84,:,:] = X_slice[targets_separated[i] == 'F',:,:][:21, :, :]

    targets_ordered = np.repeat(np.repeat(['L','R','LR','F'], 21).reshape(1,-1),X_separated.shape[0], axis=0)
    electrodes_filtered = [el.replace('.', '') for el in electrodes]

    # np.save("./data/filtered_data/signals", X_final)
    # np.save("./data/filtered_data/targets", targets_final)
    # np.save("./data/filtered_data/signals_separated", X_separated)
    # np.save("./data/filtered_data/targets_separated", targets_separated)
    # np.save("./data/filtered_data/signals_ordered_6s", X_ordered)
    # np.save("./data/filtered_data/targets_ordered_6s", targets_ordered)
    # np.save("./data/filtered_data/electrodes", electrodes_filtered)


if __name__ == "__main__":
    load_data()
