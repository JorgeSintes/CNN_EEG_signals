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


def load_data(
    path="./data/raw_data/", no_of_subjects=109, Fs=160, no_channels=64, t=4, max_chunks=19
):

    # t is the time at which the signal will be cut. Most of them last around 4.1-4.2 s.
    # but some of them are shorter and have to be padded with zeros so that all of them
    # have the same shape.

    # max_chunks is the actual maximum number of useful chunks a run has (not resting).
    # Keep in mind most of them will have less and therefore X_separated, y_separated
    # will have lots of empty occurences. Use X_final and y_final for the actual datapoints

    runs = ["R04", "R06", "R08", "R10", "R12", "R14"]

    X = np.zeros((no_of_subjects, len(runs), max_chunks, no_channels, t * Fs))
    targets = np.zeros((no_of_subjects, len(runs), max_chunks), dtype="U2")
    electrodes = None

    for subject in range(no_of_subjects):
        for run in range(len(runs)):
            # Open file
            subject_name = f"S{(subject+1):03d}"
            run_name = runs[run]
            file = pyedflib.EdfReader(
                path + subject_name + "/" + subject_name + run_name + ".edf"
            )

            # Needed parameters
            annotations = file.readAnnotations()[2]
            chop_times = file.readAnnotations()[0] * Fs
            chunks = min(
                len(annotations) // 2, max_chunks
            )  ### (only take those chunks with T1 or T2)

            electrodes = file.getSignalLabels()

            # Get 2d matrix of signals
            signal_2d = np.zeros((file.signals_in_file, file.getNSamples()[0]))
            for channel in range(file.signals_in_file):
                signal_2d[channel, :] = file.readSignal(channel)

            # Get labels
            for i in range(chunks):
                targets[subject, run, i] = get_label(annotations[2 * i + 1], run_name)
                chop_time = int(chop_times[2 * i + 1])
                # This long function is just in case the signal_2d is shorter than t*Fs, we append 0 until it reaches the size
                X[subject, run, i, :, :] = np.append(
                    signal_2d[:, chop_time : (chop_time + t * Fs)],
                    np.zeros(
                        (
                            no_channels,
                            max(
                                t * Fs
                                - signal_2d[:, chop_time : (chop_time + t * Fs)].shape[
                                    1
                                ],
                                0,
                            ),
                        )
                    ),
                    axis=1,
                )

            file.close()

    X_separated = X
    targets_separated = targets

    X_mixed = X.reshape((no_of_subjects * len(runs) * max_chunks, no_channels, t * Fs))
    targets_mixed = targets.reshape((no_of_subjects * len(runs) * max_chunks))

    # There is some empty records given that not all the signals had the same value of chunks.
    keep = np.argwhere(targets_mixed != "").flatten()

    X_final = X_mixed[keep]
    targets_final = targets_mixed[keep]

    electrodes_filtered = [el.replace('.','') for el in electrodes]

    np.save("./data/filtered_data/signals", X_final)
    np.save("./data/filtered_data/targets", targets_final)
    np.save("./data/filtered_data/electrodes", electrodes_filtered)


if __name__ == "__main__":
    load_data()
