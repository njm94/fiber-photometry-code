import tdt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from roipoly import RoiPoly
from matplotlib.animation import FFMpegWriter
from scipy import signal
import skvideo.io
from scipy.io import savemat
from scipy.optimize.minpack import curve_fit
import scipy.stats as stats
import re


def set_path(path):
    """
    Generates path recursively
    """
    if not os.path.exists(path):
        if len(path.split(os.path.sep)) == 1:
            raise Exception("Root path has been reached.")

        set_path(os.path.sep.join(path.split(os.path.sep)[:-1]))
        os.mkdir(path)


def between_peaks(f_grad):
    midpoint = f_grad.size/2
    start = None
    end = None
    peak3 = None

    while peak3 is None:
        idx = f_grad.argmax()
        if idx < midpoint:
            if start is None:
                start = idx
                f_grad[f_grad.argmax()] = 0
            else:
                if idx < start:
                    f_grad[f_grad.argmax()] = 0
                elif idx - start == 1:
                    start = idx
                    f_grad[f_grad.argmax()] = 0
                else:
                    peak3 = idx
        else:
            if end is None:
                end = idx
                f_grad[f_grad.argmax()] = 0
            else:
                if idx > end:
                    f_grad[f_grad.argmax()] = 0
                elif idx - end == -1:
                    end = idx
                    f_grad[f_grad.argmax()] = 0
                else:
                    peak3 = idx

    start += 1
    # end += 1

    data_range = (start, end)
    return data_range


def remove_dark_frames(image_stack):
    f_grad = np.abs(np.diff(np.mean(image_stack, axis=(1, 2))))
    start, end = between_peaks(f_grad)

    return image_stack[start:end, :, :], start, end


def time_in_section(tracks, fps=20, lower=0, upper=np.inf):
    """
    Determine time mouse spends in particular chamber. Note that video data and numpy data are flipped along y-axis
    :param tracks: mouse tracks (y position)
    :param fps: behavior video frame rate
    :param lower: lower bound for chamber of interest
    :param upper: upper bound for chamber of interest
    :return: time spent in chamber
    """
    return sum(i>lower and i<upper for i in tracks)/fps


def get_tracks(video_name, output_path, ksize=4, threshold=15, chamber_mask=None):
    """
    Track the position of the mouse as it explores the 3-chamber environment.
    :param ksize: kernel size for morphological opening
    :param threshold: pixel threshold for separating mouse from environment
    :param chamber_mask: ROI to manually mask only regions where the mouse can walk. Cups are not included in ROI
    :param save_figs: True or False. Save a figure showing mouse tracks
    :param save_outputs:
    :param shadow:
    :return: x_pos, y_pos: x and y positions of the mouse as a function of time
    :return: chamber_mask: return the manually drawn ROI so it does not need to be drawn again if processing in batch
    """

    # REWRITE THIS GETTING RID OF OPENCV IMAGE READINg
    for file in os.listdir(output_path):
        if video_name + "_trimmed.avi" in file:
            video_file = output_path + os.path.sep + file
    video_data = skvideo.io.vread(video_file)

    out = cv2.VideoWriter(output_path + os.path.sep + video_name + "_marked.avi",
                          cv2.VideoWriter_fourcc(*'MP42'),
                          100,
                          (int(video_data.shape[2]), int(video_data.shape[1])),
                          True)

    x_pos = []
    y_pos = []

    for i in range(video_data.shape[0]):
        frame = video_data[i, :, :, :]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if chamber_mask is None:
            plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
            plt.title("Draw chamber ROI")
            roi = RoiPoly(color='r')
            chamber_mask = roi.get_mask(gray)

        gray[~chamber_mask] = 255
        kernel = np.ones((ksize, ksize), np.uint8)
        _, fgMask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        fgMask = cv2.cvtColor(cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel), cv2.COLOR_GRAY2RGB)

        if fgMask.any():
            coords = np.where(fgMask)
            cx = np.median(coords[1])
            cy = np.median(coords[0])
            frame = cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        else:
            cx = x_pos[-1]
            cy = y_pos[-1]

        x_pos.append(cx)
        y_pos.append(cy)

        cv2.imshow(video_name, frame)

        out.write(frame)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    cv2.destroyAllWindows()

    y_bounds, x_bounds = np.where(chamber_mask)
    one_chamber_size = np.round((np.max(y_bounds) - np.min(y_bounds))/3)
    top_middle_edge = np.max(y_bounds) - one_chamber_size
    bot_middle_edge = np.min(y_bounds) + one_chamber_size

    top_time = time_in_section(y_pos, upper=bot_middle_edge)
    bottom_time = time_in_section(y_pos, lower=top_middle_edge)

    plt.figure()
    plt.plot(x_pos, -np.array(y_pos))
    plt.title(video_name)
    plt.xlabel('Top time: ' + str(top_time) + ' Bottom time: ' + str(bottom_time))
    plt.xlim(np.min(x_bounds), np.max(x_bounds))
    plt.ylim(-np.max(y_bounds), -np.min(y_bounds))
    plt.savefig(output_path + os.path.sep + video_name + '_tracks.pdf')
    plt.close()

    out.release()

    track_name = output_path + os.path.sep + video_name + "_tracks.npy"
    np.save(track_name, [x_pos, y_pos])

    mdict = {"x": x_pos, "y": y_pos}
    savemat(output_path + os.path.sep + video_name + "_tracks.mat", mdict)

    return x_pos, y_pos, chamber_mask


def xt(arr, fs, axis=0):
    t = np.arange(0, (arr.shape[axis])/fs, 1/fs)
    if len(t) == arr.shape[0]+1:
        print('Time vector is one sample greater than array size')
        t = t[:-1]
    return t


def get_fiber_data(data_block):
    reader = tdt.TDTbin2py
    data = reader.read_block(data_block)

    t = xt(data.streams._465A.data, data.streams._465A.fs)
    t_on = data.epocs.TIG_.onset[1]
    t_off = data.epocs.TIG_.offset[1]
    idx_on = np.argmin(np.abs(t-t_on))
    idx_off = np.argmin(np.abs(t-t_off))

    return data.streams._465A.data,  data.streams._405A.data, data.streams._560B.data, data.streams._465A.fs, t_on, t_off, idx_on, idx_off


def get_LED_on(vid):
    plt.figure()
    plt.imshow(vid[0, :, :, 0], cmap='gray', vmin=0, vmax=255)
    plt.title("Draw LED ROI")
    roi = RoiPoly(color='r')
    led_mask = roi.get_mask(vid[0, :, :, 0])
    led_flag = np.mean(vid[:, led_mask], axis=(1, 2))

    # plt.figure(), plt.plot(led_flag), plt.show()

    return between_peaks(np.abs(np.diff(led_flag)))


def process_fiber_data(data_block, output_path, video_name):
    data465, data405, data560, fs, t_on, t_off, idx_on, idx_off = get_fiber_data(data_block)
	
    qc_path = output_path + os.path.sep + 'quality_control'
    set_path(qc_path)

    plt.figure(), plt.plot(data405), plt.vlines(x=[idx_on, idx_off], ymin=np.min(data405), ymax=np.max(data405)), plt.title('raw 405')
    plt.savefig(qc_path + os.path.sep + video_name + '_raw405'), plt.close()

    plt.figure(), plt.plot(data465), plt.vlines(x=[idx_on, idx_off], ymin=np.min(data465), ymax=np.max(data465)), plt.title('raw465')
    plt.savefig(qc_path + os.path.sep + video_name + '_raw465'), plt.close()

    plt.figure(), plt.plot(data560), plt.vlines(x=[idx_on, idx_off], ymin=np.min(data560), ymax=np.max(data560)), plt.title('raw560')
    plt.savefig(qc_path + os.path.sep + video_name + '_raw560'), plt.close()

    if any(fname.endswith(video_name + '_trimmed_nframes.npy') for fname in os.listdir(output_path)):
        bvid_nframes = np.load(output_path + os.path.sep + video_name + '_trimmed_nframes.npy')
        data465 = signal.resample(data465[idx_on:idx_off], bvid_nframes)
        data405 = signal.resample(data405[idx_on:idx_off], bvid_nframes)
        data560 = signal.resample(data560[idx_on:idx_off], bvid_nframes)
        dur = t_off - t_on
        pi_fps = bvid_nframes / dur
        t = xt(data465, pi_fps)
    else:
        t = xt(data465, fs)

	
    dff405, best_fit405 = deltaFF(data405, t)
    plt.figure(), plt.plot(t, data405), plt.plot(t, best_fit405(t)),
    plt.savefig(qc_path + os.path.sep + video_name + '_exponential_fit405'), plt.close()
	
    dff465, best_fit465 = deltaFF(data465, t)
    plt.figure(), plt.plot(t, data465), plt.plot(t, best_fit465(t)),
    plt.savefig(qc_path + os.path.sep + video_name + '_exponential_fit465'), plt.close()

    dff560, best_fit560 = deltaFF(data560, t)
    plt.figure(), plt.plot(t, data560), plt.plot(t, best_fit560(t)),
    plt.savefig(qc_path + os.path.sep + video_name + '_exponential_fit_560'), plt.close()
	
    plt.figure(), plt.plot(t, dff405, color=[138/255, 43/255, 226/255]), 
    plt.plot(t, dff465-4, color=[0/255,191/255,255/255]), 
    plt.plot(t, dff560-8, color=[154/255,205/255,50/255])
	
	# compute rolling correlation between 465 and 405
    df = pd.DataFrame({'df405': dff405, 'df465': dff465})
    rolling_corr = df['df405'].rolling(100).corr(df['df465'])
    plt.figure(), plt.plot(rolling_corr), plt.ylim([-1.2, 1.2]), plt.title('Corr 405-465')
    plt.savefig(qc_path + os.path.sep + video_name + '_rolling_correlations')

    dff_name = output_path + os.path.sep + video_name + "_fiber_data"
    np.savez(dff_name, dff405=dff405, dff465=dff465, dff560=dff560, data465=data465, data405=data405, data560=data560, fs=fs, time=t)

    mdict = {"dff405": dff405, "dff465": dff465, "dff560": dff560, "data405": data405, "data465": data465, "data560": data560, "fs": fs, "time": t}
    savemat(output_path + os.path.sep + video_name + "_fiber_data.mat", mdict)

    return dff465, t, dff560

def deltaFF(data, t, method='exp_fit'):
    # if method == 'poly':
    #     reg = np.polyfit(dat2, dat1, 1)
    #     a, b = reg
    #     controlFit = a*dat2 + b
    #     dff = np.divide(np.subtract(dat1, controlFit), controlFit)
    # elif method == 'subtract':
    #     dff1 = np.divide(np.subtract(dat1, np.mean(dat1)), np.mean(dat1))
    #     dff2 = np.divide(np.subtract(dat2, np.mean(dat2)), np.mean(dat2))
    #     dff = dff1 - dff2
    # elif method == 'exp_fit':
    #

    guess_a, guess_b, guess_c = np.max(data), -0.05, np.min(data)
    guess = [guess_a, guess_b, guess_c]
    exp_decay = lambda x, A, b, y0: A * np.exp(x * b) + y0
    params, cov = curve_fit(exp_decay, t, data, p0=guess, maxfev=5000)
    A, b, y0 = params
    best_fit = lambda x: A * np.exp(b * x) + y0

    dff = data-best_fit(t) + 100  # add DC offset so mean of corrected signal is positive
    dff = (dff - np.mean(dff))/np.mean(dff)

    return stats.zscore(dff), best_fit


def make_combined_video(output_path, video_name, t, dff465, dff560):
    for file in os.listdir(output_path):
        if "_marked.avi" in file:
            behavior_video = output_path + os.path.sep + file
        elif "_trimmed.avi" in file:
            behavior_video = output_path + os.path.sep + file

    x = input("Use jRCaMP1b channel? y/[n]")
    if x == "y" or x == "Y":
        use_red = True
    else:
        use_red = False


    b_vid = skvideo.io.vread(behavior_video)
    b_vid = np.swapaxes(b_vid, 1, 2)
	
    writer = FFMpegWriter(fps=25)
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(311)
    ax1.axis("off")

    ax2 = fig.add_subplot(312)
    ax2.plot(t, dff465, color=[0/255,191/255,255/255])
    if use_red:
        ax2.plot(t, dff560, color=[154/255,205/255,50/255])
    ax2.set_ylabel(r'ΔF/F ($\sigma$)', fontweight="bold")
    ax2.set_xlabel('Time (s)', fontweight="bold")
    ax2.set_title('Fiber signal', fontweight="bold", fontsize=15)
    ax2.set_xlim([t[0] - 30, t[0] + 30])
    ax2.set_ylim([np.nanmin(dff465) - 0.005, np.nanmax(dff465) + 0.005])
    vl2 = ax2.axvline(0, ls='-', color='k', lw=1)

    ax3 = fig.add_subplot(313)
    ax3.plot(t, dff465, color=[0/255,191/255,255/255])
    if use_red:
        ax3.plot(t, dff560, color=[154/255,205/255,50/255])
    ax3.set_ylabel(r'ΔF/F ($\sigma$)', fontweight="bold")
    ax3.set_xlabel('Time (s)', fontweight="bold")
    ax3.set_xlim([t[0], t[-1]])
    ax3.set_ylim([np.nanmin(dff465) - 0.005, np.nanmax(dff465) + 0.005])
    vl3 = ax3.axvline(0, ls='-', color='k', lw=1)

    l1 = ax1.imshow(b_vid[0, :, :, :])
    print(output_path + os.path.sep + video_name + "_combined_fiber_behavior.mp4")

    with writer.saving(fig, output_path + os.path.sep + video_name + "_combined_fiber_behavior.mp4", 100):
        for ix in range(np.shape(b_vid)[0]):
            l1.set_data(np.flipud(b_vid[ix, :, :, :]))
            ax1.set_title('Behavior vid', fontweight="bold", fontsize=15)

            vl2.set_xdata([t[ix], t[ix]])
            ax2.set_xlim(t[ix] - 30, t[ix] + 30)

            vl3.set_xdata([t[ix], t[ix]])

            writer.grab_frame()
    plt.close(fig)


def make_annotations(boolean, t, color='g', alpha=0.25):
    event_edges = np.diff(boolean)
    if boolean[0]:
        event_edges[0] = True

    if boolean[-1]:
        event_edges[-1] = True

    event_starts = np.where(event_edges)[0][::2]
    event_ends = np.where(event_edges)[0][1::2]
    for i in range(len(event_starts)):
        plt.axvspan(t[event_starts[i]], t[event_ends[i]], facecolor=color, alpha=alpha)


def make_annotated_plot(x_pos, y_pos, t, dff465, output_path, video_name, experiment_type="3-chamber"):
    plt.figure(figsize=(15, 8)), 

    one_chamber_size = np.round((np.max(y_pos) - np.min(y_pos)) / 3)
    top_middle_edge = np.max(y_pos) - one_chamber_size
    bot_middle_edge = np.min(y_pos) + one_chamber_size

    top_time = time_in_section(y_pos, upper=bot_middle_edge)
    bottom_time = time_in_section(y_pos, lower=top_middle_edge)

    plt.subplot(211)
    
	
    if experiment_type == "3-chamber":
        plt.plot(y_pos, x_pos, 'k')
        plt.axvspan(np.nanmin(y_pos), (np.nanmin(y_pos) + 1/3 * (np.nanmax(y_pos) - np.nanmin(y_pos))), facecolor='r', alpha=0.25)
        plt.axvspan((np.nanmax(y_pos) - 1/3 * (np.nanmax(y_pos) - np.nanmin(y_pos))), np.nanmax(y_pos), facecolor='b', alpha=0.25)
        plt.title('Left time: ' + str(top_time) + ' Right time: ' + str(bottom_time))
        
        in_left = y_pos < (np.nanmin(y_pos) + 1 / 3 * (np.nanmax(y_pos) - np.nanmin(y_pos)))
        in_right = y_pos > (np.nanmax(y_pos) - 1 / 3 * (np.nanmax(y_pos) - np.nanmin(y_pos)))
        
    elif experiment_type == "open-field":
        plt.plot(x_pos, -y_pos, 'k')
        loc = re.findall(r"-.{2}-", video_name)[0][1:-1]
        if loc[0] == "U":
            y_corner = y_pos < (np.nanmax(y_pos) + np.nanmin(y_pos))/2
            low_y = (np.nanmax(y_pos) + np.nanmin(y_pos))/2
            high_y = np.nanmin(y_pos)
        elif loc[0] == "L":
            y_corner = y_pos > (np.nanmax(y_pos) + np.nanmin(y_pos))/2
            low_y = np.nanmax(y_pos)
            high_y = (np.nanmax(y_pos) + np.nanmin(y_pos))/2
        else:
            y_corner = False
        
        if loc[1] == "L":
            x_corner = x_pos < (np.nanmax(x_pos) + np.nanmin(x_pos))/2
            low_x = np.nanmin(x_pos)
            high_x = (np.nanmax(x_pos) + np.nanmin(x_pos))/2
        elif loc[1] == "R":
            x_corner = x_pos > (np.nanmax(x_pos) + np.nanmin(x_pos))/2
            low_x = (np.nanmax(x_pos) + np.nanmin(x_pos))/2
            high_x = np.nanmax(x_pos)
        else:
            x_corner = False
        
        plt.axvspan(xmin=low_x, xmax=high_x, ymin=low_y, ymax=high_y, facecolor='r', alpha=0.25)
        
        in_corner = x_corner & y_corner

		
    plt.xticks([])
    plt.yticks([])

    plt.subplot(212)
    plt.plot(t, dff465, 'k')
    if experiment_type == "3-chamber":
        make_annotations(in_left, t, color='r')
        make_annotations(in_right, t, color='b')
    elif experiment_type == "open-field":
	    make_annotations(in_corner, t, color='r')

    plt.ylabel(r'ΔF/F ($\sigma$)', fontsize=18)
    plt.xlabel('Time (s)', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.savefig(output_path + os.path.sep + video_name + '_DFFannotated.png')


def trim_video(behavior_raw, output_path):
    video_name = behavior_raw.split("\\")[-1][:-5]

    # Read behavior video and crop to LED ON/OFF times
    print("Reading behavior video...")
    video_data = skvideo.io.vread(behavior_raw)
    data_range = get_LED_on(video_data)
    video_data = video_data[data_range[0]:data_range[1], :, :, :]

    # Save chamber image as max projection. Use subset of video to reduce time
    mid = int(video_data.shape[0] / 2)
    chamber = np.max(video_data[mid - 300:mid + 300, :, :, 0], axis=0).astype('uint8')
    print("Writing trimmed video...")
    cv2.imwrite(output_path + os.path.sep + video_name + "_chamber.jpg", chamber)

    # Save number of behavior frames
    np.save(output_path + os.path.sep + video_name + "_trimmed_nframes", video_data.shape[0])

    # Save trimmed behavior video
    out = cv2.VideoWriter(output_path + os.path.sep + video_name + "_trimmed.avi",
                          cv2.VideoWriter_fourcc(*'MP42'),
                          100,
                          (int(video_data.shape[2]), int(video_data.shape[1])),
                          True)
    for i in range(video_data.shape[0]):
        frame = cv2.cvtColor(video_data[i, :, :, :], cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
    print("Done")


