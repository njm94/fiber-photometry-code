import tdt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from roipoly import RoiPoly
from matplotlib.animation import FFMpegWriter
from scipy import signal
import skvideo.io


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
    end += 1

    data_range = (start, end)
    return data_range


def remove_dark_frames(image_stack):
    f_grad = np.abs(np.diff(np.mean(image_stack, axis=(1, 2))))
    start, end = between_peaks(f_grad)

    return image_stack[start:end, :, :], start, end


def xt(arr, fs, axis=0):
    return np.arange(0, (arr.shape[axis])/fs, 1/fs)


def running_mean(x, n):
    arr = []
    for i in range(len(x)-n):
        # if i < n:
        #     tmp = np.mean(x[0:i])
        # elif i + n > len(x):
        #     tmp = np.mean(x[i:])
        # else:
        #     tmp = np.mean(x[i:i+n])
        tmp = np.mean(x[i:i+n])

        arr.append(tmp)
    return arr


def get_fiber_data(data_block):
    reader = tdt.TDTbin2py
    data = reader.read_block(data_block)

    t = xt(data.streams._465A.data, data.streams._465A.fs)
    led_on = data.epocs.TIG_.onset[1]
    led_off = data.epocs.TIG_.offset[1]
    t_on = np.argmin(np.abs(t-led_on))
    t_off = np.argmin(np.abs(t-led_off))

    return data.streams._465A.data,  data.streams._405A.data, data.streams._465A.fs, t_on, t_off


def get_LED_on(video):
    video_data = skvideo.io.vread(video)
    plt.figure()
    plt.imshow(video_data[0, :, :, 0], cmap='gray', vmin=0, vmax=255)
    plt.title("Draw LED ROI")
    roi = RoiPoly(color='r')
    led_mask = roi.get_mask(video_data[0, :, :, 0])
    led_flag = np.mean(video_data[:, led_mask], axis=(1, 2))
    # plt.figure(), plt.plot(led_flag), plt.show()
    led_grad = np.abs(np.diff(led_flag))
    thresh = np.mean(led_grad)+15*np.std(led_grad)

    indices = np.where(led_grad>thresh)
    if len(indices[0]) == 2:
        start = indices[0][0]
        end = indices[0][1]
    elif len(indices[0]) > 2:
        print("Too many indices found. You choose.")
        plt.figure(), plt.plot(led_flag), plt.plot(led_grad), plt.show()
        print(indices[0])
        start = indices[0][int(input('Enter the index that the signal should start on:'))]
        end = indices[0][int(input('Enter the index that the signal should end on:'))]
    else:
        print("Deal with this problem if it ever occurs")
        start = None
        end = None

    data_range = (start+1, end+1)

    return data_range, video_data


def calculate_df_f0_moving(data, window=144, axis=0):
    """
    Calculate df/f0, the fractional change in intensity for each pixel
    and the variance of df/f0 with a moving baseline (f0)
    """
    # frames = frames.astype(np.float32)
    # baseline = np.cumsum(frames, axis, dtype=np.float32)
    # baseline[n:] = baseline[n:] - baseline[:-n]
    # baseline[n-1:] = baseline[n-1:] / n
    #
    # prelim = np.arange(n)
    # baseline[:n - 1] = baseline[:n - 1] / prelim[1:, None, None]
    # df_f0 = np.divide(
    #     np.subtract(frames, baseline), baseline
    # )

    x = pd.DataFrame(data)
    x_mov = x.rolling(window=window, center=False)

    return df_f0


def deltaFF(dat1, dat2, method='subtract'):
    if method == 'poly':
        reg = np.polyfit(dat2, dat1, 1)
        a, b = reg
        controlFit = a*dat2 + b
        dff = np.divide(np.subtract(dat1, controlFit), controlFit)
    elif method == 'subtract':
        dff1 = np.divide(np.subtract(dat1, np.mean(dat1)), np.mean(dat1))
        dff2 = np.divide(np.subtract(dat2, np.mean(dat2)), np.mean(dat2))
        dff = dff1 - dff2

    return dff*100


def make_combined_video(behavior_video, data_block, window=1):
    out_filename = behavior_video.split('.')[0] + "_combined.mp4"
    b_vid = skvideo.io.vread(behavior_video)
    b_vid = np.swapaxes(b_vid, 1, 2)

    data465, data405, fs, t_on, t_off = get_fiber_data(data_block)
    test465 = pd.DataFrame(data465)
    test465_mov = test465.rolling(window=int(window*fs), center=True)
    data465 = test465_mov.mean()[t_on:t_off]

    test405 = pd.DataFrame(data405)
    test405_mov = test405.rolling(window=int(window*fs), center=True)
    data405 = test405_mov.mean()[t_on:t_off]

    dff = deltaFF(np.array(data465).flatten(), np.array(data405).flatten())
    dff = signal.resample(dff, np.shape(b_vid)[0])

    t = xt(dff, 20)

    writer = FFMpegWriter(fps=40)
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(211)
    ax1.axis("off")
    ax1.set_title(behavior_video.split('\\')[-1].split('_')[0] + '-' + behavior_video.split('\\')[-1].split('_')[1])
    ax2 = fig.add_subplot(212)
    ax2.plot(t, dff, 'k')
    ax2.set_ylabel('ΔF/F (%)', fontweight="bold")
    ax2.set_xlabel('Time (s)', fontweight="bold")
    ax2.set_title('Fiber signal', fontweight="bold", fontsize=15)
    ax2.set_xlim([t[0]-30, t[0]+30])
    ax2.set_ylim([np.min(dff) - 0.005, np.max(dff) + 0.005])
    vl = ax2.axvline(0, ls='-', color='r', lw=1)

    l1 = ax1.imshow(b_vid[0, :, :, :])

    with writer.saving(fig, out_filename, 100):
        for ix in range(np.shape(b_vid)[0]):
            l1.set_data(np.flipud(b_vid[ix, :, :, :]))

            vl.set_xdata([t[ix], t[ix]])
            ax2.set_xlim(t[ix]-30, t[ix]+30)

            writer.grab_frame()
    plt.close(fig)


def shorten_video(video):
    print("Reading behavior video")
    data_range, video_data = get_LED_on(video)
    print(data_range)
    video_name = video[0:video.find('.')] + "_short"
    vid_start, vid_end = data_range

    H = video_data.shape[1]
    W = video_data.shape[2]
    out_filename = video_name + ".avi"
    out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'MP42'), 100, (int(W), int(H)), True)

    for i in range(vid_start, vid_end):
        frame = cv2.cvtColor(video_data[i, :, :, :], cv2.COLOR_BGR2RGB)
        cv2.imshow(video, frame)
        out.write(frame)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    cv2.destroyAllWindows()
    out.release()



behavior_raw = r'Y:\nick\fiber\3-chamber-SDT\vCA1\Thy1\CXL2\20230313\20230313-CXL2-3.h264'
data_block = r"Y:\nick\fiber\3-chamber-SDT\vCA1\Thy1\CXL2\20230313\test-230313-151713-CXL2-3"
behavior_video = r'Y:\nick\fiber\3-chamber-SDT\vCA1\Thy1\CXL2\20230313\20230313-CXL2-3.avi'



shorten_video(behavior_raw)
make_combined_video(behavior_video, data_block, window=0.05)
