clear, clc

[trimmed_video_file, output_path] = uigetfile('*.avi','Select trimmed video file for analysis.', 'Y:\nick\fiber_photometry\vCA1\Thy1');
fiber_file = uigetfile('*.mat', 'Select fiber photometry data.', output_path);
load([output_path, fiber_file])

cd(output_path)
analysis_path = [output_path, filesep, 'analysis'];
if ~isfolder(analysis_path)
    mkdir(analysis_path)
end
    
disp('Record index of frames at events of interest')
implay([output_path, trimmed_video_file])

%%

fs = 20;
events = [1616, 7688];
event_labels = {'Novel male introduced', 'Novel female introduced'};
baseline = round(30*fs);
post_stim = round(60*fs);
% save([analysis_path, filesep, 'events.mat'], 'events', 'event_labels')

epochs = [];
for i =1:length(events)
    epochs(i,:) = dff465(events(i)-baseline:events(i)+post_stim);
end
t = xt(epochs, fs, 2)-30;
figure, plot(t, epochs(1,:))
hold on
plot(t, epochs(2,:)-3)

%%


t = xt(epochs, fs, 2)-3;
figure
for i = 1:size(epochs,1)
    plot(t, epochs(i,:)), hold on
    xline(0), hold off
    h = waitforbuttonpress;
end
%%
event_labels = {'Pick up', 'In air', 'On platform', 'Climb down', 'Hanging by paw'};
% range_of_interest = events(1)-round(baseline):events(end)+round(baseline);
t = xt(dff465, fs);
% figure, plot(t, dff465(range_of_interest))
figure, plot(t,dff465)
hold on, xline(t(events))
