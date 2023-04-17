
chamber = imread('20221125-CD4-CXL_CXR_f_n_chamber.jpg');

figure, imshow(chamber), hold on, plot(x,y)

figure, plot(data465)

%%
figure
t = xt(data465, fs);
dff = (data465 - mean(data465))./mean(data465);

plot(t, dff);
grid on;

% Convert X and Y into a table, which is the form fitnlm() likes the input data to be in.
tbl = table(t', dff');
% Define the model as Y = a + exp(-b*x)
% Note how this "x" of modelfun is related to big X and big Y.
% x((:, 1) is actually X and x(:, 2) is actually Y - the first and second columns of the table.
modelfun = @(b,x) b(1) * exp(-b(2)*x(:, 1)) + b(3);  
beta0 = [max(dff), 0.05, min(dff)]; % Guess values to start with.  Just make your best guess.
% Now the next line is where the actual model computation is done.
mdl = fitnlm(tbl, modelfun, beta0);

% Now the model creation is done and the coefficients have been determined.
% YAY!!!!

% Extract the coefficient values from the the model object.
% The actual coefficients are in the "Estimate" column of the "Coefficients" table that's part of the mode.
coefficients = mdl.Coefficients{:, 'Estimate'}
% Create smoothed/regressed data using the model:
yFitted = coefficients(1) * exp(-coefficients(2)*t) + coefficients(3);
% Now we're done and we can plot the smooth model as a red line going through the noisy blue markers.
hold on;
plot(t, yFitted, 'r-', 'LineWidth', 2);
grid on;

figure, plot(t, dff-yFitted)

%%
clc
pi_fps = length(y)/t(end);
pi_t = xt(x, pi_fps);
data_detrend = dff - yFitted;

data_interp = interp1(t, data_detrend, pi_t);

[peaks, locs, w, p] = findpeaks(data_interp, 'MinPeakProminence', 0.05, 'MinPeakWidth', 5);
figure, plot(data_interp, 'LineWidth', 2), grid on, hold on, xline(locs)%xline(pi_t(locs))
%%

figure, 
subplot(1,2,1), 
imshow(chamber), hold on,
plot(x,y)
scatter(x(locs), y(locs), p*1000, 'MarkerEdgeColor', 'k', 'MarkerFaceColor',  'r', 'MarkerFaceAlpha', 0.4)

subplot(1,2,2),
imshow(chamber), hold on,
plot(x,y)
rand_idx = randperm(length(x), length(locs));
scatter(x(rand_idx), y(rand_idx), p*1000, 'MarkerEdgeColor', 'k', 'MarkerFaceColor',  'g', 'MarkerFaceAlpha', 0.4)

%% extract peak videos

filename = '20221125-CD4-CXL_CXR_f_n_marker.avi';
v = VideoReader(filename);


frames = read(v, [1 Inf]);

%%
frames_subset = [];
for i = 1:length(locs)
    frames_subset = cat(4, frames_subset, frames(:, :, :, locs(i)-40:locs(i)+40));
end



