% Deep Residual Learning Meets OFDM Channel Estimation
% SNR represents Es/N0, Es/N0 = Eb/N0 * log2(M)
% For SNR, the power of pilot is ignored when calculating the varience of
% noise and the pilots suffer from the noise on equal level with data
% Power is balanced since IFFT/FFT is applied
% InsertDCNull is true and the effect caused by DCNull is considered in the
% SNR_OFDM, which is adjusted to be SNR + 10 * log(Num_of_subcarriers_used
% / Num_of_FFT)
% h_channel is different for each frame and stored in H, or read from h_set

SNR_Range = -5:5:25;
Num_of_frame_each_SNR = 5000;

MSE_LS_over_SNR = zeros(length(SNR_Range), 1);
MSE_MMSE_over_SNR = zeros(length(SNR_Range), 1);
MSE_DNN_over_SNR = zeros(length(SNR_Range), 1);
MSE_ReEsNet_over_SNR = zeros(length(SNR_Range), 1);
MSE_Improved_DNN_over_SNR = zeros(length(SNR_Range), 1);

% Import Deep Neuron Network
load('SL.mat'); % SL102_10filter SL

% Import Deep Neuron Network
load('ReEsNet_50000.mat');

% Import Deep Neuron Network
load('ReEsNet.mat');

Rgg_value = zeros(73, 73, 4);

for SNR = SNR_Range

Num_of_symbols = 12;
Num_of_pilot = 2;
Frame_size = Num_of_symbols + Num_of_pilot;

M = 4; % QPSK
k = log2(M);

Num_of_subcarriers = 72;
Num_of_FFT = Num_of_subcarriers + 1;
length_of_CP = 16;

Pilot_interval = 7;
Pilot_starting_location = 1;

Pilot_location = [(1:3:Num_of_subcarriers)', (2:3:Num_of_subcarriers)'];
Pilot_value_user = 1 + 1j;

length_of_symbol = Num_of_FFT + length_of_CP;

MaxDopplerShift = 97;

Num_of_QPSK_symbols = Num_of_subcarriers * Num_of_symbols * Num_of_frame_each_SNR;
Num_of_bits = Num_of_QPSK_symbols * k;

LS_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
MMSE_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
DNN_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
ReEsNet_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);
Improved_DNN_MSE_in_frame = zeros(Num_of_frame_each_SNR, 1);

t_LS = zeros(Num_of_frame_each_SNR, 1);
t_MMSE = zeros(Num_of_frame_each_SNR, 1);
t_DNN = zeros(Num_of_frame_each_SNR, 1);
t_Improved = zeros(Num_of_frame_each_SNR, 1);

for Frame = 1:Num_of_frame_each_SNR

% Data generation
N = Num_of_subcarriers * Num_of_symbols;
data = randi([0 1], N, k);
Data = reshape(data, [], 1);
dataSym = bi2de(data);

% QPSK modulator
QPSK_symbol = QPSK_Modualtor(dataSym);
QPSK_signal = reshape(QPSK_symbol, Num_of_subcarriers, Num_of_symbols);

% Pilot inserted
[data_in_IFFT, data_location] = Pilot_Insert(Pilot_value_user, Pilot_starting_location, Pilot_interval, Pilot_location, Frame_size, Num_of_FFT, QPSK_signal);
[data_for_channel, ~] = Pilot_Insert(1, Pilot_starting_location, Pilot_interval, kron((1 : Num_of_subcarriers)', ones(1, Num_of_pilot)), Frame_size, Num_of_FFT, (ones(Num_of_subcarriers, Num_of_symbols)));
data_for_channel(1, :) = 1;

% OFDM Transmitter
[Transmitted_signal, ~] = OFDM_Transmitter(data_in_IFFT, Num_of_FFT, length_of_CP);
[Transmitted_signal_for_channel, ~] = OFDM_Transmitter(data_for_channel, Num_of_FFT, length_of_CP);

% Channel

% AWGN Channel
SNR_OFDM = SNR + 10 * log10((Num_of_subcarriers / Num_of_FFT));
%awgnChan = comm.AWGNChannel('NoiseMethod', 'Signal to noise ratio (SNR)');
%awgnChan.SNR = SNR_OFDM;

% Multipath Rayleigh Fading Channel

%PathDelays = [0 50 120 200 230 500 1600 2300 5000] * 1e-9; % ETU
%AveragePathGains = [-1.0 -1.0 -1.0 0.0 0.0 0.0 -3.0 -5.0 -7.0]; % ETU
PathDelays = [0 30 70 90 110 190 410] * 1e-9; % EPA
AveragePathGains = [0 -1 -2 -3 -8 -17.2 -20.8]; % EPA
%PathDelays = [0 30 150 310 370 710 1090 1730 2510] * 1e-9; % EVA
%AveragePathGains = [0 -1.5 -1.4 -3.6 -0.6 -9.1 -7 -12 -16.9]; % EVA

rayleighchan = comm.RayleighChannel(...
    'SampleRate', 1065000, ...
    'PathDelays', PathDelays, ...
    'AveragePathGains', AveragePathGains, ...
    'MaximumDopplerShift', randi([0, MaxDopplerShift]), ...
    'PathGainsOutputPort', true);

chaninfo = info(rayleighchan); 
coeff = chaninfo.ChannelFilterCoefficients;
Np = length(rayleighchan.PathDelays);
state = zeros(size(coeff, 2) - 1, size(coeff, 1)); % initializing the delay filter state

[Multitap_Channel_Signal_user, Path_gain] = rayleighchan(Transmitted_signal);
fracdelaydata = zeros(size(Transmitted_signal, 1), Np);

for j = 1 : Np
    [fracdelaydata(:,j), state(:,j)] = filter(coeff(j, :), 1, Transmitted_signal_for_channel, state(:,j)); % fractional delay filter state is taken care of here.
end

SignalPower = mean(abs(Multitap_Channel_Signal_user) .^ 2);
Noise_Variance = SignalPower / (10 ^ (SNR_OFDM / 10));

Nvariance = sqrt(Noise_Variance / 2);
n = Nvariance * (randn(length(Transmitted_signal), 1) + 1j * randn(length(Transmitted_signal), 1)); % Noise generation

Multitap_Channel_Signal = Multitap_Channel_Signal_user + n;

Multitap_Channel_Signal_user_for_channel = sum(Path_gain .* fracdelaydata, 2);

% OFDM Receiver
[Unrecovered_signal, RS_User] = OFDM_Receiver(Multitap_Channel_Signal, Num_of_FFT, length_of_CP, length_of_symbol, Multitap_Channel_Signal_user);
[~, RS] = OFDM_Receiver(Multitap_Channel_Signal_user_for_channel, Num_of_FFT, length_of_CP, length_of_symbol, Multitap_Channel_Signal_user_for_channel);
Pilot_location_symbols = Pilot_starting_location : Pilot_interval : Frame_size;

[Received_pilot, ~] = Pilot_extract(RS_User(2:end, :), Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);
H_Ref = Received_pilot ./ Pilot_value_user;

% Channel estimation

% Perfect knowledge on Channel

% LS
[Received_pilot_LS, ~] = Pilot_extract(Unrecovered_signal(2:end, :), Pilot_location, Num_of_pilot, Pilot_location_symbols, data_location);

H_LS = Received_pilot_LS / Pilot_value_user;

H_LS_frame = imresize(H_LS, [Num_of_subcarriers, Frame_size]);

MSE_LS_frame = mean(abs(H_LS_frame - RS(2:end, :)).^2, 'all');

% MMSE

% linear MMSE

H_MMSE = zeros(Num_of_subcarriers, Num_of_pilot);

for i = 1:size(Pilot_location, 2)
H_pilot = H_Ref(:, i);
Rhh = H_pilot * H_pilot';
H_MMSE(:, i) = (RS(2:end, i) * (H_pilot')) * pinv(Rhh + (1 / 10^(SNR / 10)) * eye(size(Rhh, 1))) * H_LS(:, i);
end

H_MMSE_frame = imresize(H_MMSE, [Num_of_subcarriers, Frame_size]);

MSE_MMSE_frame = mean(abs(H_MMSE_frame - RS(2:end, :)).^2, 'all');

% ReEsNet A
Res_feature_signal(:, :, 1) = real(Received_pilot_LS / Pilot_value_user);
Res_feature_signal(:, :, 2) = imag(Received_pilot_LS / Pilot_value_user);

H_DNN_feature = predict(ReEsNet_A, Res_feature_signal);

H_DNN = H_DNN_feature(:, :, 1) + 1j * H_DNN_feature(:, :, 2);
MSE_DNN_frame = mean(abs(H_DNN - RS(2:end, :)).^2, 'all');

% ReEsNet B
Res_feature_signal(:, :, 1) = real(Received_pilot_LS / Pilot_value_user);
Res_feature_signal(:, :, 2) = imag(Received_pilot_LS / Pilot_value_user);

H_DNN_feature = predict(ReEsNet_B, Res_feature_signal);

H_DNN = H_DNN_feature(:, :, 1) + 1j * H_DNN_feature(:, :, 2);
MSE_DNN_frame_ReEsNet = mean(abs(H_DNN - RS(2:end, :)).^2, 'all');

% Improved Deep learning
Res_feature_signal(:, :, 1) = real(Received_pilot_LS / Pilot_value_user);
Res_feature_signal(:, :, 2) = imag(Received_pilot_LS / Pilot_value_user);

H_DNN_feature = predict(SL, Res_feature_signal);%Improved_DNN_Trained

H_DNN = H_DNN_feature(:, :, 1) + 1j * H_DNN_feature(:, :, 2);
MSE_Improved_DNN_frame = mean(abs(H_DNN - RS(2:end, :)).^2, 'all');

% LS MSE calculation in each frame
LS_MSE_in_frame(Frame, 1) = MSE_LS_frame;

% MMSE MSE calculation in each frame
MMSE_MSE_in_frame(Frame, 1) = MSE_MMSE_frame;

% DNN MSE calculation in each frame
DNN_MSE_in_frame(Frame, 1) = MSE_DNN_frame;

% DNN MSE calculation in each frame
ReEsNet_MSE_in_frame(Frame, 1) = MSE_DNN_frame_ReEsNet;

% DNN MSE calculation in each frame
Improved_DNN_MSE_in_frame(Frame, 1) = MSE_Improved_DNN_frame;

end

% BER calculation
MSE_LS_over_SNR(SNR_Range == SNR, 1) = sum(LS_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_MMSE_over_SNR(SNR_Range == SNR, 1) = sum(MMSE_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_DNN_over_SNR(SNR_Range == SNR, 1) = sum(DNN_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_ReEsNet_over_SNR(SNR_Range == SNR, 1) = sum(ReEsNet_MSE_in_frame, 1) / Num_of_frame_each_SNR;

MSE_Improved_DNN_over_SNR(SNR_Range == SNR, 1) = sum(Improved_DNN_MSE_in_frame, 1) / Num_of_frame_each_SNR;

end
