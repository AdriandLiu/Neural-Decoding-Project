%{
This function used the data saved in ms for a cell firing and converts the transients to binary. 
Adapted from code by Tori-Lynn Temple

function  spatialCoding = Binarize(neuron) 
dt = median(diff(ms.time))/1000; % Conversion from ms to s
Fs = 1/dt;
%}

%binarize traces from neuron.C.raw according to threshold set by standard
%deviations of low-pass filtered data using Butterworth filter
spike_data_C_raw=neuron.C_raw;
binarizedTraces_C_raw=zeros(size(spike_data_C_raw));  %binarized raw traces
z_threshold = 2;    %Zscore (std dev) threshold, typically ~2
[bFilt,aFilt] = butter(2,  2/(30/2), 'low');    %creates lowpass filter on order of 2 with normalized cutoff frequency 2/(30/2) where 2 is the cutoff freqeuency and 30 is the sampling frequency
for segment_C_raw = 1:size(spike_data_C_raw,1)                                        %Cycle through all segments in neuron.S
    %converts sparse matrix into readable array for the segment
    raw_trace_C_raw=(spike_data_C_raw(segment_C_raw,:));   %array the size of frames for the segment(e.g. 4727)
    filt_trace_C_raw = zscore(filtfilt(bFilt,aFilt,raw_trace_C_raw));   %filtfilt fxn performs zero phase digital filtering to preserve temporal component (e.g. spike at time=1000 becomes 900 with normal filter, but will remain at t=1000 with this filtering method)
    d1_trace_C_raw = diff(filt_trace_C_raw);    %first derivative of the trace (dx/dt)
    d1_trace_C_raw(end+1) = 0;
    %d2_trace = diff(d1_trace);  %second derivative of the trace
    %d2_trace(end+1) = 0;

    binary_trace_C_raw = filt_trace_C_raw*0;    %array for binarized trace
    binary_trace_C_raw(filt_trace_C_raw>z_threshold & d1_trace_C_raw>0) = 1;  %all spikes above the z threshold are set to 1

    binarizedTraces_C_raw(segment_C_raw, :) = binary_trace_C_raw;
end

%binarize traces from neuron.C
spike_data_C=neuron.C;
binarizedTraces_C=zeros(size(spike_data_C));  %binarized filtered traces
z_threshold = 2;    %Zscore (std dev) threshold, typically ~2
[bFilt,aFilt] = butter(2,  2/(30/2), 'low');    %creates lowpass filter on order of 2 with normalized cutoff frequency 2/(30/2) where 2 is the cutoff freqeuency and 30 is the sampling frequency
for segment_C = 1:size(spike_data_C,1)                                        %Cycle through all segments in neuron.S
    %converts sparse matrix into readable array for the segment
    raw_trace_C=(spike_data_C(segment_C,:));   %array the size of frames for the segment(e.g. 4727)
    filt_trace_C = zscore(filtfilt(bFilt,aFilt,raw_trace_C));   %filtfilt fxn performs zero phase digital filtering to preserve temporal component (e.g. spike at time=1000 becomes 900 with normal filter, but will remain at t=1000 with this filtering method)
    d1_trace_C = diff(filt_trace_C);    %first derivative of the trace (dx/dt)
    d1_trace_C(end+1) = 0;
    %d2_trace = diff(d1_trace);  %second derivative of the trace
    %d2_trace(end+1) = 0;

    binary_trace_C = filt_trace_C*0;    %array for binarized trace
    binary_trace_C(filt_trace_C>z_threshold & d1_trace_C>0) = 1;  %all spikes above the z threshold are set to 1

    binarizedTraces_C(segment_C, :) = binary_trace_C;
end

%binarize traces from neuron.S
spike_data_S=neuron.S;
binarizedTraces_S=zeros(size(spike_data_S));  %binarized filtered traces
z_threshold = 2;    %Zscore (std dev) threshold, typically ~2
[bFilt,aFilt] = butter(2,  2/(30/2), 'low');    %creates lowpass filter on order of 2 with normalized cutoff frequency 2/(30/2) where 2 is the cutoff freqeuency and 30 is the sampling frequency
for segment_S = 1:size(spike_data_S,1)                                        %Cycle through all segments in neuron.S
    %converts sparse matrix into readable array for the segment
    raw_trace_S=zeros(size(spike_data_S(segment_S,:)));   %array the size of frames for the segment(e.g. 4727)
    indices=find(spike_data_S(segment_S,:));    %array of the non-zero spikes' indices for the segment
    trace_values=nonzeros(spike_data_S(segment_S,:)); %array of the non-zero values corresponding to above indices
    for i=length(raw_trace_S) %loops through all frames in the segment
        for j=length(indices)   %loops through non-zero indices of the segment's spike data
            raw_trace_S(indices)=trace_values(j); %converts zero to non-zero spike value for corresponding index
        end
    end
    filt_trace_S = zscore(filtfilt(bFilt,aFilt,raw_trace_S));   %filtfilt fxn performs zero phase digital filtering to preserve temporal component (e.g. spike at time=1000 becomes 900 with normal filter, but will remain at t=1000 with this filtering method)
    d1_trace_S = diff(filt_trace_S);    %first derivative of the trace (dx/dt)
    d1_trace_S(end+1) = 0;
    %d2_trace = diff(d1_trace);  %second derivative of the trace
    %d2_trace(end+1) = 0;

    binary_trace_S = filt_trace_S*0;    %array for binarized trace
    binary_trace_S(filt_trace_S>z_threshold & d1_trace_S>0) = 1;  %all spikes above the z threshold are set to 1

    binarizedTraces_S(segment_S, :) = binary_trace_S;                                 %output
end 

%to check traces, plot(binarizedTraces_C_raw(segment number (e.g. 1, 2,
%etc.), :)) and plot(neuron.C_raw(segment number (e.g. 1, 2, etc.), :))
%compare 1v4, 2v5, 3v6
figure(1)
plot(neuron.C_raw(1, :))
figure(2)
plot(neuron.C(1, :))
figure(3)
plot(neuron.S(1, :))
figure(4)
plot(binarizedTraces_C_raw(1,:))
figure(5)
plot(binarizedTraces_C(1,:))
figure(6)
plot(binarizedTraces_S(1,:))

%save binarizedTraces for C, C_raw and S + neuron
save('binarized_data.mat', 'binarizedTraces_C', 'binarizedTraces_C_raw', 'binarizedTraces_S', 'neuron')