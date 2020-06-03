[segments,frames]=size(neuron.S);
total_frames=['Total number of frames =',num2str(frames)];
clc;
disp(total_frames)
prompt = 'Start frame: ';
start_frame=input(prompt);
prompt = 'End frame: ';
end_frame=input(prompt);
   for x = 1:segments;
        [pkgS,locsS]=findpeaks(neuron.S(x,start_frame:end_frame));
        Ncount(x)=sum(locsS>0);
        pkgSsegments=full(pkgS);
        NheightS(x)=sum(pkgSsegments)/sum(pkgSsegments>0);
        [pkgC,locsC]=findpeaks(neuron.C(x,start_frame:end_frame));
        pkgCsegments=full(pkgC);
        NheightC(x)=sum(pkgCsegments)/sum(pkgCsegments>0);
   end
segments
meanspikes=mean(Ncount)
%%meanS=sum(NheightS)/sum(NheightS>0)
%%meanC=sum(NheightC)/sum(NheightC>0)
meanS=sum(NheightS(~isnan(NheightS)))/sum(NheightS>0)
meanC=sum(NheightC(~isnan(NheightC)))/sum(NheightC>0)
active_segments=sum(NheightS>0)
studied_frames=end_frame-start_frame
