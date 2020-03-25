function ret= filesort(filenames)
ret1= [];
exp1='[0-9]+.avi';               %gets the last bit and extension of the file
exp2='[0-9]+';                   %gets the label number of the file 
[match,prefix]=regexp(filenames{1},exp1,'match','split'); %nomatch is the file prefix 
for i=1:length(filenames)
    temp1=regexp(filenames{i},exp1,'match');
    temp2=regexp(temp1,exp2,'match');
    ret1=[ret1,temp2];
end

ret2=[];
for i=1:length(ret1)
    ret2=[ret2;ret1{i}];
end
ret2=sort(str2double(ret2));

ret=[];
for i=1:length(ret2)
   ret=[ret;strcat(prefix,int2str(ret2(i,1)),'.avi')];
end
ret=ret(:,1);