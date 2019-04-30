fileList = dir('binarized*.mat');
mouse = dir("*.mat");
for k1 = 1:length(mouse)
    names{k1} = mouse(k1).name;  
end
a = regexp(names,'\d{4}','match');
mouse_num = string(a(1,1)); %%Mouse number

for i = 1:length(fileList)
    data = load(fileList(i).name); %%change to woring directory
    csvwrite(mouse_num + ' binarizedC.csv',data.binarizedTraces_C)
end