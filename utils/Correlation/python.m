function [result, status] = python(varargin)

%Put this file inside D:\MATLAB\toolbox\matlab\general
%May differ from your OS.


if nargin > 0
    [varargin{:}] = convertStringsToChars(varargin{:});
end

cmdString = '';

% Add input to arguments to operating system command to be executed.
% (If an argument refers to a file on the MATLAB path, use full file path.)
for i = 1:nargin
    thisArg = varargin{i};
    if i==1
        if exist(thisArg, 'file')==2
            % This is a valid file on the MATLAB path
            if isempty(dir(thisArg))
                % Not complete file specification
                % - file is not in current directory
                % - OR filename specified without extension
                % ==> get full file path
                thisArg = which(thisArg);
            end
        else
            % First input argument is PerlFile - it must be a valid file
            error(message('MATLAB:perl:FileNotFound', thisArg));
        end
    end
    arg = [];
    % Wrap thisArg in double quotes if it contains spaces
    if i ~= 4 && ~isempty(thisArg)
        arg = strcat(arg,thisArg); % #ok<AGROW>
    else
        data = varargin(i);
        save('dataC.mat','data');
    end


    % Add argument to command string
    
    cmdString = [cmdString, ' ', arg]; %#ok<AGROW>
    cmdString = deblank(cmdString);
end
% Check that the command string is not empty
if isempty(cmdString)
    error(message('MATLAB:perl:NoPerlCommand'));
end


% Check that perl is available if this is not a PC or isdeployed
if ~ispc || isdeployed
    if ispc
        checkCMDString = 'perl -v';
    else
        checkCMDString = 'which perl';
    end
    [cmdStatus, ~] = system(checkCMDString);
    if cmdStatus ~=0
        error(message('MATLAB:perl:NoExecutable'));
    end
end

% Execute Python script
cmdString = ['python' cmdString];
if ispc && ~isdeployed
    % Add python to the path
    pyInst = fullfile('C:\Users\Donghan\AppData\Local\Programs\Python\Python37');
    % Change this to Python installation folder
    cmdString = ['set PATH=',pyInst, ';%PATH%&' cmdString];
end

[status, result] = system(cmdString);

% Check for errors in shell command
if nargout < 2 && status~=0
    error(message('MATLAB:perl:ExecutionError', result, cmdString));
end


