function [result, status] = perl(varargin)
%PERL Execute Perl command and return the result.
%   PERL(PERLFILE) calls perl script specified by the file PERLFILE
%   using appropriate perl executable.
%
%   PERL(PERLFILE,ARG1,ARG2,...) passes the arguments ARG1,ARG2,...
%   to the perl script file PERLFILE, and calls it by using appropriate
%   perl executable.
%
%   RESULT=PERL(...) outputs the result of attempted perl call.  If the
%   exit status of perl is not zero, an error will be returned.
%
%   [RESULT,STATUS] = PERL(...) outputs the result of the perl call, and
%   also saves its exit status into variable STATUS.
%
%   If the Perl executable is not available, it can be downloaded from:
%     http://www.cpan.org
%
%   See also SYSTEM, JAVA, MEX.

%   Copyright 1990-2018 The MathWorks, Inc.

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

% Execute Perl script
cmdString = ['python' cmdString];
if ispc && ~isdeployed
    % Add perl to the path
    perlInst = fullfile('C:\Users\Donghan\AppData\Local\Programs\Python\Python37');
    cmdString = ['set PATH=',perlInst, ';%PATH%&' cmdString];
end

[status, result] = system(cmdString);

% Check for errors in shell command
if nargout < 2 && status~=0
    error(message('MATLAB:perl:ExecutionError', result, cmdString));
end


