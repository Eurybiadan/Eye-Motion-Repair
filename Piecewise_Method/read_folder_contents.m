function [ file_list isdir numfiles ] = read_folder_contents( root_dir,extension )
% Robert Cooper 12 - 05 - 09
% This function extracts all the filenames from a folder with the desired
% extension.
% v2: an optional output argument is whether or not the path is directory
%     This is of the same length and is associated directly with a file
% Input:
% @ root_dir:
% This argument is the root directory that one wishes to be read.
% @ extension:
% This argument allows the caller to specify the extension for which he/she
% is looking.
%
% Copyright (C) 2009-2018 Robert Cooper
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.


x=1;
if ~exist('extension','var')
    file_list=dir(fullfile(root_dir,['*']));
    file_list=file_list(3:end);
else
    if( iscell(extension) )
        file_list = [];
        for i=1:length(extension)
           file_list = [file_list; dir(fullfile(root_dir,['*',extension{i}]));];
        end
    else
        file_list=dir(fullfile(root_dir,['*',extension]));
    end
end
numoffiles=length(file_list);


% This should NOT be used with large numoffiles sizes (large memory
% footprint)
filenames=cell(numoffiles,1);
is_dir=cell(numoffiles,1);

for x=1:1:numoffiles

    temp=file_list(x,1).name;
    temp_isdir=file_list(x,1).isdir;
    filenames{x}=temp; % -2 offset to correct for . and .. in the beginning
    is_dir{x}=temp_isdir;
    x=x+1;

end

numfiles=numoffiles;
file_list=filenames(1:numoffiles); 
isdir=is_dir(1:numoffiles);

end

