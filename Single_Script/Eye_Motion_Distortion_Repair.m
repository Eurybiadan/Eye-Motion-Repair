function [] = Eye_Motion_Distortion_Repair(motion_path, fName, crop_ROI, framemotion, static_grid_distortion)
% EYE_MOTION_DISTORTION_REPAIR(motion_path, fName, static_grid_distortion,framemotion, static_grid_distortion)
%
% [] = EYE_MOTION_DISTORTION_REPAIR(motion_path, fName, static_grid_distortion,framemotion, static_grid_distortion)
%
% Inputs:
%
%    @motion_path: The path to the file we're going to repair.
%
%    @fName: The name of the file we're going to repair.
%
%    @crop_ROI: The region of the output image in [x y w h]. If this has
%    multiple rows, then we will search for which row matches of the
%    dimensions of the image we're working with.
%
%    @framemotion: The amount of motion in a given strip, extracted from 
%                  the dmp file, structured as follows:
%       An i*3 x j matrix detailing how each row (i) of each image was
%       translated to align to the reference frame.There are 3 rows of data
%       for each transformed image. For each set of 3 rows the first row
%       are the row indices in the reference frame that the image was
%       aligned to. The second row is the x motion *relative to that index*
%       that the image moved, and the third row is the y motion for the
%       same thing.
%
%    @static_grid_distortion: The residual static distortion (calculated 
%                             from Static_Distortion_Repair) that we'll
%                             remove.
%    
% Copyright (C) 2018 Robert Cooper
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
%

    % Check this version of EMR against git.
%     fid = fopen(fullfile(getparent(which(mfilename)),'.VERSION'),'r');
%     if fid ~= -1
%         thisver = fscanf(fid,'%s');
%         fclose(fid);
% 
%         git_version_check( 'Eurybiadan','Eye-Motion-Repair', thisver, 'WarningOnly',true )
%     else
%         warning('Failed to detect .VERSION file, unable to determine if running the newest version.')
%     end

    repeats = 1;
    outlier_cutoff = 20;
    
    
    if strcmp(fName(end-2:end), 'tif')
        
        imStk = cell(1);
        imStk{1} = imread(fullfile(motion_path, fName));
        
    elseif strcmp(fName(end-2:end), 'avi')
        
        vidobj = VideoReader( fullfile(motion_path, fName) );
        vid_length = round(vidobj.Duration*vidobj.FrameRate);
        
        imStk = cell(1, vid_length);
        
        i=1;
        while hasFrame(vidobj)
            imStk{i} = readFrame(vidobj);
            i=i+1;
        end
        
    end

    
    crop_ROI = cell2mat(crop_ROI)    ;
    tmp_rois = zeros([length(crop_ROI)/4, 4]);
    for i=1:4:length(crop_ROI)-1        
        tmp_rois(1+floor(i/4),:) = crop_ROI(i:i+3);
    end
    crop_ROI=tmp_rois;
    ROI_sizes = [crop_ROI(:,2)-crop_ROI(:,1) crop_ROI(:,4)-crop_ROI(:,3)];
    im_size = size(imStk{1});
    
    for i=1:size(ROI_sizes,1)
        if all(ROI_sizes(i,:) == im_size)
            crop_ROI = crop_ROI(i,:)+1;
            break;
        end
    end

    
    tmp = zeros( length(framemotion), length(framemotion{1}) );
    % Convert from list into regular array
    for i=1:length(framemotion)
        tmp(i,:) = cell2mat(framemotion{i});
    end
     framemotion = tmp;
clear tmp;
    
    %Ref_Frame
    ref_largest_slow_axis = max(framemotion(1,:));
    ref_smallest_slow_axis = min(framemotion(1,:));

    all_slow_axis_ref_ind = 1:3:size(framemotion,1);
    all_fast_axis_trans_ind = 2:3:size(framemotion,1);
    all_slow_axis_trans_ind = 3:3:size(framemotion,1);
    
    % Find the largest slow axis pixel
    largest_slow_axis = max(max(framemotion(all_slow_axis_ref_ind,:)));
    smallest_slow_axis = min(min(framemotion(all_slow_axis_ref_ind,:)));
    
    slow_axis_size = largest_slow_axis-smallest_slow_axis+1;
    all_xmotion  = cell(slow_axis_size, repeats);
    all_ymotion  = cell(slow_axis_size, repeats);

    % Determine the index that the row corresponds to
    framemotion(all_slow_axis_ref_ind,:) = 1+framemotion(all_slow_axis_ref_ind,:)-smallest_slow_axis; 
    %min(min(framemotion(all_slow_axis_ref_ind,:)))
    
    for r=1:repeats
        xmotion  = cell(slow_axis_size,1);
        ymotion  = cell(slow_axis_size,1);
        
        rng('shuffle');
        if repeats == 1
            randselects = 1:length(all_slow_axis_ref_ind);
        else
            randselects = randperm(length(all_slow_axis_ref_ind), floor(2*length(all_slow_axis_ref_ind)/3) );
        end
        
        slow_axis_ref_ind = all_slow_axis_ref_ind(randselects);
        fast_axis_trans_ind = all_fast_axis_trans_ind(randselects);
        slow_axis_trans_ind = all_slow_axis_trans_ind(randselects);

        for j=1:length(slow_axis_ref_ind)
	    
            % Find the max for the row, because a row may not go the full
            % length (could be shorter than the width of the matrix
            [maxref, maxrefind] = max(framemotion( slow_axis_ref_ind(j), : ));
		
            for k=1: maxrefind
%		disp([ num2str(k) ',' num2str(framemotion( slow_axis_ref_ind(j), k )) ',' num2str(framemotion( fast_axis_trans_ind(j), k )) ])
                xmotion{framemotion( slow_axis_ref_ind(j), k )}  = [ xmotion{framemotion( slow_axis_ref_ind(j), k )}, ...
                                                                     framemotion( fast_axis_trans_ind(j), k ) ];
                ymotion{framemotion( slow_axis_ref_ind(j), k )}  = [ ymotion{framemotion( slow_axis_ref_ind(j), k )}, ...
                                                                     framemotion( slow_axis_trans_ind(j), k ) ];
            end

        end

        for j=1:length(all_xmotion)
            all_xmotion{j,r} = [all_xmotion{j,r} xmotion{j}];
            all_ymotion{j,r} = [all_ymotion{j,r} ymotion{j}];
        end
    end

    startingind = ref_smallest_slow_axis-smallest_slow_axis+1;
    
    % Clamp the values in case Python overestimates the size
    if crop_ROI(2)>size(all_xmotion,1) || crop_ROI(2)>size(all_ymotion,1)
        crop_ROI(2) = min(size(all_xmotion,1), size(all_ymotion,1));
    end
    
    if crop_ROI(1)<1
        crop_ROI(1)=1;
    end
    
     % Clip out the rows that aren't part of our reference frame
    % so that it matches the cropped output image!
    all_xmotion = all_xmotion( crop_ROI(1):crop_ROI(2),:);
    all_ymotion = all_ymotion( crop_ROI(1):crop_ROI(2),:);



    %% View and adjust each row's translation so that we have something we can
    % move
    theinds=(1:size(all_xmotion,1))';

    xmotion_norm=cell(size(all_xmotion,1),repeats);
    ymotion_norm=cell(size(all_xmotion,1),repeats);
    xmotion_vect=zeros(size(all_xmotion,1),repeats);
    ymotion_vect=zeros(size(all_xmotion,1),repeats);

    estimatevar=zeros(size(all_xmotion,1),repeats);

%     imwrite(im, fullfile(motion_path,[fName(1:end-4) '_motion.tif']),'WriteMode','overwrite');
    for r=1:repeats

        for i=1:size(all_xmotion,1)
            % Do a simple outlier removal- should be changed to something like std
            % dev
            nooutx = all_xmotion{i,r}(all_xmotion{i,r}<outlier_cutoff & all_ymotion{i,r}<outlier_cutoff & ...
                     all_xmotion{i,r}>-outlier_cutoff & all_ymotion{i,r}>-outlier_cutoff);
            noouty = all_ymotion{i,r}(all_xmotion{i,r}<outlier_cutoff & all_ymotion{i,r}<outlier_cutoff & ...
                     all_xmotion{i,r}>-outlier_cutoff & all_ymotion{i,r}>-outlier_cutoff);

        % For displaying the row offsets
%             figgy=figure(1);
%             plot(all_xmotion{i,r},all_ymotion{i,r},'.');hold on;
%             plot(median(nooutx),median(noouty),'kx');
%             plot(0,0,'r.'); hold off;
%             axis square; axis([-outlier_cutoff outlier_cutoff -outlier_cutoff outlier_cutoff]); 
%             title([ 'median x: ' num2str(median(all_xmotion{i,r})) ' median y: ' num2str(median(all_ymotion{i,r})) ]);
%              drawnow;             
%              frame= getframe(figgy);
%              imwrite(frame.cdata, fullfile(motion_path,[fName(1:end-4) '_motion.tif']),'WriteMode','append');
%              writeVideo(v,f);

            estimatevar(i,r) = sqrt(var(all_xmotion{i,r}).^2 + var(all_ymotion{i,r}).^2);
            xmotion_norm{i,r} = all_xmotion{i,r}-median(all_xmotion{i,r});
            ymotion_norm{i,r} = all_ymotion{i,r}-median(all_ymotion{i,r});

            xmotion_vect(i,r) = median(all_xmotion{i,r});
            ymotion_vect(i,r) = median(all_ymotion{i,r});

            if isnan(xmotion_vect(i,r)) || isnan(ymotion_vect(i,r))
                xmotion_vect(i,r) = 0;
                ymotion_vect(i,r) = 0;
                disp('NANI!?');
            end

        end
    end
    

%     figyer = figure(2);
%     plot(xmotion_vect);    
%     saveas(figyer,fullfile(motion_path,[fName(1:end-4) '_xmotion.tif']));
%     figyer = figure(2);
%     plot(ymotion_vect);    
%     saveas(figyer,fullfile(motion_path,[fName(1:end-4) '_ymotion.tif']));
% save(fullfile(motion_path,[fName(1:end-4) '_snapshot.mat']));

    % Smooth our vectors a little bit- should help in situations where
    % there are large jumps.
    ft = fittype( 'smoothingspline' );
    opts = fitoptions( 'Method', 'SmoothingSpline' );
    opts.Normalize = 'on';
    opts.SmoothingParam = 0.99998;
    
    
    
    % Fit model to data.
    [fitresult] = fit( theinds, xmotion_vect, ft, opts );
    xmotion_vect = fitresult(theinds);    
    [fitresult] = fit( theinds, ymotion_vect, ft, opts );
    ymotion_vect = fitresult(theinds);

    
    for i=1:size(xmotion_vect,1)

        xgriddistortion(i,:) = repmat(median(xmotion_vect(i,:)), [1 size(imStk{1},2)] ); %The ref should be all 0s

		if ~isempty(static_grid_distortion)
            ygriddistortion(i,:) = repmat(median(ymotion_vect(i,:))+static_grid_distortion(i+crop_ROI(1)-1), [1 size(imStk{1},2)] );
        else
            ygriddistortion(i,:) = repmat(median(ymotion_vect(i,:)), [1 size(imStk{1},2)] );
        end
    end
    
    disp_field = cat(3,xgriddistortion,ygriddistortion);

    warpedStk = zeros(size(disp_field,1),size(disp_field,2),length(imStk));
    
    for i=1:length(imStk)
        warpedStk(:,:,i) = imwarp(double(imStk{i}),disp_field,'FillValues',NaN);
    end


    if ~exist( fullfile(motion_path, 'Repaired'), 'dir' )
        mkdir(fullfile(motion_path, 'Repaired'))
    end
    
    warpedIm = mean(warpedStk,3);
    
    
    % Crop out any border regions
    imregions= bwconncomp(warpedIm>0);
    cropbox = regionprops(imregions,'Area','BoundingBox');
    [maxarea, maxind] = max([cropbox.Area]); % Take the bigger of the two areas
    cropbox = cropbox(maxind).BoundingBox;
    
    warpedStk = warpedStk( round(cropbox(2)):round(cropbox(4)), round(cropbox(1)):round(cropbox(3)), : );    

    if length(imStk)== 1
        
        saveTransparentTif(warpedStk(:,:,1),fullfile(motion_path,'Repaired', [fName(1:end-4) '_repaired.tif']));
        
%         imwrite(warpedStk, fullfile(motion_path,'Repaired', [fName(1:end-4) '_repaired.tif']), 'Compression','lzw');
    else
        vidobj = VideoWriter( fullfile(motion_path,'Repaired', [fName(1:end-4) '_repaired.avi']), 'Grayscale AVI' );

        open(vidobj);
        writeVideo(vidobj,uint8(warpedStk));
        close(vidobj);
    end
    
end
