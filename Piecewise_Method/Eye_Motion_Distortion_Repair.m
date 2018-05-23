function [] = Eye_Motion_Distortion_Repair(motion_path, fName, static_grid_distortion)
    
    repeats = 1;
    outlier_cutoff = 20;

    
    flipName = fliplr(fName);
    
    [tok,remain]=strtok(flipName,'_');
    for i=1:4
        [tok,remain]=strtok(remain,'_');
    end
    
    commonName = fliplr(remain);
    
    xformsFname = [commonName 'transforms.csv'];
   
    modalities = {'confocal','split_det','avg'};    
    repcheck = perms(1:length(modalities));
    
    if ~exist( fullfile(motion_path, xformsFname),'file')
        for i=1:size(repcheck,1)
            tmpname = strrep(xformsFname, modalities{repcheck(i,1)}, modalities{repcheck(i,2)});
            if exist( fullfile(motion_path, tmpname),'file')
                xformsFname = tmpname;
               break; 
            end
            tmpname = strrep(xformsFname, modalities{repcheck(i,1)}, modalities{repcheck(i,3)});
            if exist( fullfile(motion_path, tmpname),'file')
               xformsFname = tmpname;
               break; 
            end
        end
    end
    
    if ~exist( fullfile(motion_path, xformsFname),'file')
        warning(['Unable to find transform file for image: ' fName ]);
        return;
    end
    
    loadmotion = dlmread( fullfile(motion_path, xformsFname) );
    
    crop_ROI = loadmotion(1,1:4)+1; % Add one so the index (designed for python) is correct in MATLAB.
    framemotion = loadmotion(2:end,:);
    
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


    %Ref_Frame
    ref_largest_slow_axis = max(framemotion(1,:));
    ref_smallest_slow_axis = min(framemotion(1,:));

    % Find the largest slow axis pixel
    largest_slow_axis = 0;
    smallest_slow_axis = 100000;
   
    if largest_slow_axis < max(max(framemotion(1:3:size(framemotion,1),:),[],2))
        largest_slow_axis = max(max(framemotion(1:3:size(framemotion,1),:),[],2));        
    end
    if smallest_slow_axis > min(framemotion(1:3:size(framemotion,1),1))
        smallest_slow_axis = min(framemotion(1:3:size(framemotion,1),1));
    end
    
    slow_axis_size = largest_slow_axis-smallest_slow_axis+1;
    all_xmotion  = cell(slow_axis_size, repeats);
    all_ymotion  = cell(slow_axis_size, repeats);


    all_slow_axis_ref_ind = 1:3:size(framemotion,1);
    all_fast_axis_trans_ind = 2:3:size(framemotion,1);
    all_slow_axis_trans_ind = 3:3:size(framemotion,1);
    
    % Determine the index that the row corresponds to
    framemotion(all_slow_axis_ref_ind,:) = 1+framemotion(all_slow_axis_ref_ind,:)-smallest_slow_axis; 
    
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

    xmotion_norm=cell(size(all_xmotion,1),repeats);
    ymotion_norm=cell(size(all_xmotion,1),repeats);
    xmotion_vect=zeros(size(all_xmotion,1),repeats);
    ymotion_vect=zeros(size(all_xmotion,1),repeats);

    estimatevar=zeros(size(all_xmotion,1),repeats);

    % imwrite(im, 'monte_warped.tif','WriteMode','overwrite');
    for r=1:repeats

        for i=1:size(all_xmotion,1)
            % Do a simple outlier removal- should be changed to something like std
            % dev
            nooutx = all_xmotion{i,r}(all_xmotion{i,r}<outlier_cutoff & all_ymotion{i,r}<outlier_cutoff & ...
                     all_xmotion{i,r}>-outlier_cutoff & all_ymotion{i,r}>-outlier_cutoff);
            noouty = all_ymotion{i,r}(all_xmotion{i,r}<outlier_cutoff & all_ymotion{i,r}<outlier_cutoff & ...
                     all_xmotion{i,r}>-outlier_cutoff & all_ymotion{i,r}>-outlier_cutoff);

        % For displaying the row offsets
%             figure(1);
%             plot(all_xmotion{i,r},all_ymotion{i,r},'.');hold on;
%             plot(median(nooutx),median(noouty),'kx');
%             plot(0,0,'r.'); hold off;
%             axis square; axis([-outlier_cutoff outlier_cutoff -outlier_cutoff outlier_cutoff]); 
%             title([ 'median x: ' num2str(median(all_xmotion{i,r})) ' median y: ' num2str(median(all_ymotion{i,r})) ]);
%              drawnow;
%              f= getframe;
%              writeVideo(v,f);

            estimatevar(i,r) = sqrt(var(all_xmotion{i,r}).^2 + var(all_ymotion{i,r}).^2);
            xmotion_norm{i,r} = all_xmotion{i,r}-median(all_xmotion{i,r});
            ymotion_norm{i,r} = all_ymotion{i,r}-median(all_ymotion{i,r});

            xmotion_vect(i,r) = median(all_xmotion{i,r});
            ymotion_vect(i,r) = median(all_ymotion{i,r});

            if isnan(xmotion_vect(i,r))
                xmotion_vect(i,r) = 0;
               disp('NAN!');
            end
            if isnan(ymotion_vect(i,r))
                ymotion_vect(i,r) = 0;
               disp('NAN!');
            end

        end
    end

    for i=1:size(xmotion_vect,1)

        xgriddistortion(i,:) = repmat(median(xmotion_vect(i,:)), [1 size(imStk{1},2)] ); %The ref should be all 0s
%         if i >= crop_ROI(1) && i<= crop_ROI(2) % Only apply the grid correction within the roi we've cropped to.           
            ygriddistortion(i,:) = repmat(median(ymotion_vect(i,:))+static_grid_distortion(i+crop_ROI(1)-1), [1 size(imStk{1},2)] );
%         else
%             ygriddistortion(i,:) = repmat(median(ymotion_vect(i,:)), [1 size(imStk{1},2)] );
%         end
    end
    
    disp_field = cat(3,xgriddistortion,ygriddistortion);

    warpedStk = uint8(zeros(size(disp_field,1),size(disp_field,2),length(imStk)));
    
    for i=1:length(imStk)
        warpedStk(:,:,i) = uint8(imwarp(imStk{i},disp_field,'FillValues',0) );
    end


    if ~exist( fullfile(motion_path, 'Redewarped'), 'dir' )
        mkdir(fullfile(motion_path, 'Redewarped'))
    end
    
    warpedIm = mean(warpedStk,3);
    
    
    % Crop out any border regions
    imregions= bwconncomp(warpedIm>0);
    cropbox = regionprops(imregions,'Area','BoundingBox');
    [maxarea, maxind] = max([cropbox.Area]); % Take the bigger of the two areas
    cropbox = cropbox(maxind).BoundingBox;
    
    warpedStk = warpedStk( round(cropbox(2)):round(cropbox(4)), round(cropbox(1)):round(cropbox(3)), : );
    
    
    fName(1:end-4)
    
    if length(imStk)== 1
        imwrite(warpedStk, fullfile(motion_path,'Redewarped', [fName(1:end-4) '_redewarped.tif']));
    else
        vidobj = VideoWriter( fullfile(motion_path,'Redewarped', [fName(1:end-4) '_redewarped.avi']), 'Grayscale AVI' );

        open(vidobj);
        writeVideo(vidobj,warpedStk);
        close(vidobj);
    end
    
end
