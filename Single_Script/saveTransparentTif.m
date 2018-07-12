function saveTransparentTif(imageToSave, saveFileName)
% saveTransparentTif(imageToSave,saveFileName)
%
% imageToSave - a 8-bit or RGB image that we wish to add transparency to.
%
% saveFileName - the full path of the file that needs to be saved with
% transparency.
%
% Method taken from Stackoverlow: https://stackoverflow.com/questions/13660108/matlab-how-to-save-tiff-with-transparency-or-png-with-no-compression?rq=1
% to create a function that outputs outwardly transparent tif images.
% Robert F Cooper created July 11,2018


%make file
tob = Tiff(saveFileName,'w');

%# you need to set Photometric before Compression
tob.setTag('Photometric',Tiff.Photometric.MinIsBlack)
tob.setTag('Compression',Tiff.Compression.LZW)

%# tell the program that channel 4 is alpha
tob.setTag('ExtraSamples',Tiff.ExtraSamples.AssociatedAlpha)

%# set additional tags (you may want to use the structure
%# version of this for convenience)
tob.setTag('ImageLength',size(imageToSave,1));
tob.setTag('ImageWidth',size(imageToSave,2));
tob.setTag('BitsPerSample',8);
tob.setTag('RowsPerStrip',16);
tob.setTag('PlanarConfiguration',Tiff.PlanarConfiguration.Separate);
tob.setTag('Software','MATLAB')

if size(imageToSave,3) == 1

    imageToSave = repmat(imageToSave(:,:,1),[1,1,2]);
    imageToSave(:,:,2) = ~isnan(imageToSave(:,:,1))*255;
    imageToSave = uint8(imageToSave);
    tob.setTag('SamplesPerPixel',2);
elseif size(imageToSave,3) == 3  
    imageToSave(:,:,4) = uint8(~isnan(imageToSave(:,:,1))*255);
    tob.setTag('SamplesPerPixel',4);
end

if(isa(imageToSave,'double') || isa(imageToSave,'single'))
    imageToSave = uint8(round(imageToSave*255));
end

%# write and close the file
tob.write(imageToSave)
tob.close
end