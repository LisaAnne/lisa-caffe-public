function create_flow_images_LRCN(base, save_base)
%create_flow_images will compute flow images from RGB images using [1].
%input: 
%     base: folder in which RGB frames from videos are stored
%     save_base: folder in which flow images should be saved
%
%[1] Brox, Thomas, et al. "High accuracy optical flow estimation based on a theory for warping." Computer Vision-ECCV 2004. Springer Berlin Heidelberg, 2004. 25-36.

  list = clean_dir(base);
  
  if ~isdir(save_base)
    mkdir(save_base)
  end
  
  for i = 1:length(list)
    if mod(i,100) == 0
      fprintf('On item %d of %d\n', i, length(list))
    end
  
    video = list{i};
  
  
    frames = clean_dir(sprintf('%s/%s',base,video));
  
    if length(frames) > 1
      if ~isdir(sprintf('%s/%s',save_base, video))
        mkdir(sprintf('%s/%s',save_base, video))
      end
      im1 = imread(sprintf('%s/%s/%s',base,video,frames{1}));
      for k = 2:length(frames)
        im2 = imread(sprintf('%s/%s/%s',base,video,frames{k}));
        flow = mex_OF(double(im1),double(im2));
      
        scale = 16;
        mag = sqrt(flow(:,:,1).^2+flow(:,:,2).^2)*scale+128;
        mag = min(mag, 255); 
        flow = flow*scale+128;
        flow = min(flow,255);
        flow = max(flow,0);
      
        [x,y,z] = size(flow);
        flow_image = zeros(x,y,3);
        flow_image(:,:,1:2) = flow;
        flow_image(:,:,3) = mag;
      
        imwrite(flow_image./255,sprintf('%s/%s/flow_image_%s',save_base,video,frames{k}))
      
        im1 = im2;
      end
    end
  end

function files = clean_dir(base)
  %clean_dir just runs dir and eliminates files in a foldr
  files = dir(base);
  files_tmp = {};
  for i = 1:length(files)
    if strncmpi(files(i).name, '.',1) == 0
      files_tmp{length(files_tmp)+1} = files(i).name;
    end
  end
  files = files_tmp; 
 

