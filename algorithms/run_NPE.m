function enhanced = run_NPE(img_low)
%RUN_NPE  Enhance a low-light image using the NPE algorithm.
%
%  Calls NPEA() from the original NPE compiled .p files (Wang et al., 2013),
%  obtained from baidut/BIMEF (github.com/baidut/BIMEF, lowlight/NPE/).
%
%  Reference:
%    Wang S., Zheng J., Hu H.-M., Li B. (2013).
%    "Naturalness Preserved Enhancement Algorithm for Non-Uniform Illumination Images."
%    IEEE Transactions on Image Processing, 22(9): 3538-3548.
%    DOI: 10.1109/TIP.2013.2261309
%
%  Input:  img_low  - Low-light color image (H x W x 3, uint8)
%  Output: enhanced - Enhanced image (H x W x 3, uint8)

    % Convert to double [0,1] — NPEA expects this range
    img_d = double(img_low) / 255;

    % Call the original NPE algorithm
    result = NPEA(img_d);

    % Convert output back to uint8
    if max(result(:)) <= 1.0
        enhanced = uint8(min(max(result * 255, 0), 255));
    else
        enhanced = uint8(min(max(result, 0), 255));
    end

end