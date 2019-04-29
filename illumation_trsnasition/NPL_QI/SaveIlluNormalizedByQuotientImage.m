function IlluminationNormalizedIm=SaveIlluNormalizedByQuotientImage(Im)
%%
% Function: Normalize the illumiantion of image by the quotien image  
% Author: XieXiaohua
% Date: 2007.7.22

%%

load EigenSubSpace;
load Standardcoordinate;
m=size(Standardcoordinate,1);
QuotientIm=QuotientImage(Im);
%Restructure
IlluminationNormalizedIm=(EigenSubSpace*Standardcoordinate+MeanX).*QuotientIm;