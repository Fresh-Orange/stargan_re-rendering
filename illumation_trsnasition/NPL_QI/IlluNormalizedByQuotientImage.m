function IlluminationNormalizedIm=IlluNormalizedByQuotientImage(Im)
%%
% Function: Normalize the illumiantion of image by the quotien image  
% Author: XieXiaohua
% Date: 2007.7.22

%%

load EigenSubSpace;
load Standardcoordinate;
%EigenSubSpace = EigenSubSpace(:,2);
EigenSubSpace(1:10,:)
%Standardcoordinate = Standardcoordinate(2, :);
Standardcoordinate
disp('size Eigen')
size(EigenSubSpace)
disp('size Standardcoordinate')
size(Standardcoordinate)
m=size(Standardcoordinate,1);
QuotientIm=QuotientImage(Im);
%Restructure
%IlluminationNormalizedIm=(MeanX).*QuotientIm;
IlluminationNormalizedIm=(EigenSubSpace*Standardcoordinate+MeanX).*QuotientIm;