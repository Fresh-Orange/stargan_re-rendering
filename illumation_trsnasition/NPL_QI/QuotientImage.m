function QuotientIm=QuotientImage(Im)
%%
% NPL-QI
% Function: calculate the quotien image  
% Author: XieXiaohua
% Date: 2007.7.9
%%
% load the subspace
disp('im')
size(Im)

load EigenSubSpace;
size(EigenSubSpace)
%Im = Im(1:10000);
disp('im')
size(Im)
disp('MeanX')
size(MeanX)
coordinate=EigenSubSpace'*(Im-MeanX);
size(coordinate)
tmp = EigenSubSpace*coordinate;
size(tmp)
illu=(EigenSubSpace*coordinate+MeanX);

% imwrite(uint8(reshape(illu,100,100)),'C:\home\sysuxiexh\SP_NPL_QI\test_NPL-QI.bmp');%test



QuotientIm=Im./illu;

% illu(find(illu<1))=1; %test
% QuotientIm=log(Im./illu+1);



