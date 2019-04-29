function IlluNormalizeAsABath(PersonList,OrderList,FileList,SavePath,ImWidth,ImHeight)
%%
% function: normalize the illumiantion of images as a bath
% Author: XieXiaohua
% Date: 2007.7.23
%
% Input: 
%        PersonList --- the Selected Person List to calculate in the database
%             for example,PersonList=[1 3 5] indicates the 1st 3rd 5th 
%             in the database are selected to be trained 
%       OrderList --- the orders of the image for one person to select
%           for example OrderList = [1 3 5 6 9]  , if indicate that               
%           read the ist,3rd,....,9th image each person
%       FileList --- the location of the Database
% Output:
%      the restructured images will be writtern into the hardisk
%%

[temp,NumOfPerson]=size(PersonList);
[temp,ImageAmountPerPerson]=size(OrderList);
clear temp;

NumOfIm=0;
for ithPerson=PersonList
    for ithImage=OrderList
        ImToCalculate=ReadImageIntoMatrix(ithPerson,ithImage,FileList);

        IlluminationNormalizedIm=IlluNormalizedByQuotientImage(ImToCalculate);
 
% IlluminationNormalizedIm=ThresholdOperation(IlluminationNormalizedIm,0.995);
        if ithPerson<10
            sPerson=sprintf('0%d',ithPerson);
        else
            sPerson=sprintf('%d',ithPerson);
        end
        if ithImage<10
            sImage=sprintf('0%d',ithImage);
        else
            sImage=sprintf('%d',ithImage);
        end
        %save data
       MAT=IlluminationNormalizedIm;
       saveName=strcat(SavePath,sPerson,sImage,'.mat');
%        save(saveName,'MAT');
       %save image
       disp('IlluminationNormalizedIm')
       size(IlluminationNormalizedIm)
        IlluminationNormalizedIm=reshape(IlluminationNormalizedIm,ImHeight,ImWidth);
        IlluminationNormalizedIm(find(IlluminationNormalizedIm<0))=0;
        IlluminationNormalizedIm(find(IlluminationNormalizedIm>255))=255;
        saveName=strcat(SavePath,'bmp\',sPerson,sImage,'.bmp');
        imwrite(uint8(IlluminationNormalizedIm),saveName);
    end
end