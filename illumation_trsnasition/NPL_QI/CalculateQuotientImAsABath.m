function CalculateQuotientImAsABath(PersonList,OrderList,FileList,SavePath,ImWidth,ImHeight)
%%
% function: calculate the quotient images as a bath
% Author: XieXiaohua
% Date: 2007.7.9
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
%      the quotients will be writtern into the hardisk 
%%
[temp,NumOfPerson]=size(PersonList);
[temp,ImageAmountPerPerson]=size(OrderList);
clear temp;

NumOfIm=0;
for ithPerson=PersonList
    for ithImage=OrderList
        NumOfIm=NumOfIm+1;
        ImToCalculate=ReadImageIntoMatrix(ithPerson,ithImage,FileList);
        % calculate the quotient image
        QuotientIm=QuotientImage(ImToCalculate);
        %normalize
%         QuotientIm(find(QuotientIm<-500))=-500;
%         QuotientIm(find(QuotientIm>500))=500;
%         QuotientIm=ThresholdOperation(QuotientIm,0.98);  
%           QuotientIm=round(255*(mat2gray(QuotientIm)));
          
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
        saveName=strcat(SavePath,sPerson,sImage,'.mat');
        MAT=QuotientIm;
        save(saveName,'MAT');
        % save image   
%          QuotientIm=reshape(QuotientIm,ImHeight,ImWidth);
%         saveName=strcat(SavePath,sPerson,sImage,'.bmp');
%         imwrite(mat2gray(QuotientIm),saveName);
    end
end