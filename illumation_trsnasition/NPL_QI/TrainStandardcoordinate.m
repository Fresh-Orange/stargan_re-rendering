function TrainStandardcoordinate(PersonList,OrderList,FileList)
%%
% refer: 刘俊，黄向生，王阳生，《基于图像子空间的改进商图像方法》，计算机科学，2005，32（8）
% Function: calculate the Standardcoordinate of normalized image on subspace  
% Author: XieXiaohua
% Date: 2007.7.23
%
% Input: 
%        PersonList --- the Selected Person List to train in the database
%             for example,PersonList=[1 3 5] indicates the 1st 3rd 5th 
%             in the database are selected to be trained 
%       OrderList --- the orders of the image for one person to select
%           for example OrderList = [1 3 5 6 9]  , if indicate that               
%           read the ist,3rd,....,9th image each person
%       FileList --- the location of the Database
% Output:
%       Save the Standardcoordinate into disk(vector,n×1）
%%
%%
ImToTrain=ReadImageIntoMatrix(PersonList,OrderList,FileList);
ImToTrain=mean(ImToTrain,2);
load EigenSubSpace;
Standardcoordinate=EigenSubSpace'*(ImToTrain-MeanX);
save Standardcoordinate Standardcoordinate;