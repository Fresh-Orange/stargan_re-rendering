function TrainSubspace(PersonList,OrderList,FileList,Dimensions)
%%
% Function: set the Subspace by main eigenvectors
% Author: XieXiaohua
% Date: 2007.7.8
%
% Input: 
%        PersonList --- the Selected Person List to train in the database
%             for example,PersonList=[1 3 5] indicates the 1st 3rd 5th 
%             in the database are selected to be trained 
%       OrderList --- the orders of the image for one person to select
%           for example OrderList = [1 3 5 6 9]  , if indicate that               
%           read the ist,3rd,....,9th image each person
%       FileList --- the location of the Database
%      Dimensions ---- the number of selected eigenvectors.
% Output:
%       EigenSubSpace --- the main eigenvectors, which will be writtern to the harddisk
%%
NumPerson=size(PersonList,2);
ImToTrain=0;
for i=PersonList
    Tem=ReadImageIntoMatrix(i,OrderList,FileList);
    ImToTrain=ImToTrain+Tem;
end
ImToTrain=ImToTrain/NumPerson;
[EigenSubSpace MeanX]=myPCA(ImToTrain,Dimensions);
% save the data
save EigenSubSpace EigenSubSpace MeanX;
