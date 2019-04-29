function [M] = ReadImageIntoMatrix(PersonList,OrderList,FileList)
%-------------------------------------------------------------------------
% Author : Xie Xiaohua
% Date : 2006.8.3
% Email : 99679462@sina.com
% SUN YAT-SEN UNIVERSITY
%-------------------------------------------------------------------------
% Read the image from ORL database and store in M
%
%-------------------------------------------------------------------------
% PersonList --- the Selected Person List to train in the database
%             for example,PersonList=[1 3 5] indicates the 1st 3rd 5th 
%             in the database are selected to be trained 
% OrderList --- the orders of the image for one person to select
%           for example OrderList = [1 3 5 6 9]  , if indicate that               
%           read the ist,3rd,....,9th image each person
% FileList --- the location of the ORL Database
%-------------------------------------------------------------------------
% M ---  the data of samples,it acts as M = [M1,M2,...........]
%      where each column is a Matrix Mi which indicate ith person and 
%      consist of [MiV1,MiV2,.....] where each vector store the image of one person.  
%      
%-------------------------------------------------------------------------
[temp,NumOfPersonTrained] = size(PersonList);
[temp,ImageAmountPerPerson] = size(OrderList); 
clear temp;
Width = 252;
Height = 258;
%initial M
M = zeros(Width*Height,NumOfPersonTrained*ImageAmountPerPerson);
%begin to read the image
ColCount = 1;
for j=1:NumOfPersonTrained    
    id = PersonList(j);
    if id<=9
        s=sprintf('0%d',id);
    else
        s=sprintf('%d',id);
    end    
    databasepathname = strcat(FileList,s);    
    for i=1:ImageAmountPerPerson
        selectImageID = OrderList(i);
        %  for different person has different feature ,one person has its featrue  
        % see the ORL database structure
        if selectImageID<=9 
            imagename = sprintf('0%d.png',selectImageID);
        else
            imagename = sprintf('%d.png',selectImageID);
        end 
        imagepathname = strcat(databasepathname,imagename);        
        [X,map]=imread(imagepathname); 
        size(X)
        reshape_X=double(reshape(X,Width*Height,1));
        M(:,ColCount)=reshape_X;
        ColCount = ColCount + 1;
    end
    clear X;clear map;clear reshape_X;
end
