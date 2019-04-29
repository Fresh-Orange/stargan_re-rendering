function [M] = Read_Data_or_Image_Into_Matrix(PersonList,OrderList,FileList)
%-------------------------------------------------------------------------
% Author : Xie Xiaohua
% Date : 2006.8.3
% Email : 99679462@sina.com
% SUN YAT-SEN UNIVERSITY
%-------------------------------------------------------------------------
% Read the image (.jpg,.bmp) or *.mat data from database and store in M
% Please note that should not put the *.bmp, *.jpg, and *.mat files at the
% same filepath.
%-------------------------------------------------------------------------
% PersonList --- the Selected Person List to train in the database
%             for example,PersonList=[1 3 5] indicates the 1st 3rd 5th 
%             in the database are selected to be trained 
% OrderList --- the orders of the image for one person to select
%           for example OrderList = [1 3 5 6 9]  , if indicate that               
%           read the ist,3rd,....,9th image each person
% FileList --- the location of Database
%-------------------------------------------------------------------------
% M ---  the data of samples,it acts as M = [M1,M2,...........]
%      where each column is a Matrix Mi which indicate ith person and 
%      consist of [MiV1,MiV2,.....] where each vector store the image of one person.  
%      
%-------------------------------------------------------------------------

NumOfPerson=length(PersonList);
ImageAmountPerPerson=length(OrderList);

imagefile=dir([FileList '*.jpg']);
numpjg=length(imagefile);
imagefile=dir([FileList '*.bmp']);
numbmp=length(imagefile);
imagefile=dir([FileList '*.mat']);
nummat=length(imagefile);

if numpjg>0
    imagetype='.jpg';
    imagefile=dir([FileList  '*.jpg']);
    disp('will read *.jpg file');
elseif numbmp>0
    imagetype='.bmp';
    imagefile=dir([FileList  '*.bmp']);
    disp('will read *.bmp file');
elseif nummat>0
    imagetype='.mat';
    imagefile=dir([FileList  '*.mat']);
    disp('will read *.mat file');
else
    disp('no *.jpg, *.bmp,or *.mat file in the path !!!!');
    M=[];
    return;
end


length_filename=length(imagefile(1).name)-4;
clear imagefile;

ColCount = 1;
M=[];
for j=1:NumOfPerson    
    id = PersonList(j);
   
    if length_filename==4       
        if id<10
           s=sprintf('0%d',id);
        else
           s=sprintf('%d',id);
        end
    elseif length_filename==6        
        if id<10
           s=sprintf('00%d',id);
        elseif id<100
           s=sprintf('0%d',id);
        else
           s=sprintf('%d',id);
        end
    end
        
    for i=1:ImageAmountPerPerson
        selectImageID = OrderList(i);
        
        if length_filename==4
            if selectImageID<10 
                imagename = sprintf('0%d',selectImageID);
            else
                imagename = sprintf('%d',selectImageID);
            end 
        elseif length_filename==6
            if selectImageID<10 
                imagename = sprintf('00%d',selectImageID);
            elseif selectImageID<100 
                imagename = sprintf('0%d',selectImageID);
            else
                imagename = sprintf('%d',selectImageID);
            end         
        end
        imagepathname = strcat(FileList,s,imagename,imagetype);
        if strcmp(imagetype,'.mat')
            load(imagepathname);         
            M(:,ColCount)=MAT;
        else
            X=imread(imagepathname);        
            reshape_X=double(X(:));
            M(:,ColCount)=reshape_X;
        end
        ColCount = ColCount + 1;
    end
end
