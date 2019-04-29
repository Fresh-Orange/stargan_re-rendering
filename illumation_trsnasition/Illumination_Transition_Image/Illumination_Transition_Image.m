function Illumination_Transition_Image
%%
% Reference: Jianyi Liu et al. Illumination Transition Image:Parameter-based Illumination Estimation and Re-rendering. ICPR2008.
%%
% Author: Xiaohua Xie
% Sun Yat-sen University
% 2009-12-12
%-----------------------------------
TrainImPath='/media/data2/laixc/AI_DATA/type_changed/';
ImPath='/media/data2/laixc/AI_DATA/multi_pie_id_crop_test_query_tradition/';
SavePath='/media/data2/laixc/AI_DATA/ITI_output/';
Width=252;
Height=258;
lamda=1;


TrainingImageList=[0,1,4,7,10,13];
ObjectIllumination=7;

% for TestPerson=1:38
%     TrainingPersonList=[1:TestPerson-1  TestPerson+1:38];      
    TrainingPersonList=1:211;
    NumTrainPerson=size(TrainingPersonList,2);
   % Training
   n=0;
   A_t_A=0;
   A_t=0;
   MCell=[];
  for p=TrainingPersonList
       n=n+1;
       M=Read_Data_or_Image_Into_Matrix(p,TrainingImageList,TrainImPath);
       A_t_A=A_t_A+M'*M;
       A_t=A_t+M';       
       MCell{n}=M;
  end
%   save Train_ITI A_t_A A_t MCell;  
%   load Train_ITI;
  
  % Compute the object illumination
  temp=Read_Data_or_Image_Into_Matrix(TrainingPersonList,ObjectIllumination,TrainImPath);
  z=inv(A_t_A)*A_t*temp;
  z=mean(z,2);
 for TestPerson=[315:330, 332:339, 341:346]
    for TestImage=[0,1,4,10,13]
         if TestPerson<10
            sPerson=sprintf('0%d',TestPerson);
%          elseif TestPerson<100
%              sPerson=sprintf('0%d',TestPerson);
        else
            sPerson=sprintf('%d',TestPerson);
        end
        if TestImage<10
            sImage=sprintf('0%d',TestImage);
%         elseif TestImage<100
%             sImage=sprintf('0%d',TestImage);
        else
            sImage=sprintf('%d',TestImage);
        end        

      %% Test
%       TestM=ReadImageIntoMatrix(TestPerson,TestImage,ImPath);    
      TestM=Read_Data_or_Image_Into_Matrix(TestPerson,TestImage,ImPath);

      % Estimate illumination
      x=inv(A_t_A)*A_t*TestM;
     
      % Compute the wighting
       B=[];
         for n=1:NumTrainPerson
               M=MCell{n};
               temp=[M*x;1];
               B=[B temp];
         end
         
         k=inv(B'*B)*B'*[TestM;1];
         k=k/sum(k);
      % relighting
      rec=0; 
      obj=0;
      for n=1:NumTrainPerson
               M=MCell{n};
               rec=rec+k(n)*M*x;
               obj=obj+k(n)*M*z;
      end
      RerendI=log2((TestM+lamda)./(rec+lamda)+1).*obj;
%      imwrite(uint8(reshape(rec,100,100)),'C:\home\sysuxiexh\SP_NPL_QI\test_ITI.bmp');%test
      RerendI=real(RerendI);
      RerendI(find(RerendI<0))=0;
      RerendI(find(RerendI>255))=255;

% Show Images

%  subplot(1,3,1);
%  imshow(mat2gray(reshape(TestM,Height,Width)));
% 
% subplot(1,3,2);
% imshow(uint8(reshape(RerendI,Height,Width)));
% 
% RealNorM=ReadImageIntoMatrix(TestPerson,ObjectIllumination,ImPath);    
% subplot(1,3,3);
% imshow(mat2gray(reshape(RealNorM,Height,Width)));

% Save Images
ImSavePage=[SavePath sPerson sImage '.bmp'];
imwrite(uint8(reshape(RerendI,Height,Width)), ImSavePage);

    end
end


