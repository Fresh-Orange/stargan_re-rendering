function LTV_Illumination_Transition_Image
%% ITI on S&L framework
% Reference: Jianyi Liu et al. Illumination Transition Image:Parameter-based Illumination Estimation and Re-rendering. ICPR2008.
%%
% Author: Xiaohua Xie
% Sun Yat-sen University
% 2010-08-03
%-----------------------------------
Train_u_Path='C:\home\sysuxiexh\Experimental_Results\LNSCT\LNSCT_YaleB\u\';
Test_u_Path='C:\home\sysuxiexh\Experimental_Results\LNSCT\LNSCT_YaleB\u\';
Test_v_Path='C:\home\sysuxiexh\Experimental_Results\LNSCT\LNSCT_YaleB\v\';
SavePath='C:\home\sysuxiexh\Experimental_Results\Result_ITI\LNSCT_ITI_FRGC_CA\';
Width=100;
Height=100;
lamda=1;


TrainingImageList=1:64;
ObjectIllumination=1;

% for TestPerson=1:38
%     TrainingPersonList=[1:TestPerson-1  TestPerson+1:38];      
    TrainingPersonList=1:38;
    NumTrainPerson=size(TrainingPersonList,2);
   % Training
   n=0;
   A_t_A=0;
   A_t=0;
   MCell=[];
  for p=TrainingPersonList
       n=n+1;
       M=Read_Data_or_Image_Into_Matrix(p,TrainingImageList,Train_u_Path);
       A_t_A=A_t_A+M'*M;
       A_t=A_t+M';       
       MCell{n}=M;
  end
%   save Train_ITI A_t_A A_t MCell;  
%   load Train_ITI;
  
  % Compute the object illumination
  temp=Read_Data_or_Image_Into_Matrix(TrainingPersonList,ObjectIllumination,Train_u_Path);
  z=inv(A_t_A)*A_t*temp;
  z=mean(z,2);
 for TestPerson=1:275
    for TestImage=1:5
         if TestPerson<10
            sPerson=sprintf('00%d',TestPerson);
         elseif TestPerson<100
             sPerson=sprintf('0%d',TestPerson);
        else
            sPerson=sprintf('%d',TestPerson);
        end
        if TestImage<10
            sImage=sprintf('00%d',TestImage);
        elseif TestImage<100
            sImage=sprintf('0%d',TestImage);
        else
            sImage=sprintf('%d',TestImage);
        end        

      %% Test
%       TestM=ReadImageIntoMatrix(TestPerson,TestImage,ImPath);    
      TestM=Read_Data_or_Image_Into_Matrix(TestPerson,TestImage,Test_u_Path);

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
      Rerendu=log2((TestM+lamda)./(rec+lamda)+1).*obj;
%      imwrite(uint8(reshape(rec,100,100)),'C:\home\sysuxiexh\SP_NPL_QI\test_ITI.bmp');%test
      Rerendu=real(Rerendu);
      
      Rerendv=Read_Data_or_Image_Into_Matrix(TestPerson,TestImage,Test_v_Path);
      RerendI=Rerendu+Rerendv;
      RerendI=exp(RerendI);
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


