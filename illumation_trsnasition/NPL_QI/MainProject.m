function MainProject

TrainPerson=1:10;
TrainIm=1:64;
ObjectIllu=1;
TrainImPath='C:\home\sysuxiexh\Database\YaleB\';
ImPath='C:\home\sysuxiexh\Database\FRGC\FRGC_renamed_UA\';
dim=20;

PersonList=[1:275];
OrderList=[1:5];
QISavePath='C:\home\sysuxiexh\Experimental_Results\NPL-QI\QI_for_Recognition\QI_FRGC_UA\';
% NorSavePath='C:\home\sysuxiexh\Experimental_Results\NPL-QI\NPL-QI_YaleB\';
ImWidth=100;
ImHeight=100;

TrainSubspace(TrainPerson,TrainIm,TrainImPath,dim);
% 
% TrainStandardcoordinate(TrainPerson,ObjectIllu,TrainImPath);
%% Calculate QI
% CalculateQuotientImAsABath(PersonList,OrderList,ImPath,QISavePath,ImWidth,ImHeight);
CalculateQuotientImAsABath_FRGC(PersonList,OrderList,ImPath,QISavePath,ImWidth,ImHeight);

%% Illumination Normalized
% IlluNormalizeAsABath(PersonList,OrderList,ImPath,NorSavePath,ImWidth,ImHeight);

% t=cputime;
% IlluNormalizeAsABath_FRGC(PersonList,OrderList,ImPath,NorSavePath,ImWidth,ImHeight);
% t1=cputime-t;
