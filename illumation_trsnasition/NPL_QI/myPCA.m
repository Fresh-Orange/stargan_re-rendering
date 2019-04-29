function [matFeatureVector meanX rvFeatureValue rate]=myPCA(matSample, NewDimensions)
% PCA.m
%
% Description:
%       �˺������ڶ�һ�������������� PCA ѵ��, ��ȡ����Ҫ����, �����ԽǾ���������� �Ʀ�i >= alpha.
% Input:
%       matSample  --  matrix 2D, ������������, ���е�ÿһ�м�Ϊһ������������
%       alpha      --  double, ��Ҫ������������������ٷֱ�, ȡֵ���� [0, 1]//��ʱû���������
%       NewDimensions --- indicate how mamy dimensionlities should be remain after 
%                   reducing dimensions .
%
% Return:
%       matFeatureVector  --  ��ȡ��������������ɵľ���, ÿһ��Ϊһ����������, ���Ѱ�����ֵ���ɴ�С����
%       rvFeatureValue    --  ��Ӧ������ֵ, ���ɴ�С����
%       nFeatureCount     --  ��ȡ���������������ܸ���//��ʱû���������
%
% History:
%          2004-3-31 19:24 written by qhn in ZSU.
%          xiexiaohua2006.8.3�����޸�
% About Me:
%       "Huining Qiu"<msctor@hotmail.com>
%%��������������������������������������������������������������������������������
X = matSample;

% ����ͼ��������ֵ
meanX = mean(X, 2); % ����ÿһ���۲��ľ�ֵ, Ҳ���Ƕ�ÿһ�����һ����ֵ
[d, M] = size(X);

% ������������Ҫ��Э������� St = cov(X) = 1/M * X * X' ����������
% Ϊ����� St ������ֵ����������,���뻻һ�ַ�ʽ, ��Ϊ�� R = M*St' = X' * X ������ֵ����������, 
% ��ʱ R �� (M x d) x (d x M) = M^2 ά�ľ���, �� St ��ά��С����, ����������ֵ�����������ǿ��е�.
% ��ϸ���Ƶ���ο��������:
%  X = X - meanX * ones(1, M);
%  R = X' * X ./ M;
%  refactor: ��ʵ��, ���� St = 1/M * (X * X') �� R = (X * X') ������ֵ��������������ȫһ����, ����Ȼ���ߵļ������,����֮.
X = X - meanX * ones(1, M); % ͬ��, ���Ƴ�ÿһ�еľ�ֵ, �����Ļ�
R = X' * X;
% ������ R ������ֵ����������
[V, D] = eig(R);
% ��Ϊ R * v = X'* X * v = d .* v �����Ƴ� X * (X' * X * v) = X * (d .* v), 
% �༴ St * (X * v) = d .* (X * v)
% �ɼ� d Ҳ�� St ������ֵ, ��Ӧ����������Ϊ X * v
% ����:
% ���� SVD ���������, ��֪ԭ�� St ������ֵ����������Ϊ
%       ui = (1 / sqrt(d)) .* X .* v

% ȡ�Խ�Ԫ�ؼ�Ϊ��������ֵ
cvEigenvalue = diag(D);
% ������ֵ��С��������
[cvEigenvalueSorted, cvEigenvalueSortedIndex] = sort(cvEigenvalue); % MATLAB: sort() ����ֻ�ܴ�С��������
iMax = length(cvEigenvalueSorted);
%###########################���Ҫ�������������ʣ�������ʱ����############################
% �Ӵ�С�ۼ�����ֵ, ������ "����"
% sumTotal = sum(cvEigenvalueSorted);
% sumEnergy = 0;
% for i = iMax : -1 : 1
%    sumEnergy = sumEnergy + cvEigenvalueSorted(i);
%    rate = (sumEnergy / sumTotal);
%    if rate >= alpha
%         break;
%    end
% end
%iMin = i;
%nEigenFaceCount = (iMax - iMin + 1);
%#########################################################################
%���㱣������������������rate >= alpha����ȡ���˶�
sumTotal = sum(cvEigenvalueSorted);
beginpoint=iMax-NewDimensions+1;
disp(beginpoint)
disp(iMax)
sumEnergy =sum(cvEigenvalueSorted(beginpoint:iMax));
rate =sumEnergy /sumTotal ;
%-------------------

nEigenFaceCount=NewDimensions;
iMin=iMax-nEigenFaceCount+1;
Wpca = zeros(d, nEigenFaceCount); 
iW = 0;
for i = iMax : -1 : iMin
    idx = cvEigenvalueSortedIndex(i);
    iW = iW + 1;
    Wpca(:, iW) = X * V(:, idx);
end

% Normalize columns
Wpca = Wpca ./ (ones(size(Wpca, 1), 1) * sqrt(sum(Wpca .^ 2, 1))); % ��ʵ���ǰ�ͶӰ�����ÿһ�ж����Ը��е� 2-norm(Ҳ����ͨ��������Ԫ�ص�ƽ������ٿ�����)
% �����ֵ
matFeatureVector = Wpca;
rvFeatureValue   = cvEigenvalueSorted(iMax : -1 : iMin);
%nFeatureCount    = nEigenFaceCount;

% -------------------------------------------------------------------------
% ���������ʱ����
clear X;
clear cvEigenvalueSortedIndex;
clear nEigenFaceCount;
clear sumTotal;
clear d   M  R  V  D;
clear iMin  iMax;
