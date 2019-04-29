function Output=ThresholdOperation(Input,KeptRate)
%% processe data by threshold operation according to the histogram
% XieXiaohua 
%%
% Input=round(Input);
if KeptRate==1
    Output=Input;
    return;
end
minval=min(min(Input));
maxval=max(max(Input));
% L=maxval-minval+1;%bin number of histogram
if (maxval-minval)>1000
L=10000;
else
L=1000;
end
[Heigh Width]=size(Input);
SumPoint=Heigh*Width;
IncludedPoint=round(SumPoint*KeptRate);
H=hist(Input,L);
% subplot(1,2,1);
% hist(Input,L)
[HighVal HighPoint]=max(H);

tempSum=HighVal;
left=HighPoint;
right=HighPoint;
while tempSum<IncludedPoint
    
    if left>1
        left=left-1;
        tempSum=tempSum+H(left);
    end
    
    if right<L
        right=right+1;
        tempSum=tempSum+H(right);
    end
    
    if left==1&&right==L
        return;
    end
end

minthreshold=minval+(left-1)*(maxval-minval)/L;
maxthreshold=minval+right*(maxval-minval)/L;

Input(find(Input<minthreshold))=minthreshold;
Input(find(Input>maxthreshold))=maxthreshold;

Output=Input;
clear Input;
% subplot(1,2,2);
% hist(Output)
        
        

