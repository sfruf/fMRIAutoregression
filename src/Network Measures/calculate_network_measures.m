%calculate basic matrix properties

%load matrices

load('../Data/Glaser_Mats.mat')

mats=mats_mean_std_1;
nROIs=size(mats,1);
pos=zeros(nROIs,nsubjs);
neg=zeros(nROIs,nsubjs);
cc=zeros(nROIs,nsubjs);

for i=1:nsubjs
%calculate strength
[pos(:,i),neg(:,i)]=strengths_und_sign(mats(:,:,i));

%calculate clustering coefficient
cc(:,i)=clustering_coef_wu_sign(mats(:,:,i),3); %3 corresponds to the Constantini & Perugini cc variant

end

save('../Data/Network_Measures.mat','pos','neg','cc')