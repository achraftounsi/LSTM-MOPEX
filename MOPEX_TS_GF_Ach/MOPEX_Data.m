clear
clc
%% Directories
MOPEX_input_files = 'C:\Users\I-SMART_Lab_03\OneDrive - stevens.edu\Desktop\MOPEX\MOPEX_TS_Ach';
MOPEX_output_files = 'C:\Users\I-SMART_Lab_03\OneDrive - stevens.edu\Desktop\MOPEX\MOPEX_TS_GF_Ach';
cd(MOPEX_output_files);
%% Basic Files and Variables
GF = readtable('MOPEX_GF.xlsx');
M = table2array(GF);
n = length (M);
i=1;
k=1;
ext ='.csv';

%% Play with data
for i =1 : n  %% loop over the MOPEX ID list: call Ankit

    ii = M(i,1); %% usgs staion index
    % MOPEX stations path
    cd(MOPEX_input_files);
    %% play with lables
    st = num2str(ii);    
    s2 = '.csv';
    str = append (st,s2);
    T = readtable(str);
    TT = table2timetable(T);
    H = height (T);
    M_GF = [];
    for k=1:H
        if month(TT.datetime(k))== 1
            M_GF(k,1) = M (i,2);
        elseif month(TT.datetime(k))== 2
            M_GF(k,1) = M (i,3);
        elseif month(TT.datetime(k))== 3
            M_GF(k,1) = M (i,4);
        elseif month(TT.datetime(k))== 4
            M_GF(k,1) = M (i,5);
        elseif month(TT.datetime(k))== 5
            M_GF(k,1) = M (i,6);
        elseif month(TT.datetime(k))== 6
            M_GF(k,1) = M (i,7);
        elseif month(TT.datetime(k))== 7
            M_GF(k,1) = M (i,8);
        elseif month(TT.datetime(k))== 8
            M_GF(k,1) = M (i,9);
        elseif month(TT.datetime(k))== 9
            M_GF(k,1) = M (i,10);
        elseif month(TT.datetime(k))== 10
            M_GF(k,1) = M (i,11);
        elseif month(TT.datetime(k))== 11
            M_GF(k,1) = M (i,12);
        elseif month(TT.datetime(k))== 12
            M_GF(k,1) = M (i,13);
        end
    end
    M_GF=array2table(M_GF);
    T_MOPEX = table;
    T_MOPEX(:,1)= T(:,1);  
    T_MOPEX(:,2)= T(:,2); 
    T_MOPEX(:,3)= T(:,3); 
    T_MOPEX(:,4)= T(:,4); 
    T_MOPEX(:,5)= T(:,5); 
    T_MOPEX(:,6)= M_GF(:,1);
    T_MOPEX(:,7)= T(:,6);
    %% Create new Table with lable
    T_MOPEX.Properties.VariableNames = {'datetime' 'P' 'PE' 'T_max' 'T_min' 'GF' 'Q'} ;                
    filename = append (st,ext);
    cd(MOPEX_output_files); 
    writetable(T_MOPEX,filename);

                    
                  
 
end 
  

    
    
    

    
    



