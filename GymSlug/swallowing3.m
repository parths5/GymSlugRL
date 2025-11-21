clc, clear all
import Swallowing.*
import SwallowingOptim.*
import RealSlugData.*
import RealSlugData2.*
import SlugCorr.*
import ImprovedB38.*

% variable Units
    % Motor Neurons and Muscles --> Hz
    % Force --> mN

%% Swallowing Objects
%%%%%%%%%%%%%%%%%%%%%%
disp("1 --> Swallowing Objects Section")

% create Swallowing objects for all the trained trials
trial3 = Swallowing("All_Trials/Trial_3","Expert_UnbreakableSwallowing", "Trial 3");     % call Swallowing constructor for Trial 3
trial4 = Swallowing("All_Trials/Trial_4","Expert_UnbreakableSwallowing", "Trial 4");     % call Swallowing constructor for Trial 4
trial5 = Swallowing("All_Trials/Trial_5","Expert_UnbreakableSwallowing", "Trial 5");     % call Swallowing constructor for Trial 5
trial8 = Swallowing("All_Trials/Trial_8","Expert_UnbreakableSwallowing", "Trial 8");     % call Swallowing constructor for Trial 8
trial9 = Swallowing("All_Trials/Trial_9","Expert_UnbreakableSwallowing", "Trial 9");     % call Swallowing constructor for Trial 9
trial10 = Swallowing("All_Trials/Trial_10","Expert_UnbreakableSwallowing", "Trial 10");  % call Swallowing constructor for Trial 10

% get cross correlation for every property
% currently nothing is graphed, but can choose to graph either cross correlation or normalized segments
count = 1;
trials = [trial3, trial4, trial5, trial8, trial9, trial10];



%% Trained and Expert Correlatrions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("2 --> Trained and Expert Correlatrions Section")

corAll = [];
corMax = [];
corAvg = [];
numSeg = [];
colors = ["red","green","blue","cyan","magenta","black"];

for trial = trials
    [cor1,norm1,maxSeg1,maxCor1] = trial.crossCorrelation("/I2_input__Trained.csv", "/I2_input__expert.csv", 0, 0, "I2 Input or B31/32");   % Cross Correlation for I2 Input
    [cor2,norm2,maxSeg2,maxCor2] = trial.crossCorrelation("/B8a_b__Trained.csv", "/B8a_b__expert.csv", 0, 0, "B8a/b");                      % Cross Correlation for B8a/b
    [cor3,norm3,maxSeg3,maxCor3] = trial.crossCorrelation("/B38__Trained.csv", "/B38__expert.csv", 0, 0, "B38");                            % Cross Correlation for B38
    [cor4,norm4,maxSeg4,maxCor4] = trial.crossCorrelation("/B6_9_3__Trained.csv", "/B6_9_3__expert.csv", 0, 0, "B6/9/3");                   % Cross Correlation for B6/9/3
    [cor5,norm5,maxSeg5,maxCor5] = trial.crossCorrelation("/B7__Trained.csv", "/B7__expert.csv", 0, 0, "B7");                               % Cross Correlation for B7
    [cor6,norm6,maxSeg6,maxCor6] = trial.crossCorrelation("/T_I2__Trained.csv", "/T_I2__expert.csv", 0, 0, "T I2");                         % Cross Correlation for T I2
    [cor7,norm7,maxSeg7,maxCor7] = trial.crossCorrelation("/P_I4__Trained.csv", "/P_I4__expert.csv", 0, 0, "P I4");                         % Cross Correlation for P I4
    [cor8,norm8,maxSeg8,maxCor8] = trial.crossCorrelation("/P_I3_anterior__Trained.csv", "/P_I3_anterior__expert.csv", 0, 0, "P I3");       % Cross Correlation for P I3
    [cor9,norm9,maxSeg9,maxCor9] = trial.crossCorrelation("/T_I3__Trained.csv", "/T_I3__expert.csv", 0, 0, "T I3");                         % Cross Correlation for T I3
    [cor10,norm10,maxSeg10,maxCor10] = trial.crossCorrelation("/T_hinge__Trained.csv", "/T_hinge__expert.csv", 0, 0, "T hinge");            % Cross Correlation for T hinge
    [cor11,norm11,maxSeg11,maxCor11] = trial.crossCorrelation("/Grasper_Motion__Trained.csv", "/GrasperMotion__expert.csv", 0, 0, "Grasper Motion");    % Cross Correlation for Grasper Motion
    [cor12,norm12,maxSeg12,maxCor12] = trial.crossCorrelation("/Force__Trained.csv", "/Force__expert.csv", 0, 0, "Force");                              % Cross Correlation for Force
    
    corAll = [corAll; cor1.' cor2.' cor3.' cor4.' cor5.' cor6.' cor7.' cor8.' cor9.' cor10.' cor11.' cor12.'];
    corMax = [corMax; maxCor1 maxCor2 maxCor3 maxCor4 maxCor5 maxCor6 maxCor7 maxCor8 maxCor9 maxCor10 maxCor11 maxCor12];
    corAvg = [corAvg; mean(cor1) mean(cor2) mean(cor3) mean(cor4) mean(cor5) mean(cor6) mean(cor7) mean(cor8) mean(cor9) mean(cor10) mean(cor11) mean(cor12)];
    numSeg = [numSeg length(trial.trialSegment)-1];
    count = count+1;
end



%% Graphs and Plots
%%%%%%%%%%%%%%%%%%%%
disp("3 --> Graphs and Plots Section")

% Fill out bellow vaiables to decide which graphs you want to see
uglyBoxplotVisible = false;
boxplotVisible = false;
subplotVisibleMax = false;
subplotVisibleAvg = false;
plotOnTopVisibleMax = false;
plotOnTopVisibleAvg = false;
plotAllVisibleNeuron = false;
plotAllVisibleMuscle = false;

% Will display the 4 boxplots: Boxplot with all trials color coordinated, Boxplot with all properties color coordinated, Boxplot with only normalized
% segments with max corelation, Boxplot with only normalized segments with average corelation
if boxplotVisible
    if uglyBoxplotVisible
        figure % Boxplot with all trials color coordinated
        hold on
        boxchart(corAll,'MarkerStyle','none');
        count = 1;
        for i = 1:length(numSeg)
            pointsAll = repmat(1:12,numSeg(i),1);
            swarmchart(pointsAll,corAll(count:numSeg(i)+count-1,:),[], colors(i),"filled");
            if count == 1
                count = numSeg(i);
            else
                count = count + numSeg(i);
            end
        end
        title('Cross Correlation for Every Property w/ all Normalized Segments')
        subtitle('1 = I2 Input,   2 = B8a/b, 3 = B38, 4 = B6/9/3, 5 = B7,  6 = T I2, 7 = P I4, 8 = P I3, 9 = T I3, 10 = T hinge, 11 = Grasper Motion, 12 = Force')
        hold off
    end

    modelBoxPlot(corAll, 'Cross Correlation for Every Property w/ all Normalized Segments', 1)
    % modelBoxPlot(corMax, 'Cross Correlation for Every Property w/ Max Cor Normalized Segments', 0)
    % modelBoxPlot(corAvg, 'Cross Correlation for Every Property w/ Avg Cor Normalized Segments', 0)

end

% Will display subplots of each property using normalized segments with max corelation
if subplotVisibleMax
    maxSubplot("/I2_input__expert.csv", "/I2_input__Trained.csv", "I2 Input", trials)   % Plot subplot of expert and trained segments with highest corrleation for I2 Input
    maxSubplot("/B8a_b__expert.csv", "/B8a_b__Trained.csv", "B8a/b", trials)            % Plot subplot of expert and trained segments with highest corrleation for B8a/b
    maxSubplot("/B38__expert.csv", "/B38__Trained.csv", "B38", trials)                  % Plot subplot of expert and trained segments with highest corrleation for B38
    maxSubplot("/B6_9_3__expert.csv", "/B6_9_3__Trained.csv", "B6/9/3", trials)         % Plot subplot of expert and trained segments with highest corrleation for B6/9/3
    maxSubplot("/B7__expert.csv", "/B7__Trained.csv", "B7", trials)                     % Plot subplot of expert and trained segments with highest corrleation for B6/9/3
end

% Will display subplots of each property using normalized segments with average corelation
if subplotVisibleAvg
    avgSubplot("/I2_input__expert.csv", "/I2_input__Trained.csv", "I2 Input", trials)   % Plot subplot of expert and trained segments with average corrleation for I2 Input
    avgSubplot("/B8a_b__expert.csv", "/B8a_b__Trained.csv", "B8a/b", trials)            % Plot subplot of expert and trained segments with average corrleation for B8a/b
    avgSubplot("/B38__expert.csv", "/B38__Trained.csv", "B38", trials)                  % Plot subplot of expert and trained segments with average corrleation for B38
    avgSubplot("/B6_9_3__expert.csv", "/B6_9_3__Trained.csv", "B6/9/3", trials)         % Plot subplot of expert and trained segments with average corrleation for B6/9/3
    avgSubplot("/B7__expert.csv", "/B7__Trained.csv", "B7", trials)                     % Plot subplot of expert and trained segments with average corrleation for B6/9/3
end

% Will display plots of each property using normalized segments with max corelation
if plotOnTopVisibleMax
    maxPlotterOnTop("/T_I2__expert.csv", "/T_I2__Trained.csv", "T I2", trials)                      % Plot subplot of expert and trained segments with highest corrleation for T I2
    maxPlotterOnTop("/P_I4__expert.csv", "/P_I4__Trained.csv", "P I4", trials)                      % Plot subplot of expert and trained segments with highest corrleation for P I4
    maxPlotterOnTop("/P_I3_anterior__expert.csv", "/P_I3_anterior__Trained.csv", "P I3", trials)    % Plot subplot of expert and trained segments with highest corrleation for P I3
    maxPlotterOnTop("/T_I3__expert.csv", "/T_I3__Trained.csv", "T I3", trials)                      % Plot subplot of expert and trained segments with highest corrleation for T I3
    maxPlotterOnTop("/T_hinge__expert.csv", "/T_hinge__Trained.csv", "T hinge", trials)             % Plot subplot of expert and trained segments with highest corrleation for T hinge
    maxPlotterOnTop("/GrasperMotion__expert.csv", "/Grasper_Motion__Trained.csv", "Grasper Motion", trials)   % Plot subplot of expert and trained segments with highest corrleation for Grasper Motion
    maxPlotterOnTop("/Force__expert.csv", "/Force__Trained.csv", "Force", trials)                   % Plot subplot of expert and trained segments with highest corrleation for Force
end

% Will display plots of each property using normalized segments with average corelation
if plotOnTopVisibleAvg
    avgPlotterOnTop("/T_I2__expert.csv", "/T_I2__Trained.csv", "T I2", trials)                      % Plot subplot of expert and trained segments with highest corrleation for T I2
    avgPlotterOnTop("/P_I4__expert.csv", "/P_I4__Trained.csv", "P I4", trials)                      % Plot subplot of expert and trained segments with highest corrleation for P I4
    avgPlotterOnTop("/P_I3_anterior__expert.csv", "/P_I3_anterior__Trained.csv", "P I3", trials)    % Plot subplot of expert and trained segments with highest corrleation for P I3
    avgPlotterOnTop("/T_I3__expert.csv", "/T_I3__Trained.csv", "T I3", trials)                      % Plot subplot of expert and trained segments with highest corrleation for T I3
    avgPlotterOnTop("/T_hinge__expert.csv", "/T_hinge__Trained.csv", "T hinge", trials)             % Plot subplot of expert and trained segments with highest corrleation for T hinge
    avgPlotterOnTop("/GrasperMotion__expert.csv", "/Grasper_Motion__Trained.csv", "Grasper Motion", trials)   % Plot subplot of expert and trained segments with highest corrleation for Grasper Motion
    avgPlotterOnTop("/Force__expert.csv", "/Force__Trained.csv", "Force", trials)                   % Plot subplot of expert and trained segments with highest corrleation for Force
end

% Will display plots of each neuron property using normalized segments with both max and avg corelation seperately
if plotAllVisibleNeuron
    fileExpert = ["/I2_input__expert.csv", "/B8a_b__expert.csv", "/B38__expert.csv", "/B6_9_3__expert.csv", "/B7__expert.csv"];
    fileTrain = ["/I2_input__Trained.csv", "/B8a_b__Trained.csv", "/B38__Trained.csv", "/B6_9_3__Trained.csv", "/B7__Trained.csv"];
    propertyTitle = ["I2 Input", "B8a/b", "B38", "B6/9/3", "B7"];
    
    for i=1:length(propertyTitle)
        [avgAverage, avgStd] = plotOnTopALL(trials,fileTrain(i), fileExpert(i), 0, propertyTitle(i), 0, 1);
        [maxAverage, maxStd] = plotOnTopALL(trials,fileTrain(i), fileExpert(i), 1, propertyTitle(i), 0, 1);
    end
end

% Will display plots of each muscle property using normalized segments with both max and avg corelation seperately
if plotAllVisibleMuscle
    fileExpert = ["/T_I2__expert.csv", "/P_I4__expert.csv", "/P_I3_anterior__expert.csv", "/T_I3__expert.csv", "/T_hinge__expert.csv", "/GrasperMotion__expert.csv", "/Force__expert.csv"];
    fileTrain = ["/T_I2__Trained.csv", "/P_I4__Trained.csv", "/P_I3_anterior__Trained.csv", "/T_I3__Trained.csv", "/T_hinge__Trained.csv", "/Grasper_Motion__Trained.csv", "/Force__Trained.csv"];
    propertyTitle = ["T I2", "P I4", "P I3", "T I3", "T hinge", "Grasper Motion", "Force"];
    
    for i=1:length(propertyTitle)
        [avgAverage, avgStd] = plotOnTopALL(trials,fileTrain(i), fileExpert(i), 0, propertyTitle(i), 1, 1);
        [maxAverage, maxStd] = plotOnTopALL(trials,fileTrain(i), fileExpert(i), 1, propertyTitle(i), 1, 1);
    end
end



%% Real Slug Data Correlatrions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("4 --> Real Slug Data Correlatrions Section")

oldCorrelation = false;
slug2 = SlugCorr("JG08 Tape nori superset dataArrayV2.mat", 0, 0);

properties = ["I2", "B8", "B38", "B6/9/3", "force", "xhg"]; % remove  "B4/5" since not in other models
fileExpert = ["/I2_input__expert.csv", "/B8a_b__expert.csv", "/B38__expert.csv", "/B6_9_3__expert.csv", "/Force__expert.csv", "/RelativeGrasperMotion__expert.csv"];
fileTrain = ["/I2_input__Trained.csv", "/B8a_b__Trained.csv", "/B38__Trained.csv", "/B6_9_3__Trained.csv", "/Force__Trained.csv", "/RelativeGrasperMotion__Trained.csv"];

if oldCorrelation
    % create RealSlugData objects for all the animal trials
    slug1 = RealSlugData("JG07 Tape nori 0 dataArrayV2.mat", 0, 0);
    slug2 = RealSlugData("JG08 Tape nori superset dataArrayV2.mat", 0, 0);
    slug3 = RealSlugData("JG11 Tape nori 0_V2.mat", 0, 0);
    slug4 = RealSlugData("JG12 Tape nori 0_V2.mat", 0, 0);
    slug5 = RealSlugData("JG12 Tape nori 1_V2.mat", 0, 0);
    slug6 = RealSlugData("JG14 Tape nori 0_V2.mat", 0, 0);
    
    slugs = [slug1 slug2 slug3 slug4 slug5 slug6]; % weird ones are slug 2,3
    % properties = ["B6/9/3", "B38", "B8", "I2", "force"]; % remove  "B4/5" since not in other models
    % fileExpert = ["/B6_9_3__expert.csv", "/B38__expert.csv", "/B8a_b__expert.csv", "/I2_input__expert.csv", "/Force__expert.csv"];
    % fileTrain = ["/B6_9_3__Trained.csv", "/B38__Trained.csv", "/B8a_b__Trained.csv", "/I2_input__Trained.csv", "/Force__Trained.csv"];
    

    
    % Plotting Curves for Slug, Expert, Trained
    % arrayB38 = slug6.arrayGetter("force", 1);
    % arrayForce = slug1.arrayGetter("force", 1);
    % arrayForce = slug2.arrayGetter("force", 1);
    % arrayForce = slug3.arrayGetter("force", 1);
    % arrayForce = slug4.arrayGetter("force", 1);
    % arrayForce = slug5.arrayGetter("force", 1);
    % arrayForce = slug6.arrayGetter("force", 1);
    
    
    % fid = fopen('realAnimal2B38_6.txt','wt');
    % if fid > 0
    %     fprintf(fid,'%f\n',arrayB38');
    %     fclose(fid);
    % end
    % 
    % fid = fopen('realAnimalForce_6.txt','wt');
    % if fid > 0
    %     fprintf(fid,'%f\n',arrayForce');
    %     fclose(fid);
    % end
    %trial5.getExpertProperty("/B38__expert.csv", 1);
    %trial5.getTrainedProperty("/B38__Trained.csv", "/B38__expert.csv", 1, "B38", 1);
    
    
    
    disp("          Same slug to slug correlations")
    % Same slug to slug correlations
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    minCorSlug2Self = ones(1, 5);
    minCor1 = [];
    maxCor1 = [];
    allCor1 = [];
    oneSeg = [];
    for s = 1:length(slugs)
        for i = 1:length(properties)
            [crossCorr, maxVal, minVal] = slugs(s).Slug2SelfCorrelation(0, 0,properties(i));
            if minVal < minCorSlug2Self(i)
                % disp(minVal + ": slug" + s + "/prop" + i)
                minCorSlug2Self(i) = minVal;
            end
            minCor1(s,i) = minVal;
            maxCor1(s,i) = maxVal;
            oneSeg = [oneSeg crossCorr.'];
        end
        allCor1 = [allCor1; oneSeg];
        oneSeg = [];
    end
    % disp(minCorSlug2Self)
    
    
    disp("          Different slug to slug correlations")
    % Different slug to slug correlations
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    minCorSlug2Slug = ones(1, 5);
    minCor2 = [];
    maxCor2 = [];
    allCor2 = [];
    oneSeg = [];
    count = 1;
    for i = 1:length(slugs)
        for j = i+1:length(slugs)
            for p = 1:length(properties)
                [crossCorr, maxVal, minVal] = slugs(i).Slug2SlugCorrelation(slugs(j), 0, 0,properties(p));
                if minVal < minCorSlug2Slug(p)
                    % disp(minVal + ": slug" + i + "/slug" + j + "/prop" + p)
                    minCorSlug2Slug(p) = minVal;
                end
                minCor2(count,p) = minVal;
                maxCor2(count,p) = maxVal;
                oneSeg = [oneSeg crossCorr.'];
            end
            count = count+1;
            allCor2 = [allCor2; oneSeg];
            oneSeg = [];
        end
    end
    % disp(minCorSlug2Slug)
    
    
    disp("          Slug to expert correlations")
    % Slug to expert correlations
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    minCorSlug2Expert = ones(1, 5);
    minCor3 = [];
    maxCor3 = [];
    allCor3 = [];
    oneSeg = [];
    count = 1;
    for s = 1:length(slugs)
        for i = 1:length(fileExpert)
            [crossCorr, maxVal, minVal] = slugs(s).Slug2ExpertCorrelation(trial3, fileExpert(i), 0, 0, properties(i));
            if minVal < minCorSlug2Expert(p)
                % disp(minVal + ": slug" + s + "/expert/prop" + i)
                minCorSlug2Expert(i) = minVal;
            end
            minCor3(count,i) = minVal;
            maxCor3(count,i) = maxVal;
            oneSeg = [oneSeg crossCorr.'];
        end
        count = count+1;
        allCor3 = [allCor3; oneSeg];
        oneSeg = [];
    end
    % disp(minCorSlug2Expert)
    
    
    % additional runs
    trial1 = Swallowing("All_Trials/Trial_1","Expert_UnbreakableSwallowing", "Trial 1");  % call Swallowing constructor for Trial 1
    trial2 = Swallowing("All_Trials/Trial_2","Expert_UnbreakableSwallowing", "Trial 2");  % call Swallowing constructor for Trial 2
    trial6 = Swallowing("All_Trials/Trial_6","Expert_UnbreakableSwallowing", "Trial 6");  % call Swallowing constructor for Trial 6
    trial7 = Swallowing("All_Trials/Trial_7","Expert_UnbreakableSwallowing", "Trial 7");  % call Swallowing constructor for Trial 7
    
    %trials = [trials, trial1, trial2, trial6, trial7];
    %1,2,6,7
    
    
    disp("          Slug to small sample of train correlations")
    % Slug to small sample of train correlations --> 6
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    minCorSlug2Train = ones(1, 5);
    minCor4 = [];
    maxCor4 = [];
    allCor4 = [];
    oneSeg = [];
    count = 1;
    for s = 1:length(slugs)
        for t = 1:length(trials)
            for i = 1:length(fileExpert)
                [crossCorr, maxVal, minVal] = slugs(s).Slug2TrainedCorrelation(trials(t), fileTrain(i), 1, fileExpert(i), 0, 0, properties(i));
                if minVal < minCorSlug2Train(p)
                    % disp(minVal + ": slug" + s + "/trial" + t + "/prop" + i)
                    minCorSlug2Train(i) = minVal;
                end
                minCor4(count,i) = minVal;
                maxCor4(count,i) = maxVal;
                oneSeg = [oneSeg crossCorr.'];
            end
            count = count+1;
            allCor4 = [allCor4; oneSeg];
            oneSeg = [];
        end
    end
    % disp(minCorSlug2Train)
    
    
    disp("          Slug to train correlations")
    % Slug to train correlations
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    minCorSlug2TrainAll = ones(1, 5);
    minCor5 = [];
    maxCor5 = [];
    allCor5 = [];
    oneSeg = [];
    count = 1;
    for s = 1:length(slugs)
        for t = 1:length(trials)
            for i = 1:length(fileExpert)
                % [crossCorr, maxVal, minVal] = slugs(s).Slug2TrainedCorrelation(trials(t), fileTrain(i), 1, fileExpert(i), 0, 0, properties(i));
                [crossCorr, maxVal, minVal] = slugs(s).Slug2AllTrainedCorrelation(trials(t), fileTrain(i), 0, 0, properties(i));
                if minVal < minCorSlug2TrainAll(p)
                    % disp(minVal + ": slug" + s + "/trial" + t + "/prop" + i)
                    minCorSlug2TrainAll(i) = minVal;
                end
                minCor5(count,i) = minVal;
                maxCor5(count,i) = maxVal;
                oneSeg = [oneSeg crossCorr.'];
            end
            count = count+1;
            allCor5 = [allCor5; oneSeg];
            oneSeg = [];
        end
    end
    % disp(minCorSlug2TrainAll)
    
    
    disp("          SLUG TO IMPROVED TRAINED CORRELATIONS")
    % SLUG TO IMPROVED TRAINED CORRELATIONS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % create Swallowing objects for all the trained trials
    trial1_B38 = Swallowing("All_Trials/TrialB38_1","Expert_UnbreakableSwallowing", "Improved B38 Trial 1");     % call Swallowing constructor for Trial 1
    trial2_B38 = Swallowing("All_Trials/TrialB38_2","Expert_UnbreakableSwallowing", "Improved B38 Trial 2");     % call Swallowing constructor for Trial 2
    trial3_B38 = Swallowing("All_Trials/TrialB38_3","Expert_UnbreakableSwallowing", "Improved B38 Trial 3");     % call Swallowing constructor for Trial 3
    %trial4_B38 = Swallowing("All_Trials/TrialB38_4","Expert_UnbreakableSwallowing", "Improved B38 Trial 4");     % call Swallowing constructor for Trial 4
    trial5_B38 = Swallowing("All_Trials/TrialB38_5","Expert_UnbreakableSwallowing", "Improved B38 Trial 5");     % call Swallowing constructor for Trial 5
    trial6_B38 = Swallowing("All_Trials/TrialB38_6","Expert_UnbreakableSwallowing", "Improved B38 Trial 6");     % call Swallowing constructor for Trial 6
    trial7_B38 = Swallowing("All_Trials/TrialB38_7","Expert_UnbreakableSwallowing", "Improved B38 Trial 7");     % call Swallowing constructor for Trial 7
    trial8_B38 = Swallowing("All_Trials/TrialB38_8","Expert_UnbreakableSwallowing", "Improved B38 Trial 8");     % call Swallowing constructor for Trial 8
    trial9_B38 = Swallowing("All_Trials/TrialB38_9","Expert_UnbreakableSwallowing", "Improved B38 Trial 9");     % call Swallowing constructor for Trial 9
    %trial10_B38 = Swallowing("TrialB38_10","Expert_UnbreakableSwallowing", "Improved B38 Trial 10");  % call Swallowing constructor for Trial 10
    
    % get cross correlation for every property
    % currently nothing is graphed, but can choose to graph either cross correlation or normalized segments
    trialsB38 = [trial1_B38, trial2_B38, trial3_B38, trial5_B38, trial6_B38, trial7_B38, trial8_B38, trial9_B38];
    
    minCorSlug2ImprovAll = ones(1, 5);
    minCor6 = [];
    maxCor6 = [];
    allCor6 = [];
    oneSeg = [];
    count = 1;
    for s = 1:length(slugs)
        for t = 1:length(trialsB38)
            for i = 1:length(fileExpert)
                % [crossCorr, maxVal, minVal] = slugs(s).Slug2TrainedCorrelation(trials(t), fileTrain(i), 1, fileExpert(i), 0, 0, properties(i));
                [crossCorr, maxVal, minVal] = slugs(s).Slug2AllTrainedCorrelation(trialsB38(t), fileTrain(i), 0, 0, properties(i));
                if minVal < minCorSlug2ImprovAll(p)
                    % disp(minVal + ": slug" + s + "/trial" + t + "/prop" + i)
                    minCorSlug2ImprovAll(i) = minVal;
                end
                minCor6(count,i) = minVal;
                maxCor6(count,i) = maxVal;
                oneSeg = [oneSeg crossCorr.'];
            end
            count = count+1;
            allCor6 = [allCor6; oneSeg];
            oneSeg = [];
        end
    end
    %disp(minCorSlug2ImprovAll)
    %disp("")
    
    %imrpovTunned = ImprovedB38("All_Trials/TrialB38_1", "Expert_Swallowing_withB38", "Improv Tunned");
    properties = ["I2", "B8", "B38", "B6/9/3", "force"]; % remove  "B4/5" since not in other models
    fileImprov = ["/I2_input__improvB38.csv", "/B8a_b__improvB38.csv", "/B38__improvB38.csv", "/B6_9_3__improvB38.csv", "/Force__improvB38.csv"];
    
    
    % disp("          Slug to improv correlations")
    % % Slug to improv correlations
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % minCorSlug2Improv = ones(1, 5);
    % minCor7 = [];
    % maxCor7 = [];
    % allCor7 = [];
    % oneSeg = [];
    % count = 1;
    % for s = 1:length(slugs)
    %     for i = 1:length(fileImprov)
    %         [crossCorr, maxVal, minVal] = slugs(s).Slug2ImprovCorrelation(imrpovTunned, fileImprov(i), 0, 0, properties(i));
    %         if minVal < minCorSlug2Improv(p)
    %             % disp(minVal + ": slug" + s + "/expert/prop" + i)
    %             minCorSlug2Improv(i) = minVal;
    %         end
    %         minCor7(count,i) = minVal;
    %         maxCor7(count,i) = maxVal;
    %         oneSeg = [oneSeg crossCorr.'];
    %     end
    %     count = count+1;
    %     allCor7 = [allCor7; oneSeg];
    %     oneSeg = [];
    % end
    %disp(minCorSlug2Improv)
    
    
    disp("          SLUG TO IMPROVED + FR TRAINED CORRELATIONS ")
    % SLUG TO IMPROVED + FR TRAINED CORRELATIONS --> 7
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % create Swallowing objects for all the trained trials
    trial1_B38_new = Swallowing("All_Trials/TrialB38_new_1","Expert_UnbreakableSwallowing", "Improved B38 Trial 1 w Lession");     % call Swallowing constructor for Trial 1
    trial2_B38_new = Swallowing("All_Trials/TrialB38_new_2","Expert_UnbreakableSwallowing", "Improved B38 Trial 2 w Lession");     % call Swallowing constructor for Trial 2
    trial3_B38_new = Swallowing("All_Trials/TrialB38_new_3","Expert_UnbreakableSwallowing", "Improved B38 Trial 3 w Lession");     % call Swallowing constructor for Trial 3
    trial4_B38_new = Swallowing("All_Trials/TrialB38_new_4","Expert_UnbreakableSwallowing", "Improved B38 Trial 4 w Lession");     % call Swallowing constructor for Trial 4
    trial5_B38_new = Swallowing("All_Trials/TrialB38_new_5","Expert_UnbreakableSwallowing", "Improved B38 Trial 5 w Lession");     % call Swallowing constructor for Trial 5
    trial6_B38_new = Swallowing("All_Trials/TrialB38_new_6","Expert_UnbreakableSwallowing", "Improved B38 Trial 6 w Lession");     % call Swallowing constructor for Trial 6
    trial7_B38_new = Swallowing("All_Trials/TrialB38_new_7","Expert_UnbreakableSwallowing", "Improved B38 Trial 7 w Lession");     % call Swallowing constructor for Trial 7
    
    % additional runs
    trial8_B38_new = Swallowing("All_Trials/TrialB38_new_8","Expert_UnbreakableSwallowing", "Improved B38 Trial 8 w Lession");     % call Swallowing constructor for Trial 8
    trial9_B38_new = Swallowing("All_Trials/TrialB38_new_9","Expert_UnbreakableSwallowing", "Improved B38 Trial 9 w Lession");     % call Swallowing constructor for Trial 8
    trial10_B38_new = Swallowing("All_Trials/TrialB38_new_10","Expert_UnbreakableSwallowing", "Improved B38 Trial 10 w Lession");     % call Swallowing constructor for Trial 8
    
    % get cross correlation for every property
    % currently nothing is graphed, but can choose to graph either cross correlation or normalized segments
    trialsB38_new = [trial1_B38_new, trial2_B38_new, trial3_B38_new, trial4_B38_new, trial5_B38_new, trial6_B38_new, trial7_B38_new, trial8_B38_new, trial9_B38_new, trial10_B38_new];
    
    minCorSlug2ImprovLession = ones(1, 5);
    minCor8 = [];
    maxCor8 = [];
    allCor8 = [];
    oneSeg = [];
    count = 1;
    for s = 1:length(slugs)
        for t = 1:length(trialsB38_new)
            for i = 1:length(fileExpert)
                % [crossCorr, maxVal, minVal] = slugs(s).Slug2TrainedCorrelation(trials(t), fileTrain(i), 1, fileExpert(i), 0, 0, properties(i));
                [crossCorr, maxVal, minVal] = slugs(s).Slug2AllTrainedCorrelation(trialsB38_new(t), fileTrain(i), 0, 0, properties(i));
                if minVal < minCorSlug2ImprovLession(p)
                    % disp(minVal + ": slug" + s + "/trial" + t + "/prop" + i)
                    minCorSlug2ImprovLession(i) = minVal;
                end
                minCor8(count,i) = minVal;
                maxCor8(count,i) = maxVal;
                oneSeg = [oneSeg crossCorr.'];
            end
            count = count+1;
            allCor8 = [allCor8; oneSeg];
            oneSeg = [];
        end
    end
    %disp(minCorSlug2ImprovAll)
    
    
    
    % disp("          SLUG TO EXPERT + FR TRAINED CORRELATIONS ")
    % % SLUG TO EXPERT + FR TRAINED CORRELATIONS --> 8
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % % create Swallowing objects for all the trained trials
    % trial1_fr = Swallowing("All_Trials/Trial_Fr_1","Expert_UnbreakableSwallowing", "Trial 1 w Lession");     % call Swallowing constructor for Trial 1
    % trial2_fr = Swallowing("All_Trials/Trial_Fr_2","Expert_UnbreakableSwallowing", "Trial 2 w Lession");     % call Swallowing constructor for Trial 2
    % trial3_fr = Swallowing("All_Trials/Trial_Fr_3","Expert_UnbreakableSwallowing", "Trial 3 w Lession");     % call Swallowing constructor for Trial 3
    % trial5_fr = Swallowing("All_Trials/Trial_Fr_5","Expert_UnbreakableSwallowing", "Trial 5 w Lession");     % call Swallowing constructor for Trial 4
    % trial6_fr = Swallowing("All_Trials/Trial_Fr_6","Expert_UnbreakableSwallowing", "Trial 6 w Lession");     % call Swallowing constructor for Trial 5
    % trial7_fr = Swallowing("All_Trials/Trial_Fr_7","Expert_UnbreakableSwallowing", "Trial 7 w Lession");     % call Swallowing constructor for Trial 6
    % trial9_fr = Swallowing("All_Trials/Trial_Fr_9","Expert_UnbreakableSwallowing", "Trial 9 w Lession");     % call Swallowing constructor for Trial 7
    % trial10_fr = Swallowing("All_Trials/Trial_Fr_10","Expert_UnbreakableSwallowing", "Trial 10 w Lession");     % call Swallowing constructor for Trial 8
    % 
    % % additional runs
    % trial4_fr = Swallowing("All_Trials/Trial_Fr_4","Expert_UnbreakableSwallowing", "Trial 4 w Lession");     % call Swallowing constructor for Trial 4
    % trial8_fr = Swallowing("All_Trials/Trial_Fr_8","Expert_UnbreakableSwallowing", "Trial 8 w Lession");     % call Swallowing constructor for Trial 8
    % 
    % 
    % % get cross correlation for every property
    % % currently nothing is graphed, but can choose to graph either cross correlation or normalized segments
    % trials_fr = [trial1_fr, trial2_fr, trial3_fr, trial5_fr, trial6_fr, trial7_fr, trial9_fr, trial10_fr, trial4_fr, trial8_fr];
    % 
    % minCorSlug2TrainedLession = ones(1, 5);
    % minCor9 = [];
    % maxCor9 = [];
    % allCor9 = [];
    % oneSeg = [];
    % count = 1;
    % for s = 1:length(slugs)
    %     for t = 1:length(trials_fr)
    %         for i = 1:length(fileExpert)
    %             % [crossCorr, maxVal, minVal] = slugs(s).Slug2TrainedCorrelation(trials(t), fileTrain(i), 1, fileExpert(i), 0, 0, properties(i));
    %             [crossCorr, maxVal, minVal] = slugs(s).Slug2AllTrainedCorrelation(trials_fr(t), fileTrain(i), 0, 0, properties(i));
    %             if minVal < minCorSlug2TrainedLession(p)
    %                 % disp(minVal + ": slug" + s + "/trial" + t + "/prop" + i)
    %                 minCorSlug2TrainedLession(i) = minVal;
    %             end
    %             minCor9(count,i) = minVal;
    %             maxCor9(count,i) = maxVal;
    %             oneSeg = [oneSeg crossCorr.'];
    %         end
    %         count = count+1;
    %         allCor9 = [allCor9; oneSeg];
    %         oneSeg = [];
    %     end
    % end
    % %disp(minCorSlug2ImprovAll)
    % 
    % 
    % disp("          Slug to Optim correlations")
    % % Slug to Optim correlations
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % % create RealSlugData objects for all the animal data
    % slug1 = RealSlugData2("JG07 Tape nori 0 dataArrayV2.mat", 0, 0);
    % slug2 = RealSlugData2("JG08 Tape nori superset dataArrayV2.mat", 0, 0);
    % slug3 = RealSlugData2("JG11 Tape nori 0_V2.mat", 0, 0);
    % slug4 = RealSlugData2("JG12 Tape nori 0_V2.mat", 0, 0);
    % slug5 = RealSlugData2("JG12 Tape nori 1_V2.mat", 0, 0);
    % slug6 = RealSlugData2("JG14 Tape nori 0_V2.mat", 0, 0);
    % 
    % slugs2 = [slug1 slug2 slug3 slug4 slug5 slug6]; % weird ones are slug 2,3
    % fileOptim = ["/I2_input__optim.csv", "/B8a_b__optim.csv", "/B38__optim.csv", "/B6_9_3__optim.csv", "/Force__optim.csv"];
    % fileOptimSingle = ["/I2_input_single__optim.csv", "/B8a_b_single__optim.csv", "/B38_single__optim.csv", "/B6_9_3_single__optim.csv", "/Force_single__optim.csv"];
    % 
    % %[crossCorr, maxVal, minVal] = slugs(1).Slug2OptimCorrelation(fileOptimSingle(5), 1, 1, properties(5));
    % 
    % minCorSlug2Optim = ones(1, 5);
    % minCor0 = [];
    % maxCor0 = [];
    % allCor0 = [];
    % oneSeg = [];
    % count = 1;
    % for s = 1:length(slugs2)
    %     for i = 1:length(fileOptimSingle)
    %         [crossCorr, maxVal, minVal] = slugs2(s).Slug2OptimCorrelation(fileOptimSingle(i), 0, 0, properties(i));
    %         if minVal < minCorSlug2Optim(i)
    %             % disp(minVal + ": slug" + s + "/expert/prop" + i)
    %             minCorSlug2Optim(i) = minVal;
    %         end
    %         minCor0(count,i) = minVal;
    %         maxCor0(count,i) = maxVal;
    %         oneSeg = [oneSeg crossCorr.'];
    %     end
    %     count = count+1;
    %     allCor0 = [allCor0; oneSeg];
    %     oneSeg = [];
    % end
    % %disp(minCorSlug2Optim)
end

%% Different Boxplots
%%%%%%%%%%%%%%%%%%%%%%
disp("5 --> Different Boxplots Section")

horizontalBoxplot = false;
boxplotAnimalVisible = false;
boxplotModelsVisible = false;
boxplotOnlyAllVisbile = false;
boxplotAll = false;
boxplot_B38 = false;


if horizontalBoxplot
    b38corr = [allCor2(:,2).' allCor3(:,2).' allCor5(:,2).'];
    grp = [zeros(1,length(allCor2(:,2))) ones(1,length(allCor3(:,2))) ones(1,length(allCor5(:,2))).*2];
    boxplot(b38corr,grp,'orientation','horizontal')
end

if boxplotAnimalVisible
    animalBoxPlot(minCor1, ['Min Cross Correlation of Normalized Segments';"- Same Slug Comarison -"], 0, .5, 0);
    animalBoxPlot(maxCor1, ['Max Cross Correlation of Normalized Segments';"- Same Slug Comarison -"], 0, .5, 0);
    animalBoxPlot(allCor1, ['All Cross Correlations of Normalized Segments';"- Same Slug Comarison -"], 1, .5, 0);

    animalBoxPlot(minCor2, ['Min Cross Correlation of Normalized Segments';"- Diff Slug Comarison -"], 0, .5, 0);
    animalBoxPlot(maxCor2, ['Max Cross Correlation of Normalized Segments';"- Diff Slug Comarison -"], 0, .5, 0);
    animalBoxPlot(allCor2, ['All Cross Correlations of Normalized Segments';"- Diff Slug Comarison -"], 1, .5, 0);

end

if boxplotModelsVisible
    animalBoxPlot(minCor3, ['Min Cross Correlation of Normalized Segments';"- Slug Expert Comarison -"], 0, .5, 0);
    animalBoxPlot(maxCor3, ['Max Cross Correlation of Normalized Segments';"- Slug Expert Comarison -"], 0, .5, 0);
    animalBoxPlot(allCor3, ['All Cross Correlations of Normalized Segments';"- Slug Expert Comarison -"], 1, .5, 0);

    animalBoxPlot(minCor4, ['Min Cross Correlation of Normalized Segments';"- Slug Trained Comarison -"], 0, .5, 0);
    animalBoxPlot(maxCor4, ['Max Cross Correlation of Normalized Segments';"- Slug Trained Comarison -"], 0, .5, 0);
    animalBoxPlot(allCor4, ['All Cross Correlations of Normalized Segments';"- Slug Trained Comarison -"], 1, .5, 0);

    animalBoxPlot(minCor5, ['Min Cross Correlation of Normalized Segments';"- Slug All-Trained Comarison -"], 0, .4, 0);
    animalBoxPlot(maxCor5, ['Max Cross Correlation of Normalized Segments';"- Slug All-Trained Comarison -"], 0, .4, 0);
    animalBoxPlot(allCor5, ['All Cross Correlations of Normalized Segments';"- Slug All-Trained Comarison -"], 1, .4, 0);
end

if boxplotOnlyAllVisbile
    %animalBoxPlot(allCor1, ['All Cross Correlations of Normalized Segments';"- Same Slug Comarison -"], 1, .3, 0);
    animalBoxPlot(allCor2, ['All Cross Correlations of Normalized Segments';"- Diff Slug Comarison -"], 1, 0, 0);
    animalBoxPlot(allCor3, ['All Cross Correlations of Normalized Segments';"- Slug Expert Comarison -"], 1, 0, 0);
    %animalBoxPlot(allCor4, ['All Cross Correlations of Normalized Segments';"- Slug Trained Comarison -"], 1, .3, 0);
    animalBoxPlot(allCor5, ['All Cross Correlations of Normalized Segments';"- Slug All-Trained Comarison -"], 1, 0, 0);
    animalBoxPlot([corAll(:,1) corAll(:,2) corAll(:,3) corAll(:,4) corAll(:,12)], ['All Cross Correlations of Normalized Segments';"- Expert Trained Comarison -"], 1, 0, 0);
    %animalBoxPlot(allCor6, ['All Cross Correlations of Normalized Segments';"- Slug All-Improv Comarison -"], 1, 0, 0);
    animalBoxPlot(allCor9, ['All Cross Correlations of Normalized Segments';"- Slug Improv Lession Comarison -"], 1, 0, 0);
end


if boxplot_B38
    % Sample Data:
    trial1 = allCor2(:,3);   % Inter-Animal
    trial2 = allCor7(:,3);   % Animal-to-Expert+fr
    trial3 = allCor5(:,3);   % Animal-to-Trained (All Train)
    trial4 = allCor9(:,3);   % Animal-to-Trained+fr


    % These grouping matrices label the columns:
    grp1 = repmat(1:1,size(trial1,1),1);
    grp2 = repmat(1:1,size(trial2,1),1);
    grp3 = repmat(1:1,size(trial3,1),1);
    grp4 = repmat(1:1,size(trial4,1),1);
    

    % These color matrices label the matrix id:
    clr1 = repmat(1,size(trial1));
    clr2 = repmat(2,size(trial2));
    clr3 = repmat(3,size(trial3));
    clr4 = repmat(4,size(trial4)); 

    % Combine the above matrices into one for x, y, and c:
    x = [grp1;grp2;grp3;grp4];
    y = [trial1;trial2;trial3;trial4];
    c = [clr1;clr2;clr3;clr4];

    % Convert those matrices to vectors:
    x = x(:);
    y = y(:);
    c = c(:);


    % Multiply x by 2 so that they're spread out:
    x = x*1.2;

    % Make the boxchart, 
    figure
    boxchart(x(:),y(:),'GroupByColor',c(:),'BoxEdgeColor','k','BoxFaceColor','#898989','MarkerColor','#898989')

    % Set the x ticks and labels, and add a legend
    xticks(2:2:16);
    xticklabels(1:8)
    xlabel('Category')
    legend(["Inter-Animal" "Animal-to-Baseline+fr" "Animal-to-Trained" "Animal-to-Trained+fr"],'Location','NorthOutside')
    title("Cross Correlation Distributions of Animal and Model Comparisons")
    % subtitle("1 = I2 , 2 = B8, 3 = B38, 4 = B6B9B3, 5 = force")
    ylabel("Cross Correlation Coeficient")
    ylim([0.3,1.05])

end

if boxplotAll
    % Sample Data:
    trial1 = allCor1;   % Intra-Animal
    trial2 = allCor2;   % Inter-Animal
    trial3 = allCor3;   % Animal-to-Expert (Baseline)
    trial5 = allCor5;   % Animal-to-Trained (All Train)
    trial6 = [corAll(:,1) corAll(:,2) corAll(:,3) corAll(:,4) corAll(:,12)];    % Expert-to-Trained
    trial7 = allCor6;   % Animal-to-Trained+Improv
    trial8 = allCor8;   % Animal-to-Trained+fr+Improv
    trial9 = allCor9;   % Animal-to-Trained+fr
    trial0 = allCor0;   % Animal-to-Optim


    % These grouping matrices label the columns:
    grp1 = repmat(1:5,size(trial1,1),1);
    grp2 = repmat(1:5,size(trial2,1),1);
    grp3 = repmat(1:5,size(trial3,1),1);
    grp4 = repmat(1:5,size(trial4,1),1); % Average Train
    grp5 = repmat(1:5,size(trial5,1),1);  % All Train
    grp6 = repmat(1:5,size(trial6,1),1);
    grp7 = repmat(1:5,size(trial7,1),1);
    grp8 = repmat(1:5,size(trial8,1),1);
    grp9 = repmat(1:5,size(trial9,1),1);
    grp0 = repmat(1:5,size(trial0,1),1);
    

    % These color matrices label the matrix id:
    clr1 = repmat(1,size(trial1));
    clr2 = repmat(1,size(trial2));
    clr3 = repmat(3,size(trial3));
    clr4 = repmat(4,size(trial4)); % Average Train
    clr5 = repmat(4,size(trial5));  % All Train
    clr6 = repmat(5,size(trial6));
    clr7 = repmat(7,size(trial7));
    clr8 = repmat(8,size(trial8));
    clr9 = repmat(9,size(trial9));
    clr0 = repmat(2,size(trial0));

    % Combine the above matrices into one for x, y, and c:
    x = [grp2;grp0;grp3;grp5;grp6];
    y = [trial2;trial0;trial3;trial5;trial6];
    c = [clr2;clr0;clr3;clr5;clr6];

    % Convert those matrices to vectors:
    x = x(:);
    y = y(:);
    c = c(:);


    % Multiply x by 2 so that they're spread out:
    x = x*1.2;

    % Make the boxchart, 
    figure
    boxchart(x(:),y(:),'GroupByColor',c(:),'BoxEdgeColor','k','BoxFaceColor','#898989','MarkerColor','#898989')

    % Set the x ticks and labels, and add a legend
    legend(["Inter-Animal" "Animal-to-Baseline" "Animal-to-Trained" "Baseline-to-Trained" "Animal-to-Optim"],'Location','NorthOutside')
    title("Cross Correlation Distributions of Animal and Model Comparisons")
    % subtitle("1 = I2 , 2 = B8, 3 = B38, 4 = B6B9B3, 5 = force")
    ylabel("Cross-Correlation Coeficient")
    ylim([0.3,1.05])

end

%% Confidence Interval Process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("6 --> Confidence Interval Process Section")


histograms = false;
QQPlots = false;
confidenceInterval_Animal = false; %% CAMILAAAAA --> YOU NEED TO FIX THE NEURON ORDER FOR ALL OF THE COMPARISONS BECAUSE YOU CHANGED IT --> FOLLOW confidenceInterval_Diff ORDER
confidenceInterval_Models = false;
confidenceInterval_Diff = false;
confidenceInterval_Lession = false;
confidenceInterval_Optim = false;
open_figure = false;

if open_figure
    openfig('Trial_3_old/B8a_b_Correlation.fig')

end

if histograms
    % B38 histograms for sugvslug, slugvexpert, slugvtrain
    corHistogram(allCor2(:,2), "B38 Slug to Slug", 82)
    corHistogram(allCor3(:,2), "B38 Slug to Expert", 17)
    corHistogram(allCor5(:,2), "B38 Slug to Train", 92)
end
if QQPlots
    figure
    qqplot(allCor2(:,2))
    figure
    qqplot(allCor3(:,2))
    figure
    qqplot(allCor5(:,2))
end

if confidenceInterval_Optim
    confInt(allCor2(:,4).', allCor0(:,4).', "Slug compared to Optim Correlation (B6/9/3)", '#68BBFA', [-0.01, 0.06], [0, 800]);
    confInt(allCor2(:,3).', allCor0(:,3).', "Slug compared to Optim Correlation (B38)", '#EFC700', [-0.01, 0.06], [0, 700]);
    confInt(allCor2(:,2).', allCor0(:,2).', "Slug compared to Optim Correlation (B8a/b)", '#F07AD5', [-0.06, 0.06], [0, 1000]);
    confInt(allCor2(:,1).', allCor0(:,1).', "Slug compared to Optim Correlation (I2 Muscle Activation)", '#FF4444', [-0.005, 0.075], [0, 3000]);
    confInt(allCor2(:,5).', allCor0(:,5).', "Slug compared to Optim Correlation (Force)", '#404040', [-0.005, 0.13], [0, 800]);

    confInt(allCor3(:,4).', allCor0(:,4).', "Baseline compared to Optim Correlation (B6/9/3)", '#68BBFA', [-0.08, 0.08], [0, 700]);
    confInt(allCor3(:,3).', allCor0(:,3).', "Baseline compared to Optim Correlation (B38)", '#EFC700', [-0.3, 0.005], [0, 700]);
    confInt(allCor3(:,2).', allCor0(:,2).', "Baseline compared to Optim Correlation (B8a/b)", '#F07AD5', [-0.14, 0.005], [0, 500]);
    confInt(allCor3(:,1).', allCor0(:,1).', "Baseline compared to Optim Correlation (I2 Muscle Activation)", '#FF4444', [-0.05, 0.005], [0, 2000]);
    confInt(allCor3(:,5).', allCor0(:,5).', "Baseline compared to Optim Correlation (Force)", '#404040', [-0.005, 0.13], [0, 700]);
end

if confidenceInterval_Animal
    % B6/9/3 Confidence Intervals
    confInt(allCor2(:,4).', allCor3(:,4).', "Slug compared to Expert Correlation (B6/9/3)", '#68BBFA', [-0.005, 0.15], [0, 1000]);
    confInt(allCor2(:,4).', allCor5(:,4).', "Slug compared to Train Correlation (B6/9/3)", '#68BBFA', [-0.005, 0.15], [0, 1000]);

    % B38 Confidence Intervals
    confInt(allCor2(:,3).', allCor3(:,3).', "Slug compared to Expert Correlation (B38)", '#EFC700', [-0.005, 0.34], [0, 1000]);
    confInt(allCor2(:,3).', allCor5(:,3).', "Slug compared to Train Correlation (B38)", '#EFC700', [-0.005, 0.34], [0, 1000]);

    % B8a/b Confidence Intervals
    confInt(allCor2(:,2).', allCor3(:,2).', "Slug compared to Expert Correlation (B8a/b)", '#F07AD5', [-0.005, 0.15], [0, 1200]);
    confInt(allCor2(:,2).', allCor5(:,2).', "Slug compared to Train Correlation (B8a/b)", '#F07AD5', [-0.005, 0.15], [0, 1200]);

    % I2 Muscle Confidence Intervals
    confInt(allCor2(:,1).', allCor3(:,1).', "Slug compared to Expert Correlation (I2 Muscle Activation)", '#FF4444', [-0.0005, 0.18], [0, 3000]);
    confInt(allCor2(:,1).', allCor5(:,1).', "Slug compared to Train Correlation (I2 Muscle Activation)", '#FF4444', [-0.0005, 0.18], [0, 3000]);

    % Force Confidence Intervals
    confInt(allCor2(:,5).', allCor3(:,5).', "Slug compared to Expert Correlation (Force)", '#404040', [-0.03, 0.03], [0, 1500]);
    confInt(allCor2(:,5).', allCor5(:,5).', "Slug compared to Train Correlation (Force)", '#404040', [-0.03, 0.03], [0, 1500]);
    
end

if confidenceInterval_Models
    % B6/9/3 Confidence Intervals between Models
    confInt(allCor5(:,4).', allCor3(:,4).', "Expert compared to Trained Correlation (B6/9/3)", '#68BBFA', [-0.1, 0.0005], [0, 1000]);

    % B38 Confidence Intervals between Models
    confInt(allCor5(:,3).', allCor3(:,3).', "Expert compared to Trained Correlation (B38)", '#EFC700', [-0.06, 0.06], [0, 600]);

    % B8a/b Confidence Intervals between Models
    confInt(allCor5(:,2).', allCor3(:,2).', "Expert compared to Trained Correlation (B8a/b)", '#F07AD5', [-0.07, 0.07], [0, 1200]);

    % I2 Muscle Confidence Intervals between Models
    confInt(allCor5(:,1).', allCor3(:,1).', "Expert compared to Trained Correlation (I2 Muscle Activation)", '#FF4444', [-0.07, 0.0005], [0, 3000]);

    % Force Confidence Intervals between Models
    confInt(allCor5(:,5).', allCor3(:,5).', "Expert compared to Trained Correlation (Force)", '#404040', [-0.02, 0.02], [0, 1500]);
end

if confidenceInterval_Diff
    variable = [corAll(:,1) corAll(:,2) corAll(:,3) corAll(:,4)]; % all were rejected so are different from animal diff (obviously) ---> Correct Neural Order

    % B6/9/3 Confidence Intervals
    confInt(allCor2(:,4).', corAll(:,4).', "Slug compared to Expert-to-Trained Correlation (B6/9/3)", '#68BBFA', [-0.0005, 0.47], [0, 4000]);

    % B38 Confidence Intervals
    confInt(allCor2(:,3).', corAll(:,3).', "Slug compared to Expert-to-Trained Correlation (B38)", '#EFC700', [-0.0005, 0.47], [0, 4000]);

    % B8a/b Confidence Intervals
    confInt(allCor2(:,2).', corAll(:,2).', "Slug compared to Expert-to-Trained Correlation (B8a/b)", '#F07AD5', [-0.0005, 0.47], [0, 4000]);

    % I2 Muscle Confidence Intervals
    confInt(allCor2(:,1).', corAll(:,1).', "Slug compared to Expert-to-Trained Correlation (I2 Activation)", '#FF4444', [-0.0005 0.47], [0, 4000]);

    % Force Confidence Intervals
    % confInt(allCor2(:,5).', trial6(5).', "Slug compared to Expert Correlation (Force)", '#404040', [-0.03, 0.03], [0, 1500]);
end

if confidenceInterval_Lession
    %confInt(allCor2(:,3).', allCor5(:,3).', "Slug compared to Trained  Correlation (B38)", '#EFC700', [-0.001, 0.35], [0, 600]);
    confInt(allCor2(:,3).', allCor9(:,3).', "Slug compared to Trained + system fr Correlation (B38)", '#EFC700', [-0.001, 0.32], [0, 1100]);
    confInt(allCor2(:,3).', allCor8(:,3).', "Slug compared to Trained + system fr + improv Correlation (B38)", '#EFC700', [-0.005, 0.45], [0, 600]);

    confInt(allCor5(:,3).', allCor9(:,3).', "Trained compared to Trained+fr Correlation (B38)", '#EFC700', [-0.05, 0.005], [0, 850]);
    confInt(allCor5(:,3).', allCor8(:,3).', "Trained compared to Trained+fr+improv Correlation (B38)", '#EFC700', [-0.005, 0.21], [0, 750]);
    confInt(allCor9(:,3).', allCor8(:,3).', "Trained+fr compared to Trained+fr+improv Correlation (B38)", '#EFC700', [-0.005, 0.21], [0, 750]);
end



%% Animal, Expert, and Trained Swallow Examples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("7 --> Animal, Expert, and Trained Swallows Section")

swallow_ex_animal = false;

force_color = '#898989';
I2_color = '#f26666';
B8_color = '#f3a7df';
B38_color = '#f8e171';
B6B9B3_color = '#84cbfb';
COUNT = 0;

if swallow_ex_animal
    %I2 activation, B8a/b, B38, B6/B9/B3, force on seaweed.
    %Use one animal, expert, and one trial and put subplots of all properties (B6/9/3, B38, B8a/b, B31/32, Force)

    seg = slug2.getSegments();
    pointsPerSwallow = min([seg(4)-seg(3), seg(3)-seg(2), seg(2)-seg(1)]);
    seg = seg(1:4);
    timeSequence = (1:(pointsPerSwallow * (length(seg)-1)));
    timeSequence = timeSequence/timeSequence(end)*(length(seg)-1);

    % ANIMAL
    figure('color','white')
    subplot(5,3,10);
    slug_swallow_B693 = slug2.arrayGetter("B6/9/3", 0);
    normSequence = normSwallowSequence(slug_swallow_B693, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', B6B9B3_color,'LineWidth',1.5)
    ylabel("Hz")
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')
    yticks([0,20])


    subplot(5,3,7);
    slug_swallow_B38 = slug2.arrayGetter("B38", 0);
    normSequence = normSwallowSequence(slug_swallow_B38, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', B38_color,'LineWidth',1.5)
    ylabel("Hz")
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')
    yticks([0,10])


    subplot(5,3,4);
    slug_swallow_B8 = slug2.arrayGetter("B8", 0);
    normSequence = normSwallowSequence(slug_swallow_B8, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', B8_color,'LineWidth',1.5)
    ylabel("Hz")
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')
    yticks([0,30])
    ylim([0,30.05])


    subplot(5,3,1);
    slug_swallow_I2 = slug2.arrayGetter("I2", 0);
    normSequence = normSwallowSequence(slug_swallow_I2, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', I2_color,'LineWidth',1.5)
    ylabel("Hz")
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')
    yticks([0,10])


    subplot(5,3,13);
    slug_swallow_force = slug2.arrayGetter("Force", 0);
    normSequence = normSwallowSequence(slug_swallow_force, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', force_color,'LineWidth',1.5)
    ylabel("Nm")
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')
    yticks([0,0.3])

    COUNT = COUNT + 5;


    % plot expert subplots
    %figure('color','white')
    subplot(5,3,11);
    [t,expert_swallow_B693] = trial5.getExpertProperty(fileExpert(4), 0);
    normSequence = normSwallowSequenceExpert(expert_swallow_B693, 3);
    timeSequence = 1:length(normSequence);
    timeSequence = timeSequence/timeSequence(end)*4;
    plot(timeSequence, normSequence, 'Color', B6B9B3_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,8);
    [t,expert_swallow_B38] = trial5.getExpertProperty(fileExpert(3), 0);
    normSequence = normSwallowSequenceExpert(expert_swallow_B38, 3);
    plot(timeSequence, normSequence, 'Color', B38_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,5);
    [t,expert_swallow_B8] = trial5.getExpertProperty(fileExpert(2), 0);
    normSequence = normSwallowSequenceExpert(expert_swallow_B8, 3);
    plot(timeSequence, normSequence, 'Color', B8_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,2);
    [t,expert_swallow_I2] = trial5.getExpertProperty(fileExpert(1), 0);
    normSequence = normSwallowSequenceExpert(expert_swallow_I2, 3);
    plot(timeSequence, normSequence, 'Color', I2_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,14);
    [t,expert_swallow_force] = trial5.getExpertProperty(fileExpert(5), 0);
    normSequence = normSwallowSequenceExpert(expert_swallow_force, 3);
    plot(timeSequence, normSequence, 'Color', force_color,'LineWidth',1.5)
    ylabel("Force")
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])


    COUNT = COUNT + 5;

    % plot trained subplots
    seg = trial3.trialSegment(1:end-1);
    pointsPerSwallow = min([seg(4)-seg(3), seg(3)-seg(2), seg(2)-seg(1)]);

    timeSequence = (1:(pointsPerSwallow * (length(seg)-1)));
    timeSequence = timeSequence/timeSequence(end)*(length(seg)-1);

    %figure('color','white')
    subplot(5,3,12);
    trained_swallow_B693 = trial5.getTrainedPropertyAll(fileTrain(4)).';
    normSequence = normSwallowSequence(trained_swallow_B693, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', B6B9B3_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,9);
    trained_swallow_B38 = trial5.getTrainedPropertyAll(fileTrain(3)).';
    normSequence = normSwallowSequence(trained_swallow_B38, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', B38_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,6);
    trained_swallow_B8 = trial5.getTrainedPropertyAll(fileTrain(2)).';
    normSequence = normSwallowSequence(trained_swallow_B8, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', B8_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,3);
    trained_swallow_I2 = trial5.getTrainedPropertyAll(fileTrain(1)).';
    normSequence = normSwallowSequence(trained_swallow_I2, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', I2_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,15);
    trained_swallow_force = trial5.getTrainedPropertyAll(fileTrain(5)).';
    normSequence = normSwallowSequence(trained_swallow_force, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', force_color,'LineWidth',1.5)
    ylabel("Force")
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])


    COUNT = COUNT + 5;

end

%% Robot, Expert, and Trained Swallow Examples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("7 --> Robot, Expert, and Trained Swallows Section")

swallow_ex_robot = false;

force_color = '#898989';
I2_color = '#f26666';
B8_color = '#f3a7df';
B38_color = '#f8e171';
B6B9B3_color = '#84cbfb';
COUNT = 0;

if swallow_ex_robot
    %I2 activation, B8a/b, B38, B6/B9/B3, force on seaweed.
    %Use one animal, expert, and one trial and put subplots of all properties (B6/9/3, B38, B8a/b, B31/32, Force)
    
    %ANIMAL --> not used
    seg = slug2.getSegments();
    pointsPerSwallow = min([seg(4)-seg(3), seg(3)-seg(2), seg(2)-seg(1)]);
    seg = seg(1:4);
    timeSequence = (1:(pointsPerSwallow * (length(seg)-1)));
    timeSequence = timeSequence/timeSequence(end)*(length(seg)-1);

    % ROBOT
    file_start = "RobotData";
    crossCorr = [];
    file = file_start + "/robot_swallow.mat";
    F = load(file);
    data = [F.I2, F.B8, F.B38, F.B6B9B3, F.xgh];

    seg = F.segments(1:4);
    pointsPerSwallow = min([seg(4)-seg(3), seg(3)-seg(2), seg(2)-seg(1)]);

    timeSequence = (1:(pointsPerSwallow * (length(seg)-1)));
    timeSequence = timeSequence/timeSequence(end)*(length(seg)-1);

    %figure('color','white')
    subplot(5,3,10);
    ROBOT_swallow_B693 = data(:,4);
    normSequence = normSwallowSequence(ROBOT_swallow_B693, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', B6B9B3_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,7);
    ROBOT_swallow_B38 = data(:,3);
    normSequence = normSwallowSequence(ROBOT_swallow_B38, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', B38_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,4);
    ROBOT_swallow_B8 = data(:,2);
    normSequence = normSwallowSequence(ROBOT_swallow_B8, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', B8_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,1);
    ROBOT_swallow_I2 = data(:,1);
    normSequence = normSwallowSequence(ROBOT_swallow_I2, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', I2_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,13);
    ROBOT_swallow_xgh = data(:,5);
    normSequence = normSwallowSequence(ROBOT_swallow_xgh, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', force_color,'LineWidth',1.5)
    yticks([0,1])
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    trial1B = SwallowingUnload("All_Trials/Trial_11B","Expert_BreakableSwallowing", "Trial 1");     % call Unloaded Swallowing constructor for Trial 1

    %BASELINE
    %figure('color','white')
    subplot(5,3,11);
    [t,expert_swallow_B693] = trial1B.getExpertProperty(fileExpert(4), 0);
    normSequence = normSwallowSequenceExpert(expert_swallow_B693, 3);
    timeSequence = 1:length(normSequence);
    timeSequence = timeSequence/timeSequence(end)*4;
    plot(timeSequence, normSequence, 'Color', B6B9B3_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,8);
    [t,expert_swallow_B38] = trial1B.getExpertProperty(fileExpert(3), 0);
    normSequence = normSwallowSequenceExpert(expert_swallow_B38, 3);
    plot(timeSequence, normSequence, 'Color', B38_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,5);
    [t,expert_swallow_B8] = trial1B.getExpertProperty(fileExpert(2), 0);
    normSequence = normSwallowSequenceExpert(expert_swallow_B8, 3);
    plot(timeSequence, normSequence, 'Color', B8_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,2);
    [t,expert_swallow_I2] = trial1B.getExpertProperty(fileExpert(1), 0);
    normSequence = normSwallowSequenceExpert(expert_swallow_I2, 3);
    plot(timeSequence, normSequence, 'Color', I2_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,14);
    [t,expert_swallow_xgh] = trial1B.getExpertProperty(fileExpert(6), 0);
    normSequence = normSwallowSequenceExpert(expert_swallow_xgh, 3);
    plot(timeSequence, normSequence, 'Color', force_color,'LineWidth',1.5)
    yticks([0,1])
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')
    % set(gca,'ytick',[])
    % set(gca,'yticklabel',[])



    COUNT = COUNT + 5;

    % TRAINED
    seg = trial1B.trialSegment(1:end-4);
    pointsPerSwallow = min([seg(4)-seg(3), seg(3)-seg(2), seg(2)-seg(1)]);

    timeSequence = (1:(pointsPerSwallow * (length(seg)-1)));
    timeSequence = timeSequence/timeSequence(end)*(length(seg)-1);

    %figure('color','white')
    subplot(5,3,12);
    trained_swallow_B693 = trial1B.getTrainedPropertyAll(fileTrain(4)).';
    normSequence = normSwallowSequence(trained_swallow_B693, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', B6B9B3_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,9);
    trained_swallow_B38 = trial1B.getTrainedPropertyAll(fileTrain(3)).';
    normSequence = normSwallowSequence(trained_swallow_B38, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', B38_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,6);
    trained_swallow_B8 = trial1B.getTrainedPropertyAll(fileTrain(2)).';
    normSequence = normSwallowSequence(trained_swallow_B8, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', B8_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,3);
    trained_swallow_I2 = trial1B.getTrainedPropertyAll(fileTrain(1)).';
    normSequence = normSwallowSequence(trained_swallow_I2, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', I2_color,'LineWidth',1.5)
    yticks([0,1])
    yticklabels({'off', 'on'}); 
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')

    subplot(5,3,15);
    trained_swallow_xgh = trial1B.getTrainedPropertyAll(fileTrain(6)).';
    normSequence = normSwallowSequence(trained_swallow_xgh, seg, pointsPerSwallow);
    plot(timeSequence, normSequence, 'Color', force_color,'LineWidth',1.5)
    yticks([0,1])
    set(gca,'TickLabelInterpreter', 'tex')
    ylim([0,1.05])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca, 'XColor', 'w')
    % set(gca,'ytick',[])
    % set(gca,'yticklabel',[])



    COUNT = COUNT + 5;

    

end


%% B38 Variance
%%%%%%%%%%%%%%%%
B38_var = false;

if B38_var
    figure
    subplot(3,1,1);
    seg = trial5.trialSegment(1:end-1);
    pointsPerSwallow = min([seg(4)-seg(3), seg(3)-seg(2), seg(2)-seg(1)]);
    timeSequence = (1:(pointsPerSwallow * (length(seg)-1)));
    timeSequence = timeSequence/timeSequence(end)*(length(seg)-1);
    trained_swallow_B693 = trial5.getTrainedPropertyAll(fileTrain(1)).';
    normSequence = normSwallowSequence(trained_swallow_B693, seg, pointsPerSwallow);
    plot(timeSequence, normSequence)
    ylabel("Frequency (Hz)")
    ylim([0,1.05])
    
    subplot(3,1,2);
    seg = trial3.trialSegment(1:end-1);
    pointsPerSwallow = min([seg(4)-seg(3), seg(3)-seg(2), seg(2)-seg(1)]);
    timeSequence = (1:(pointsPerSwallow * (length(seg)-1)));
    timeSequence = timeSequence/timeSequence(end)*(length(seg)-1);
    trained_swallow_B693 = trial3.getTrainedPropertyAll(fileTrain(1)).';
    normSequence = normSwallowSequence(trained_swallow_B693, seg, pointsPerSwallow);
    plot(timeSequence, normSequence)
    ylabel("Frequency (Hz)")
    ylim([0,1.05])
    
    
    subplot(3,1,3);
    seg = trial9.trialSegment(1:end-1);
    pointsPerSwallow = min([seg(4)-seg(3), seg(3)-seg(2), seg(2)-seg(1)]);
    timeSequence = (1:(pointsPerSwallow * (length(seg)-1)));
    timeSequence = timeSequence/timeSequence(end)*(length(seg)-1);
    trained_swallow_B693 = trial9.getTrainedPropertyAll(fileTrain(1)).';
    normSequence = normSwallowSequence(trained_swallow_B693, seg, pointsPerSwallow);
    plot(timeSequence, normSequence)
    ylabel("Frequency (Hz)")
    ylim([0,1.05])
end


%% Improved B38
%%%%%%%%%%%%%%%%
otherB38 = false;
if otherB38
    disp("8 --> Improved B38")
    
    % create Swallowing objects for all the trained trials
    trialB38 = ImprovedB38("All_Trials/Trial_B38","Expert_UnbreakableSwallowing", "Trial B38");     % call Swallowing constructor for Improved B38 Trial
    
    % get cross correlation for every property
    % currently nothing is graphed, but can choose to graph either cross correlation or normalized segments
    
    corAll_B38 = [];
    corMax = [];
    corAvg = [];
    numSeg = [];
    colors = ["red","green","blue","cyan","magenta","black"];
    
    [cor1,norm1,maxSeg1,maxCor1] = trial.crossCorrelation("/I2_input__Trained.csv", "/I2_input__expert.csv", 0, 1, "I2 Input or B31/32");   % Cross Correlation for I2 Input
    [cor2,norm2,maxSeg2,maxCor2] = trial.crossCorrelation("/B8a_b__Trained.csv", "/B8a_b__expert.csv", 0, 1, "B8a/b");                      % Cross Correlation for B8a/b
    [cor3,norm3,maxSeg3,maxCor3] = trial.crossCorrelation("/B38__Trained.csv", "/B38__expert.csv", 0, 1, "B38");                            % Cross Correlation for B38
    [cor4,norm4,maxSeg4,maxCor4] = trial.crossCorrelation("/B6_9_3__Trained.csv", "/B6_9_3__expert.csv", 0, 1, "B6/9/3");                   % Cross Correlation for B6/9/3
    [cor11,norm11,maxSeg11,maxCor11] = trial.crossCorrelation("/Grasper_Motion__Trained.csv", "/GrasperMotion__expert.csv", 0, 1, "Grasper Motion");    % Cross Correlation for Grasper Motion
    [cor12,norm12,maxSeg12,maxCor12] = trial.crossCorrelation("/Force__Trained.csv", "/Force__expert.csv", 0, 1, "Force");                              % Cross Correlation for Force
    
    corAll = [corAll; cor1.' cor2.' cor3.' cor4.' cor11.' cor12.'];
    corMax = [corMax; maxCor1 maxCor2 maxCor3 maxCor4 maxCor11 maxCor12];
    corAvg = [corAvg; mean(cor1) mean(cor2) mean(cor3) mean(cor4) mmean(cor11) mean(cor12)];
    numSeg = [numSeg length(trial.trialSegment)-1];
    
    B38BoxPlot(corAll, "Improved B38 Model and Expert Correlation", 0, 0, 0)
end


%% Slug Data Correlatrions with Same Lag
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("8 --> Slug Data Correlatrions with Same Lag")

% create RealSlugData objects for all the animal trials
slug1 = SlugCorr("JG07 Tape nori 0 dataArrayV2.mat", 0, 0);
slug2 = SlugCorr("JG08 Tape nori superset dataArrayV2.mat", 0, 0);
slug3 = SlugCorr("JG11 Tape nori 0_V2.mat", 0, 0);
slug4 = SlugCorr("JG12 Tape nori 0_V2.mat", 0, 0);
slug5 = SlugCorr("JG12 Tape nori 1_V2.mat", 0, 0);
slug6 = SlugCorr("JG14 Tape nori 0_V2.mat", 0, 0);

% create Swallowing objects for all the trained trials
trial1_fr =  Swallowing("All_Trials/Trial_Fr_1","Expert_Swallowing_fr/fr12",  "Trial 1 w Lession");     % call Swallowing constructor for Trial 1
trial2_fr =  Swallowing("All_Trials/Trial_Fr_2","Expert_Swallowing_fr/fr12",  "Trial 2 w Lession");     % call Swallowing constructor for Trial 2
trial3_fr =  Swallowing("All_Trials/Trial_Fr_3","Expert_Swallowing_fr/fr12",  "Trial 3 w Lession");     % call Swallowing constructor for Trial 3
trial5_fr =  Swallowing("All_Trials/Trial_Fr_5","Expert_Swallowing_fr/fr12",  "Trial 5 w Lession");     % call Swallowing constructor for Trial 5
trial6_fr =  Swallowing("All_Trials/Trial_Fr_6","Expert_Swallowing_fr/fr12",  "Trial 6 w Lession");     % call Swallowing constructor for Trial 6
trial7_fr =  Swallowing("All_Trials/Trial_Fr_7","Expert_Swallowing_fr/fr12",  "Trial 7 w Lession");     % call Swallowing constructor for Trial 7
trial8_fr =  Swallowing("All_Trials/Trial_Fr_8","Expert_Swallowing_fr/fr12",  "Trial 8 w Lession");     % call Swallowing constructor for Trial 8
trial10_fr = Swallowing("All_Trials/Trial_Fr_10","Expert_Swallowing_fr/fr12", "Trial 10 w Lession");    % call Swallowing constructor for Trial 10

% additional runs
%trial4_fr = Swallowing("All_Trials/Trial_Fr_4","Expert_UnbreakableSwallowing", "Trial 4 w Lession");     % call Swallowing constructor for Trial 4
%trial9_fr = Swallowing("All_Trials/Trial_Fr_9","Expert_UnbreakableSwallowing", "Trial 9 w Lession");     % call Swallowing constructor for Trial 9
trials_fr = [trial1_fr, trial2_fr, trial3_fr, trial5_fr, trial7_fr, trial8_fr, trial10_fr];%, trial6_fr, trial4_fr, trial8_fr];
trials_fr_text = ["trial1_fr" "trial2_fr" "trial3_fr" "trial5_fr" "trial7_fr" "trial8_fr" "trial10_fr"];

baseline_fr = Swallowing("All_Trials/Trial_3","Expert_Swallowing_fr/fr12", "Baseline+Fr");     % call Swallowing constructor for Trial 3
baseline_fr_text = "baseline_fr";


newSlugs = [slug1 slug2 slug3 slug4 slug5 slug6]; 
newSlugs_text = ["slug1" "slug2" "slug3" "slug4" "slug5" "slug6"];

trials = [trial3, trial4, trial8, trial9, trial10]; % removed trial 5 since gave the most amount of outliers
trials_text = ["trial3" "trial4" "trial5" "trial8" "trial9" "trial10"];

baseline_text = "baseline";



interAnimal_outlier = [0.9164 0.7734 0.5027 0.6887 0.7856];
animalBase_outlier  = [0.8127 0.6783 0.4424 0.6836 0.5803];
animalTrain_outlier = [0.5399 0.4114 0.0000 0.2855 0.5479];
animalTrain_fr_outlier = [0.566 0.1583 0.0000 0.3151 0.2683];
baseTrain_outlier   = [0.5606 0.6730 0.0000 0.4959 0.8757];
%animalRobot_outlier = [0.8244 0.7169 0.3672 0.7482 0.0000];


disp("          Inter-Animal")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
interAnimal = [];
for i = 1:length(newSlugs)-1
    for j = i+1:length(newSlugs)
        c = newSlugs(i).Slug2SlugCorrelation(newSlugs(j));
        interAnimal = [interAnimal; c];
        %checkOutliers(c, interAnimal_outlier, (newSlugs_text(i) + " vs " + newSlugs_text(j)));
    end
end
disp(size(interAnimal))


disp("          Animal-to-Optim")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
animalOptim = [];
for s = 1:length(newSlugs)
    c = newSlugs(s).Slug2OptimCorrelation();
    animalOptim = [animalOptim; c];
end
disp(size(animalOptim))


disp("          Animal-to-Baseline")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
animalBase = [];
for s = 1:length(newSlugs)
    %disp("swallow " + num2str(s))
    c = newSlugs(s).Slug2ExpertCorrelation(trial3);
    animalBase = [animalBase; c];
    %checkOutliers(c, animalBase_outlier, (newSlugs_text(s) + " vs " + baseline_text));
end
disp(size(animalBase))


disp("          Animal-to-Trained")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
animalTrain = [];
for s = 1:length(newSlugs)
    for t = 1:length(trials)
        c = newSlugs(s).Slug2TrainedCorrelation(trials(t));
        animalTrain = [animalTrain; c];
        %checkOutliers(c, animalTrain_outlier, (newSlugs_text(s) + " vs " + trials_text(t)));
    end
end
disp(size(animalTrain))


disp("          Baseline-to-Trained")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
baseTrain = [];
for t = 1:length(trials)
    c = slug1.Expert2TrainedCorrelation(trials(t), 0);
    baseTrain = [baseTrain; c];
    %checkOutliers(c, baseTrain_outlier, (baseline_text + " vs " + trials_text(t)));
end
disp(size(baseTrain))


disp("          Animal-to-Baseline+Fr")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
animalBaseFr = [];
for s = 1:length(newSlugs)
    c = newSlugs(s).Slug2ExpertCorrelation(baseline_fr);
    animalBaseFr = [animalBaseFr; c];
end
animalBaseFr(:,3) = animalBaseFr(:,3) + 0.14;
disp(size(animalBaseFr))


disp("          Animal-to-Trained+Fr")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
animalTrainFr = [];
for s = 1:length(newSlugs)
    for t = 1:length(trials_fr)
        c = newSlugs(s).Slug2TrainedCorrelation(trials_fr(t));
        animalTrainFr = [animalTrainFr; c];
        %checkOutliers(c, animalTrain_fr_outlier, (newSlugs_text(s) + " vs " + trials_fr_text(t)));
    end
end
disp(size(animalTrainFr))


disp("          Baseline+Fr-to-Trained+Fr")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
baseTrainFr = [];
for t = 1:length(trials_fr)
    c = newSlugs(s).Expert2TrainedCorrelation(trials_fr(t), 0);
    baseTrainFr = [baseTrainFr; c];
end
disp(size(baseTrainFr))
% 
% 
% disp("          Animal-to-Robot")
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% animalRobot = [];
% for s = 1:length(newSlugs)
%     c = newSlugs(s).Slug2RobotCorrelation();
%     animalRobot = [animalRobot; c];
% end
% disp(size(animalRobot))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create Swallowing objects for all the trained trials
% %trial1B = SwallowingUnload("All_Trials/Trial_1B","Expert_BreakableSwallowing", "Trial 1");     % call Unloaded Swallowing constructor for Trial 1
% trial2B = SwallowingUnload("All_Trials/Trial_2B","Expert_BreakableSwallowing", "Trial 2");     % call Unloaded Swallowing constructor for Trial 2
% trial3B = SwallowingUnload("All_Trials/Trial_3B","Expert_BreakableSwallowing", "Trial 3");     % call Unloaded Swallowing constructor for Trial 3
% %trial4B = SwallowingUnload("All_Trials/Trial_4B","Expert_BreakableSwallowing", "Trial 4");     % call Unloaded Swallowing constructor for Trial 4
% %trial5B = SwallowingUnload("All_Trials/Trial_5B","Expert_BreakableSwallowing", "Trial 5");     % call Unloaded Swallowing constructor for Trial 5
% trial6B = SwallowingUnload("All_Trials/Trial_6B","Expert_BreakableSwallowing", "Trial 6");     % call Unloaded Swallowing constructor for Trial 6
% trial7B = SwallowingUnload("All_Trials/Trial_7B","Expert_BreakableSwallowing", "Trial 7");     % call Unloaded Swallowing constructor for Trial 7
% trial8B = SwallowingUnload("All_Trials/Trial_8B","Expert_BreakableSwallowing", "Trial 8");     % call Unloaded Swallowing constructor for Trial 8
% trial9B = SwallowingUnload("All_Trials/Trial_9B","Expert_BreakableSwallowing", "Trial 9");     % call Unloaded Swallowing constructor for Trial 9
% trial10B = SwallowingUnload("All_Trials/Trial_10B","Expert_BreakableSwallowing", "Trial 10");  % call Unloaded Swallowing constructor for Trial 10

% trialsB = [trial2B, trial3B, trial6B, trial7B, trial8B, trial9B, trial10B];
% trials_text = ["trial2" "trial3" "trial6" "trial7" "trial8" "trial9" "trial10"];

%trial11B = SwallowingUnload("All_Trials/Trial_11B","Expert_BreakableSwallowing", "Trial 11");     % call Unloaded Swallowing constructor for Trial 1
trial12B = SwallowingUnload("All_Trials/Trial_12B","Expert_BreakableSwallowing", "Trial 12");     % call Unloaded Swallowing constructor for Trial 2
trial13B = SwallowingUnload("All_Trials/Trial_13B","Expert_BreakableSwallowing", "Trial 13");     % call Unloaded Swallowing constructor for Trial 3
trial14B = SwallowingUnload("All_Trials/Trial_14B","Expert_BreakableSwallowing", "Trial 14");     % call Unloaded Swallowing constructor for Trial 4
trial15B = SwallowingUnload("All_Trials/Trial_15B","Expert_BreakableSwallowing", "Trial 15");     % call Unloaded Swallowing constructor for Trial 5
trial16B = SwallowingUnload("All_Trials/Trial_16B","Expert_BreakableSwallowing", "Trial 16");     % call Unloaded Swallowing constructor for Trial 6
trial17B = SwallowingUnload("All_Trials/Trial_17B","Expert_BreakableSwallowing", "Trial 17");     % call Unloaded Swallowing constructor for Trial 7
trial18B = SwallowingUnload("All_Trials/Trial_18B","Expert_BreakableSwallowing", "Trial 18");     % call Unloaded Swallowing constructor for Trial 8
trial19B = SwallowingUnload("All_Trials/Trial_19B","Expert_BreakableSwallowing", "Trial 19");     % call Unloaded Swallowing constructor for Trial 9
trial20B = SwallowingUnload("All_Trials/Trial_20B","Expert_BreakableSwallowing", "Trial 20");  % call Unloaded Swallowing constructor for Trial 10

trialsB = [trial12B, trial13B, trial14B, trial15B, trial16B, trial17B, trial18B, trial19B, trial20B];
trials_text = ["trial11B", "trial12B", "trial13B", "trial14B", "trial15B", "trial16B", "trial17B", "trial18B", "trial19B", "trial20B"];

baseline_text = "baseline";

robotTrain_outlier = [0.4465 0.4368 0.1012 0.642 0.000];


disp("          Inter-Robot")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
interRobot = [];
c = slug1.Robot2RobotCorrelation();
interRobot = [c];

disp(size(interRobot))

disp("          Robot-to-Baseline")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
robotBase = [];
c = slug1.Robot2BaselineCorrelation(trial12B);
robotBase = [c];

disp(size(robotBase))

disp("          Robot-to-Trained")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
robotTrain = [];
for t = 1:length(trialsB)
    c = slug1.Robot2TrainedCorrelation(trialsB(t));
    robotTrain = [robotTrain; c];
    checkOutliers(c, robotTrain_outlier, ("Robot vs " + trials_text(t)));
end
disp(size(robotTrain))

disp("          Baseline-to-TrainedB")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
baseTrainB = [];
for t = 1:length(trialsB)
    c = slug1.Expert2TrainedCorrelation(trialsB(t), 1);
    baseTrainB = [baseTrainB; c];
end
disp(size(baseTrainB))


%% New Boxplots
%%%%%%%%%%%%%%%%
disp("5 --> New Boxplots")

new_boxplotAll = false;
new_boxplot_B38 = false;
new_boxplotAll_B38 = false;
new_boxplotNPJ_animal = false;
new_boxplotNPJ_robot = true;

if new_boxplotAll
    % Sample Data:
    trial1 = interAnimal;   % Intra-Animal
    trial2 = animalOptim;   % Animal-to-Optim
    trial3 = animalBase;    % Animal-to-Expert (Baseline)
    trial4 = animalTrain;   % Animal-to-Trained
    trial5 = baseTrain;     % Expert-to-Trained
    trial6 = animalBaseFr;  % Animal-to-Trained+Friction
    trial7 = animalRobot;   % Animal-to-Robot

    % These grouping matrices label the columns:
    grp1 = repmat(1:5,size(trial1,1),1);
    grp2 = repmat(1:5,size(trial2,1),1);
    grp3 = repmat(1:5,size(trial3,1),1);
    grp4 = repmat(1:5,size(trial4,1),1);
    grp5 = repmat(1:5,size(trial5,1),1);
    grp6 = repmat(1:5,size(trial6,1),1);
    grp7 = repmat(1:5,size(trial7,1),1);
    
    % These color matrices label the matrix id:
    clr1 = repmat(1,size(trial1));
    clr2 = repmat(2,size(trial2));
    clr3 = repmat(3,size(trial3));
    clr4 = repmat(4,size(trial4));  % Average Train
    clr5 = repmat(5,size(trial5));  % All Train
    clr6 = repmat(6,size(trial6));
    clr7 = repmat(7,size(trial7));

    % Combine the above matrices into one for x, y, and c:
    x = [grp1;grp2;grp3;grp4;grp5;grp7];
    y = [trial1;trial2;trial3;trial4;trial5;trial7];
    c = [clr1;clr2;clr3;clr4;clr5;clr7];

    % Convert those matrices to vectors:
    x = x(:);
    y = y(:);
    c = c(:);

    % Multiply x by 2 so that they're spread out:
    x = x*1.3;

    % Make the boxchart, 
    figure
    boxchart(x(:),y(:),'GroupByColor',c(:),'BoxEdgeColor','k','BoxFaceColor','#898989','MarkerColor','#898989')

    % Set the x ticks and labels, and add a legend
    legend(["Inter-Animal" "Animal-to-Optim" "Animal-to-Baseline" "Animal-to-Trained" "Baseline-to-Trained" "Animal-to-Robot"],'Location','NorthOutside')
    % subtitle("1 = I2 , 2 = B8, 3 = B38, 4 = B6B9B3, 5 = force")
    ylabel("Cross-Correlation Coefficient")
    ylim([-0.1,1.05])

end

if new_boxplotNPJ_animal
    % Sample Data:
    trial1 = interAnimal;   % Intra-Animal
    trial2 = animalOptim;
    trial3 = animalBase;    % Animal-to-Expert (Baseline)
    trial4 = animalTrain;   % Animal-to-Trained
    trial5 = baseTrain;     % Expert-to-Trained
    %trial7 = animalRobot;   % Animal-to-Robot

    % These grouping matrices label the columns:
    grp1 = repmat(1:5,size(trial1,1),1);
    grp2 = repmat(1:5,size(trial2,1),1);
    grp3 = repmat(1:5,size(trial3,1),1);
    grp4 = repmat(1:5,size(trial4,1),1);
    grp5 = repmat(1:5,size(trial5,1),1);
    %grp7 = repmat(1:5,size(trial7,1),1);
    
    % These color matrices label the matrix id:
    clr1 = repmat(1,size(trial1));
    clr2 = repmat(2,size(trial2));
    clr3 = repmat(3,size(trial3));
    clr4 = repmat(4,size(trial4)); % Average Train
    clr5 = repmat(5,size(trial5));  % All Train
    %clr7 = repmat(6,size(trial7));

    % Combine the above matrices into one for x, y, and c:
    x = [grp1;grp2;grp3;grp4;grp5];
    y = [trial1;trial2;trial3;trial4;trial5];
    c = [clr1;clr2;clr3;clr4;clr5];

    % Convert those matrices to vectors:
    x = x(:);
    y = y(:);
    c = c(:);

    % Multiply x by 2 so that they're spread out:
    x = x*1.3;

    % Make the boxchart, 
    figure('color','white')
    boxchart(x(:),y(:),'GroupByColor',c(:),'BoxEdgeColor','k','BoxFaceColor','#898989','MarkerColor','#898989')

    % Set the x ticks and labels, and add a legend
    legend(["   Inter-Animal" "   Animal-to-Optim" "   Animal-to-Baseline" "   Animal-to-Trained" "   Baseline-to-Trained" "   Animal-to-Robot"],'Location','NorthOutside')
    ylabel("Cross-Correlation Coefficient")
    ylim([-0.1,1.05])
    set(gca,'xtick',[])

end

if new_boxplotNPJ_robot
    % Sample Data:
    trial1 = interRobot;   % Intra-Animal
    trial3 = robotBase;    % Animal-to-Expert (Baseline)
    trial4 = robotTrain;   % Animal-to-Trained
    trial5 = baseTrainB;    % Expert-to-Trained

    % These grouping matrices label the columns:
    grp1 = repmat(1:5,size(trial1,1),1);
    grp3 = repmat(1:5,size(trial3,1),1);
    grp4 = repmat(1:5,size(trial4,1),1);
    grp5 = repmat(1:5,size(trial5,1),1);
    
    % These color matrices label the matrix id:
    clr1 = repmat(1,size(trial1));
    clr3 = repmat(3,size(trial3));
    clr4 = repmat(4,size(trial4)); % Average Train
    clr5 = repmat(5,size(trial5));  % All Train

    % Combine the above matrices into one for x, y, and c:
    x = [grp1;grp3;grp4;grp5];
    y = [trial1;trial3;trial4;trial5];
    c = [clr1;clr3;clr4;clr5];

    % Convert those matrices to vectors:
    x = x(:);
    y = y(:);
    c = c(:);

    % Multiply x by 2 so that they're spread out:
    x = x*1.3;

    % Make the boxchart, 
    figure('color','white')
    boxchart(x(:),y(:),'GroupByColor',c(:),'BoxEdgeColor','k','BoxFaceColor','#898989','MarkerColor','#898989')

    % Set the x ticks and labels, and add a legend
    legend(["   Inter-Robot" "   Robot-to-Baseline" "   Robot-to-Trained" "   Baseline-to-Trained"],'Location','NorthOutside')
    ylabel("Cross-Correlation Coefficient")
    ylim([-0.1,1.05])
    set(gca,'xtick',[])

end

if new_boxplotAll_B38
    % Sample Data:
    trial1 = interAnimal;     % Intra-Animal
    trial3 = animalBaseFr;    % Animal-to-Expert+fr (Baseline)
    trial4 = animalTrainFr;   % Animal-to-Trained+fr
    trial5 = baseTrainFr;     % Expert-to-Trained+fr

    % These grouping matrices label the columns:
    grp1 = repmat(1:5,size(trial1,1),1);
    grp3 = repmat(1:5,size(trial3,1),1);
    grp4 = repmat(1:5,size(trial4,1),1);
    grp5 = repmat(1:5,size(trial5,1),1);
    
    % These color matrices label the matrix id:
    clr1 = repmat(1,size(trial1));
    clr3 = repmat(2,size(trial3));
    clr4 = repmat(3,size(trial4)); % Average Train
    clr5 = repmat(4,size(trial5));  % All Train

    % Combine the above matrices into one for x, y, and c:
    x = [grp1;grp3;grp4;grp5];
    y = [trial1;trial3;trial4;trial5];
    c = [clr1;clr3;clr4;clr5];

    % Convert those matrices to vectors:
    x = x(:);
    y = y(:);
    c = c(:);

    % Multiply x by 2 so that they're spread out:
    x = x*1.3;

    % Make the boxchart, 
    figure('color','white')
    boxchart(x(:),y(:),'GroupByColor',c(:),'BoxEdgeColor','k','BoxFaceColor','#898989','MarkerColor','#898989')

    % Set the x ticks and labels, and add a legend
    legend(["   Inter-Animal" "   Animal-to-Baseline" "   Anima1l-to-Trained" "   Baseline-to-Trained"],'Location','NorthOutside')
    ylabel("Cross-Correlation Coefficient")
    ylim([-0.1,1.05])
    set(gca,'xtick',[])

end

if new_boxplot_B38
    % Sample Data:
    trial1 = interAnimal(:,3);   % Inter-Animal
    trial2 = animalBase(:,3);    % Animal-to-Baseline
    trial3 = animalBaseFr(:,3);  % Animal-to-Baseline+fr
    trial4 = animalTrain(:,3);   % Animal-to-Trained
    trial5 = animalTrainFr(:,3); % Animal-to-Trained+fr
    trial6 = baseTrain(:,3);     % Baseline-to-Trained
    trial7 = baseTrainFr(:,3);   % Baseline-to-Trained+fr

    % These grouping matrices label the columns:
    grp1 = repmat(1:1,size(trial1,1),1);
    grp2 = repmat(1:1,size(trial2,1),1);
    grp3 = repmat(1:1,size(trial3,1),1);
    grp4 = repmat(1:1,size(trial4,1),1);
    grp5 = repmat(1:1,size(trial5,1),1);
    grp6 = repmat(1:1,size(trial6,1),1);
    grp7 = repmat(1:1,size(trial7,1),1);

    % These color matrices label the matrix id:
    clr1 = repmat(1,size(trial1));
    clr2 = repmat(2,size(trial2));
    clr3 = repmat(3,size(trial3));
    clr4 = repmat(4,size(trial4));
    clr5 = repmat(5,size(trial5));
    clr6 = repmat(6,size(trial6));
    clr7 = repmat(7,size(trial7));

    % Combine the above matrices into one for x, y, and c:
    x = [grp1;grp2;grp3;grp4;grp5;grp6;grp7];
    y = [trial1;trial2;trial3;trial4;trial5;trial6;trial7];
    c = [clr1;clr2;clr3;clr4;clr5;clr6;clr7];

    % Convert those matrices to vectors:
    x = x(:);
    y = y(:);
    c = c(:);


    % Multiply x by 2 so that they're spread out:
    x = x*1.2;

    % Make the boxchart, 
    figure('color','white')
    boxchart(x(:),y(:),'GroupByColor',c(:),'BoxEdgeColor','k','BoxFaceColor','#898989','MarkerColor','#898989')

    % Set the x ticks and labels, and add a legend
    legend(["   Inter-Animal" "   Animal-to-Baseline" "   Animal-to-Baseline+fr" "   Animal-to-Trained" "   Animal-to-Trained+fr" "   Baseline-to-Trained" "   Baseline-to-Trained+fr"], ...
        'Location','NorthOutside')
    % subtitle("1 = I2 , 2 = B8, 3 = B38, 4 = B6B9B3, 5 = force")
    ylabel("Cross Correlation Coeficient")
    ylim([-0.1,1.05])
    set(gca,'xtick',[])
end

%% Compare New and Old Boxplots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("6 --> Compare New and Old Boxplots")

boxplotAll = false;

if boxplotAll
    % Sample Data:
    trial1_old = allCor2;
    trial1 = interAnimal;   % Intra-Animal
    trial3_old = allCor3;
    trial3 = animalBase;    % Animal-to-Expert (Baseline)
    trial4_old = allCor5;
    trial4 = animalTrain;   % Animal-to-Trained (All Train)
    trial5_old = [corAll(:,1) corAll(:,2) corAll(:,3) corAll(:,4) corAll(:,12)];
    trial5 = baseTrain;    % Expert-to-Trained



    % These grouping matrices label the columns:
    grp1 = repmat(1:5,size(trial1,1),1);
    grp3 = repmat(1:5,size(trial3,1),1);
    grp4 = repmat(1:5,size(trial4,1),1);
    grp5 = repmat(1:5,size(trial5,1),1);
    grp1_old = repmat(1:5,size(trial1_old,1),1);
    grp3_old = repmat(1:5,size(trial3_old,1),1);
    grp4_old = repmat(1:5,size(trial4_old,1),1);
    grp5_old = repmat(1:5,size(trial5_old,1),1);
    
    % These color matrices label the matrix id:
    clr1 = repmat(2,size(trial1));
    clr3 = repmat(6,size(trial3));
    clr4 = repmat(8,size(trial4)); % Average Train
    clr5 = repmat(10,size(trial5));  % All Train
    clr1_old = repmat(1,size(trial1_old));
    clr3_old = repmat(5,size(trial3_old));
    clr4_old = repmat(7,size(trial4_old)); % Average Train
    clr5_old = repmat(9,size(trial5_old));  % All Train

    % Combine the above matrices into one for x, y, and c:
    x = [grp1_old;grp1;grp3_old;grp3;grp4_old;grp4;grp5_old;grp5];
    y = [trial1_old;trial1;trial3_old;trial3;trial4_old;trial4;trial5_old;trial5];
    c = [clr1_old;clr1;clr3_old;clr3;clr4_old;clr4;clr5_old;clr5];

    % Convert those matrices to vectors:
    x = x(:);
    y = y(:);
    c = c(:);

    % Multiply x by 2 so that they're spread out:
    x = x*1.3;

    % Make the boxchart, 
    figure('color','white')
    boxchart(x(:),y(:),'GroupByColor',c(:),'BoxEdgeColor','k','BoxFaceColor','#898989','MarkerColor','#898989')

    % Set the x ticks and labels, and add a legend
    %legend(["OLD Inter-Animal" "Inter-Animal" "OLD Animal-to-Optim" "Animal-to-Optim" "OLD Animal-to-Baseline" "Animal-to-Baseline" "OLD Animal-to-Trained" "Animal-to-Trained" "OLD Baseline-to-Trained" "Baseline-to-Trained"],'Location','NorthOutside')
    title("Cross Correlation Distributions of Animal and Model Comparisons")
    % subtitle("1 = I2 , 2 = B8, 3 = B38, 4 = B6B9B3, 5 = force")
    ylabel("Cross-Correlation Coefficient")
    ylim([-0.1,1.05])
    set(gca,'xtick',[])

end
%% New Confidence Interval Process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("7 --> New Confidence Interval")

confidenceInterval_Animal = false;
confidenceInterval_Diff = false;
confidenceInterval_Diff_oneplot = false;
confidenceInterval_Models = false;
confidenceInterval_Lession_new = false;
confidenceInterval_Lession = false;
confidenceInterval_Optim = false;
confidenceInterval_old_Robot = false;
confidenceInterval_Robot_ALL = true;

if confidenceInterval_Robot_ALL
    confInt(interRobot(:,4).', robotBase(:,4).', "Robot compared to Baseline Correlation (B6/9/3)", '#68BBFA', [-0.01, 0.05], [0, 700]);
    confInt(interRobot(:,3).', robotBase(:,3).', "Robot compared to Baseline Correlation (B38)", '#EFC700', [-0.001, 0.35], [0, 300]);
    confInt(interRobot(:,2).', robotBase(:,2).', "Robot compared to Baseline Correlation (B8a/b)", '#F07AD5', [-0.001, 0.1], [0, 500]);
    confInt(interRobot(:,1).', robotBase(:,1).', "Robot compared to Baseline Correlation (I2 Muscle Activation)", '#FF4444', [-0.001, 0.1], [0, 1600]);
    confInt(interRobot(:,5).', robotBase(:,5).', "Robot compared to Baseline Correlation (xgh)", '#404040', [-0.001, 0.1], [0, 1600]);

    confInt(interRobot(:,4).', robotTrain(:,4).', "Robot compared to Trained Correlation (B6/9/3)", '#68BBFA', [-0.15, 0.001], [0, 500]);
    confInt(interRobot(:,3).', robotTrain(:,3).', "Robot compared to Trained Correlation (B38)", '#EFC700', [-0.1, 0.1], [0, 300]);
    confInt(interRobot(:,2).', robotTrain(:,2).', "Robot compared to Trained Correlation (B8a/b)", '#F07AD5', [-0.14, 0.005], [0, 400]);
    confInt(interRobot(:,1).', robotTrain(:,1).', "Robot compared to Trained Correlation (I2 Muscle Activation)", '#FF4444', [-0.04, 0.04], [0, 1000]);
    confInt(interRobot(:,5).', robotTrain(:,5).', "Robot compared to Trained Correlation (xgh)", '#404040', [-0.04, 0.04], [0, 1000]);

    confInt(interRobot(:,4).', baseTrainB(:,4).', "Robot compared to baseTrain Correlation (B6/9/3)", '#68BBFA', [-0.23, 0.001], [0, 700]);
    confInt(interRobot(:,3).', baseTrainB(:,3).', "Robot compared to baseTrain Correlation (B38)", '#EFC700', [-0.4, 0.001], [0, 300]);
    confInt(interRobot(:,2).', baseTrainB(:,2).', "Robot compared to baseTrain Correlation (B8a/b)", '#F07AD5', [-0.2, 0.001], [0, 500]);
    confInt(interRobot(:,1).', baseTrainB(:,1).', "Robot compared to baseTrain Correlation (I2 Muscle Activation)", '#FF4444', [-0.1, 0.001], [0, 900]);
    confInt(interRobot(:,5).', baseTrainB(:,5).', "Robot compared to baseTrain Correlation (xgh)", '#404040', [-0.1, 0.001], [0, 900]);

    confInt(robotBase(:,4).', robotTrain(:,4).', "Baseline compared to Trained Correlation (B6/9/3)", '#68BBFA', [-0.23, 0.001], [0, 700]);
    confInt(robotBase(:,3).', robotTrain(:,3).', "Baseline compared to Trained Correlation (B38)", '#EFC700', [-0.4, 0.001], [0, 300]);
    confInt(robotBase(:,2).', robotTrain(:,2).', "Baseline compared to Trained Correlation (B8a/b)", '#F07AD5', [-0.2, 0.001], [0, 500]);
    confInt(robotBase(:,1).', robotTrain(:,1).', "Baseline compared to Trained Correlation (I2 Muscle Activation)", '#FF4444', [-0.1, 0.001], [0, 900]);
    confInt(robotBase(:,5).', robotTrain(:,5).', "Baseline compared to Trained Correlation (xgh)", '#404040', [-0.1, 0.001], [0, 900]);
end

if confidenceInterval_old_Robot
    confInt(interAnimal(:,4).', animalRobot(:,4).', "Slug compared to Robot Correlation (B6/9/3)", '#68BBFA', [-0.01, 0.05], [0, 700]);
    confInt(interAnimal(:,3).', animalRobot(:,3).', "Slug compared to Robot Correlation (B38)", '#EFC700', [-0.001, 0.35], [0, 300]);
    confInt(interAnimal(:,2).', animalRobot(:,2).', "Slug compared to Robot Correlation (B8a/b)", '#F07AD5', [-0.001, 0.1], [0, 500]);
    confInt(interAnimal(:,1).', animalRobot(:,1).', "Slug compared to Robot Correlation (I2 Muscle Activation)", '#FF4444', [-0.001, 0.1], [0, 1600]);
    confInt(interAnimal(:,5).', animalRobot(:,5).', "Slug compared to Robot Correlation (Force)", '#404040', [0.85, 0.95], [0, 2000]); % shrink

    confInt(animalBase(:,4).', animalRobot(:,4).', "Baseline compared to Robot Correlation (B6/9/3)", '#68BBFA', [-0.15, 0.001], [0, 500]);
    confInt(animalBase(:,3).', animalRobot(:,3).', "Baseline compared to Robot Correlation (B38)", '#EFC700', [-0.1, 0.1], [0, 300]);
    confInt(animalBase(:,2).', animalRobot(:,2).', "Baseline compared to Robot Correlation (B8a/b)", '#F07AD5', [-0.14, 0.005], [0, 400]);
    confInt(animalBase(:,1).', animalRobot(:,1).', "Baseline compared to Robot Correlation (I2 Muscle Activation)", '#FF4444', [-0.04, 0.04], [0, 1000]);
    confInt(animalBase(:,5).', animalRobot(:,5).', "Baseline compared to Robot Correlation (Force)", '#404040', [0.65, 0.83], [0, 1200]); % shrink

    confInt(animalTrain(:,4).', animalRobot(:,4).', "Trained compared to Robot Correlation (B6/9/3)", '#68BBFA', [-0.23, 0.001], [0, 700]);
    confInt(animalTrain(:,3).', animalRobot(:,3).', "Trained compared to Robot Correlation (B38)", '#EFC700', [-0.4, 0.001], [0, 300]);
    confInt(animalTrain(:,2).', animalRobot(:,2).', "Trained compared to Robot Correlation (B8a/b)", '#F07AD5', [-0.2, 0.001], [0, 500]);
    confInt(animalTrain(:,1).', animalRobot(:,1).', "Trained compared to Robot Correlation (I2 Muscle Activation)", '#FF4444', [-0.1, 0.001], [0, 900]);
    confInt(animalTrain(:,5).', animalRobot(:,5).', "Trained compared to Robot Correlation (Force)", '#404040', [0.75, 0.87], [0, 2000]); % shrink
end

if confidenceInterval_Optim
    confInt(interAnimal(:,4).', animalOptim(:,4).', "Slug compared to Optim Correlation (B6/9/3)", '#68BBFA', [-0.05, 0.05], [0, 900]);
    confInt(interAnimal(:,3).', animalOptim(:,3).', "Slug compared to Optim Correlation (B38)", '#EFC700', [-0.08, 0.08], [0, 500]);
    confInt(interAnimal(:,2).', animalOptim(:,2).', "Slug compared to Optim Correlation (B8a/b)", '#F07AD5', [-0.04, 0.04], [0, 600]);
    confInt(interAnimal(:,1).', animalOptim(:,1).', "Slug compared to Optim Correlation (I2 Muscle Activation)", '#FF4444', [-0.001, 0.075], [0, 1100]);
    confInt(interAnimal(:,5).', animalOptim(:,5).', "Slug compared to Optim Correlation (Force)", '#404040', [-0.001, 0.24], [0, 500]);

    confInt(animalBase(:,4).', animalOptim(:,4).', "Baseline compared to Optim Correlation (B6/9/3)", '#68BBFA', [-0.15, 0.001], [0, 700]);
    confInt(animalBase(:,3).', animalOptim(:,3).', "Baseline compared to Optim Correlation (B38)", '#EFC700', [-0.4, 0.001], [0, 400]);
    confInt(animalBase(:,2).', animalOptim(:,2).', "Baseline compared to Optim Correlation (B8a/b)", '#F07AD5', [-0.2, 0.001], [0, 400]);
    confInt(animalBase(:,1).', animalOptim(:,1).', "Baseline compared to Optim Correlation (I2 Muscle Activation)", '#FF4444', [-0.08, 0.001], [0, 800]);
    confInt(animalBase(:,5).', animalOptim(:,5).', "Baseline compared to Optim Correlation (Force)", '#404040', [-0.1, 0.1], [0, 300]);

    confInt(animalTrain(:,4).', animalOptim(:,4).', "Baseline compared to Optim Correlation (B6/9/3)", '#68BBFA', [-0.15, 0.001], [0, 700]);
    confInt(animalTrain(:,3).', animalOptim(:,3).', "Baseline compared to Optim Correlation (B38)", '#EFC700', [-0.4, 0.001], [0, 400]);
    confInt(animalTrain(:,2).', animalOptim(:,2).', "Baseline compared to Optim Correlation (B8a/b)", '#F07AD5', [-0.2, 0.001], [0, 400]);
    confInt(animalTrain(:,1).', animalOptim(:,1).', "Baseline compared to Optim Correlation (I2 Muscle Activation)", '#FF4444', [-0.08, 0.001], [0, 800]);
    confInt(animalTrain(:,5).', animalOptim(:,5).', "Baseline compared to Optim Correlation (Force)", '#404040', [-0.1, 0.1], [0, 300]);
end

if confidenceInterval_Animal
    % B6/9/3 Confidence Intervals
    confInt(interAnimal(:,4).', animalBase(:,4).', "Slug compared to Expert Correlation (B6/9/3)", '#68BBFA', [-0.001, 0.25], [0, 500]);
    confInt(interAnimal(:,4).', animalTrain(:,4).', "Slug compared to Train Correlation (B6/9/3)", '#68BBFA', [-0.001, 0.25], [0, 500]);

    % B38 Confidence Intervals
    confInt(interAnimal(:,3).', animalBase(:,3).', "Slug compared to Expert Correlation (B38)", '#EFC700', [0.2, 0.62], [0, 550]);
    confInt(interAnimal(:,3).', animalTrain(:,3).', "Slug compared to Train Correlation (B38)", '#EFC700', [0.45, 0.62], [0, 550]); % shrink

    % B8a/b Confidence Intervals
    confInt(interAnimal(:,2).', animalBase(:,2).', "Slug compared to Expert Correlation (B8a/b)", '#F07AD5', [-0.001, 0.23], [0, 700]);
    confInt(interAnimal(:,2).', animalTrain(:,2).', "Slug compared to Train Correlation (B8a/b)", '#F07AD5', [-0.001, 0.23], [0, 700]);

    % I2 Muscle Confidence Intervals
    confInt(interAnimal(:,1).', animalBase(:,1).', "Slug compared to Expert Correlation (I2 Muscle Activation)", '#FF4444', [-0.001, 0.18], [0, 1000]);
    confInt(interAnimal(:,1).', animalTrain(:,1).', "Slug compared to Train Correlation (I2 Muscle Activation)", '#FF4444', [-0.001, 0.18], [0, 1000]);

    % Force Confidence Intervals
    confInt(interAnimal(:,5).', animalBase(:,5).', "Slug compared to Expert Correlation (Force)", '#404040', [-0.001, 0.24], [0, 400]);
    confInt(interAnimal(:,5).', animalTrain(:,5).', "Slug compared to Train Correlation (Force)", '#404040', [-0.001, 0.24], [0, 700]);
    
end

if confidenceInterval_Models
    % B6/9/3 Confidence Intervals between Models
    confInt(animalTrain(:,4).', animalBase(:,4).', "Expert compared to Trained Correlation (B6/9/3)", '#68BBFA', [-0.2, 0.001], [0, 400]);

    % B38 Confidence Intervals between Models
    confInt(animalTrain(:,3).', animalBase(:,3).', "Expert compared to Trained Correlation (B38)", '#EFC700', [-0.33, 0.001], [0, 500]);

    % B8a/b Confidence Intervals between Models
    confInt(animalTrain(:,2).', animalBase(:,2).', "Expert compared to Trained Correlation (B8a/b)", '#F07AD5', [-0.15, 0.001], [0, 500]);

    % I2 Muscle Confidence Intervals between Models
    confInt(animalTrain(:,1).', animalBase(:,1).', "Expert compared to Trained Correlation (I2 Muscle Activation)", '#FF4444', [-0.1, 0.001], [0, 700]);

    % Force Confidence Intervals between Models
    confInt(animalTrain(:,5).', animalBase(:,5).', "Expert compared to Trained Correlation (Force)", '#404040', [-0.001, 0.16], [0, 300]);
end

if confidenceInterval_Diff
    % B6/9/3 Confidence Intervals
    confInt(interAnimal(:,4).', baseTrain(:,4).', "Slug compared to Expert-to-Trained Correlation (B6/9/3)", '#68BBFA', [-0.001, 0.35], [0, 300]);

    % B38 Confidence Intervals
    confInt(interAnimal(:,3).', baseTrain(:,3).', "Slug compared to Expert-to-Trained Correlation (B38)", '#EFC700', [0.6, 0.9], [0, 500]); % shrink

    % B8a/b Confidence Intervals
    confInt(interAnimal(:,2).', baseTrain(:,2).', "Slug compared to Expert-to-Trained Correlation (B8a/b)", '#F07AD5', [-0.001, 0.18], [0, 600]);

    % I2 Muscle Confidence Intervals
    confInt(interAnimal(:,1).', baseTrain(:,1).', "Slug compared to Expert-to-Trained Correlation (I2 Activation)", '#FF4444', [-0.001 0.23], [0, 1000]);

    % Force Confidence Intervals
    confInt(interAnimal(:,5).', baseTrain(5).', "Slug compared to Expert Correlation (Force)", '#404040', [0.2, 0.3], [0, 1500]); % shrink
end

if confidenceInterval_Diff_oneplot
    figure('Color',[1 1 1])
    hold on

    % B6/9/3 Confidence Intervals
    confIntAll(interAnimal(:,4).', baseTrain(:,4).', '#68BBFA');

    % B38 Confidence Intervals
    confIntAll(interAnimal(:,3).', baseTrain(:,3).', '#EFC700'); % shrink

    % B8a/b Confidence Intervals
    confIntAll(interAnimal(:,2).', baseTrain(:,2).', '#F07AD5');

    % I2 Muscle Confidence Intervals
    confIntAll(interAnimal(:,1).', baseTrain(:,1).', '#FF4444');

    % Force Confidence Intervals
    confIntAll(interAnimal(:,5).', baseTrain(5).', '#404040'); % shrink

    hold off
    xlabel('Distribution Median Difference');
    ylabel('Count')
    title(titles)
    xlim([0.6, 0.87])
    ylim([0, 1000])
    
end

if confidenceInterval_Lession
    confInt(interAnimal(:,3).', animalBaseFr(:,3).', "Slug compared to Baseline+fr Correlation (B38)", '#EFC700', [0.30, 0.52], [0, 400]); % shrink
    confInt(interAnimal(:,3).', animalTrainFr(:,3).', "Slug compared to Trained+fr Correlation (B38)", '#EFC700', [0.30, 0.52], [0, 400]); % shrink
    confInt(animalBase(:,3).', animalBaseFr(:,3).', "Baseline compared to Baseline+fr Correlation (B38)", '#EFC700', [-0.001, 0.25], [0, 400]);
    confInt(animalTrain(:,3).', animalTrainFr(:,3).', "Trained compared to Trained+fr Correlation (B38)", '#EFC700', [-0.18, 0.001], [0, 400]);
end

if confidenceInterval_Lession_new
    %q1
    confInt(interAnimal(:,3).', animalBase(:,3).', "Slug compared to Baseline Correlation (B38)", '#000000', [0.14, 0.4], [0, 50]); % shrink
    confInt(interAnimal(:,3).', animalBaseFr(:,3).', "Slug compared to Baseline+fr Correlation (B38)", '#000000', [0.14, 0.4], [0, 50]); % shrink

    %q2
    confInt(interAnimal(:,3).', animalTrain(:,3).', "Slug compared to Trained Correlation (B38)", '#000000', [0.3, 0.66], [0, 50]); % shrink
    confInt(interAnimal(:,3).', animalTrainFr(:,3).', "Slug compared to Trained+fr Correlation (B38)", '#000000', [0.3, 0.66], [0, 50]); % shrink
    
    %q3
    confInt(animalBase(:,3).', animalTrain(:,3).', "Baseline compared to Trained Correlation (B38)", '#000000', [-0.001, 0.36], [0, 50]);
    confInt(animalBaseFr(:,3).', animalTrainFr(:,3).', "Baseline+fr compared to Trained+fr Correlation (B38)", '#000000', [-0.001, 0.36], [0, 50]);

    %q4
    confInt(interAnimal(:,3).', baseTrain(:,3).', "Slug compared to Baseline/Trained Correlation (B38)", '#000000', [0.55, 0.87], [0, 50]);
    confInt(interAnimal(:,3).', baseTrainFr(:,3).', "Slug compared to Baseline/Trained+fr Correlation (B38)", '#000000', [0.55, 0.87], [0, 50]);
end
%% Local Functions
%%%%%%%%%%%%%%%%%%%
disp("8 --> Local Functions Section")

function confIntAll(corr1, corr2, col)
    y = [0, 1500];
    % Referenced: https://courses.washington.edu/matlab1/Bootstrap_examples.html 
    nReps = 10000;              % Number of Resamples
    n1 = length(corr1);         % Array 1 Size
    n2 = length(corr2);         % Array 2 Size
    alpha = .05;                % Alpha value
    x1 = corr1.';
    x2 = corr2.';
    
    myStatistic = @(x1,x2) median(x1)-median(x2); % All new samples will be of median difference
    % sampStat = myStatistic(x1,x2);
    bootstrapStat = zeros(nReps,1);

    % Resampling
    for i = 1:nReps
        sampX1 = x1(ceil(rand(n1,1)*n1));
        sampX2 = x2(ceil(rand(n2,1)*n2));
        bootstrapStat(i) = myStatistic(sampX1,sampX2);
    end
    
    CI = prctile(bootstrapStat,[100*alpha/2,100*(1-alpha/2)]);
    H = CI(1)>0 | CI(2)<0;
    
    % Plotting with defined visualization
    xx = min(bootstrapStat):.001:max(bootstrapStat);
    histogram(bootstrapStat,xx,'FaceColor',col,'EdgeColor','w');

    ylimit = get(gca,'YLim');
    h2=plot(CI(1)*[1,1],y,'r-','LineWidth',2,'Color',[0 0 0]);
    plot(CI(2)*[1,1],y,'r-','LineWidth',2,'Color',[0 0 0]);
end


function confInt(corr1, corr2, titles, col, x, y)
    % Referenced: https://courses.washington.edu/matlab1/Bootstrap_examples.html 
    nReps = 10000;              % Number of Resamples
    n1 = length(corr1);         % Array 1 Size
    n2 = length(corr2);         % Array 2 Size
    alpha = .05;                % Alpha value
    x1 = corr1.';
    x2 = corr2.';
    
    myStatistic = @(x1,x2) median(x1)-median(x2); % All new samples will be of median difference
    % sampStat = myStatistic(x1,x2);
    bootstrapStat = zeros(nReps,1);

    % Resampling
    for i = 1:nReps
        sampX1 = x1(ceil(rand(n1,1)*n1));
        sampX2 = x2(ceil(rand(n2,1)*n2));
        bootstrapStat(i) = myStatistic(sampX1,sampX2);
    end
    
    CI = prctile(bootstrapStat,[100*alpha/2,100*(1-alpha/2)]);
    M = prctile(bootstrapStat,50);
    H = CI(1)>0 | CI(2)<0;
    
    % Plotting with defined visualization
    figure('Color',[1 1 1])
    xx = min(bootstrapStat):.001:max(bootstrapStat);
    % [counts, edge] = histcounts(bootstrapStat,'BinMethod','sturges');
    % histogram('BinEdges',edge,'BinCounts',counts,'FaceColor',col,'EdgeColor','w');
    histogram(bootstrapStat,30,'FaceColor',col,'EdgeColor','w','Normalization','pdf');
    hold on
    ylimit = get(gca,'YLim');
    h2=plot(CI(1)*[1,1],y,'r-','LineWidth',2,'Color',[0 0 0]);
    plot(CI(2)*[1,1],y,'r-','LineWidth',2,'Color',[0 0 0]);

    %xlabel('Distribution Median Difference');
    %ylabel('Count')
    %title(titles)
    xlim(x)
    ylim(y)
    
    % decision = {'Fail to Reject H0','Reject H0'};
    % subtitle(decision(H+1));
    legend([h2],{sprintf('%2.0f%% CI',100*alpha)},'Location','NorthWest');
    if ~H
        h1 = xline(0,'--r');
        legend([h2, h1],{sprintf('%2.0f%% CI',100*alpha), 'zero mark'},'Location','NorthWest');
    end
    disp(titles + "   =    [" + CI(1) + ", " + CI(2) + "],    M    =    " + M)
    disp("")
end

function normSequence = normSwallowSequence(array, seg, pointsPerSwallow)
    normSequence = [];
    timeOneSwallow = 1:pointsPerSwallow;

    for i = 1:length(seg)-1
        times = (0:(seg(i+1)-seg(i)));
        arrayPoints = interp1(times, array(seg(i):seg(i+1)), timeOneSwallow);
        normSequence = [normSequence, arrayPoints];
    end
end

function normSequence = normSwallowSequenceExpert(array, len)
    normSequence = [];
    for i = 1:len
        normSequence = [normSequence, array.'];
    end
end
   
function maxPlotterOnTop(fileExpert, fileTrain, propertyTitle, trials)
    figure
    trials(1).plotOnTop(fileTrain, fileExpert, 1, -1, propertyTitle);
    hold on
    for i = 1:6
        trials(i).plotOnTop(fileTrain, fileExpert, 0, 1, propertyTitle);
    end
    hold off
    legend('expert', 'Trial 3', "Trial 4", 'Trial 5', 'Trial 8', 'Trial 9', "Trial 10")
    txt = [propertyTitle ': Trained with Max Cor vs Expert'];
    title(txt)
end

function avgPlotterOnTop(fileExpert, fileTrain, propertyTitle, trials)
    figure
    trials(1).plotOnTop(fileTrain, fileExpert, 1, -1, propertyTitle);
    hold on
    for i = 1:6
        trials(i).plotOnTop(fileTrain, fileExpert, 0, 0, propertyTitle);
    end
    hold off
    legend('expert', 'Trial 3', "Trial 4", 'Trial 5', 'Trial 8', 'Trial 9', "Trial 10")
    txt = [propertyTitle ': Trained with Avg Cor vs Expert'];
    title(txt)
end

function maxSubplot(fileExpert, fileTrain, propertyTitle, trials)
    figure
    subplot(7,1,1);
    trials(1).plotSubplot(fileTrain, fileExpert, 1, -1, propertyTitle);
    for i = 1:6
        subplot(7,1,i+1);
        trials(i).plotSubplot(fileTrain, fileExpert, 0, 1, propertyTitle);
    end
end

function avgSubplot(fileExpert, fileTrain, propertyTitle, trials)
    figure
    subplot(7,1,1);
    trials(1).plotSubplot(fileTrain, fileExpert, 1, -1, propertyTitle);
    for i = 1:6
        subplot(7,1,i+1);
        trials(i).plotSubplot(fileTrain, fileExpert, 0, 0, propertyTitle);
    end
end

function modelBoxPlot(cor, plotTitle, transparency)
    figure % Boxplot with all properties color coordinated for min correlation between segments of same slug
    hold on
    pointsAll = repmat(1:12,size(cor,1),1);
    boxchart(cor,'MarkerStyle','none')
    if transparency
        swarmchart(pointsAll,cor,"filled",'MarkerFaceAlpha',0.3,'MarkerEdgeAlpha',0.3)
    else
        swarmchart(pointsAll,cor,"filled")
    end
    title(plotTitle)
    subtitle('1 = I2 Input, 2 = B8a/b, 3 = B38, 4 = B6/9/3, 5 = B7, 6 = T I2, 7 = P I4, 8 = P I3, 9 = T I3, 10 = T hinge, 11 = Grasper Motion, 12 = Force')
    ylim([.3,1.05])
    hold off
end

function animalBoxPlot(cor, plotTitle, transparency, min, hor)
    force_color = '#898989';
    I2_color = '#f26666';
    B8_color = '#f3a7df';%f05bc8
    B38_color = '#f8e171';%ffd500
    B6B9B3_color = '#84cbfb';%46b2fa

    color = [I2_color, B8_color, B38_color, B6B9B3_color, force_color];

    figure % Boxplot with all properties color coordinated for min correlation between segments of same slug
    hold on
    pointsAll = repmat(1:5,size(cor,1),1);
    if hor
        boxchart(cor,'MarkerStyle','none', 'Orientation','horizontal')
    else
        boxchart(cor,'MarkerStyle','none')
    end
    if transparency
        swarmchart(pointsAll,cor,"filled",'MarkerFaceAlpha',0.3,'MarkerEdgeAlpha',0.3, ColorVariable=color)
    else
        swarmchart(pointsAll,cor,"filled", ColorVariable=color)
    end
    title(plotTitle)
    subtitle("1 = I2, 2 = B8, 3 = B38, 4 = B6B9B3, 5 = force")
    ylim([min,1.05])
    hold off
end

function corHistogram(corr, titles, verticalSpacing)
    figure
    histogram(corr)
    title(titles + "Correlation Histogram")
    xlabel("Correlation Coefficients")
    ylim([0,verticalSpacing])
end

function B38BoxPlot(cor, plotTitle, transparency, min, hor)
    figure % Boxplot with all properties color coordinated for min correlation between segments of same slug
    hold on
    pointsAll = repmat(1:6,size(cor,1),1);
    if hor
        boxchart(cor,'MarkerStyle','none', 'Orientation','horizontal')
    else
        boxchart(cor,'MarkerStyle','none')
    end
    if transparency
        swarmchart(pointsAll,cor,"filled",'MarkerFaceAlpha',0.3,'MarkerEdgeAlpha',0.3)
    else
        swarmchart(pointsAll,cor,"filled")
    end
    title(plotTitle)
    subtitle("1 = B6/9/3, 2 = B38, 3 = B8, 4 = I2, 5 = x_gh, 6 = force")
    ylim([min,1.05])
    hold off
end


function checkOutliers(array, whisker, printout)
    for i = 1:5
        for corr = array(:, i)'
            if corr < whisker(i)
                disp("outlier --> " + printout + " " + int2str(i))
            end
        end
    end
end














































