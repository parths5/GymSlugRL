classdef Swallowing
    %SWALLOWING: In this class you can run all the functions needed to run cross correlation correctly
    

    properties
        trialFileName   % Name of trained directory
        expertFileName  % Name of expert directory
        trialSegment    % List of indeces of every trained swallow
        expertSegment   % List of beginning an end indeces of expert swallow
        graphTitle      % Trial title for Graphs
    end



%% CONSTRUCTOR
    methods
        function obj = Swallowing(trialName, expertName, title)
            %SWALLOWING Construct an instance of this class
            obj.trialFileName = trialName;
            obj.expertFileName = expertName;
            obj.trialSegment = obj.Segmentation("trained",0);
            obj.expertSegment = obj.Segmentation("expert",0);
            obj.graphTitle = title;
        end
        


%% SEGMENTATION
        function segmentList = Segmentation(obj, type, graph)
            %SEGMENTATION Will output segmentList which defined the beginning of every swallow with indeces
            %   type: defines if you are looking for expert of trained list since they will be different (string)
            %   graph: will give option to graph segmented force (0 is no, 1 is yes)

            segmentList = [];
            % Get data from csv files
            if type == "trained"
                % Get time, force, and I2 Input from trained files
                file = obj.trialFileName + "/Force__Trained.csv";
                F = dlmread(file,' ');
                time = F(1:1000,1);
                force = F(1:1000,2);
                file = obj.trialFileName + "/I2_input__Trained.csv";
                F = dlmread(file,' ');
                data = F(1:1000,2);
            else
                % Get time, force, and I2 Input from expert files
                file = obj.expertFileName + "/Force__expert.csv";
                F = dlmread(file,' ');
                time = F(1:1000,1);
                force = F(1:1000,2);
                file = obj.expertFileName + "/I2_input__expert.csv";
                F = dlmread(file,' ');
                data = F(1:1000,2);
            end
            
            %Max Force Peaks
            maxForce = [];
            for i = 2:length(force)-1
                if force(i-1) < force(i) && force(i) > force(i + 1) && force(i) > 0.2
                        maxForce(end+1) = i;
                end
            end
        
            %I2 Activation Peaks(B31/32) 
            indexAll = 1:length(time);
            indexAll = indexAll(((data(2:end) > 0.05) .* (data(1:end-1) < 0.05)) > 0.5);
            indexAll = indexAll(find((indexAll>maxForce(1)) .* (indexAll<maxForce(end))));
            indexAll = indexAll(find(force(indexAll)<0.3));
        
            %Correct Index for segmentList
            m = 1;
            for i = indexAll
                check = 1;
                while check & m < length(maxForce)
                    if i > maxForce(m) && i < maxForce(m+1)
                        check = 0;
                        segmentList(end+1) = i;
                        m=m+1;
                    elseif i < maxForce(m)
                        check=0;
                    else
                        m=m+1;
                    end
                end
            end

            if type == "expert"
                segmentList = segmentList(end-1:end);
            end

            if graph
                figure;
                hold on;
                for i = 1:length(segmentList)-1
                    plot(time(segmentList(i):segmentList(i+1)), force(segmentList(i):segmentList(i+1)));
                end
                title(type);
                xlabel('Time (s)');
                ylabel('Force (mN)');
                hold off;
            end
        end % End of Segmentation Function



%% CORRELATION
        function [crossCorr,normalizer, maxSeg, maxVal] = crossCorrelation(obj, fileTrain, fileExpert, graphPlot, graphCorr, propertyTitle)
            %CROSSCORRELATION Will find the correlation all segments of specified property by outputting list crossCorr
            %   fileTrain: will be file name of trained property we want to analyze
            %   fileExpert: will be file name of expert property we want to analyze
            %   graphPlot: will give option to graph expert and trial plots over one another (0 is no, 1 is yes)
            %   graphCorr: will give option to graph cross correlation graph over lag (0 is no, 1 is yes)
            %   propertyTitle: will be string used to title graphs

            % Get data from csv files anf apply found segments
            file = obj.expertFileName + fileExpert;
            F = dlmread(file,' ');
            timeExp = F(obj.expertSegment(1):obj.expertSegment(2),1);
            timeExp = timeExp - timeExp(1);
            timeExp = timeExp/timeExp(end);
            dataExp = F(obj.expertSegment(1):obj.expertSegment(2),2);
           
            file = obj.trialFileName + fileTrain;
            F = dlmread(file,' ');
            crossCorr = [];
            maxVal = 0;
            for i = 1:length(obj.trialSegment)-1
                time = F(obj.trialSegment(i):obj.trialSegment(i+1),1);
                time = time - time(1);
                time = time/time(end);
                data = F(obj.trialSegment(i):obj.trialSegment(i+1),2);
                normalizer = norm(dataExp)*norm(data);

                % Cross Correlation
                [c,lag] = xcorr(dataExp, data);
                c = c/normalizer;       % Normalize Correlation Coefficient
                [max_cor, m] = max(c);  % Find Max Correlation (peak)
                max_lag = lag(m);       % Max Lag
                crossCorr(end+1) = max_cor;

                if graphPlot
                    figure
                    plot(time, data, timeExp, dataExp)
                    title(propertyTitle + ": Expert vs Trained Segment")
                end
                
                if graphCorr
                    figure
                    plot(lag, c, max_lag, max_cor,'o');
                    txt = ['Max Correlation Coefficient: ' num2str(max_cor) ' @ ' int2str(max_lag)];
                    subtitle(txt);
                    title(propertyTitle + " Cross Correlation");
                    xlabel('lag');
                    txt = ['Correlation (norm by ' num2str(normalizer) ')'];
                    ylabel(txt);
                end

                if max_cor > maxVal
                    maxVal = max_cor;
                    maxSeg = [obj.trialSegment(i), obj.trialSegment(i+1)];
                end
            end
        end



%% GRAPHS
        function complete = plotSubplot(obj, fileTrain, fileExpert, fileType, graphType, propertyTitle)
            %PLOTSUBPLOT Will plot all the desired motor neurons in a subplot with bottom being the expert and the rest trained
            %   fileTrain: will be file name of trained property we want to analyze
            %   fileExpert: will be file name of expert property we want to analyze
            %   fileType: will define what file type we want to plot (1 is expert, 0 is trained)
            %   graphType: will define if plotting segments with max correlation or segment closest to average (1 is max, 0 is average, -1 is expert)
            %   propertyTitle: will be string used to title graphs
            complete = 0;
            if fileType
                % Will plot expert segment for subplot
                exp = obj.expertFileName + fileExpert;
                F = dlmread(exp,' ');
                timeExp = F(obj.expertSegment(1):obj.expertSegment(2),1);
                timeExp = timeExp - timeExp(1);
                timeExp = timeExp/timeExp(end);
                dataExp = F(obj.expertSegment(1):obj.expertSegment(2),2);

                plot(timeExp, dataExp,"red",'LineWidth',3)
                ylabel(propertyTitle + ": Expert       ")
                set(get(gca,'YLabel'),'Rotation',0)
                ylim([-2 2])
                complete = 1;
            else
                % Will plot trained segment for subplot
                [cor,norm,maxSeg,maxCor] = obj.crossCorrelation(fileTrain, fileExpert, 0, 0, propertyTitle);                
               
                file = obj.trialFileName + fileTrain;
                F = dlmread(file,' ');
                if graphType
                    % Will plot trained segmens with max correlation
                    time = F(maxSeg(1):maxSeg(2),1);
                    time = time - time(1);
                    time = time/time(end);
                    data = F(maxSeg(1):maxSeg(2),2);
    
                    plot(time, data,'LineWidth',3);
                    ylabel({propertyTitle + ": Trained " + obj.graphTitle + "       "; " w/ Max Corr of " + maxCor + "       "})
                    set(get(gca,'YLabel'),'Rotation',0)
                    ylim([-2 2])
                    complete = 1;
                else
                    % Will plot trained segment closest to average correlation
                    meanCor = mean(cor);
                    diff = mean(cor);
                    closestMeanCor = mean(cor);
                    seg = 0;
                    meanSeg = [];
                    for i = 1:length(cor)
                        seg = seg+1;
                        if abs(meanCor - cor(i)) < diff
                            diff = abs(meanCor-cor(i));
                            closestMeanCor = cor(i);
                            meanSeg = [obj.trialSegment(seg) obj.trialSegment(seg+1)];
                        end
                    end
                    time = F(meanSeg(1):meanSeg(2),1);
                    time = time - time(1);
                    time = time/time(end);
                    data = F(meanSeg(1):meanSeg(2),2);
    
                    plot(time, data,'LineWidth',3);
                    ylabel({propertyTitle + ": Trained " + obj.graphTitle + "       "; " w/ Corr Avg of " + meanCor + " which       "; " is the closest to Corr of " + closestMeanCor + "       "})
                    set(get(gca,'YLabel'),'Rotation',0)
                    ylim([-2 2])
                    complete = 1;
                end
            end
        end



        function complete = plotOnTop(obj, fileTrain, fileExpert, fileType, graphType, propertyTitle)
            %%PLOTONTOP Will plot all the desired muscle movement in one graph with all the expert and selected trained curves on top of each other
            %   fileTrain: will be file name of trained property we want to analyze
            %   fileExpert: will be file name of expert property we want to analyze
            %   fileType: will define what file type we want to plot (1 is expert, 0 is trained)
            %   graphType: will define if plotting segments with max correlation or segment closest to average (1 is max, 0 is average, -1 is expert)
            %   propertyTitle: will be string used to title graphs
            complete = 0;
            if fileType
                % Will plot expert segment for subplot
                exp = obj.expertFileName + fileExpert;
                F = dlmread(exp,' ');
                timeExp = F(obj.expertSegment(1):obj.expertSegment(2),1);
                timeExp = timeExp - timeExp(1);
                timeExp = timeExp/timeExp(end);
                dataExp = F(obj.expertSegment(1):obj.expertSegment(2),2);

                plot(timeExp, dataExp,"red",'LineWidth',3)
                %ylabel(propertyTitle + ": Expert       ")
                %set(get(gca,'YLabel'),'Rotation',0)
                complete = 1;
            else
                % Will plot trained segment for subplot
                [cor,norm,maxSeg,maxCor] = obj.crossCorrelation(fileTrain, fileExpert, 0, 0, propertyTitle);                
               
                file = obj.trialFileName + fileTrain;
                F = dlmread(file,' ');
                if graphType
                    % Will plot trained segmens with max correlation
                    time = F(maxSeg(1):maxSeg(2),1);
                    time = time - time(1);
                    time = time/time(end);
                    data = F(maxSeg(1):maxSeg(2),2);
    
                    s = plot(time, data,'LineWidth',3);
                    %ylabel({propertyTitle + ": Trained " + obj.graphTitle + "       "; " w/ Max Corr of " + maxCor + "       "})
                    %set(get(gca,'YLabel'),'Rotation',0)
                    s.Color(4) = 0.25;
                    complete = 1;
                else
                    % Will plot trained segment closest to average correlation
                    meanCor = mean(cor);
                    diff = mean(cor);
                    %closestMeanCor = mean(cor);
                    seg = 0;
                    meanSeg = [];
                    for i = 1:length(cor)
                        seg = seg+1;
                        if abs(meanCor - cor(i)) < diff
                            diff = abs(meanCor-cor(i));
                            %closestMeanCor = cor(i);
                            meanSeg = [obj.trialSegment(seg), obj.trialSegment(seg+1)];
                        end
                    end
                    time = F(meanSeg(1):meanSeg(2),1);
                    time = time - time(1);
                    time = time/time(end);
                    data = F(meanSeg(1):meanSeg(2),2);
    
                    s = plot(time, data,'LineWidth',3);
                    s.Color(4) = 0.25;
                    complete = 1;
                end
            end
        end



        function [averageData, stdev] = plotOnTopALL(trials, fileTrain, fileExpert, graphType, propertyTitle, plotAvg, plotStdDev)
            %%PLOTONTOPALL Will plot all the desired muscle movement of all trials in one graph with all the expert and selected trained curves on top of each other
            %   trials: will be an array of Swallow objects
            %   fileTrain: will be file name of trained property we want to analyze
            %   fileExpert: will be file name of expert property we want to analyze
            %   graphType: will define if plotting segments with max correlation or segment closest to average (1 is max, 0 is average)
            %   propertyTitle: will be string used to title graphs
            %   plotAvg: will give option to inlcude average curve on plot (1 is yes, 0 is no)
            %   plotStdDev: will give option to include standard deviation on plot (1 is yes, 0 is no)
            firstValAdded = false;
            allTrialData = [];
            allTrialTime = [];
            for trial = trials
                % Will plot trained segment for subplot
                [cor,norm,maxSeg,maxCor] = trial.crossCorrelation(fileTrain, fileExpert, 0, 0, propertyTitle);                
               
                file = trial.trialFileName + fileTrain;
                F = dlmread(file,' ');
                if graphType
                    % Will plot trained segmens with max correlation
                    time = F(maxSeg(1):maxSeg(2),1);
                    time = time - time(1);
                    time = time/time(end);
                    data = F(maxSeg(1):maxSeg(2),2);Segment " + j + " slug2
                    
                    if ~firstValAdded
                        allTrialTime = time;
                        allTrialData = [allTrialData, data];
                        firstValAdded = true;
                    else
                        newData = interp1(time, data, allTrialTime);
                        allTrialData = [allTrialData, newData];
                    end
                    txt = [propertyTitle ': Avg and Std Dev of Trained Max Cor'];
                else
                    % Will plot trained segment closest to average correlation
                    meanCor = mean(cor);
                    diff = mean(cor);
                    %closestMeanCor = mean(cor);
                    seg = 0;
                    meanSeg = [];
                    for i = 1:length(cor)
                        seg = seg+1;
                        if abs(meanCor - cor(i)) < diff
                            diff = abs(meanCor-cor(i));
                            %closestMeanCor = cor(i);
                            meanSeg = [trial.trialSegment(seg), trial.trialSegment(seg+1)];
                        end
                    end
                    time = F(meanSeg(1):meanSeg(2),1);
                    time = time - time(1);
                    time = time/time(end);
                    data = F(meanSeg(1):meanSeg(2),2);

                    if ~firstValAdded
                        allTrialTime = time;
                        allTrialData = [allTrialData, data];
                        firstValAdded = true;
                    else
                        newData = interp1(time, data, allTrialTime);
                        allTrialData = [allTrialData, newData];
                    end
                    txt = [propertyTitle ': Avg and Std Dev of Trained Avg Cor'];
                end
            end
            averageData = mean(allTrialData,2);
            stdev = std(allTrialData,0,2);
            
            figure
            if plotAvg
                plot(allTrialTime,averageData,'LineWidth',3)
            end
            hold on
            for data = allTrialData
                s = plot(allTrialTime,data,'LineWidth',3);
                s.Color(4) = 0.25;
            end
            if plotStdDev
                plot(allTrialTime,averageData+stdev,"red",'LineWidth',3)
                plot(allTrialTime,averageData-stdev,"red",'LineWidth',3)
            end
            hold off
            legend('average', 'Trial 3', "Trial 4", 'Trial 5', 'Trial 8', 'Trial 9', "Trial 10", '+std dev', '-std dev')
            title(txt)
        end

        
%% GETTERS
        function [timeExp,dataExp] = getExpertProperty(obj, fileExpert, graph)
            %%GETEXPERTPROPERTY Will plot all the desired muscle movement of all trials in one graph with all the expert and selected trained curves on top of each other
            %   fileExpert: will be file name of expert property we want to analyze

            exp = obj.expertFileName + fileExpert;
            F = dlmread(exp,' ');
            timeExp = F(obj.expertSegment(1):obj.expertSegment(2),1);
            timeExp = timeExp - timeExp(1);
            timeExp = timeExp/timeExp(end);
            dataExp = F(obj.expertSegment(1):obj.expertSegment(2),2);

            if graph
                figure
                plot(timeExp, dataExp)
                ylim([-.05,1.05])
            end
        end


        function [time,data] = getTrainedProperty(obj, fileTrain, fileExpert, graphType, propertyTitle, graph)
            %%GETTRAINEDPROPERY Will plot all the desired muscle movement of all trials in one graph with all the expert and selected trained curves on top of each other
            %   fileTrain: will be file name of trained property we want to analyze
            %   fileExpert: will be file name of expert property we want to analyze
            %   graphType: will define if plotting segments with max correlation or segment closest to average (1 is max, 0 is average)
            %   propertyTitle: will be string used to title graphs

            [cor,norm,maxSeg,maxCor] = obj.crossCorrelation(fileTrain, fileExpert, 0, 0, propertyTitle);                
               
            file = obj.trialFileName + fileTrain;
            F = dlmread(file,' ');
            if graphType
                % Will plot trained segmens with max correlation
                time = F(maxSeg(1):maxSeg(2),1);
                time = time - time(1);
                time = time/time(end);
                data = F(maxSeg(1):maxSeg(2),2);

            else
                % Will plot trained segment closest to average correlation
                meanCor = mean(cor);
                diff = mean(cor);
                %closestMeanCor = mean(cor);
                seg = 0;
                meanSeg = [];
                for i = 1:length(cor)
                    seg = seg+1;
                    if abs(meanCor - cor(i)) < diff
                        diff = abs(meanCor-cor(i));
                        %closestMeanCor = cor(i);
                        meanSeg = [obj.trialSegment(seg), obj.trialSegment(seg+1)];
                    end
                end
                time = F(meanSeg(1):meanSeg(2),1);
                time = time - time(1);
                time = time/time(end);
                data = F(meanSeg(1):meanSeg(2),2);
            end

            if graph
                figure
                plot(time, data)
                ylim([-.05,1.05])
            end
        end


        function [time,data] = getTrainedPropertySeg(obj, fileTrain, SegmentNum)
            %%GETTRAINEDPROPERY Will plot all the desired muscle movement of all trials in one graph with all the expert and selected trained curves on top of each other
            %   fileTrain: will be file name of trained property we want to analyze
            %   SegmentNum: will define if plotting segments with max correlation or segment closest to average (1 is max, 0 is average)
               
            file = obj.trialFileName + fileTrain;
            F = dlmread(file,' ');
            time = F(obj.trialSegment(SegmentNum):obj.trialSegment(SegmentNum+1),1);
            time = time - time(1);
            time = time/time(end);
            data = F(obj.trialSegment(SegmentNum):obj.trialSegment(SegmentNum+1),2);
        end

        function data = getTrainedPropertyAll(obj, fileTrain)
            %%GETTRAINEDPROPERY Will plot all the desired muscle movement of all trials in one graph with all the expert and selected trained curves on top of each other
            %   fileTrain: will be file name of trained property we want to analyze
            %   SegmentNum: will define if plotting segments with max correlation or segment closest to average (1 is max, 0 is average)
               
            file = obj.trialFileName + fileTrain;
            F = dlmread(file,' ');
            data = F(:,2);
        end
    end
end





























