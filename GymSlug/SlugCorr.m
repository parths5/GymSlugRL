classdef SlugCorr
    %REAL SLUG DATA: In this class you can modify animal data so that it can be compared with trained and expert data

    properties
        fullMatFileName     % Name of animal data mat file
        matObject           % Holds all property arrays --> B3, B38, B4/5, B6/9, B8, I2 (B31/32), force      % Will combine B3 and B6/9 by adding the two arrays together
        swallowSegments     % List of indeces of every animal swallow
        time                % Array with the time span of the experiment

        allProps
        expertProps
        trainProps
        optimProps
    end


    methods
        function obj = SlugCorr(file, graph, matProperties)
            %REALSLUGDATA: Construct an instance of this class
            %   file: will be file name of animal data we want to analyze
            %   graph: will give option to graph segmented force (0 is no, 1 is yes)
            %   matProperties: will give option to print mat file properties (0 is no, 1 is yes)

            obj.allProps = ["I2", "B8", "B38", "B6/9/3", "force", "xgh"];
            obj.expertProps = ["/I2_input__expert.csv", "/B8a_b__expert.csv", "/B38__expert.csv", "/B6_9_3__expert.csv", "/Force__expert.csv", "/RelativeGrasperMotion__expert.csv"];
            obj.trainProps = ["/I2_input__Trained.csv", "/B8a_b__Trained.csv", "/B38__Trained.csv", "/B6_9_3__Trained.csv", "/Force__Trained.csv", "/RelativeGrasperMotion__Trained.csv"];
            obj.optimProps = ["/I2_input_single__optim.csv", "/B8a_b_single__optim.csv", "/B38_single__optim.csv", "/B6_9_3_single__optim.csv", "/Force_single__optim.csv"];

            obj.fullMatFileName = fullfile("RealSlugData", file);
            if matProperties
                disp(obj.fullMatFileName)
            end

            if ~exist(obj.fullMatFileName, 'file')
                message = sprintf('%s does not exist', obj.fullMatFileName);
                uiwait(warndlg(message));
            else
                % Load mat file
                s = load(obj.fullMatFileName);
                obj.matObject = s;
                if matProperties
                    disp(obj.matObject)
                end
                
                % Interpolate I2 graph so to smoothen it out (remove any miniscule steps)
                obj.time = [0:length(s.I2)-1]/length(s.I2);
                ti = [0:1/10000:1];
                I2_inter = interp1(obj.time, s.I2, ti);
                I2 = interp1(ti, I2_inter, obj.time);
                
                % Begin segmentation using B31/32 activation (aslo known as when I2 muscle begins to tense up) 
                obj.swallowSegments = [];
                for i= 11:length(s.I2)-11
                    if s.I2(i) < 0.1 && s.I2(i+1) >= 0.1
                        obj.swallowSegments = [obj.swallowSegments, i+1];
                    end
                    if I2(i-1) > I2(i) && I2(i) < I2(i+1) && I2(i) > 0.1 && I2(i) < 2
                        if obj.time(i+1) - obj.time(obj.swallowSegments(end)) > 0.02
                            obj.swallowSegments = [obj.swallowSegments, i+1];
                        end
                    end
                end
                
                %Graph segmented graph for each swallow
                if graph
                    figure
                    hold on;
                    for i = 1:length(obj.swallowSegments)-1
                        plot(obj.time(obj.swallowSegments(i):obj.swallowSegments(i+1)), s.B38(obj.swallowSegments(i):obj.swallowSegments(i+1)));
                    end
                    title(file);
                    xlabel('Time (s)');
                    ylabel('Force (mN)');
                    hold off;
                end
            end
        end




        function crossCorr = Slug2SlugCorrelation(obj, obj2)
            %SLUG_2_SLUG_CORRELATION: find correlation coefficient between one animal and another animal data
            %   obj2: RealSlugData object for specified slug (not the same)
            
            crossCorr = [];

            %disp("loops: " + num2str(length(obj.swallowSegments)) + " * " + num2str(length(obj2.swallowSegments)))
            
            for i = 1:length(obj.swallowSegments)-1
                for j = 1:length(obj2.swallowSegments)-1
                    % Get time for segment i of first slug
                    time1 = obj.time(obj.swallowSegments(i):obj.swallowSegments(i+1));
                    time1 = time1 - time1(1);
                    time1 = time1/time1(end);

                    allCorr = [];
                    sumCorr = zeros(1, length(time1)*2-1);

                    % Get data for all properties
                    for p = 1:5
                        data1 = obj.arrayGetter(obj.allProps(p), 0);
                        data1 = data1(obj.swallowSegments(i):obj.swallowSegments(i+1));
                        
                        % Get time and data for segment j of second slug
                        time2 = obj2.time(obj2.swallowSegments(j):obj2.swallowSegments(j+1));
                        time2 = time2 - time2(1);
                        time2 = time2/time2(end);
                        data2 = obj2.arrayGetter(obj.allProps(p), 0);
                        data2 = data2(obj2.swallowSegments(j):obj2.swallowSegments(j+1));
                        data2 = interp1(time2, data2, time1);

    
                        % Perform Cross Correlation
                        normalizer = norm(data1)*norm(data2);
                        [c,lag] = xcorr(data1, data2);
                        c = c/normalizer;               % Normalize Correlation Coefficient
                        c = c.';
                        allCorr = [allCorr; c];
                        sumCorr = sumCorr + c;
                    end
                    [z, m] = max(sumCorr);      % Find Max Correlation (peak)
                    crossCorr = [crossCorr; allCorr(:,m).']; 
                end
            end
        end



        function crossCorr = Slug2OptimCorrelation(obj)
            %SLUG_2_OPTIM_CORRELATION: find correlation coefficient between animal data and expert data

            file_start = "Optimized_Swallows";
            crossCorr = [];

            %disp("loops: " + num2str(length(obj.swallowSegments)))

            for i= 1:length(obj.swallowSegments)-1
                time1 = obj.time(obj.swallowSegments(i):obj.swallowSegments(i+1));
                time1 = time1 - time1(1);
                time1 = time1/time1(end);

                allCorr = [];
                sumCorr = zeros(1, length(time1)*2-1);

                % Get data for all properties
                for p = 1:5
                    data1 = obj.arrayGetter(obj.allProps(p), 0);
                    data1 = data1(obj.swallowSegments(i):obj.swallowSegments(i+1));
                    data1 = data1/max(data1);
    
                    file = file_start + obj.optimProps(p);
                    F = dlmread(file,' ');
                    dataOptim = F(1:end,2);
                    timeOptim = F(1:end,1);
                    timeOptim = timeOptim/timeOptim(end);
                    time2 = time1;
                    data2 = interp1(timeOptim, dataOptim, time1);
                    data2 = data2/max(data2);

                    % Perform Cross Correlation
                    normalizer = norm(data1)*norm(data2);
                    [c,lag] = xcorr(data1, data2);
                    c = c/normalizer;               % Normalize Correlation Coefficient
                    c = c.';
                    allCorr = [allCorr; c];
                    sumCorr = sumCorr + c;
                end
                [z, m] = max(sumCorr);      % Find Max Correlation (peak)
                crossCorr = [crossCorr; allCorr(:,m).']; 
            end
        end



        function crossCorr = Slug2ExpertCorrelation(obj, objE)
            %SLUG_2_EXPERT_CORRELATION: find correlation coefficient between animal data and expert data
            %   objE: Swallowing object for any trained trial --> same expert info is embedded in all trained trials

            crossCorr = [];
            colors = ["#FF4444", "#F07AD5", "#EFC700", "#68BBFA", "#404040"];

            %disp("loops: " + num2str(length(obj.swallowSegments)))

            for i= 1:length(obj.swallowSegments)-1
                time1 = obj.time(obj.swallowSegments(i):obj.swallowSegments(i+1));
                time1 = time1 - time1(1);
                time1 = time1/time1(end);

                allCorr = [];
                sumCorr = zeros(1, length(time1)*2-1);
                lag_max = [];

                % Get data for all properties
                for p = 1:5
                    data1 = obj.arrayGetter(obj.allProps(p), 0);
                    data1 = data1(obj.swallowSegments(i):obj.swallowSegments(i+1));
                    data1 = data1/max(data1);
    
                    [timeExp,dataExp] = objE.getExpertProperty(obj.expertProps(p), 0);
                    time2 = time1;
                    data2 = interp1(timeExp, dataExp, time1);
                    data2 = data2/max(data2);
                    
                    % Perform Cross Correlation
                    normalizer = norm(data1)*norm(data2);
                    [c,lag] = xcorr(data1, data2);
                    c = c/normalizer;               % Normalize Correlation Coefficient
                    c = c.';
                    allCorr = [allCorr; c];
                    sumCorr = sumCorr + c;

                    [max_cor, m] = max(c);          % Find Max Correlation (peak)
                    lag_max = [lag_max, m];         % Max Lag
                    time = 1:length(time1);
                    opt_m = 33421;

                    graph_corr = false;
                    if graph_corr
                        figure
                        hold on                        
                        %plot([still;data1],'Color', [.7 .7 .7],'linewidth',2)
                        plot(time + length(time1), data1,'Color', [.7 .7 .7],'linewidth',2)
                        xline(length(time1)-1,'--k','linewidth',2)
                        xline(opt_m,'--', 'Color', "#8B0000",'linewidth',2)
                        %plot([front; data2'; back],'Color', colors(p),'linewidth',2)
                        plot(time + opt_m, data2','Color', colors(p),'linewidth',2)
                        hold off
                    end
                end
                [z, m] = max(sumCorr);      % Find Max Correlation (peak)
                crossCorr = [crossCorr; allCorr(:,m).'];
                if graph_corr
                    disp((m-length(time1))/length(time1)*100)
                    disp((lag_max-length(time1))/length(time1)*100)
                end
            end
        end



        function crossCorr = Slug2TrainedCorrelation(obj, objT)
            %SLUG_2_ALL_TRAINED_CORRELATION: find correlation coefficient between animal data and all trained data
            %   objT: Swallowing object for specified trained trial

            crossCorr = [];

            %disp("loops: " + num2str(length(objT.trialSegment)) + " * " + num2str(length(obj.swallowSegments)))

            for SegmentNum = 1:length(objT.trialSegment)-1
                for i= 1:length(obj.swallowSegments)-1
                    time1 = obj.time(obj.swallowSegments(i):obj.swallowSegments(i+1));
                    time1 = time1 - time1(1);
                    time1 = time1/time1(end);

                    allCorr = [];
                    sumCorr = zeros(1, length(time1)*2-1);
    
                    % Get data for all properties
                    for p = 1:5
                        data1 = obj.arrayGetter(obj.allProps(p), 0);
                        data1 = data1(obj.swallowSegments(i):obj.swallowSegments(i+1));
                        data1 = data1/max(data1);
        
                        [timeTrain,dataTrain] = objT.getTrainedPropertySeg(obj.trainProps(p), SegmentNum);
                        time2 = time1;
                        data2 = interp1(timeTrain, dataTrain, time1);
                        data2 = data2/max(data2);

                        % Perform Cross Correlation
                        normalizer = norm(data1)*norm(data2);
                        [c,lag] = xcorr(data1, data2);
                        c = c/normalizer;               % Normalize Correlation Coefficient
                        c = c.';
                        allCorr = [allCorr; c];
                        sumCorr = sumCorr + c;
                    end
                    [z, m] = max(sumCorr);      % Find Max Correlation (peak)
                    crossCorr = [crossCorr; allCorr(:,m).'];
    
                end
            end
        end


        function crossCorr = Expert2TrainedCorrelation(obj, objT, breakable)
            %EXPERT_2_ALL_TRAINED_CORRELATION: find correlation coefficient between expert data and all trained data
            %   objT: Swallowing object for specified trained trial (same object for baseline)
            %   breakable: if 1=unlaoded swallow so p5 = xgh, else p5 = force

            crossCorr = [];

            %disp("loops: " + num2str(length(objT.trialSegment)))

            for SegmentNum = 1:length(objT.trialSegment)-1
                [timeExp,o] = objT.getExpertProperty(obj.expertProps(1), 0);
                time1 = timeExp/timeExp(end);

                allCorr = [];
                sumCorr = zeros(1, length(time1)*2-1);

                % Get data for all properties
                for p = 1:5
                    if (p == 5) & breakable
                        p = 6;
                    end
                    [o,dataExp] = objT.getExpertProperty(obj.expertProps(p), 0);
                    data1 = dataExp/max(dataExp);
    
                    [timeTrain,dataTrain] = objT.getTrainedPropertySeg(obj.trainProps(p), SegmentNum);
                    time2 = time1;
                    data2 = interp1(timeTrain, dataTrain, time1);
                    data2 = data2/max(data2);

                    % Perform Cross Correlation
                    normalizer = norm(data1)*norm(data2);
                    [c,lag] = xcorr(data1, data2);
                    c = c/normalizer;               % Normalize Correlation Coefficient
                    c = c.';
                    allCorr = [allCorr; c];
                    sumCorr = sumCorr + c;
                end
                [z, m] = max(sumCorr);      % Find Max Correlation (peak)
                crossCorr = [crossCorr; allCorr(:,m).'];

            end
        end


        function crossCorr = Slug2RobotCorrelation(obj)
            %SLUG_2_ROBOT_CORRELATION: find correlation coefficient between animal data and robot data (NO FORCE)

            file_start = "RobotData";
            crossCorr = [];
            file = file_start + "/robot_swallow.mat";
            F = load(file);
            data = [F.I2, F.B8, F.B38, F.B6B9B3];

            time = zeros(length(F.dt),1);
            time(2:end) = (time(1:end-1) + cumsum(F.dt(1:end-1)))/1000;

            %disp("loops: " + num2str(length(F.segments)) + " * " + num2str(length(obj.swallowSegments)))

            for SegmentNum = 1:length(F.segments)-1
                for i= 1:length(obj.swallowSegments)-1
                    time1 = obj.time(obj.swallowSegments(i):obj.swallowSegments(i+1));
                    time1 = time1 - time1(1);
                    time1 = time1/time1(end);
    
                    allCorr = [];
                    sumCorr = zeros(1, length(time1)*2-1);
    
                    % Get data for all properties
                    c_length = 0;
                    for p = 1:5
                        if p == 5
                            c = zeros(1,c_length);
                        else
                            data1 = obj.arrayGetter(obj.allProps(p), 0);
                            data1 = data1(obj.swallowSegments(i):obj.swallowSegments(i+1));
                            data1 = data1/max(data1);
    
                            d = data(:,p);
                            dataRobot = d(F.segments(SegmentNum):F.segments(SegmentNum+1));
                            dataRobot = dataRobot';
                            timeRobot = time(F.segments(SegmentNum):F.segments(SegmentNum+1));
                            timeRobot = timeRobot - timeRobot(1);
                            timeRobot = timeRobot/timeRobot(end);
                            timeRobot = timeRobot';

                            time2 = time1;
                            data2 = interp1(timeRobot, dataRobot, time1);
                            data2 = data2/max(data2);
        
                            % Perform Cross Correlation
                            normalizer = norm(data1)*norm(data2);
                            [c,lag] = xcorr(data1, data2);
                            c = c/normalizer;               % Normalize Correlation Coefficient
                            c = c.';
                            c_length = length(c);
                            % figure
                            % plot(time1, data1, time2, data2)
                            % title(obj.allProps(p) + ": Slug/Robot")
                        end
                        allCorr = [allCorr; c];
                        sumCorr = sumCorr + c;
                    end
                    [z, m] = max(sumCorr);      % Find Max Correlation (peak)
                    crossCorr = [crossCorr; allCorr(:,m).']; 
                end
            end
        end

        

        function crossCorr = Robot2RobotCorrelation(obj)
            %SLUG_2_ROBOT_CORRELATION: find correlation coefficient between animal data and robot data (NO FORCE)
            file_start = "RobotData";
            crossCorr = [];
            file = file_start + "/robot_swallow.mat";
            F = load(file);
            data = [F.I2, F.B8, F.B38, F.B6B9B3, F.xgh];

            time = zeros(length(F.dt),1);
            time(2:end) = (time(1:end-1) + cumsum(F.dt(1:end-1)))/1000;

            %disp("loops: " + num2str(length(F.segments)) + " * " + num2str(length(obj.swallowSegments)))

            for SegmentNum = 1:4
                for SegmentNum2 = 5:length(F.segments)-1
                    %disp(num2str(SegmentNum) + " vs " + num2str(SegmentNum2))
                    timeRobot = time(F.segments(SegmentNum):F.segments(SegmentNum+1));
                    timeRobot = timeRobot - timeRobot(1);
                    timeRobot = timeRobot/timeRobot(end);
                    time1 = timeRobot';
    
                    allCorr = [];
                    sumCorr = zeros(1, length(time1)*2-1);
    
                    % Get data for all properties
                    c_length = 0;
                    for p = 1:5
                        %disp("property: " + num2str(p))
                        d = data(:,p);
                        data1 = d(F.segments(SegmentNum):F.segments(SegmentNum+1));
                        data1 = data1';

                        data2 = d(F.segments(SegmentNum2):F.segments(SegmentNum2+1));
                        data2 = data2';
                        timeRobot = time(F.segments(SegmentNum2):F.segments(SegmentNum2+1));
                        timeRobot = timeRobot - timeRobot(1);
                        timeRobot = timeRobot/timeRobot(end);
                        timeRobot = timeRobot';

                        time2 = time1;
                        data2 = interp1(timeRobot, data2, time1);
                        data2 = data2/max(data2);
    
                        % Perform Cross Correlation
                        normalizer = norm(data1)*norm(data2);
                        [c,lag] = xcorr(data1, data2);
                        c = c/normalizer;               % Normalize Correlation Coefficient
                        %c = c.';
                        %disp(size(c))
                        c_length = length(c);
                        % figure
                        % plot(time1, data1, time2, data2)
                        % title(obj.allProps(p) + ": Slug/Robot")
                        
                        %disp("update allCorr")
                        allCorr = [allCorr; c];
                        %disp(size(allCorr))
                        sumCorr = sumCorr + c;
                    end
                    [z, m] = max(sumCorr);      % Find Max Correlation (peak)
                    crossCorr = [crossCorr; allCorr(:,m).']; 
                end
            end
        end


        function crossCorr = Robot2BaselineCorrelation(obj, objE)
            %SLUG_2_ROBOT_CORRELATION: find correlation coefficient between animal data and robot data (NO FORCE)
            file_start = "RobotData";
            crossCorr = [];
            file = file_start + "/robot_swallow.mat";
            F = load(file);
            data = [F.I2, F.B8, F.B38, F.B6B9B3, F.xgh];

            time = zeros(length(F.dt),1);
            time(2:end) = (time(1:end-1) + cumsum(F.dt(1:end-1)))/1000;

            %disp("loops: " + num2str(length(F.segments)) + " * " + num2str(length(obj.swallowSegments)))

            for SegmentNum = 1:length(F.segments)-1
                timeRobot = time(F.segments(SegmentNum):F.segments(SegmentNum+1));
                timeRobot = timeRobot - timeRobot(1);
                timeRobot = timeRobot/timeRobot(end);
                time1 = timeRobot';

                allCorr = [];
                sumCorr = zeros(1, length(time1)*2-1);

                % Get data for all properties
                c_length = 0;
                for p = 1:5
                    %disp("property: " + num2str(p))
                    d = data(:,p);
                    data1 = d(F.segments(SegmentNum):F.segments(SegmentNum+1));
                    data1 = data1';

                    if p == 5
                        p_base = 6;
                    else
                        p_base = p;
                    end
                    [timeExp,dataExp] = objE.getExpertProperty(obj.expertProps(p_base), 0);
                    time2 = time1;
                    data2 = interp1(timeExp, dataExp, time1);
                    data2 = data2/max(data2);

                    % Perform Cross Correlation
                    normalizer = norm(data1)*norm(data2);
                    [c,lag] = xcorr(data1, data2);
                    c = c/normalizer;               % Normalize Correlation Coefficient
                    %c = c.';
                    %disp(size(c))
                    c_length = length(c);
                    % figure
                    % plot(time1, data1, time2, data2)
                    % title(obj.allProps(p) + ": Slug/Robot")
                    
                    %disp("update allCorr")
                    allCorr = [allCorr; c];
                    %disp(size(allCorr))
                    sumCorr = sumCorr + c;
                end
                [z, m] = max(sumCorr);      % Find Max Correlation (peak)
                crossCorr = [crossCorr; allCorr(:,m).']; 
            end
        end


        function crossCorr = Robot2TrainedCorrelation(obj, objT)
            %EXPERT_2_ALL_TRAINED_CORRELATION: find correlation coefficient between expert data and all trained data
            %   objT: Swallowing object for specified trained trial (same object for baseline)

            file_start = "RobotData";
            crossCorr = [];
            file = file_start + "/robot_swallow.mat";
            F = load(file);
            data = [F.I2, F.B8, F.B38, F.B6B9B3, F.xgh];

            time = zeros(length(F.dt),1);
            time(2:end) = (time(1:end-1) + cumsum(F.dt(1:end-1)))/1000;

            %disp("loops: " + num2str(length(F.segments)) + " * " + num2str(length(obj.swallowSegments)))

            for SegmentNum2 = 1:length(objT.trialSegment)-1
                for SegmentNum = 1:length(F.segments)-1
                    timeRobot = time(F.segments(SegmentNum):F.segments(SegmentNum+1));
                    timeRobot = timeRobot - timeRobot(1);
                    timeRobot = timeRobot/timeRobot(end);
                    time1 = timeRobot';

                    allCorr = [];
                    sumCorr = zeros(1, length(time1)*2-1);
    
                    % Get data for all properties
                    c_length = 0;
                    for p = 1:5
                        %disp("property: " + num2str(p))
                        d = data(:,p);
                        data1 = d(F.segments(SegmentNum):F.segments(SegmentNum+1));
                        data1 = data1';
                        if p == 5
                            p_train = 6;
                        else
                            p_train = p;
                        end
                        [timeTrain,dataTrain] = objT.getTrainedPropertySeg(obj.trainProps(p_train), SegmentNum2);
                        time2 = time1;
                        data2 = interp1(timeTrain, dataTrain, time1);
                        data2 = data2/max(data2);

                        % Perform Cross Correlation
                        normalizer = norm(data1)*norm(data2);
                        [c,lag] = xcorr(data1, data2);
                        c = c/normalizer;               % Normalize Correlation Coefficient
                        %c = c.';
                        %disp(size(c))
                        c_length = length(c);
                        % figure
                        % plot(time1, data1, time2, data2)
                        % title(obj.allProps(p) + ": Slug/Robot")
                        
                        %disp("update allCorr")
                        allCorr = [allCorr; c];
                        %disp(size(allCorr))
                        sumCorr = sumCorr + c;
                    end
                    [z, m] = max(sumCorr);      % Find Max Correlation (peak)
                    crossCorr = [crossCorr; allCorr(:,m).']; 
                end
            end
        end



        function array = arrayGetter(obj, property, graph)
            %ARRAYGETTER: getter function
            %   property: will be file name of animal data we want to analyze
            %   graph: will give option to graph segmented force (0 is no, 1 is yes)
            % B3, B38, B4/5, B6/9, B8, I2 (B31/32), force 
            if property == "B6/9/3"
                array = obj.matObject.B3 + obj.matObject.B69;
            elseif property == "B38"
                array = obj.matObject.B38;
            elseif property == "B4/5"
                array = obj.matObject.B45;
            elseif property == "B8"
                array = obj.matObject.B8;
            elseif property == "I2"
                array = obj.matObject.I2;
            else
                array = obj.matObject.force;
            end

            if graph
                time = obj.time(obj.swallowSegments(1):obj.swallowSegments(2));
                time = time - time(1);
                time = time/time(end);
                % fid = fopen('realAnimalTime_6.txt','wt');
                % if fid > 0
                %     fprintf(fid,'%f\n',time');
                %     fclose(fid);
                % end
                %figure
                %plot(time, array(obj.swallowSegments(1):obj.swallowSegments(2)));
                figure
                hold on;
                for i = 1:length(obj.swallowSegments)-1
                    plot(obj.time(obj.swallowSegments(i):obj.swallowSegments(i+1)), array(obj.swallowSegments(i):obj.swallowSegments(i+1)));
                    
                    % saving individual normalized swallows
                    % txt = "Animal6_B6B9B3_" + int2str(i) + '.txt';
                    % fid = fopen(txt,'wt');
                    % if fid > 0
                    %     fprintf(fid,'%f\n', array(obj.swallowSegments(i):obj.swallowSegments(i+1))/max(array));
                    %     fclose(fid);
                    % end

                    % saving individual un-normalized swallows
                    txt = "RealAnimal6_force_" + int2str(i) + '.txt';
                    fid = fopen(txt,'wt');
                    if fid > 0
                        fprintf(fid,'%f\n', array(obj.swallowSegments(i):obj.swallowSegments(i+1))); %/max(array));
                        fclose(fid);
                    end
                end
                title(property);
                hold off;
            end
        end


        function array = getSegments(obj)
            array = obj.swallowSegments;
        end
    end
end































