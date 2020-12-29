import cv2
import numpy as np
import pandas as pd
import math

class Plot:
    def social_distancing_view(self,frame, distances_mat, array_boxes_detected, risk_count):
        
        red = (0, 0, 255)
        green = (0, 255, 0)
        yellow = (0, 255, 255)
        for i in range(len(array_boxes_detected)):
            
            frame = cv2.rectangle(frame,(array_boxes_detected[i][1],array_boxes_detected[i][0]),(array_boxes_detected[i][3],array_boxes_detected[i][2]),green,2)
                            
        for i in range(len(distances_mat)):

            per1 = distances_mat[i][0]
            per2 = distances_mat[i][1]
            closeness = distances_mat[i][2]
            
            if closeness == 1:
                x,y,w,h = per1[:]
                frame = cv2.rectangle(frame,(y,x),(h,w),yellow,2)

                x1,y1,w1,h1 = per2[:]
                frame = cv2.rectangle(frame,(y,x),(h,w),yellow,2)
                #frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),yellow, 2) 
                
        for i in range(len(distances_mat)):

            per1 = distances_mat[i][0]
            per2 = distances_mat[i][1]
            closeness = distances_mat[i][2]
            
            if closeness == 0:
                x,y,w,h = per1[:]
                frame = cv2.rectangle(frame,(y,x),(h,w),red,2)
                x1,y1,w1,h1 = per2[:]
                frame = cv2.rectangle(frame,(y,x),(h,w),red,2)
                #frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),red, 2)
                
        pad = np.full((140,frame.shape[1],3), [110, 110, 100], dtype=np.uint8)
        cv2.putText(pad, "Bounding box shows the level of risk to the person.", (50, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 0), 2)
        cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        frame = np.vstack((frame,pad))               
        return frame

    def plotStatistic(self,risk_counts,framesPerTimeSpan,videoDuration,timeSpan):
        riskCountsN =[]
        durP = []
        for i in range(0,len(risk_counts),int(framesPerTimeSpan)):
            countPerTiSt = risk_counts[i:i+int(framesPerTimeSpan)]
            riskCountsN.append((sum([(r[0]) for r in countPerTiSt]),sum([(r[1]) for r in countPerTiSt]),sum([(r[2]) for r in countPerTiSt])))
        for j in range(0,int(videoDuration),int(timeSpan)):
            if (j+int(timeSpan))>int(videoDuration):
                timeRem = videoDuration % j
                timSta = [j,j+int(timeRem)]
            else:
                timSta = [j,j+int(timeSpan)]
            durP.append(str(timSta))
        raw_data = {"HIGH_RISK":[r[0] for r in riskCountsN],"LOW_RISK":[r[1] for r in riskCountsN],"SAFE":[r[2] for r in riskCountsN],"timeSpan":durP}
        df = pd.DataFrame(raw_data, columns = ["HIGH_RISK", "LOW_RISK","SAFE","timeSpan"])
        print("dtaframe",df)
        ax = df.plot.bar(x="timeSpan", y=["HIGH_RISK", "LOW_RISK", "SAFE"], color=['#ee0000', '#eedd00', '#00dd88'], figsize=(17, 11))
        max_y = max([df[attr].values.max() for attr in ["HIGH_RISK", "LOW_RISK", "SAFE"]])
        min_y = max([df[attr].values.min() for attr in ["HIGH_RISK", "LOW_RISK", "SAFE"]])
        print("max_y",max_y)
        print("\n", "min_y",min_y)
        ax.set_yticks([math.ceil(i) for i in np.linspace(0, max_y, min_y)])
        ax.set_xticklabels(list(df['timeSpan'].values), rotation=0)
        fig = ax.get_figure()
        fig.savefig('../statistics/socialDistancingStatistics2.png')