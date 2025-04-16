import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, send_file
import io

 

# Initialize Flask app
app = Flask(__name__)

# Initialize Faker
fake = Faker() # To create a dummy students name for every KAFA Al Huda class

# Define constants
sktam_student_startno = 0 # START of unique students ID's range (Total 120 students records for all class)
sktam_student_endno = 80 # END of unique students ID's range (Total 120 students records for all class)
sktam_lowscorebaseline = 30 # Historical minimum exam scores (%)
sktam_highscorebaseline = 101 # Historical maximum exam scores (%)
sktam_minattendanceratebaseline = 70 # Historical minimum attendance (%)
sktam_maxattendanceratebaseline = 101 # Historical maximum attendance (%)
sktam_minparticipationbaseline = 50 # Historical minimum for overall student participation (%)
sktam_maxparticipationbaseline = 101 # Historucal maximum for overall student participation (%)
sktam_scorethreshold = 85 # Define the threshold for an A score for every subject
sktam_minscorethreshold = 60  # Define the threshold for underperformance

sktam_student_ids = range(sktam_student_startno, sktam_student_endno)  # 120 unique students ID's range
student_uniqueid = [f"{i}" for i in sktam_student_ids] # pre-assign students matrix number by each class
sktam_classlist = ["Saidina Ali", "Saidina Othman", "Saidina Uthman", "Saidina Umar"] # List of KAFA Al Huda classes
sktam_students_per_class = 20  # Number of students per class

# Generate student names and matrix numbers
sktam_student_profiles = [fake.simple_profile() for _ in range(sktam_student_startno,sktam_student_endno)]
sktam_student_names = [profile['name'] for profile in sktam_student_profiles]
sktam_student_genders = [profile['sex'] for profile in sktam_student_profiles]
student_matrixnumbers = [f"KAFAALHUDAKULIM-ID{specific_student_uniqueid}" for specific_student_uniqueid in student_uniqueid]




# Generate systematic data
data = {
    'sktam_student_id': [],
    'student_matrixnumber': [],
    'student_name': [],
    'student_gender': [],
    'sktam_classes': [],
    'Attendance_Rate': np.random.uniform(sktam_minattendanceratebaseline, sktam_maxattendanceratebaseline, sktam_student_endno),  # Percentage
    'Participation_score': np.random.uniform(sktam_minparticipationbaseline, sktam_maxparticipationbaseline, sktam_student_endno),  # Percentage
    
    # PREVIOUS SCORES RECORD - START
    'prev_score_AlQuran': np.random.uniform(sktam_lowscorebaseline, sktam_highscorebaseline, sktam_student_endno),  # Scores between 30 and 100 for specific subject and number of students
    'prev_score_Akidah': np.random.uniform(sktam_lowscorebaseline, sktam_highscorebaseline, sktam_student_endno),  # Scores between 30 and 100 for specific subject and number of students
    'prev_score_Sirah': np.random.uniform(sktam_lowscorebaseline, sktam_highscorebaseline, sktam_student_endno),  # Scores between 30 and 100 for specific subject and number of students
    'prev_score_Adab': np.random.uniform(sktam_lowscorebaseline, sktam_highscorebaseline, sktam_student_endno),  # Scores between 30 and 100 for specific subject and number of students
    'prev_score_JawiDanKhat': np.random.uniform(sktam_lowscorebaseline, sktam_highscorebaseline, sktam_student_endno),  # Scores between 30 and 100 for specific subject and number of students
    'prev_score_BahasaArab': np.random.uniform(sktam_lowscorebaseline, sktam_highscorebaseline, sktam_student_endno),  # Scores between 30 and 100 for specific subject and number of students
    'prev_score_Ibadah': np.random.uniform(sktam_lowscorebaseline, sktam_highscorebaseline, sktam_student_endno),  # Scores between 30 and 100 for specific subject and number of students
    'prev_score_PCHI': np.random.uniform(sktam_lowscorebaseline, sktam_highscorebaseline, sktam_student_endno),  # Scores between 30 and 100 for specific subject and number of students
    'prev_score_AmaliSolat': np.random.uniform(sktam_lowscorebaseline, sktam_highscorebaseline, sktam_student_endno),  # Scores between 30 and 100 for specific subject and number of students
    # PREVIOUS SCORES RECORD - END
    
  
}



# Create binary target variables for "A" and underperformed scores based on a hypothetical future score
for subject in ['AlQuran', 'Akidah', 'Sirah', 'Adab', 'JawiDanKhat', 'BahasaArab', 'Ibadah', 'PCHI', 'AmaliSolat']:
    future_score = np.clip(data[f'prev_score_{subject}'] * 1.2, None, 100)
    data[f'A_score_{subject}'] = (future_score >= sktam_scorethreshold).astype(int)
    data[f'UnderperformScore_{subject}'] = (future_score < sktam_minscorethreshold).astype(int)
    



counter_datasetindx = 0
for i in range(sktam_student_endno):
    
    if counter_datasetindx <= len(sktam_student_ids):
        #print(f"Index data > {counter_datasetindx}")
        student_index = i % len(sktam_student_ids)  # Cycle through the list of students
        sktam_class_index = (student_index // sktam_students_per_class) % len(sktam_classlist)  # Determine class based on student index
        data['sktam_student_id'].append(sktam_student_ids[student_index])
        data['student_matrixnumber'].append(student_matrixnumbers[student_index])
        data['student_name'].append(sktam_student_names[student_index])
        data['student_gender'].append(sktam_student_genders[student_index])
        data['sktam_classes'].append(sktam_classlist[sktam_class_index])
        #data['Future_Score'].append(sktam_futurescoresbaselines[sktam_class_index])
        
        counter_datasetindx +=1
    else:
        break


    
        




# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('scoredstudent_performance_sample.csv', index=False)

print("Sample data generated with A scores and underperformed scores and saved to 'scoredstudent_performance_sample.csv'.")



#
##
###
####
##### MACHINE LEARNING SECTION for Performed SCORES

# Features and target variables
features = [
    'Attendance_Rate', 'Participation_score',
    'prev_score_AlQuran', 'prev_score_Akidah', 'prev_score_Sirah', 'prev_score_Adab',
    'prev_score_JawiDanKhat', 'prev_score_BahasaArab', 'prev_score_Ibadah',
    'prev_score_PCHI', 'prev_score_AmaliSolat'
]
targets = [f'A_score_{subject}' for subject in ['AlQuran', 'Akidah', 'Sirah', 'Adab', 'JawiDanKhat', 'BahasaArab', 'Ibadah', 'PCHI', 'AmaliSolat']]

X = df[features]
y = df[targets]

# Split the data into training and testing sets, keeping track of indices
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the multi-output classification model
model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
ascore_accuracyreport = classification_report(y_test, y_pred, target_names=targets, output_dict=True)
#print(f'Accuracy: {accuracy}')
#print(classification_report(y_test, y_pred, target_names=targets))

# Predict "A" scores for all students
all_predictions = model.predict(X)



# Count the number of students predicted to get an "A" in each subject
a_score_counts = np.sum(all_predictions, axis=0)
a_score_dict = {target: count for target, count in zip(targets, a_score_counts)}

#print("Number of students predicted to get an 'A' in each subject:")
#for subject, count in a_score_dict.items():
#    print(f"{subject}: {count}")
    


# Create a DataFrame to display which students are predicted to get an "A" in each subject
predictions_df = pd.DataFrame(all_predictions, columns=targets)
predictions_df['student_name'] = df['student_name']

# Prepare data for rendering
students_with_ascoree = {subject: predictions_df[predictions_df[subject] == 1]['student_name'].tolist() for subject in targets}

# Display students predicted to get an "A" in each subject
#for subject in targets:
#    students_with_a = predictions_df[predictions_df[subject] == 1]['student_name'].tolist()
#    print(f"Students predicted to get an 'A' in {subject}: {students_with_a}")



# Count the number of "A" scores each student is predicted to receive
predictions_df['Total_A'] = predictions_df[targets].sum(axis=1)

# Group students by the number of "A" scores
#grouped_by_allascores_count = predictions_df.groupby('Total_A').size().reset_index(name='Number_of_Students')

# Group students by the number of "A" scores
grouped_by_a_count = predictions_df.groupby('Total_A')['student_name'].apply(list).reset_index()

# Display the results
students_dict = {}  
for _, row in grouped_by_a_count.iterrows():  
    num_a = row['Total_A']  
    students = row['student_name']  
    if num_a not in students_dict:  
        students_dict[num_a] = [students]  
    else:  
        students_dict[num_a].append(students)  

print(f"{students_dict}")

# Bar Chart: Number of students with each number of A's
plt.figure(figsize=(10, 6))
sns.countplot(x='Total_A', data=predictions_df, palette='viridis')
plt.title('Number of Students with Each Number of A\'s')
plt.xlabel('Number of A\'s')
plt.ylabel('Number of Students')
#plt.show()



#####
####
###
##
#






#
##
###
####
##### MACHINE LEARNING SECTION for UNDERPERFORMED SCORES

# Features and target variables
features_underperformed = [
    'Attendance_Rate', 'Participation_score',
    'prev_score_AlQuran', 'prev_score_Akidah', 'prev_score_Sirah', 'prev_score_Adab',
    'prev_score_JawiDanKhat', 'prev_score_BahasaArab', 'prev_score_Ibadah',
    'prev_score_PCHI', 'prev_score_AmaliSolat'
]
targets_underperformed = [f'UnderperformScore_{subject}' for subject in ['AlQuran', 'Akidah', 'Sirah', 'Adab', 'JawiDanKhat', 'BahasaArab', 'Ibadah', 'PCHI', 'AmaliSolat']]

X_underperformed = df[features_underperformed]
y_underperformed = df[targets_underperformed]

# Split the data into training and testing sets, keeping track of indices
X_underperformed_train, X_underperformed_test, y_underperformed_train, y_underperformed_test = train_test_split(
    X_underperformed, y_underperformed, test_size=0.2, random_state=42
)

# Train the multi-output classification model
underperformed_model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
underperformed_model.fit(X_underperformed_train, y_underperformed_train)

# Evaluate the model
y_underperformed_pred = underperformed_model.predict(X_underperformed_test)
underperformed_accuracy = accuracy_score(y_underperformed_test, y_underperformed_pred)
underperformed_accuracyreport = classification_report(y_underperformed_test, y_underperformed_pred, target_names=targets, output_dict=True)
print(f'Underperformed Accuracy: {underperformed_accuracy}')
print(classification_report(y_underperformed_test, y_underperformed_pred, target_names=targets_underperformed))

# Predict underperformance for all students
underperformed_all_predictions = underperformed_model.predict(X_underperformed)

# Create a DataFrame to display which students are predicted to underperform in each subject
underperformed_predictions_df = pd.DataFrame(underperformed_all_predictions, columns=targets_underperformed)
underperformed_predictions_df['student_name'] = df['student_name']

# Count the number of underperforming students in each subject
underperform_counts = underperformed_predictions_df[targets_underperformed].sum().to_dict()

# Provide advice for underperforming students
def provide_advice(row):  
    advice = []  
    rank_attendance = []
    rank_participation = []
    rank_subject = []
    
    # Advice classification based on the attendance percentage data
    if row['Attendance_Rate'] <= 50:  
        advice.append("(HIGH)Action required to improve attendance else students struggle to understand the lessons | ") 
        rank_attendance.append("1") 
    elif row['Attendance_Rate'] > 50 and row['Attendance_Rate'] <= 75:  
        advice.append("(MEDIUM) Keep improve attendance for better participation in class | ")  
        rank_attendance.append("2") 
    elif row['Attendance_Rate'] > 75 and row['Attendance_Rate'] <= 100:  
        advice.append("(LOW)Good attendance performance! | ")  
        rank_attendance.append("3") 
    else:
        advice.append("(UNKNOWN)Attendance data not available | ")
        rank_attendance.append("0") 
    
    # Advice classification based on the participation in class
    if row['Participation_score'] <= 50:  
        advice.append("(HIGH)Action required to improve participation in class else students struggle silently on what is being taught or teachers may not know if students really understand the lesson | ")  
        rank_participation.append("1")
    elif row['Participation_score'] > 50 and row['Participation_score'] <= 75:  
        advice.append("(MEDIUM)Keep improve participation for better understand lesson in class | ")  
        rank_participation.append("2")
    elif row['Participation_score'] > 75 and row['Participation_score'] <= 100:  
        advice.append("(LOW)Good participation in class | ")  
        rank_participation.append("3")
    else:
        advice.append("(UNKNOWN)Participation data not available | ")
        rank_participation.append("0")
    
    # Advice classification based on the underperform students score
    for underperformedsubject in targets_underperformed:  
        if underperformedsubject in row and row[underperformedsubject] == 1:  
            advice.append(f"Focus on {underperformedsubject.split('_')[1]}. |")
            rank_subject.append("1")  
    
    print(f"Attendance Rank: {rank_attendance}")
    print(f"Participation Rank: {rank_participation}")
    print(f"Subject Rank: {rank_subject}")
    
    # Convert the string ranks to integers  
    attendance_ranks_toint = [int(intrankattendance) for intrankattendance in rank_attendance] 
    participation_ranks_toint = [int(intrankparticipation) for intrankparticipation in rank_participation] 
    subject_ranks_toint = [int(intranksubject) for intranksubject in rank_subject]  
    
    
    # Sum the ranks and calculate overall rank based on total UPKK subject 
    total_attendance_rank = sum(attendance_ranks_toint) 
    total_participation_rank = sum(participation_ranks_toint) 
    total_subject_rank_raw = sum(subject_ranks_toint) 
    
    # Calculate the expression (sum(subject_ranks) / 9) * 3  
    result_total_subject_rank = (total_subject_rank_raw / 9) * 3
    # Format result_total_subject_rank to 3 decimal places without rounding using f-string  
    result_total_subject_rank_decimal = f"{result_total_subject_rank:.3f}"
    # Convert formatted_result back to a float for comparison  
    result_total_subject_rank = float(result_total_subject_rank_decimal) 
    
    
  
    
    # Print the total (0 - LOW, 1 - MEDIUM, 3 - CRITICAL)
    print(f"Overall UPKK Subject Rank:", result_total_subject_rank)  
    
    # Generate overall underperform student performance
    # Attendance_Rank    Participation_Rank   Overall_Subject_Rank  
    #  0 - Critical         0 - Critical        0 - Good
    #  1 - Very Bad         1 - Very Bad        1 - Bad
    #  2 - Bad              2 - Bad             2 - Very Bad
    #  3 - Good             3 - Good            3 - Critical
    

    if total_attendance_rank <= 1:
        #advice.append(f"OVERALL SUMMARY: (HIGH) VERY BAD ATTENDANCE")
        
        if total_participation_rank <= 1:
            #advice.append(f"OVERALL SUMMARY: (HIGH) VERY BAD PARTICIPATION")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Keep it up to maintain subject scoring. Our trained model suggest student to improve attendance and participation in class, so students may improve their subject scores.")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Our trained model suggest teachers may need to investigate the topic that students struggle. In the same time students must improve their attendance and participation in the class. so students may improve their subject scores.")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Our trained model suggest teachers may take serious mitigation plan since students really struggle, understand students issue is mandatory to understand which area teachers can help including their attendance, participation, and subject scores")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Our trained model suggest teachers may take serious mitigation plan since students really struggle, understand students issue is mandatory to understand which area teachers can help including their attendance, participation, and subject scores")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
                
        elif total_participation_rank > 1 and total_participation_rank <= 2:
            #advice.append(f"OVERALL SUMMARY: (MEDIUM) AVERAGE PARTICIPATION")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Keep it up to maintain subject scoring. Our trained model suggest student to improve attendance and participation in class, so students may improve their subject scores.")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Our trained model suggest teachers may need to investigate the topic that students struggle. In the same time students must improve their attendance and participation in the class. so students may improve their subject scores.")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Our trained model suggest teachers may take serious mitigation plan since students really struggle, understand students issue is mandatory to understand which area teachers can help including their attendance, participation, and subject scores")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Our trained model suggest teachers may take serious mitigation plan since students really struggle, understand students issue is mandatory to understand which area teachers can help including their attendance, participation, and subject scores")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
                
        elif total_participation_rank > 2 and total_participation_rank <= 3:
            #advice.append(f"OVERALL SUMMARY: (LOW) GOOD PARTICIPATION")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Keep it up to maintain subject scoring and participation in class. Our trained model suggest student to improve attendance, so students may improve their subject scores.")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Even good participation in class, our trained model suggest teachers may need to investigate the topic that students struggle. In the same time students must improve their attendance. so students may improve their subject scores.")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Even good participation in class but really bad attendance, our trained model suggest teachers may take serious mitigation plan since students really struggle, understand students issue is mandatory to understand which area teachers can help including their attendance, participation, and subject scores")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Even good participation in class, but really bad in attendance and subject scores. Our trained model suggest teachers may take serious mitigation plan since students really struggle, understand students issue is mandatory to understand which area teachers can help including their attendance, participation, and subject scores")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
                
        else:
            #advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Keep it up to maintain subject scoring. Our trained model suggest student to improve attendance and teachers need to check if the participation data enter correctly in your database because this data not found, so students may improve their subject scores.")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Our trained model suggest teachers may need to investigate the topic that students struggle. In the same time students must improve their attendance and teachers need to check if the participation data enter correctly in your database because this data not found. so students may improve their subject scores.")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Really bad attendance, our trained model suggest teachers may take serious mitigation plan since students really struggle, understand students issue is mandatory to understand which area teachers can help including their attendance and teachers need to check if the participation data enter correctly in your database because this data not found in order to improve subject scores")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Really bad in attendance and subject scores. Our trained model suggest teachers may take serious mitigation plan since students really struggle, understand students issue is mandatory to understand which area teachers can help including their attendance and teachers need to check if the participation data enter correctly in your database because this data not found in order to improve subject scores")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the participation and subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
            
    elif total_attendance_rank > 1 and total_attendance_rank <= 2:
        #advice.append(f"OVERALL SUMMARY: (MEDIUM) AVERAGE ATTENDANCE")
        
        if total_participation_rank <= 1:
            #advice.append(f"OVERALL SUMMARY: (HIGH) VERY BAD PARTICIPATION")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Keep maintain and improve subject score. Our trained model suggest student to improve the attendance and teacher required to investigate the best way to increase student participation in class")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Our trained model suggest student to improve student attendance and participation in class. Also, teachers need to identify which topic student struggle")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Our trained model suggest student to improve attendance. Most important teacher need to have mitigation plan to improve student participation in class in order to improve subject scores")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Our trained model highly recommended teachers to understand student attendance issue, must improve student participation in class and understand which topic student struggle to understand")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
                
        elif total_participation_rank > 1 and total_participation_rank <= 2:
            #advice.append(f"OVERALL SUMMARY: (MEDIUM) AVERAGE PARTICIPATION")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Our trained model suggest student to improve attendance and participation in class. Also, keep maintain and improve student scores")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Our trained mode suggest student to improve the attendance and participation in class. Also, teacher please understand which topic student struggle to understand")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Bad subject scores. Student must improve theri attendance and also participation in class. Most important, teachers need to understand which topic student struggle to understand")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Very bad subject score. Teacher need to investigate immediately how to improve student attendance, participation in class and also topics that student struggle to understand in order to improve student subject score")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
                
        elif total_participation_rank > 2 and total_participation_rank <= 3:
            #advice.append(f"OVERALL SUMMARY: (LOW) GOOD PARTICIPATION")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Good job on student subject scores and participation in class. However please improve student attendance")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Nice participation in class. But, student need to keep improving their attendance and understand which topic student struggle")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Nice participation in class but subject score still low. Teacher need to help student to improve their attendance and also understand topic that student struggle in order to improve their subject score")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Nice participation in class, but subject score is very bad. Teacher need to take action on improving student subject score by understand the topic that student struggle and in the same time please help to improve student attendance and their participation in class")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
                
        else:
            #advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Please maintain and improve student score. But student need to improve their attendance. Teacher please check the participation data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Please improve student score and attendance. Teacher please check the participation data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Teacher need to investigate the topic that student struggle and improve student attendance. Teacher please check the participation data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Subject score is very bad. Our trained model suggest teacher to help understand topic that student struggle and please help to improve student attendance. Teacher please check the participation data in your database and make sure it is enter correctly")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the participation and subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
                
    elif total_attendance_rank > 2 and total_attendance_rank <= 3:
        #advice.append(f"OVERALL SUMMARY: (LOW) GOOD ATTENDANCE")
        
        if total_participation_rank <= 1:
            #advice.append(f"OVERALL SUMMARY: (HIGH) VERY BAD PARTICIPATION")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Keep it up a good attendance and subject score. But student really need to improve their participation in class")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Nice attendance. Please improve student participation and understand topic student struggle")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Nice attendance. Teacher need to understand student participation in class and understand topic that student struggle")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Nice attendance. Teacher need to really understand which topic student struggle and improve student participation in class")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
                
        elif total_participation_rank > 1 and total_participation_rank <= 2:
            #advice.append(f"OVERALL SUMMARY: (MEDIUM) AVERAGE PARTICIPATION")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Good attendance. Maintain student subject scores. In the meantime please improve student participation in class")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Good attendance. Student need to improve their subject score and participation in class")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Good attendance. But teacher really need to help understand which topic student struggle to understand and in the same time help student to improve student participation in class")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Very bad subject score even the attendance is good. Teacher must investigate on topic that student struggle and improve student participation in class")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
                
        elif total_participation_rank > 2 and total_participation_rank <= 3:
            #advice.append(f"OVERALL SUMMARY: (LOW) GOOD PARTICIPATION")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Good attendance and participation in class. Keep this hard work and maintain student subject score")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Good attendance and participation in class. Keep improve student subject score")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Help needed from teacher to further understand which topic student lack. Keep it up a good attendance and participation in class")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Very bad subject score. Teacher need to understand which topic student struggle. Keep it up a good attendance and participation in class")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the participation and subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
                
        else:
            #advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Good attendance and keep maintain and improve student subject scores. Teacher please check the participation data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Good attendance. Student need to improve subject score. Teacher please check the participation data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Good attendance. Teacher must understand which topic student struggle. Teacher please check the participation data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Good attendance. Teacher must understand which topic student struggle. Teacher please check the participation data in your database and make sure it is enter correctly")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the data in your database and make sure the attendance, participation, and subject scoring is enter properly")
                
    else:
        #advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND")
        
        if total_participation_rank <= 1:
            #advice.append(f"OVERALL SUMMARY: (HIGH) VERY BAD PARTICIPATION")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Very bad participation in class but still able to maintain subject score. Keep it up. Teacher please check the attendance data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Very bad participation in class and student need to improve their subject score. Teacher please check the attendance data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECTVery bad participation in class. Teacher need to understand which topic student struggle. Teacher please check the attendance data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Very bad participation in class and also subject score. Teacher must understand which topic student struggle. Teacher please check the attendance data in your database and make sure it is enter correctly")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the attendance and subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
                
        elif total_participation_rank > 1 and total_participation_rank <= 2:
            #advice.append(f"OVERALL SUMMARY: (MEDIUM) AVERAGE PARTICIPATION.")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Need to improve participation in class. Maintain and keep improve subject score. Teacher please check the attendance data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Need to improve participation in class and subject score. Teacher please check the attendance data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Teacher need to understand which topic student struggle and improve student participation in class. Teacher please check the attendance data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Teacher really need to understand which topic student struggle and improve student participation in class. Teacher please check the attendance data in your database and make sure it is enter correctly")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the attendance and subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
                
        elif total_participation_rank > 2 and total_participation_rank <= 3:
            #advice.append(f"OVERALL SUMMARY: (LOW) GOOD PARTICIPATION")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Good participation and maintain subject score. Teacher please check the attendance data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Good participation. Student need to improve subject score. Teacher please check the attendance data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Even good participation, teacher need to investigate which topic student struggle to understand. Teacher please check the attendance data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Even good participation in class but student still have many topic that they do not understand. Teacher need to investigate and improve it. Teacher please check the attendance data in your database and make sure it is enter correctly")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the attendance and subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
                
        else:
            #advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND")
            
            if result_total_subject_rank == 0:
                advice.append(f"OVERALL SUMMARY: (LOW) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Maintain subject score. Teacher please check the attendance and participation data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 0 and result_total_subject_rank <= 1:
                advice.append(f"OVERALL SUMMARY: (MEDIUM) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Improve subject score. Teacher please check the attendance and participation data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 1 and result_total_subject_rank <= 2:
                advice.append(f"OVERALL SUMMARY: (HIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Teacher need to further understand which topic student struggle. Teacher please check the attendance and participation data in your database and make sure it is enter correctly")
            elif result_total_subject_rank > 2 and result_total_subject_rank <= 3:
                advice.append(f"OVERALL SUMMARY: (VERYHIGH) {total_subject_rank_raw} UNDERPERFORM SUBJECT. Teacher really need to take action to identify which topic student struggle. Teacher please check the attendance and participation data in your database and make sure it is enter correctly")
            else:
                advice.append(f"OVERALL SUMMARY: (HIGH) DATA NOT FOUND. Our trained model suggest teachers to look at the attendance, participation and subject score data in your database and make sure the attendance, participation, and subject scoring is enter properly")
        
    
    # Return to detail advice
    return " ".join(advice)  




underperformed_predictions_df['Advice'] = df.apply(provide_advice, axis=1)
data[f'Advice'] = df.apply(provide_advice, axis=1)


# Display students predicted to underperform and their advice
underperforming_listofstudents = underperformed_predictions_df[underperformed_predictions_df[targets_underperformed].any(axis=1)]
print(underperforming_listofstudents[['student_name', 'Advice']])

students_advice_list = underperforming_listofstudents[['student_name', 'Advice']].to_dict('records')  


# Create DataFrame
df = pd.DataFrame(data)


# Save to CSV
df.to_csv('allscoredstudent_performance_sample.csv', index=False)

print("Sample data generated with A scores and underperformed scores and saved to 'allscoredstudent_performance_sample.csv'.")

#####
####
###
##
#

@app.route('/')
def index():
    # Pass the accuracy and classification report to the template
    return render_template('index.html', accuracy=accuracy, report=ascore_accuracyreport, underperformed_accuracy=underperformed_accuracy, underperformed_accuracyreport=underperformed_accuracyreport, students_advice_list=students_advice_list, a_score_dict=a_score_dict, students_dict=students_dict, students_with_ascoree=students_with_ascoree)


@app.route('/plot_underperformed.png')
def plot_underperformed():
    # Create subplots using matplotlib
    underperformedfig, underperformedaxs = plt.subplots(2, 1, figsize=(10, 12))

    # Bar chart for underperforming students
    underperformedbars = underperformedaxs[0].bar(underperform_counts.keys(), underperform_counts.values(), color='skyblue', label='Underperforming Students')
    underperformedaxs[0].set_title('Number of Underperforming Students by Subject')
    underperformedaxs[0].set_xlabel('Subject')
    underperformedaxs[0].set_ylabel('Number of Students')
    underperformedaxs[0].legend()
    underperformedaxs[0].tick_params(axis='x', rotation=45)

    # Add data labels
    for specificunderperformedbar in underperformedbars:
        underperformedyval = specificunderperformedbar.get_height()
        underperformedaxs[0].text(specificunderperformedbar.get_x() + specificunderperformedbar.get_width()/2, underperformedyval + 0.5, int(underperformedyval), ha='center', va='bottom')

    # Line plot for attendance and participation
    underperformedaxs[1].plot(df['student_name'], df['Attendance_Rate'], label='Attendance Rate', color='green', marker='o')
    underperformedaxs[1].plot(df['student_name'], df['Participation_score'], label='Participation Score', color='orange', marker='x')
    underperformedaxs[1].set_title('Attendance and Participation Rates')
    underperformedaxs[1].set_xlabel('Student')
    underperformedaxs[1].set_ylabel('Percentage')
    underperformedaxs[1].legend()
    underperformedaxs[1].tick_params(axis='x', rotation=90)

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a BytesIO object and return it as a response
    underperformedoutput = io.BytesIO()
    plt.savefig(underperformedoutput, format='png')
    underperformedoutput.seek(0)
    plt.close(underperformedfig) # Close the figure 
    return send_file(underperformedoutput, mimetype='image/png')

    



if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)