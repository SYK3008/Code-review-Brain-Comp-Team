# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:29:27 2022

@author: Saeyeon
"""

# %% Plot fig 5C (within-subject correlations with observed data, shuffing keeping pairs, full shuffling)


#add a column in subj_correlations dataframe corresponding to the index (subject names)
subj_correlations['subject name'] = list_subject

#create a new dataframe to get accuracies for each subject, at each trial ("the accuracy of probability estimates was computed at the trial level as the distance between the subject's and IO's estimates)   
subj_accuracies  = pd.DataFrame(index = data['subject'], 
                                columns= ['subject name', 'accuracy_estimation', 'accuracy_confidence','accuracy_estimation_pairs_shuffling','accuracy_confidence_pairs_shuffling', 
                                          'accuracy_estimation_full_shuffling','accuracy_confidence_full_shuffling'])
subj_accuracies['subject name'] = data['subject'].values

for i in range(len(data)):
    
    #estimation accuracy
    estimation_OI = data.iloc[i, data.columns.get_loc('io_pred')]#OI
    estimation_subject = data.iloc[i, data.columns.get_loc('sub_pred')]#subject
   
    subj_accuracies.iloc[i, subj_accuracies.columns.get_loc('accuracy_estimation')] = math.dist((estimation_OI, estimation_subject) ,(estimation_OI, estimation_OI))
    
    
    #confidence accuracy
    confidence_OI = data.iloc[i, data.columns.get_loc('io_conf')]#OI
    confidence_subject = data.iloc[i, data.columns.get_loc('sub_conf')]#subject
    a = subj_correlations.loc[subj_correlations['subject name'] == data.iloc[i, 0]]['sub_io_conf_slope'].values[0]
    b = subj_correlations.loc[subj_correlations['subject name'] == data.iloc[i, 0]]['sub_io_conf_int'].values[0]
    confidence_predicted = confidence_OI*a + b
    
    subj_accuracies.iloc[i, subj_accuracies.columns.get_loc('accuracy_confidence')] = math.dist((confidence_OI, confidence_subject) ,(confidence_OI, confidence_predicted)) 

      
#compute correlations btw acc_estimation and acc_confidence
subj_correlations_accuracies = pd.DataFrame(index=list_subject,
                                 columns=['subject name', 'observed data', 'shuffling keeping pairs',
                                          'full shuffling'])
subj_correlations_accuracies['subject name'] = list_subject

for subject in list_subject :
    subj_correlations_accuracies.loc[subject]['observed data'] = np.corrcoef(subj_accuracies.loc[subj_accuracies['subject name'] == subject]['accuracy_estimation'].astype(float),
                        subj_accuracies.loc[subj_accuracies['subject name'] == subject]['accuracy_confidence'].astype(float))[0, 1]

    
            
'''shuffling keeping pairs and full shuffling  '''          

for subject in list_subject :
    subj_pred = data.loc[data['subject'] == subject]['sub_pred'].array #subj_pred is an array containing all the probabilty estimates of the given subject derived from the observed data      
    io_pred = data.loc[data['subject'] == subject]['io_pred'].array
    io_conf = data.loc[data['subject'] == subject]['io_conf'].array
    subj_conf = data.loc[data['subject'] == subject]['sub_conf'].array
              
    #at each trial, compute the estimates given by the subject and by the IO (for probabilty estimates and confidence reports) 
    array_subject = np.empty((0, 2), float) # 2D-array for the proba estimates and confidence reports of the subject
    array_IO = np.empty((0, 2), float) # 2D-array for the proba estimates and confidence reports of the IO 
    
    indices = len(subj_pred)  # indices is the number of trials for a given subject 
    
    for i in range(indices):
        array_subject = np.append(array_subject, np.array([[subj_pred[i], subj_conf[i]]]), axis=0)
        array_IO = np.append(array_IO, np.array([[io_pred[i], io_conf[i]]]), axis=0)
    

    #shuffing (ie permutation) keeping pairs
    array_subject_shuffled = np.random.permutation(array_subject) # 2D-array of the subject (proba estimate, conf report)
    array_IO_shuffled = np.random.permutation(array_IO) # 2D-array of the IO (proba estimate, conf report)
    print('array_subject_shuffled',  array_subject_shuffled)
    print('array_IO_shuffled', array_IO_shuffled)
    
    #compute the new accuracies after shuffling keeping pairs 
    acc_pred = []
    acc_conf = []
      
    a, b, r, p_value, std_err = linregress (pd.DataFrame(array_IO_shuffled)[1], pd.DataFrame(array_subject_shuffled)[1])
    
    
    for j in range(indices) :
        acc_pred.append(abs(array_subject_shuffled[j][0] - array_IO_shuffled[j][0]))
        
        #computing accuracy for confidence: a is the slope and b is the intercept of the reg of subjective confidences on IO confidences 
        confidence_predicted = array_IO_shuffled[j][1]*a + b
    
        acc_conf.append(math.dist((array_IO_shuffled[j][1], array_subject_shuffled[j][1]),(array_IO_shuffled[j][1], confidence_predicted)))
            
    
    #full shuffling (ie full permutation)
    subj_data_shuffled = np.random.permutation(subj_pred)
    io_pred_shuffled = np.random.permutation(io_pred)
    io_conf_shuffled = np.random.permutation(io_conf)
    subj_conf_shuffled = np.random.permutation(subj_conf)
    
    acc_pred_full = []
    acc_conf_full = []
    
    c, d, R, p, stderr = linregress(io_conf_shuffled, subj_conf_shuffled)
    
    for k in range(indices) :
        acc_pred_full.append(abs(subj_data_shuffled[k] - io_pred_shuffled[k]))
        
        #computing accuracy for confidence: c is the slope and d is the intercept of the reg of subjective confidences on IO confidences
        confidence_predicted_full = io_conf_shuffled[k]*c + d
    
        acc_conf_full.append(math.dist((io_conf_shuffled[k], subj_conf_shuffled[k]),(io_conf_shuffled[k], confidence_predicted_full)))
        
 
    #correlation per subject
    subj_correlations_accuracies.loc[subject]['shuffling keeping pairs'] = np.corrcoef(np.array(acc_pred), np.array(acc_conf))[0,1]
    subj_correlations_accuracies.loc[subject]['full shuffling'] = np.corrcoef(np.array(acc_pred_full), np.array(acc_conf_full))[0,1]
    
    
'''plot error bars'''

#subj_correlations_accuracies2 = subj_correlations_accuracies.filter(like='PCB2015', axis=0)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
x=np.array(['Observed data', 'Shuffling \n keeping pairs', 'Full shuffling'])
y=np.array([subj_correlations_accuracies.mean()['observed data'], subj_correlations_accuracies.mean()['shuffling keeping pairs'], subj_correlations_accuracies.mean()['full shuffling']])
yerr = np.array([subj_correlations_accuracies.sem()['observed data'],subj_correlations_accuracies.sem()['shuffling keeping pairs'], subj_correlations_accuracies.sem()['full shuffling']])
ax.errorbar(x,y,yerr=yerr,
                 fmt='o', capsize=8,
                 markersize=8,
                 color="black",
                 zorder=2)

'''t-tests on two related samples''' 

paired_ttest_pred1 = scipy.stats.ttest_rel(subj_correlations_accuracies['observed data'],
                                           subj_correlations_accuracies['shuffling keeping pairs'], 
                                     axis = 0, alternative = 'less')

print('t test observed data and shuffling keeping pairs : ', paired_ttest_pred1, '\n')

paired_ttest_pred2 = scipy.stats.ttest_rel(subj_correlations_accuracies['shuffling keeping pairs'],
                                           subj_correlations_accuracies['full shuffling'], 
                                     axis = 0, alternative = 'less')

print('t test shuffling keeping pairs and ful shuffling : ', paired_ttest_pred2, '\n')

paired_ttest_pred3 = scipy.stats.ttest_rel(subj_correlations_accuracies['observed data'],
                                           subj_correlations_accuracies['full shuffling'], 
                                     axis = 0, alternative = 'less')

print('t test shuffling keeping pairs and full shuffling : ', paired_ttest_pred3, '\n')