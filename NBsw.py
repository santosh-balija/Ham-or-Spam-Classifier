import os
import string
import math

# Appending the file paths in a list   
def load_file_paths(path):
    
    file_paths = []
    for root, dirs, file in os.walk(path):
        for f in file:
            if '.txt' in f:
                file_paths.append(os.path.join(root, f))
    return file_paths

# Get all the words from file_paths
def all_words(File_paths,stop_words):
    
    list_of_words = []
    for file in File_paths:
        f = open(file,'r',errors='ignore')
        words = f.read().replace('\n',' ')
        list_of_words.extend(words.split())
    list_of_words = [''.join(c for c in s if c not in string.punctuation) for s in list_of_words]
    list_of_words = [x for x in list_of_words if x]
    list_of_words = [x for x in list_of_words if x not in stop_words]
    return list_of_words
    
def count(words):
    d = {}
    for i in words:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d
        

def conditional_prob(words,label,words_combined_unique):     
    conditional_prob_final = {} 
    b = len(words_combined_unique)                      
    DictWords = count(words)
    for w in words_combined_unique:
        if w in DictWords:
            conditional_prob_final[w+':'+label] =  float((DictWords[w] + 1)/(len(words) + b)) 
        else:
            conditional_prob_final[w+':'+label] =  float(1/(len(words) + b))   
    return conditional_prob_final

def NB(prior,file_paths_test,cond_prob,true_label,stop_words):
    c_p = 0;
    in_p = 0;
    for file in file_paths_test:
        words= []
        f = open(file,'r',errors ='ignore')     
        w = f.read().replace('\n',' ')
        words.extend(w.split())
        words = [''.join(c for c in s if c not in string.punctuation) for s in words]   
        words = [x for x in words if x]
        words = [x for x in words if x not in stop_words] 
        words_unique = count(words)
        label = ['ham','spam']
        predval = {}
        for lb in label:
            if (lb == 'ham'):
                pr = prior[0]
            else:
                pr = prior[1]
            sumprob = 0

            for w,c in words_unique.items():
                if (w+':'+lb) in cond_prob:
                    sumprob += c*(math.log(cond_prob[w+':'+lb]))
            predval[lb] = sumprob + math.log(pr) 
        max_value = max(predval.values()) 
        predlabel = (([k for k, v in predval.items() if v == max_value]))
        if (predlabel[0]==true_label):
            c_p = c_p + 1
        else:
            in_p = in_p + 1
    return c_p, in_p
            
if __name__=='__main__':
    
    pathstop = '.\\stopwords.txt'
    f = open(pathstop, 'r')  
    # To append all the stopwords in a list
    stop_words = []
    for line in f.readlines():
        stop_words.append(line.strip())
    
    # file paths for both ham and spam train fils
    ham_train_files = load_file_paths('assignment3_train/train/ham')
    spam_train_files = load_file_paths('assignment3_train/train/spam')
    
    #Words in Ham and Spam train files 
    ham_train_words  =  all_words(ham_train_files,stop_words)
    spam_train_words = all_words(spam_train_files,stop_words)
    
    # combined train words in ham and spam
    words_combined = ham_train_words + spam_train_words
    
    # unique words in words_combined 
    words_combined_unique = list(dict.fromkeys(words_combined))  #getting unique words by eliminating duplicates
    

    ham_conditional_p = dict()
    spam_conditional_p = dict()
    
    # conditional probabilites of ham and spam train words
    ham_conditional_p = conditional_prob(ham_train_words,'ham',words_combined_unique)  
    spam_conditional_p = conditional_prob(spam_train_words,'spam',words_combined_unique)  
   
    
    # combined conditional probabilities
    combined_conditional = {**ham_conditional_p,**spam_conditional_p}    
    
   
    # priors of ham and spam
    prior_ham = len(ham_train_files)/(len(ham_train_files)+len(spam_train_files))
    prior_spam = len(spam_train_files)/(len(ham_train_files)+len(spam_train_files))
    
    prior = [prior_ham, prior_spam]
    
    ham_test_files = load_file_paths('assignment3_test/test/ham_test')
    spam_test_files = load_file_paths('assignment3_test/test/spam_test')
    
    # ham accuracy
    c_p_ham, in_p_ham = NB(prior,ham_test_files,combined_conditional,'ham',stop_words) 
    ham_accuracy = c_p_ham/(c_p_ham + in_p_ham)
    
    # spam_accuracy
    c_p_spam, in_p_spam = NB(prior,spam_test_files,combined_conditional,'spam',stop_words) 
    spam_accuracy = c_p_spam/(c_p_spam+in_p_spam)
    
    # total _accuracy
    total_accuracy = (c_p_ham + c_p_spam)/(c_p_ham + in_p_ham + c_p_spam + in_p_spam)
    
    print("HamAccuracy:",ham_accuracy*100)
    print("SpamAccuracy:",spam_accuracy*100)
    print("TotalAccuracy:",total_accuracy*100) 
    
    
    
    
    
    
    
     
