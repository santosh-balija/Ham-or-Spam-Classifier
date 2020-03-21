import os
import string
import numpy as np

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
    

def Sigmoid(x):
    sig = 1.0/(1 + np.exp(-x))
    return sig

def update_weights(l,iterations, weights, words, true_labels):
                     
        for i in range(iterations): 
            p = np.dot(words,weights)            
            t = np.array(p,dtype=np.float32)
            s = Sigmoid(t)
            labels = true_labels
            a = np.dot(words.transpose(),labels - s)  
            weights = weights + (0.001*(a))-(0.001*l*weights)  
            
        return weights

def LR(test_data, label, weights):
    y = np.array(np.dot(test_data,weights),dtype=np.float32)
    c_h =0
    c_s = 0
    in_h = 0
    in_s = 0
    p_labels=[]
  
    for i in y:
        if i>0:
            p_labels.append(1)
        else:
            p_labels.append(0)
    
    for i in range(len(label)):
        if label[i] == p_labels[i]:
            if label[i] == 0:
                c_s = c_s + 1
            else:
                c_h = c_h + 1
        else:        
            if p_labels[i] == 0:
                in_s = in_s + 1
            else:
                in_h = in_h + 1
                
    
    return c_h,c_s,in_h,in_s

if __name__=='__main__':
    stop_words = []
    
    pathstop = '.\\stopwords.txt'
    f = open(pathstop, 'r')  
    # To append all the stopwords in a list
    stop_words = []
    for line in f.readlines():
        stop_words.append(line.strip())
    
    # file paths for both ham and spam train fils
    ham_train_files = load_file_paths('assignment3_train/train/ham')
    spam_train_files = load_file_paths('assignment3_train/train/spam')
    
    all_train = ham_train_files + spam_train_files
    #Words in Ham and Spam train files 
    ham_train_words  =  all_words(ham_train_files,stop_words)
    spam_train_words = all_words(spam_train_files,stop_words)
    # combined train words in ham and spam
    words_combined = ham_train_words + spam_train_words
    # unique words in words_combined 
    words_combined_unique = list(dict.fromkeys(words_combined)) 
    

    
    c =0;
    train_dict = dict()  
    
    for file in all_train:
        all_words_train = []
        f = open(file,'r',errors ='ignore')     
        words = f.read().replace('\n',' ')
        all_words_train.extend(words.split())
        all_words_train = [''.join(c for c in s if c not in string.punctuation) for s in all_words_train]   
        all_words_train = [x for x in all_words_train if x]
        all_words_train = [x for x in all_words_train if x not in stop_words]
        
        r = [0]*len(words_combined_unique)
        for i in range(len(words_combined_unique)):        
            if words_combined_unique[i] in all_words_train:
                feature_count = all_words_train.count(words_combined_unique[i])
                r[i] = feature_count               
            else:
                r[i] = 0
        r.insert(0,1)                
        train_dict[c]=r
        c = c + 1
    
        

    
    labels = []             
    for i in range(len(ham_train_files)):
        labels.append(1)
        
    for i in range(len(spam_train_files)):
        labels.append(0)
    
    labels_np = np.asarray(labels)
    
    
    words_each_file = list(train_dict.values())
    
    words_each_file_np = np.asarray(words_each_file)
 
    ham_test_files = load_file_paths('assignment3_test/test/ham_test')
    spam_test_files = load_file_paths('assignment3_test/test/spam_test')
    
    all_test = ham_test_files + spam_test_files
    
    test_dict = dict()
    d = 0
    for file in all_test:
        all_words_test = []
        f = open(file,'r',errors ='ignore')     
        wordstest = f.read().replace('\n',' ')
        all_words_test.extend(wordstest.split())
        all_words_test = [''.join(c for c in s if c not in string.punctuation) for s in  all_words_test]   
        all_words_test = [x for x in all_words_test if x]
        all_words_test = [x for x in all_words_test if x not in stop_words]
        
        q = [0]*len(words_combined_unique)
        for i in range(len(words_combined_unique)):        
            if words_combined_unique[i] in all_words_test:
                feature_count = all_words_test.count(words_combined_unique[i])
                q[i]=feature_count               
            else:
                q[i]=0
                
        q.insert(0,1)                
        test_dict[d]=q
        d = d + 1
    
    labels_1 = []             
    for i in range(len(ham_test_files)):
        labels_1.append(1)
    for i in range(len(spam_test_files)):
        labels_1.append(0)
    
    words_each_file_t = list(test_dict.values())
    
    words_each_file_t_np = np.asarray(words_each_file_t)
    
    lamda = [0.05,0.09,2.5]
    
    iterations = 100
    
    for l in lamda:
        weights = np.zeros(words_each_file_np.shape[1])

        new_weights = update_weights(l,iterations,weights,words_each_file_np,labels_np)      
   
        c_p_ham, c_p_spam, in_p_ham, in_p_spam = LR( words_each_file_t_np, labels_1, new_weights )
    
        ham_accuracy = c_p_ham/(c_p_ham + in_p_ham)
    
        spam_accuracy = c_p_spam/(c_p_spam+in_p_spam)
    
        total_accuracy = (c_p_ham + c_p_spam)/(c_p_ham + in_p_ham + c_p_spam + in_p_spam)
    
        print("HamAccuracy :",ham_accuracy*100)
    
        print("SpamAccuracy :",spam_accuracy*100)
        
        print("TotalAccuracy :",total_accuracy*100)
        
        print("\n")
    