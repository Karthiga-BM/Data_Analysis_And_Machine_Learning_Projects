
import seaborn as sns
import os
import  numpy as np


my_file = os.path.join("Confusion Matrix Plot Directory")

#Confusion Matrix for Logistic Regression:
labels = np.array([['3124',  '645'],['1977' ,'1409']])
sns.heatmap(lr_cm, annot=labels, fmt='')
mpl.show()
mpl.figure.savefig("Confusion Matrix Plot Directory\Logistic Regression Confusion Matrix for 100 features")


labels = np.array([['2923' , '846'],['1413' ,'1973']])
sns.heatmap(lr_cm, annot=labels, fmt='')
mpl.show()
mpl.figure.savefig("Confusion Matrix Plot Directory\Logistic Regression Confusion Matrix for 500 features")

labels = np.array([['2984' , '785'],['1175', '2211']])
sns.heatmap(lr_cm, annot=labels, fmt='')
mpl.show()
mpl.figure.savefig("Confusion Matrix Plot Directory\Logistic Regression Confusion Matrix for 1000 features")

labels = np.array([['3077' , '692'],[' 856' ,'2530']])
sns.heatmap(lr_cm, annot=labels, fmt='')
mpl.show()
mpl.figure.savefig("Confusion Matrix Plot Directory\Logistic Regression Confusion Matrix for 5000 features")

labels = np.array([['3117'  ,'652'],[ '829' ,'2557']])
sns.heatmap(lr_cm, annot=labels, fmt='')
mpl.show()
mpl.figure.savefig("Confusion Matrix Plot Directory\Logistic Regression Confusion Matrix for 8000 features")
# #Confusion Matrix for Naive Bayes:
# labels = np.array([['3062', '707],[1962', '1424']])
# sns.heatmap(nb_cm, annot=labels, fmt='')
# mpl.show()
# mpl.figure.savefig("Confusion Matrix Plot Directory\Naive Bayes Confusion Matrix for 100 features")
#
#
# labels = np.array([['2840' , '929'],['1348 ' ,'2038']])
# sns.heatmap(nb_cm, annot=labels, fmt='')
# mpl.show()
# mpl.figure.savefig("Confusion Matrix Plot Directory\Naive Bayes Confusion Matrix for 500 features")
#
# labels = np.array([['2857' , '912'],['1111 ', '2275']])
# sns.heatmap(nb_cm, annot=labels, fmt='')
# mpl.show()
# mpl.figure.savefig("Confusion Matrix Plot Directory\Naive Bayes Confusion Matrix for 1000 features")
#
# labels = np.array([['2951  ' , '818'],[' 770 ' ,'2616']])
# sns.heatmap(nb_cm, annot=labels, fmt='')
# mpl.show()
# mpl.figure.savefig("Confusion Matrix Plot Directory\Naive Bayes Confusion Matrix for 5000 features")
#
# labels = np.array([['2991  '  ,'778'],[ '746 ' ,'2640']])
# sns.heatmap(nb_cm, annot=labels, fmt='')
# mpl.show()
# mpl.figure.savefig("Confusion Matrix Plot Directory\Naive Bayes Confusion Matrix for 8000 features")