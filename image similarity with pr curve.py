import cv2 as cv
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1

from collections import OrderedDict
directory = 'D:\Personal Kashif\mphil 2019\Courses & Books\Second semester\Selected topics in IR\oxbuild_images'
#directory = 'D:\Personal Kashif\mphil 2019\Courses & Books\Second semester\Selected topics in IR\oxford-500'
#directory = 'D:\Personal Kashif\mphil 2019\Courses & Books\Second semester\Selected topics in IR\one image'
directory1 = 'D:\Personal Kashif\mphil 2019\Courses & Books\Second semester\Selected topics in IR\gt_files_170407'
#directory = 'C:/Users/IT Section/AppData/Local/Programs/Python/Python37'
index={}
images={}
result= {}
dd={}
rank_file=[]
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png")or filename.endswith(".jpeg"):
        print(os.path.join(directory,filename))
        image = cv.imread(os.path.join(directory,filename))
        images[os.path.join(filename)] = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        hist1 = cv.calcHist([image],[0],None,[256],[0,256])# Calculaing histogram for each file 
        hist1 = cv.normalize(hist1, hist1).flatten() # Normalaize the histograms
        index[filename] = hist1
        
    else:
        continue
#print(images)
tot_rel = 0
rel_retrieved=0
Pre=[]# store the precision value
Rec=[] #store the Recall  value
file=''
tot_retrieved=0
counter=0

f=open(os.path.join(directory1,'all_souls_1_good.txt'), "r")
tot_relevant=24 # as there are 24 files in all souls 1 good.txt will be treated as tp
contents=''
contents=f.read()
print(contents)
for (k, hist) in index.items():
    tot_retrieved+=1
    # compute the distance between the two histograms
    # using the method and update the results dictionary
    d = cv.compareHist(index["all_souls_000013.jpg"], hist, 1)
    result[k]=d
#result=(sorted(result.items(), key = lambda kv:(kv[1], kv[0])))
dd = OrderedDict(sorted(result.items(), key=lambda x: x[1]))#sorting the result dictionary as per min Distance
print(dd)    
    
for x in dd:
    file=x # Assigning file name to file
    pos=x.find(".")
    index=x.index(".")
    #print(pos)
    #print(index)
    file=file.rstrip(".jpg")# Removing the extension from the file name
    
    if (file in contents): # Check whether the fie name exist in the contents i.e all tp images
        rank_file.append(file)
        counter=counter+1
        rel_retrieved= rel_retrieved+1        
        Pre.append(rel_retrieved/counter)
        Rec.append(rel_retrieved/tot_relevant)
    else :
        counter=counter+1
        Pre.append(rel_retrieved/counter)
        Rec.append(rel_retrieved/tot_relevant)
print(Pre)
print(Rec)
print("Total match",rel_retrieved)
print(rank_file)

    

# plot the precision-recall curves
plt.plot(Rec, Pre, linestyle='--' )
plt.plot(Rec, Pre, marker='.')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
fig=plt1.figure(figsize=(1, 1))

# show the query image

ax = fig.add_subplot(1, 1, 1)
fig = plt1.figure("Query")
ax.imshow(images["all_souls_000013.jpg"])
plt1.axis("off")
k=1
fig=plt1.figure(figsize=(8, 8))
fig = plt1.figure("Search Result")
n=0
for i in dd:
    n=n+1
    if n==21:
        break
    else:
        print(i)
        ax = fig.add_subplot(4, 5, k)
        ax.imshow(images[i])
        k=k+1
        plt1.xticks(())
        plt1.yticks(())

plt.show()
plt1.show()



     
     
