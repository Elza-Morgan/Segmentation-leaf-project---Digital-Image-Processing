from cgitb import reset
from email.mime import image
from importlib.resources import path
from multiprocessing import shared_memory
import cv2
import numpy 
import matplotlib as plt
import os
from skimage import exposure
import warnings
from sklearn.metrics import jaccard_score
warnings.filterwarnings("ignore")
sumTotal =0
averageTotal =0
counterTotal = 10 #number of folders that contain the images

##.......................................................functionFolder1And9................................................
def functionFolder1And9(CameFolder): ## camefolder variable contians  the name of the folder
    global sumTotal
    count = 0
    count2 =0
    all_images = [] # that contains the images
    all_image_GroundTruth = []
    folder = "C:/Users/zizoj/OneDrive/Desktop/project digital/_Output/_Output/"+CameFolder ##name of the file is concat with folder path
    GroundTruthFolder = "C:/Users/zizoj/OneDrive/Desktop/project digital/_Ground_Truth/_GroundTruth/"+CameFolder ##file for comparision

    ## file varibale---> vraibel string poitning first element in the list 
    for file in os.listdir(folder): ## standing at the first folder,,,   listdir---> counts how many images inside and converts them to number in order to count
        if(file.endswith('.JPG')): ## this if condtion says in this folder contains format .jpg it will enter the if condition || checks the format of the image
            img = cv2.imread(os.path.join(folder,file)) #os---> in order to access the files inside your laptop / joins the folder that 
            if img is not None: #checking that there is something in the file
                all_images.append(img) ##adding it to the array
            count +=1 ## moves onto the next image in the list

    for file in os.listdir(GroundTruthFolder):
        if(file.endswith('.JPG')):
            all_image_GroundTruth.append(os.path.join(GroundTruthFolder,file))
            count2 +=1

    print('Total:', count)
    print('Total:', count2)

    average = 0
    sum = 0
    length = len(all_images) ##size/ length of the array
    for i in range(length):
        pickFirst = all_images[i].copy() ## took copy
        ##remove noise
        median = cv2.medianBlur(pickFirst, 3) ## median--> it;s non-linear used to cure salt and paper images, 3---> matrix 3x3
        cv2.imwrite("C:/Users/zizoj/OneDrive/Desktop/project digital/function for folder 1 and 9/image_phase 1_before conversion to HSV_median/" +str(i)+".JPG",median) #option 1

        result = median.copy()          ##covnerter
        image = cv2.cvtColor(result, cv2.COLOR_BGR2HSV) ## convert the image to hsv color, because it's more range colours and also threshhold uses hsv colours
        ## phase 1
        cv2.imwrite("C:/Users/zizoj/OneDrive/Desktop/project digital/function for folder 1 and 9/image_phase 1_after conversion to HSV_median/" +str(i)+".JPG",image)



        #..........................................Threshhold type: Colour MASK...................... 
        #Targeting specific colour 
        lower_yellow = [30, 20, 0] # lower boundary for yellow color-->just setup
        upper_green = [70, 255, 255] # upper boundary for green color-->just setup
        lower = numpy.array(lower_yellow, dtype = "uint8") #numpy-->used to make it easy on us while using arrays
        upper = numpy.array(upper_green, dtype = "uint8")
        mask = cv2.inRange(image, lower, upper) #used method used to create range according what I give it to the method, that's the process itself
        cv2.imwrite("C:/Users/zizoj/OneDrive/Desktop/project digital/function for folder 1 and 9/mask/" +str(i)+".JPG",mask)
        
        result = cv2.bitwise_and(image, image, mask = mask) #returns binary mask shows the colour that is in the range and makes the rest black which are not found in the target colour
            # in order to convert the image to gray we need to convert the image to binary form
        cv2.imwrite("C:/Users/zizoj/OneDrive/Desktop/project digital/function for folder 1 and 9/image_phase3_bitwise/" +str(i)+".JPG",result) #option 1

        #converts the image to grayscale to be able to process
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        eroded = cv2.erode(gray, (1,1) , iterations=1)  # 1X1 erosion matrix ### without it didn't work properly, iteration trun only once, to shrink or remove color white
                #used from opencv directly
        cv2.imwrite("C:/Users/zizoj/OneDrive/Desktop/project digital/function for folder 1 and 9/image_phase4_erod/" +str(i)+".JPG",eroded) #option 1

        #...Threshholding....                              type of the image beacuse we are working on 0's and 1's
        (thresh,threshtest) = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) ## otsu it automaticaly finds the best parts to threshold (cuts the image and takes the white and leaves rest black), that's opening and clsoing outso
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # 5x5 kernel matrix
        closing = cv2.morphologyEx(threshtest, cv2.MORPH_CLOSE, kernel) # like filling in the gap in whitr to black to form a clear shape
        cv2.imwrite("C:/Users/zizoj/OneDrive/Desktop/project digital/function for folder 1 and 9/image_phase4_closing/" +str(i)+".JPG",closing)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel) # works same as erode but that's type of morphology inteaded 
        cv2.imwrite("C:/Users/zizoj/OneDrive/Desktop/project digital/function for folder 1 and 9/image_phase4_opening/" +str(i)+".JPG",opening) #option 1

        median2 = cv2.medianBlur(closing, 15) ## to remove the noise again but this time after converting it to binary 

        image_test = median2                              #converter to grayscale in order to read the images in grey 
        cv2.imwrite("C:/Users/zizoj/OneDrive/Desktop/project digital/function for folder 1 and 9/image_final_process/" +str(i)+".JPG",image_test) ## prints the output on external folder, using option 1


        image1_original = cv2.imread(all_image_GroundTruth[i],cv2.IMREAD_GRAYSCALE) #images in black and white
        print(all_image_GroundTruth[i]) #prints the image with name and folder which is the path
        image1_np_original = numpy.array(image1_original).ravel() #function ravel converts the image to 1d array in order for the jaccard_score to understand it.
        image2_np_test = numpy.array(image_test).ravel()
        output = jaccard_score(image1_np_original, image2_np_test, average='weighted') #'samples' 'macro' 'micro' 'weighted' None

        print(output)
        sum += output
    average = (sum / count)*100
    sumTotal += average ## sum for all the folder
    print("Average of class: ",average)
    average = 0
    sum = 0
    

##..............................................................function............................................................    
def function(CameFolder, sharpness, Camethresh, matrixSize):
    global sumTotal

    count = 0
    count2 =0
    all_images = []
    all_image_GroundTruth = []

    folder = "C:/Users/zizoj/OneDrive/Desktop/project digital/_Output/_Output/"+CameFolder
    GroundTruthFolder = "C:/Users/zizoj/OneDrive/Desktop/project digital/_Ground_Truth/_GroundTruth/"+CameFolder
    for file in os.listdir(folder):
        if(file.endswith('.JPG')):
            img = cv2.imread(os.path.join(folder,file),cv2.IMREAD_GRAYSCALE) #converted to grey from the beggining 
            if img is not None:
                all_images.append(img)
            count +=1

    for file in os.listdir(GroundTruthFolder):
        if(file.endswith('.JPG')):
            all_image_GroundTruth.append(os.path.join(GroundTruthFolder,file))
            count2 +=1
            #to check their number is equale
    print('Total:', count)
    print('Total:', count2)

    average = 0
    sum = 0
    length = len(all_images)
    for i in range(length):
    #remove noise

        pickFirst = all_images[i].copy()
        median = cv2.medianBlur(pickFirst,7) #to remove noise,matrix
        cv2.imwrite("C:/Users/zizoj/OneDrive/Desktop/project digital/function/image_phase1_median/" +str(i)+".JPG",median) 
 
    #sharp the image
        sharpen_filter1 = numpy.array([[-1,-1,-1], [-1, sharpness,-1],[-1,-1,-1]]) ## used to sharp the image
        sharped_img = cv2.filter2D(median, -1, sharpen_filter1) # depth=-1 represnting the result image and have same depth as the source image and this method is used to add the sharpnes onto the image 
        result1 = sharped_img.copy()
        cv2.imwrite("C:/Users/zizoj/OneDrive/Desktop/project digital/function/image_phase2_sharp/" +str(i)+".JPG",result1) ##option2

    #Thresholding ---> Binary threshold
        thresh = Camethresh #mnimum
        maxValue = 255                                      #method used as binary
        (th, dst) = cv2.threshold(result1, thresh, maxValue, cv2.THRESH_BINARY)
        cv2.imwrite("C:/Users/zizoj/OneDrive/Desktop/project digital/function/image_phase3_thresh/" +str(i)+".JPG",dst) ##option 2
        

    #Opening otsu
        imageO = dst
        (T, threshInv) = cv2.threshold(imageO, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (matrixSize,matrixSize))
        opening = cv2.morphologyEx(threshInv, cv2.MORPH_OPEN, kernel)
        cv2.imwrite("C:/Users/zizoj/OneDrive/Desktop/project digital/function/image_phase4_opening/" +str(i)+".JPG",opening) ## option 2


        print(all_image_GroundTruth[i])

        image_original = cv2.imread(all_image_GroundTruth[i],cv2.IMREAD_GRAYSCALE)
        img_original=numpy.array(image_original).ravel()
        img_test=numpy.array(opening).ravel()
        output = jaccard_score(img_original, img_test, average='weighted') #'samples' 'macro' 'micro' 'weighted' None

        print(output)
        sum += output
    
    average = (sum / count)*100
    print("Average of class: ",average)


    sumTotal += average ## sum for all the folder
    sum=0
    average=0


while True:
    option = int(input("enter value from 0-9 : "))
    
    if option == 0:     ##----->>>>> 0.93
        result = "Arjun_(P1)"
        print("Folder = ",result)
        function(result,11, 79 ,5)
 
    elif option == 1:     #-->94
        result =  "Alstonia_Scholaris_(P2)"
        print("Folder ", result)
        functionFolder1And9(result) 

    elif option == 2:     ##----->>>>>83
        result = "Jamun_(P5)"
        print("Folder ", result)
        function(result,10, 115,5)

    elif option == 3:     ##----->>>> 84
        result = "Jatropha_(P6)"
        print("Folder ", result)
        function(result,11, 110,5)

    elif option == 4:     ##----->>>>> 83
        result = "Pongamia_Pinnata_(P7)"
        print("Folder ", result)
        function(result,11, 115,5)

    elif option == 5:     ##----->>>>> 90
        result = "Basil_(P8)"
        print("Folder ", result)
        function(result,10, 115, 5)

    elif option == 6:     ##----->>>>> 91
        result = "Pomegranate_(P9)"
        print("Folder ", result)
        function(result,10, 115, 5)
    
    elif option == 7:     ##----->>>>> 92
        result = "Lemon_(P10)"
        print("Folder ", result)
        function(result,10, 115, 5)
    
    elif option == 8:     ##----->>>>> 84
        result = "Chinar_(P11)"
        print("Folder ", result)
        function(result,10, 115, 5)

    elif option == 9:     ##----->>>>> 92
        result = "Mango_(P0)"
        print("Folder ", result)
        functionFolder1And9(result) 

    elif option ==10:
        averageTotal = sumTotal / counterTotal ## avergae for overall folder classes
        print("Average Total Classes: ", averageTotal)
        averageTotal = 0
    else:
        print("Incorrect option")

