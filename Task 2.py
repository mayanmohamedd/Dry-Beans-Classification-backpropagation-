from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import Implementations
from sklearn.metrics import accuracy_score

master = Tk() 
master.geometry('800x600')
X_train,y_train,X_test,y_test = Implementations.Cleaning_Data.Clean_Transform()

def getInfo(): 
    
    NumbOfHiddenLayers = int(h.get())
    
    Neurons=[]
    nn = (n.get()).split(',') 
    for i in nn:
        Neurons.append(int(i))
    

    LearningRate = float(lr.get())
    NumOfEpochs = int(m.get())
    boolBias = bool(bias.get())
    
    Function = Activation_func.current()  # 0 sigmoid, 1 tangent

    weights_hidden, bias_hidden, weights_output, bias_output,pred_output = Implementations.train_neural_network(X_train,y_train,Neurons,NumbOfHiddenLayers,Function,LearningRate,NumOfEpochs,boolBias)
    accuracy_train = accuracy_score(y_train, pred_output)
    print("Accuracy of Train:", accuracy_train)
    
    predicted_output = Implementations.test_neural_network(X_test,weights_hidden, bias_hidden, weights_output, bias_output,Function)
    accuracy_test = accuracy_score(y_test, predicted_output)
    print("Accuracy of Test:", accuracy_test)

    confusion_m = Implementations.Cleaning_Data.confusion_matrix(y_test, predicted_output)
    print("Confusion Matrix: ")
    print(confusion_m)
    return


# ttk.Label(master, text = "GFG Combobox Widge",  
#           background = 'green', foreground ="white",  
#           font = ("Times New Roman", 15)).grid(row = 0, column = 1) 

Label(master, text='Please Enter Number Of Hidden Layers', padx=10, pady=10).grid(sticky=W)
h = Entry(master)
h.place(x=390,y=15)
h.focus_set()

Label(master, text='Please Enter Of Neurons In Each Hidden Layer (Separated by Comas)',padx=10, pady=10).grid(sticky=W)
n = Entry(master)
n.place(x=390,y=50)
n.focus_set()


Label(master, text='Please Enter Learning Rate (Eta)',padx=10, pady=10).grid(sticky=W)
lr = Entry(master)
lr.place(x=390,y=90)
lr.focus_set()

Label(master, text='Please Enter Number Of Epochs',padx=10, pady=10).grid(sticky=W)
m = Entry(master)
m.place(x=390,y=130)
m.focus_set()


bias = BooleanVar()
Checkbutton(master, text='Add Bias', variable=bias,padx=10).grid(row=10, sticky=W)

Label(master, text='Choose Activation Function',padx=10, pady=10).grid(sticky=W)
func = StringVar()
Activation_func = ttk.Combobox(master, width = 27, textvariable = func)
Activation_func['values']=('Sigmoid','Hyperbolic Tangent Sigmoid')
Activation_func.grid( column=250,row = 11) 


GetInfoButton = Button(master, text='Run The Algorithm', width=30, command=getInfo)
GetInfoButton.place(x=250 , y=300)


master.mainloop() 