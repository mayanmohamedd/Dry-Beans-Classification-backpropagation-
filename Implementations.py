import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

class HelperMethods:
    def init(input_size, hidden_sizes, hidden_layer, output_size):

       
        for i in range(hidden_layer):
            if i == 0:
                weights_hidden = [np.random.randn(input_size, hidden_sizes[0])]
                #print("wo", weights_hidden[i].shape)
                bias_hidden = [np.zeros((1, hidden_sizes[0]))]

            else:

                weights_hidden.append(np.random.randn(hidden_sizes[i - 1], hidden_sizes[i]))
                #print("wo", weights_hidden[i].shape)
                bias_hidden.append(np.zeros((1, hidden_sizes[i])))



        weights_output = np.random.randn(hidden_sizes[-1], output_size)
        bias_output = np.zeros((1, output_size))

        return weights_hidden, bias_hidden, weights_output, bias_output
    
    
    def sigmoid(x , derivative=False):
        if derivative:
            return HelperMethods.sigmoid(x) * (1 - HelperMethods.sigmoid(x))
        else:
            return 1 / (1 + np.exp(-x))
    # Derivative of Sigmoid
    def der_sigmoid(x):
        return HelperMethods.sigmoid(x) * (1 - HelperMethods.sigmoid(x))
    
    
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def der_tanh(x):
        return 1 - (HelperMethods.tanh(x) ** 2)
    

        
    def activation_functions(x, function_type, derivative=False):
        if function_type == 0 :
            if derivative:
                return HelperMethods.der_sigmoid(x)
            else:
                return HelperMethods.sigmoid(x)

        elif function_type == 1:
            if derivative:
                return HelperMethods.der_tanh(x)
            else:
                return HelperMethods.tanh(x)
        else:
            raise ValueError("Invalid activation function type")

    def calc_net(x, wh, bh ,function_type  ):
        # hidden_layer_outputs = [X]
        layer_input = np.dot(x, wh) + bh
        layer_output = HelperMethods.activation_functions(layer_input , function_type , derivative=False )

        return layer_output


    def forward(x, wh, bh, wo, bo ,function_type ):
        hidden_layer_outputs = [x]

        for i in range(len(wh)):

            layer_output = HelperMethods.calc_net(hidden_layer_outputs[i], wh[i], bh[i] , function_type )       
            hidden_layer_outputs.append(layer_output)



        predicted_output = HelperMethods.calc_net(hidden_layer_outputs[-1], wo, bo, function_type )


        one_hot_predictions = np.zeros_like(predicted_output)
        one_hot_predictions[np.arange(len(predicted_output)), predicted_output.argmax(axis=1)] = 1

        return one_hot_predictions, hidden_layer_outputs
    
    def backward(weights_output,weights_hidden, pred_otput,  hidden_layer_outputs, wo,bo , ytrain, activationfn, w, bias, lr , flag) :


        error = ytrain - pred_otput
        sK = error * HelperMethods.activation_functions(pred_otput, activationfn, derivative=True)
        sh = [np.dot(sK, weights_output.T) * HelperMethods.activation_functions(hidden_layer_outputs[-1], activationfn, derivative=True)]

        for i in reversed(range(1, len(hidden_layer_outputs) - 1)):
            sh.append(
                np.dot(sh[-1], weights_hidden[i].T) * HelperMethods.activation_functions(hidden_layer_outputs[i], activationfn, derivative=True))


        wo += np.dot(hidden_layer_outputs[-1].T, sK) * lr
        if(flag== True) :
            bias_out = np.sum(sK, axis=0, keepdims=True)
            bo += bias_out * lr


        sh = sh[::-1]
        for i in range(len(w)):
            w[i] += hidden_layer_outputs[i].T.dot(sh[i]) * lr
            
            if (flag == True):
                bias[i] += np.sum(sh[i], axis=0, keepdims=True) * lr


    
class Cleaning_Data:
    def Clean_Transform():
        df = pd.read_excel('Dry_Bean_Dataset.xlsx', engine='openpyxl')
        df['MinorAxisLength'].fillna(df['MinorAxisLength'].median(), inplace=True)
        x = df.iloc[: ,0:5]
        y = df['Class']
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        y = y.values.reshape(-1, 1)

        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse=False)

        # Fit and transform the data
        y = encoder.fit_transform(y)
        y=pd.DataFrame(y)

        # c1=data[0:50] #bombai
        x0 = x[0:50]
        y0 = y[0:50]

        x1 = x[50:100]
        y1 = y[50:100]

        x2 = x[100:150]
        y2 = y[100:150]

        from sklearn.model_selection import train_test_split
        X_train0, X_test0, y_train0, y_test0 = train_test_split(x0, y0, test_size=0.4, random_state=0)  # bomai

        from sklearn.model_selection import train_test_split
        X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.4, random_state=0)  # cali

        from sklearn.model_selection import train_test_split
        X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.4, random_state=0)  # sira

        X_train = np.concatenate([X_train0, X_train1, X_train2], axis=0)
        y_train = np.concatenate([y_train0, y_train1 ,y_train2], axis=0)

        X_test = np.concatenate([X_test0, X_test1 ,X_test2 ], axis=0)
        y_test = np.concatenate([y_test0, y_test1 , y_test2], axis=0)

        return X_train,y_train,X_test,y_test
    
    def confusion_matrix(y_true, y_pred):
        num_classes = y_true.shape[1]
        matrix = np.zeros((num_classes, num_classes), dtype=int)

        for true, pred in zip(y_true, y_pred):
            true_class = np.argmax(true)
            pred_class = np.argmax(pred)
            matrix[true_class, pred_class] += 1

        return matrix
    





def train_neural_network(x_train, y_train, hidden_sizes, hidden_layer, activation_fn, learning_rate, epochs , flag ):
    # weights_hidden ,bias_hidden ,weights_output ,bias_output = init( 5, [10, 15 , 10], 3 ,3)
    weights_hidden, bias_hidden, weights_output, bias_output = HelperMethods.init(5, hidden_sizes, hidden_layer, 3)

    for epoch in range(epochs):
        # pred_otput , hidden_layer_outputs = forward(X_train,weights_hidden,bias_hidden,weights_output ,bias_output , 0)
        predicted_output, hidden_layer_outputs = HelperMethods.forward(x_train, weights_hidden, bias_hidden,weights_output, bias_output, activation_fn)
        #backward( pred_otput,  hidden_layer_outputs, weights_output,bias_output , y_train, 0 , weights_hidden, bias_hidden, 0.001 , True)
        HelperMethods.backward(weights_output,weights_hidden,predicted_output, hidden_layer_outputs,weights_output, bias_output,y_train,activation_fn, weights_hidden,bias_hidden,learning_rate, flag)

    return weights_hidden, bias_hidden, weights_output, bias_output,predicted_output


# Testing the model
def test_neural_network(x_test, weights_hidden, bias_hidden, weights_output, bias_output , active_fun):
    predicted_output, _ = HelperMethods.forward(x_test, weights_hidden, bias_hidden, weights_output, bias_output ,active_fun )
    return predicted_output





