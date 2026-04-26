import RF_model
from data_processor import preprocess, create_3d_dataset, processed_data


Model = RF_model.RandomForestFromScratch()

train_data, train_y  = create_3d_dataset()


Model.fit(train_data,train_y )




#This is the classifier function 
def digit_classify(data, Model = Model):
    """
    This function teake your test data and use our best model to classify the 3D digits

    Parameter: Your test data
    Return: Output lables of class[0,...,9]
    """
    data  =  preprocess(data)
    #Call the Random Forest predict to classify
    return Model.predict_single_sample(data)

