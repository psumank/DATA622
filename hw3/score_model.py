__author__  = 'Suman'

#When this is called using python score_model.py in the command line, this will ingest the .pkl random forest file and apply the model
#to the locally saved scoring dataset csv. There must be data check steps and clear commenting for each step inside the .py file.
#The output for running this file is a csv file with the predicted score, as well as a png or text file output that contains the model accuracy report
#(e.g. sklearn's classification report or any other way of model evaluation)

#import required packages
import train_model as tm
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('PDF')
import matplotlib.pylab as plt


def load_model(modelfile):
    model_pkl = open(modelfile, 'rb')
    model = tm.pickle.load(model_pkl)
    print("loaded the model ", model)
    return model

def predict_and_save(model, testdf, filename):
    prediction = model.predict(testdf)
    output = tm.pd.DataFrame(testdf.index)
    output['Survived'] = prediction
    output.to_csv(filename)
    return prediction

def classifaction_report_csv(report, filename):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = tm.pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(filename, index = False)

def scoring_analysis(df, model, filename_classification, filename_roc):
    # Split the train dataset into train & test. And find out the score. Model accuracy report.
    # lets first define the target variable.
    y = df.Survived
    features = df.columns.values[1:]
    print(df.head())
    X_train, X_test, y_train, y_test = tm.train_test_split(df[features], y, test_size=0.20, stratify=y)
    y_pred = model.predict(X_test)
    classifaction_report_csv(classification_report(y_test, y_pred), filename_classification)
    print_roc_curve(y_test, y_pred, filename_roc)

def print_roc_curve(y_test, y_pred, filename_roc):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    print(fpr, tpr, thresholds)
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(filename_roc)
    # plt.show()
    fig.savefig(filename_roc)



if __name__ == '__main__':
    #read the downloaded csv file.
    dftest = tm.pd.read_csv('test.csv', index_col=0)
    #impute the data
    dftest = tm.imputedata(dftest)
    #get dummies
    dftest = tm.getdummies(dftest)

    print(dftest.head())

    kNN_pkl_filename = 'kNN_classifier.pkl'
    rf_pkl_filename = 'rf_classifier.pkl'

    if (tm.validate(dftest)):
        print("Ok, we got proper test dataset to operate on!, lets predict and score/evaluate the models...")
        kNNModel = load_model(kNN_pkl_filename)
        prediction_kNN = predict_and_save(kNNModel, dftest,  'predictions_kNN.csv')
        rfModel = load_model(rf_pkl_filename)
        prediction_rf = predict_and_save(rfModel, dftest, 'predictions_RForest.csv')

        #Scoring analysis for kNN model
        df = tm.read_and_clean_training_data()
        scoring_analysis(df, kNNModel, 'classifaction_report_kNN.csv', 'ROC_kNN.png')

        # Scoring analysis for random forest model
        scoring_analysis(df, rfModel, 'classifaction_report_rForest.csv', 'ROC_rForest.png')

        #from the above score and roc curve , its clear that the random forest provides us better accuracy!
    else:
        print("Please review the test data  !, Tidy the test data set !!")