__author__ = 'Suman'

#import required packages
import requests
from ruamel.yaml import YAML
import os
import sys


def readcredentials(yamlfile):
    """
    read credentials from the config file (yaml), and provide the creds object back.
    :param yamlfile:
    creds.yaml file must be in the below format:
    -------
    creds:
       username: here-goes-your-user-name
       password: here-goes-your-password
    -------
    :return: creds dictionary object
    """
    try:
        code = YAML().load(open(yamlfile).read())
        print(code['creds']['username'])
        return {'UserName' : code['creds']['username'] , 'Password': code['creds']['password']}
    except IOError:
        print('No such file %s' % yamlfile)


#function to download file from given URL
def downloadfile(url, credentials, filename):
    """
    download the file from the given 'url' using the 'credentials', and output the contents as 'filename'
    :param url:
    :param credentials:
    :param filename:
    :return:
    """
    try:
        reader = requests.get(url)
        #print('reader is ', reader.url)
        #headers = {'Accept' : 'text/csv'}
        #reader = requests.post(reader.url, data=credentials, headers = headers )
        f = open(filename, 'wb')
        # read in 1 MB chunks.
        for chunk in reader.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    except ValueError:
        print('%s is not a vlid URL.' % url)
    finally:
        f.close()

#function to return record count
def recordcount(filename):
    """
    For a given filename, count the number of records.
    :param filename:
    :return: number of lines
    """
    try:
        f = open(filename)
        return len(f.readlines())
    except IOError:
        print ('No such file %s' % filename)
    finally:
        f.close()

if __name__ == '__main__':
    #initialize inputs.
    train_url = "https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/train.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1508113266&Signature=Weizkt%2F0JHltbqV5beCLIDelHvAFPxtK%2FetUBoqxaQfBTcnEVt5uREgTnXehUo4vlOMLWhU5AkNZoOUCkYJc5YyMjCpMIJFh1ptNtWOW09FBwo8mwuNiEQj9qXG33O65EqR2KtAiAMFc%2B8KKX34QKWbF4NtbNdRPMUTFx0WD2wjsajD5fQuAQ1LtT%2Bky13nrp1HTF5V7zyFoidiEfxszvwwzdhPIPFPlQf9u5gpjy31R6JUx2wdxnjeYI%2FrttTDHAppWyuBetvp3RfbUL9xsgw7ilfFE6O6y2M2BUeQcsHVQeNgW9OcJkqmVjnCIdSQXmPkAU2LQn%2FIcyM9NoGANJg%3D%3D"
    #"https://www.kaggle.com/c/titanic/download/train.csv"
    test_url = "https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/test.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1508113268&Signature=b6dXXxlx3b65JKQFs2rwTp%2FtE9uZo93m07wF0FIKhvI1T%2FRvuW3o%2FbV1LdggCEUldofQjtU6ytVyt3%2FtxcKqIEI1NG4wxJ9dDTwW%2Fbr4mDyIw1ckESOUkl%2By%2BLb1qhd%2BxIDyjDA7JIj8aUJeXp%2BN67rjFKcwn5Hov4rwvrZt%2BvSUfTW%2BTMIAi9wPyNecXog2BsphBft5zeQluo9ts3yzLfZNYaHPpeWNJ2djIpXiPAsKJ77AHhT%2Bt4fMw0e3l5PI%2BxKVC0%2F5JmWALdsqN%2Fzl2iDQkHUlYc6M3gwZ6yIPIT6p3Lfle5nsk1k%2F6hIW6ieTO0%2B7ifo8K10gJTmSFmqE%2BA%3D%3D"
    #"https://www.kaggle.com/c/titanic/download/test.csv"
    creds_config = 'creds.yaml'
    train_filename = "train.csv"
    test_filename = "test.csv"

    #handling ssl certificate location
    os.environ['REQUESTS_CA_BUNDLE'] = os.path.join('/etc/ssl/certs/', 'ca-bundle.crt')

    #read the config file (yaml) for creds.
    kaggle_creds = readcredentials(creds_config)

    # download train and test files.
    downloadfile(train_url, kaggle_creds, train_filename)
    downloadfile(test_url, kaggle_creds, test_filename)

    #validate the data.
    print("Number of training records: %d" % recordcount(train_filename))
    print("Number of test records: %d" % recordcount(test_filename))