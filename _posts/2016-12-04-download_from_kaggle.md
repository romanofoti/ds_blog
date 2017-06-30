---
layout: post
title: Downloading datasets from Kaggle using Python
---

In this brief post, I will outline a simple procedure to automate the download of datasets from Kaggle.
This script may be useful when one wants to run a model from a remote machine (e.g. a AWS instance) and does not want to spend time moving files between local and remote machines.

How can you achieve this?

Well, there may be several ways, but I make use of two classes. The first, called KaggleRequest() below, handles the request from the Kaggle website, the second one, called DataDownloader() is the one that passes the link of the files we want to download.

So, let us see how each of works.

## Data Downloader

I start with this one, because it is the class with which one interacts directly. In the simplified version I am putting here is really just a simple function that takes a dictionary of urls and the location of the file where you stored your credentials (more on this later). One must provide the url(s) to the kaggle dataset(s) as value(s) in string format in this dictionary, whose keys are just a description of the dataset.

For instance, the three lines below will do the trick:

```python
cred_file = path_to_your_credential_file
url_dc = {
          'train.csv.zip': 'https://www.kaggle.com/c/allstate-claims-severity/download/train.csv.zip',
          'test.csv.zip': 'https://www.kaggle.com/c/allstate-claims-severity/download/test.csv.zip',
          }

DataDownloader(cred_file=cred_file).download_from_kaggle(url_dc)
```

How does the class look like?

```python
class DataDownloader():

    def __init__(self, cred_file=None, verbose=True, logger=None):
        self.cred_file = cred_file
        self.verbose = verbose
        self.logger = logger
    #end

    def download_from_kaggle(self, url_dc=None):
        '''
        Downloads and unzips datasets from Kaggle

        '''
        logger = self.logger
        if url_dc==None:      
            log_or_print('Error: Dictionary of downloading URLs needs to be provided!', logger=logger)
        #end
        for ds, url in zip(url_dc.keys(), url_dc.values()):
            log_or_print('Downloading and unzipping %s ...' %ds, logger=logger)
            KaggleRequest(logger=logger).retrieve_dataset(url)
        #end
        return
    #end

#end
```

## Request Handler

This class handles the request. In the form below, the class can handle a logger and has a verbose or silent mode. Additionally, it requires the following libraries: sys, requests, base64, zipfile.

Here is the gist of the class:

 - You initialize it with information about logger, verbose status and credentials
 - The method retrieve_dataset does the lifting, by establishing the connection with Kaggle, posting the request and downloading the data
 - The name of the dataset can be provided by the user. If not, it is inferred by the url.
 - The method unzip is invoked to unzip the dataset (Kaggle provides zipfiles).
 - The method decrypt is used to decrypt the credentials from the file where you stored them

About the credential encryption/decryption, there is really not much cybersecurity here, just something to make sure that your Kaggle username and password are not in plain view for anyone who happens to peek into your credentials file. The way I do this, is by encoding in base64 my username and password and them saving them in a text file on the same line separated by a comma. Obviously, if you know they are base64 encrypted, it would be easy to steal them, but for the purpose of their intended use it is more than sufficient. Just do not copy them around.

Here is the KaggleRequest() class:


```python
class KaggleRequest():
    '''
    Connects to Kaggle and downloads datasets.
    '''

    def __init__(self, credentials_file=None, verbose=True, logger=None):
        self.verbose = verbose
        self.logger = logger
        if not credentials_file:
            self.credentials_file = './kaggle_cred.cred'
        #end
    #end

    def decrypt(self, credentials_file):
        '''
        This function retrieves the encrypted credential file
        and returns a dictionary with decripted username and password
        '''
        cred_file = open(credentials_file, 'r')
        cred_lines_encry_ls = cred_file.read().split(',')
        try:
            creds_dc = {'UserName': base64.b64decode(cred_lines_encry_ls[0]), 
                        'Password': base64.b64decode(cred_lines_encry_ls[1])}
        except:
            if self.verbose:
                if not self.logger:
                    print 'Problem decrypting credentials. Request terminated.'
                    sys.stdout.flush()
                else:
                    self.logger.info('Problem decrypting credentials. Request terminated.')
                #end
            #end
            return
        #end
        return creds_dc
    #end

    def unzip(self, filename):
        '''
        Unzips a file
        '''
        output_path = '/'.join([level for level in filename.split('/')[0:-1]]) + '/'
        with zipfile.ZipFile(filename, "r") as z:
            z.extractall(output_path)
        #end
        z.close()
        if self.verbose:
            log_or_print('File successfully unzipped!', logger=logger)
        #end
        return
    #end

    def retrieve_dataset(self, data_url, local_filename=None, chunksize=512, unzip=True):
        '''
        Connects to Kaggle website, downloads the dataset one chunk at a time
        and saves it locally.
        '''
        if not data_url:
            if self.verbose:
                log_or_print('A data URL needs to be provided.', logger=logger)
            #end
        if not local_filename:
            try:
                local_filename = './' + data_url.split('/')[-1]
                if self.verbose:
                    log_or_print('Dataset name inferred from data_url. It is going to be saved in the default location.', logger=logger)
                #end
            except:
                if self.verbose:
                    log_or_print('Could not infer data name, request terminated.', logger=logger)
                #end
                return
            #end
        #end
        kaggle_info = self.decrypt(self.credentials_file)
        chunks = chunksize * 1024
        req = requests.get(data_url) # attempts to download the CSV file and gets rejected because we are not logged in
        req = requests.post(req.url, data=kaggle_info, stream=True) # login to Kaggle and retrieve the data
        f = open(local_filename, 'w')
        for chunk in req.iter_content(chunk_size=chunks): # Reads one chunk at a time into memory
            if chunk: # Filtering out keep-alive new chunks
                f.write(chunk)
            #end
        #end
        f.close()
        if self.verbose:
            log_or_print('Data successfully downloaded!', logger=logger)
        #end
        if unzip:
            self.unzip(local_filename)
        #end
        return
    #end

#end
```

Obviously, there is much room for customization and improvement, but I hope you will find this helpful to get started.

#### Appendix

As you may have noticed, I make use of the function log_or_print() throughout. It is a helper function that I created for myself to make sure I either print messages on the screen or log them in a logger file. If a logger - initialized and configured as a python logger - is passed, then the log is saved in the logfile, otherwise it is simply printed as command line output.

Here is the function:

```python
def log_or_print(message, logger=None):
    if not logger:
        print message
        sys.stdout.flush()
    else:
        logger.info(message)
    #end
    return
#end
```

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-101907146-1', 'auto');
  ga('send', 'pageview');

</script>