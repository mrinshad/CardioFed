
#### pip

Write the command below in your terminal to install the dependencies according to the configuration file requirements.txt.

```shell
pip install -r requirements.txt
```

## Run Federated Learning with XGBoost and Flower

Afterwards you are ready to start the Flower server as well as the clients.
You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning.
To do so simply open two more terminal windows and run the following commands.

Start client 1 in the first terminal:

```shell
python3 client.py --partition-id=0 --dataset=./Datasets/data1.csv
python3 client.py --partition-id=1 --dataset=./Datasets/data2.csv 
python3 client.py --partition-id=2 --dataset=./Datasets/data3.csv
python3 client.py --partition-id=3 --dataset=./Datasets/data4.csv