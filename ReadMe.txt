In order to run app you need to satisfy following Conditions
1 Python 3.6 need to be install in your system.
2 Pip is also need to be install in your system along with it(Some time you need to intall seperately). If you have some trouble with your pip try to re install python and carefully tick all require options.
3 After successfully installing Python and pip you need to install streamlit by writing following command in your terminal/command prompt
	"pip install streamlit"
Few Libraries also need to be preinstal before you run this app
1 Numpy write command pip install numpy
2 Pandas write command pip install pandas
3 Matplotlib write command pip install matplotlib
4 Sklearn write command pip install sklearn
5 Seaborn write command pip install seaborn 

All above command you need to write in command promt or on terminal based on your operaion system,
After installing all require libraries transfer your command prompt/terminal execution to your folder where your app.py file is present. Make sure rows.csv file are in same directory along with your app.py file;
After migrating cmd/terminal execution to your desire dirctory or folder type following command in order to run app.py file
"streamlit run app.py"

after you run this command it automatically open your default browser and run app, if due to any reason if it does not open just type localhost:8501

port_id i.e 8501 may be same or may be different just write localhost: and followed by port_id display on your cmd/terminal. Dont try to press ctrl+c or cmd+c as it will stop execution of app and you need to re run command "streamlit run app.py"