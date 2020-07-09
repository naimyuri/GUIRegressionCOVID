import pandas as pd
from sklearn import metrics, linear_model, neighbors
from sklearn.metrics import explained_variance_score
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import tkinter as tk
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkcalendar import *
import datetime
from datetime import date

currency_exchange = pd.read_csv('HistoryExchange.csv')
df = pd.DataFrame(currency_exchange,columns=['Date','Days','Active Cases MY','New Cases MY','Recovered Cases MY','Death Cases MY','Active Cases IND','New Cases IND','Recovered Cases IND','Death Cases IND','Active Cases SG','New Cases SG','Recovered Cases SG','Death Cases SG','Active Cases JP','New Cases JP','Recovered Cases JP','Death Cases JP','Rate']) 

root= tk.Tk()
root.title("MYR to USD Forecasting")

selectedDate = StringVar()

canvas1 = tk.Canvas(root, width = 550, height = 230, bg='white')
canvas1.pack(expand=1)

clmns = [ 'New Cases MY',
          'New Cases SG',
          'New Cases JP',
          'New Cases IND',
          'Death Cases MY',
          'Death Cases SG',
          'Death Cases JP',
          'Death Cases IND',
          'Active Cases MY',
          'Active Cases SG',
          'Active Cases JP',
          'Active Cases IND',
          'Recovered Cases MY',
          'Recovered Cases SG',
          'Recovered Cases JP',
          'Recovered Cases IND'
          ]

techniques = [ 'SVR',
               'Ada Boost',
               'Linear Model',
               'Random Forest',
               'Decision Tree',
               'K-Nearest Neighbors',
               'Stochastic Gradient Descent',
               'Compare Cross Decomposition'
               ]

clicked = StringVar()
clicked.set(clmns[0])

clicked1 = StringVar()
clicked1.set(techniques[0])

label3 = tk.Label(root, text='Regression Techniques: ', font='courier 8')
canvas1.create_window(215, 40, window=label3)

drop1 = OptionMenu(canvas1, clicked1, *techniques)
drop1.config(bg='floral white' , font='courier 8')
drop1['menu'].config(bg='floral white' , font='courier 8')
canvas1.create_window(415, 40, window=drop1)

label4 = tk.Label(root, text='Attributes: ', font='courier 8')
canvas1.create_window(255, 70, window=label4)

drop = OptionMenu(canvas1, clicked, *clmns)
drop.config(bg='floral white' , font='courier 8')
drop['menu'].config(bg='floral white' , font='courier 8')
canvas1.create_window(385, 70, window=drop)

def select_date():
    top = Toplevel(root)
    cal = Calendar(top, font='courier 8',
                   selectmode='day', year = 2020, month =1, day=26,
                   mindate=datetime.date(2020, 1, 26))
    cal.pack(fill='both', expand=True)
    def cal_date(e):
        result = cal.selection_get() - date(2020, 1, 26)
        selectedDate.set(result.days)
        top.destroy()
        
    cal.bind("<<CalendarSelected>>", cal_date)
    
    
label1 = tk.Label(root, text='Input days since outbreak (1/26/2020): ', font='courier 8')
canvas1.create_window(160, 100, window=label1)

entry1 = tk.Entry (root, font='courier 8', state='disabled', text=selectedDate)
canvas1.create_window(370, 100, window=entry1)

button3 = tk.Button (root,bg='floral white', text='Select Date',command=select_date, font='courier 8')
canvas1.create_window(500, 100, window=button3)

label2 = tk.Label(root, text='Input amount of cases: ', font='courier 8')
canvas1.create_window(216, 120, window=label2)

entry2 = tk.Entry (root, font='courier 8')
canvas1.create_window(370, 120, window=entry2)

label_info = tk.Label(root, bg='floral white', justify='left', font='courier 8', text='MY = Malaysia\nSG = Singapore\nJP = Japan\nIND = Indonesia')
canvas1.create_window(60, 170, window=label_info)

def values():
    global scatter3
    figure3 = plt.Figure(figsize=(5,4), dpi=100)
    ax3 = figure3.add_subplot()

    X = df[['Days',clicked.get()]].astype(float) 
    Y = df['Rate'].astype(float)

    if clicked1.get() == 'Linear Model':
        regr = linear_model.LinearRegression()
        print(clicked1.get() + ' of ' + clicked.get())
    elif clicked1.get() == 'SVR':
        regr = SVR(kernel='rbf')
        print(clicked1.get() + ' of ' + clicked.get())
    elif clicked1.get() == 'Stochastic Gradient Descent':
        regr = make_pipeline(StandardScaler(),SGDRegressor(alpha=0.1,max_iter=500, tol=1e-3))
        print(clicked1.get() + ' of ' + clicked.get())
    elif clicked1.get() == 'Compare Cross Decomposition':
        regr = PLSRegression(n_components=2)
        print(clicked1.get() + ' of ' + clicked.get())
    elif clicked1.get() == 'Random Forest':
        regr = RandomForestRegressor(max_depth=2, random_state=0)
        print(clicked1.get() + ' of ' + clicked.get())
    elif clicked1.get() == 'Ada Boost':
        regr = AdaBoostRegressor(random_state=0, n_estimators=500)
        print(clicked1.get() + ' of ' + clicked.get())
    elif clicked1.get() == 'Decision Tree':
        regr = DecisionTreeRegressor(max_depth=3)
        print(clicked1.get() + ' of ' + clicked.get())
    elif clicked1.get() == 'K-Nearest Neighbors':
        regr = KNeighborsRegressor(n_neighbors=3)
        print(clicked1.get() + ' of ' + clicked.get())

    regr.fit(X, Y)
    New_Days = float(entry1.get())
    New_Case = float(entry2.get())

    global label_Prediction
    global label_accuracy
    Prediction_result  = ('Predicted Rate: ', regr.predict([[New_Days ,New_Case]]))
    label_Prediction = tk.Label(root,bg='floral white',font='courier 8', text= Prediction_result)
    canvas1.create_window(260, 180, window=label_Prediction)

    print(' with accuracy of ', explained_variance_score(Y, regr.predict(df[['Days',clicked.get()]].astype(float))))

    accuracy_result = ('Accuracy: ', explained_variance_score(Y, regr.predict(df[['Days',clicked.get()]].astype(float))))
    label_accuracy = tk.Label(root,bg='floral white',font='courier 8', text = accuracy_result )
    canvas1.create_window(260, 210, window=label_accuracy)

    ax3.scatter(df['Days'], df['Rate'], color='black')
    ax3.plot(df['Days'], regr.predict(df[['Days',clicked.get()]].astype(float)), color = 'r')
    scatter3 = FigureCanvasTkAgg(figure3, root)
    scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    ax3.legend(['Predicted Data','Actual Data']) 
    ax3.set_xlabel('Days since COVID-19 outbreak')
    ax3.set_title(clicked1.get() + ' Regression of ' + clicked.get() +' vs Rate')
    button1['state'] = 'disabled'
    button2['state'] = 'active'

def clear():
    scatter3.get_tk_widget().pack_forget()
    label_Prediction['text'] = ''
    label_accuracy['text'] = ''
    button1['state'] = 'active'
    button2['state'] = 'disabled'

button1 = tk.Button (root,bg='floral white', text='Predict Rate',command=values, font='courier 8') 
canvas1.create_window(345, 150, window=button1)

button2 = tk.Button (root, state='disabled', text='Clear',bg='floral white',command=clear, font='courier 8')
canvas1.create_window(420, 150, window=button2)

root.mainloop()
