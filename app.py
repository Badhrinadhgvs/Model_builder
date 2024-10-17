import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
import pickle

warnings.filterwarnings('ignore')

linear = LinearRegression()

st.html('<h1>Simple Linear Regression Model Builder</h1>')

df=st.file_uploader("Select Dataset",type='.csv')

dep=st.text_input('Enter dependent variable column')

indep = st.text_input('Enter independent variable column')



def chart(X_train,y_train):
  fig,ax = plt.subplots()
  
  ax.scatter(X_train,y_train)
  ax.plot(X_train,linear.predict(X_train),color='r')
  plt.title("Best Fit Line")
  ax.grid()
  
  return fig

def check(a):
  out = a.isna().sum()
  if out == 0:
    return a
  else:
    a.fillna(np.mean(a),inplace=True)
    return a
    
if df and dep and indep:
  df = pd.read_csv(df)
  a=df.columns
  a=a.str.split(';')
  for i,col in enumerate(a):
    if col == dep:
      dep = a[i]
  for i,col in enumerate(a):
    if col == indep:
      indep = a[i]
  dep = df[dep]
  indep = df[indep]
  dep = check(dep)
  indep = check(indep)
  test_size = st.number_input("Select test size: (range : 0.0 to 1.0)")
  st.html("<p><b>Note</b> : Check accuracy after entering test_size</p>")
  if test_size == 0.0:
    test_size = 0.20 #default test size
  X_train,X_test,y_train,y_test=train_test_split(pd.DataFrame(indep),pd.DataFrame(dep),test_size=test_size,random_state=42)
  linear.fit(X_train,y_train)
  pred = linear.predict(X_test)
  acc = r2_score(pred,y_test)
  st.success(f"Accuraccy: {acc} %")
      
  val = st.number_input("Enter value to predict",value=0)
  if val:
    res = linear.predict([[val]])
    st.success(f"predicted output : {int(np.round(res))}")
  fig = chart(X_train,y_train)
  st.pyplot(fig,clear_figure=True)
  model_data = pickle.dumps(linear)
  st.download_button(
    label="Download Model",
    data=model_data,
    file_name="LinearModel.pkl",
    mime='application/octet-stream')
  st.write("Note : Check 👈 to know how to use model")  
  

footer_html = """
<style>
  .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #1f2937; /* Dark background color */
    color: white;
    text-align: center;
    padding: 10px 0;
  }
  .footer a {
    color: #9ca3af; /* Light gray color for links */
    margin: 0 10px;
    text-decoration: none;
  }
  .footer a:hover {
    color: white; /* White color on hover */
  }
</style>
<footer class="footer">
    <div>
        <p>&copy; 2024 Created by <b>Venkata Sai Badhrinadh.</p>
        <div>
            <a href="https://www.linkedin.com/in/badhrinadhgvs/">Linkedin</a>
            <a href="https://github.com/Badhrinadhgvs">Github</a>
            <a href="https://vsbpersonalportfolio.pythonanywhere.com/
            ">About & Contact</a>
        </div>
    </div>
</footer>
"""

# Display the footer

st.markdown(footer_html, unsafe_allow_html=True)
    
  
  
  
  
    

  
  
  
  
  
  
  


    
    
    

  

