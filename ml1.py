from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st
import pandas as pd
import sqlite3
from random import randint
import matplotlib.pyplot as plt
class Model:
    
    
    def __init__(self, datafile = "C:\ML\Train2.csv", model_type = None):
        
        self.df = pd.read_csv(datafile)
        
class Model_2:
    def __init__(self) -> None:
               
        self.df=pd.read_csv("C:\ML\Personal_Loan.csv")      

        

       

class Home_L(Model):
    def __init__(self) -> None:
        pass
    def streamlit(self,df1):
        pass
        #st.write("homeL")
    def Home(self,df1):
    #st.header("Loan Prediction ML")
    # front end elements of the web page 
        html_temp = """ 
        <div style ="background-color:yellow;padding:13px;border-radius:50px;"> 
        <h1 class ="neon"style ="color:black;text-align:center;"> Loan Prediction ML App</h1> 
        </div> <br><hr style = "border: 1px solid red;">
        <b style ="font-size: 30px;"><b style ="font-size: 50px;">W</b>elcome to Home Loan Prediction</b>
            """

        # display the front end aspect
        st.markdown(html_temp, unsafe_allow_html = True) 

        if st.checkbox("Show Dataset"):
            st.write("### Enter the number of rows to view")
            rows = st.number_input("", min_value=0,value=5)
            if rows > 0:
                st.dataframe(df1.head(rows))



        st.subheader("Data Visualization")
    

        if st.checkbox("Plots"):
            st.subheader("Pie chart of Dataset ")
            type_of_plot = st.selectbox("Select Type of Plot",["Gender","Married","Education","Property_Area","Self_Employed","Credit_History","Loan Status"])
            
            st.success("Generating A Pie Plot")
            if type_of_plot == 'Gender':
                st.write(df1.iloc[:,1].value_counts().plot.pie(autopct="%1.1f%%"))
                st.pyplot()
            elif type_of_plot == 'Married':
                st.write(df1.iloc[:,2].value_counts().plot.pie(autopct="%1.1f%%"))
                st.pyplot()
            elif type_of_plot == 'Education':
                st.write(df1.iloc[:,4].value_counts().plot.pie(autopct="%1.1f%%"))
                st.pyplot()
            elif type_of_plot == 'Property_Area':
                st.write(df1.iloc[:,11].value_counts().plot.pie(autopct="%1.1f%%"))
                st.pyplot()
            elif type_of_plot == 'Self_Employed':
                st.write(df1.iloc[:,5].value_counts().plot.pie(autopct="%1.1f%%"))
                st.pyplot()
            elif type_of_plot == "Credit_History":
                st.write(df1.iloc[:,10].value_counts().plot.pie(autopct="%1.1f%%"))
                st.pyplot()
            elif type_of_plot == "Loan Status":
                st.write(df1.iloc[:,12].value_counts().plot.bar())
                st.pyplot()
        


        
        





        if st.checkbox("Show Line Chart "):
            st.line_chart(df1)
    
   
    
    def get_parameters(self,loantype):
        if loantype == "home":
            self.df = pd.read_csv("C:\ML\Train2.csv")
            X = np.array(self.df[['Loan_ID','Gender', 'Married','Education','Self_Employed','ApplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']])
            Y = np.array(self.df['Loan_Status'])     
            self.X=X
            self.Y=Y
            from sklearn.impute import SimpleImputer   #check null values
            si=SimpleImputer(strategy='most_frequent');
            X[:,0:7]=si.fit_transform(X[:,0:7])
            X[:,9:11]=si.fit_transform(X[:,9:11])
            si1=SimpleImputer(strategy='mean');
            X[:,6:9]=si1.fit_transform(X[:,6:9])
            for i in range(0,10):
                if  i==0 or i==1 or i==2 or i==3 or i==4 or i==9 :
                    from sklearn.preprocessing import LabelEncoder
                    le=LabelEncoder();
                    X[:,i]=le.fit_transform(X[:,i])
                else:
                 continue
            from sklearn.preprocessing import StandardScaler   
            ss=StandardScaler()
            X=ss.fit_transform(X)
            from sklearn.model_selection import train_test_split 
            X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25,random_state=69)
            self.X_train=X_train
            self.X_test=X_test
            self.Y_train=Y_train
            self.Y_test=Y_test

            from sklearn.linear_model import LogisticRegression
            Lr=LogisticRegression(random_state=45,fit_intercept=False).fit(self.X,self.Y)
            self.Lr=Lr
            mod=self.Lr.fit(self.X_train,self.Y_train)
            pred=self.Lr.predict(self.X_test)
            from sklearn.metrics import confusion_matrix
            CM = confusion_matrix(self.Y_test,pred)
            print(CM)




            html_temp = """ 
            <div style ="background-color:lightblue;padding:13px;border-radius:20px;"> 
            <h1 style ="color:black;text-align:center;"> Home Loan Prediction </h1> 
            </div> <br><hr style = "border: 1px solid red;">
            """
            st.markdown(html_temp, unsafe_allow_html = True) 
            # display the front end aspect
            Loan_Id = st.number_input("Loan ID")
            Gender = st.selectbox('Gender',("Male","Female"))
            Genders=Gender
            
            Married = st.selectbox('Marital Status',("Unmarried","Married")) 
            Marrieds=Married
            #Dependents = st.selectbox('Dependents',("0","1","2","3")) 
            Education = st.selectbox('Education',('Graduate','Not Graduate'))
            Educations=Education
            ApplicantIncome = st.number_input("Applicants monthly income") 
            #CoapplicantIncome = st.number_input("CoapplicantIncome")
            LoanAmount = st.number_input("Total loan amount")
            Loan_Amount_Term = st.number_input("Loan amount term")
            Credit_History = st.selectbox('Credit_History',("Unclear Debts","No Unclear Debts"))
            Credit_Historys=Credit_History
            Property_Area = st.selectbox('Property Area',('Urban','SemiUrban','Rural'))
            Property_Areas=Property_Area
            Self_Employed = st.selectbox('Self Employed',('Yes','No'))
            Self_Employeds=Self_Employed
            if Gender == "Male":
                Gender = 1
            else:
                Gender = 0
            if Married == "Unmarried":
                Married = 0
            else:
                Married = 1

            if Education == "Graduate":
                Education = 1
            else:
                Education = 0 
            if Property_Area == "Urban":
                Property_Area = 0
            elif Property_Area == "SemiUrban":
                Property_Area = 1
            else:
                Property_Area =2
            if Self_Employed == "Yes":
                Self_Employed = 1
            else:
                Self_Employed = 0    
            if Credit_History == "Unclear Debts":
                Credit_History = 0
            else:
                Credit_History = 1 

            #Lr=LogisticRegression(random_state=45,fit_intercept=False).fit(self.X,self.Y)
            #self.Lr=Lr
            prediction = self.Lr.predict( 
                        [[Loan_Id,Gender, Married,Education, ApplicantIncome, LoanAmount,Loan_Amount_Term,Property_Area,Self_Employed, Credit_History]])
            if prediction == 'Y':
                pred = 'Approved'
            else:
                pred = 'Rejected'


            if st.button("Predict"): 
                if LoanAmount  == 0 or ApplicantIncome == 0  or Loan_Amount_Term ==0 or Loan_Id == 0:
                    st.warning("Some field are Required")
                else:
                    #result = prediction(Loan_ID,Gender, Married,Education, ApplicantIncome, LoanAmount,Loan_Amount_Term,Property_Area,Self_Employed, Credit_History) 
                    st.success('Your loan is {}'.format(pred))
                    print(LoanAmount)
                    st.write(LoanAmount)
                    to_add = {"Gender":[Genders],"Married":[Marrieds],"Education":[Educations],"ApplicantIncome":[ApplicantIncome],"LoanAmount":[LoanAmount],"Loan_Amount_Term":[Loan_Amount_Term],"Property_Area":[Property_Areas],"Self_Employed":[Self_Employeds],"Credit_History":[Credit_Historys]}
                    to_add = pd.DataFrame(to_add)
                    to_add.to_csv("C:\ML\Preprocessing\Home_Loan_db.csv",mode='a',header = False,index= False) 
                    print("s")
                    to_add = {"pred":[pred]}
                    to_add = pd.DataFrame(to_add)
                    to_add.to_csv("C:\ML\Preprocessing\Loanstatus.csv",mode='a',header = False,index= False)     
                    print("s")


            if st.checkbox("Generate Pie Plot"):

                df1 = pd.read_csv("C:\ML\Preprocessing\Home_Loan_db.csv")
                df2 = pd.read_csv("C:\ML\Preprocessing\Loanstatus.csv")
                st.write(df1)
                st.subheader("Pie chart of prediction ")
                type_of_plot = st.selectbox("Select Type of Plot",["Gender","Married","Education","Property_Area","Self_Employed","Credit_History","Loan status"])

                st.success("Generating A Pie Plot")
                if type_of_plot == 'Gender':
                    st.write(df1.iloc[:,0].value_counts().plot.pie(autopct="%1.1f%%"))
                    st.pyplot()
                elif type_of_plot == 'Married':
                    st.write(df1.iloc[:,1].value_counts().plot.pie(autopct="%1.1f%%"))
                    st.pyplot()
                elif type_of_plot == 'Education':
                    st.write(df1.iloc[:,2].value_counts().plot.pie(autopct="%1.1f%%"))
                    st.pyplot()
                elif type_of_plot == 'Property_Area':
                    st.write(df1.iloc[:,6].value_counts().plot.pie(autopct="%1.1f%%"))
                    st.pyplot()
                elif type_of_plot == 'Self_Employed':
                    st.write(df1.iloc[:,7].value_counts().plot.pie(autopct="%1.1f%%"))
                    st.pyplot()
                elif type_of_plot == "Credit_History":
                    st.write(df1.iloc[:,8].value_counts().plot.pie(autopct="%1.1f%%"))
                    st.pyplot()
                elif type_of_plot == "Loan status":
                    st.write(df2.iloc[:,0].value_counts().plot.bar())
                    st.pyplot()
            
        
        if loantype == "personal":
            self.df=pd.read_csv("C:\ML\Preprocessing\Personal_Loan.csv")
            self.X = np.array(self.df[['ID','Age', 'Experience','Income','Family','Education','Mortgage']])
            self.Y = np.array(self.df['Personal Loan'])  


            from sklearn.preprocessing import StandardScaler   
            ss=StandardScaler()
            self.X=ss.fit_transform(self.X)

            from sklearn.model_selection import train_test_split 
            X_train, X_test, Y_train, Y_test = train_test_split(self.X,self.Y, test_size=0.25, random_state=40)
            self.X_train=X_train
            self.X_test=X_test
            self.Y_train=Y_train
            self.Y_test=Y_test

            from sklearn.linear_model import LogisticRegression
            self.slr=LogisticRegression()
            model=self.slr.fit(self.X_train,self.Y_train)
            prediction=self.slr.predict(self.X_test)

            from sklearn.metrics import confusion_matrix
            cm=confusion_matrix(self.Y_test,prediction)
            print(cm)
            from sklearn.metrics import accuracy_score
            ac=accuracy_score(self.Y_test,prediction)
            print(ac)
            html_temp = """ 
            <div style ="background-color:lightblue;padding:13px;border-radius:20px;"> 
            <h1 style ="color:black;text-align:center;"> Personal Loan Prediction </h1> 
            </div> <br><hr style = "border: 1px solid red;">
            """
            st.markdown(html_temp, unsafe_allow_html = True) 
            # display the front end aspect
            #ID =int(st.number_input("ID"))
            Id= int(st.number_input("Loan id"))
            Age = int(st.number_input('Age'))
            Ages=Age
            Experience =int( st.number_input('Experience')) 
            Experiences=Experience
            Income = int(st.number_input('Income')) 
            Incomes=Income
            Family = int(st.number_input("Family")) 
            Families=Family
            Mortgage = int(st.number_input("Mortgage")) 
            Mortgages=Mortgage
            Education = st.selectbox('Education',('Ungraduate','Graduate','Professional'))
            Educations=Education
            if Education == "Ungraduate":
                Education = 1
            elif Education == "Graduate":
                Education = 2
            elif Education == "Professional":
                Education = 3
            

            #Lr=LogisticRegression(random_state=45,fit_intercept=False).fit(self.X,self.Y)
            #self.Lr=Lr
            prediction = self.slr.predict( 
                        [[Id,Age, Experience,Income,Family,Education,Mortgage]])
            if prediction == 1:
                pred = 'Approved'
            else:
                pred = 'Rejected'


            if st.button("Predict"): 
                if Age  == 0 or Experience == 0  or Income ==0 or Family ==0 or Mortgage ==0 :
                    st.warning("Some field are Required")
                else:
                    #result = prediction(Loan_ID,Gender, Married,Education, ApplicantIncome, LoanAmount,Loan_Amount_Term,Property_Area,Self_Employed, Credit_History) 
                    st.success('Your loan is {}'.format(pred))
                    
                    to_add = {"Ages":[Ages],"Experiences":[Experiences],"Incomes":[Incomes],"Families":[Families],"Mortgages":[Mortgages],"Educations":[Educations]}
                    to_add = pd.DataFrame(to_add)
                    to_add.to_csv("C:\ML\Preprocessing\Personal_Loan_db.csv",mode='a',header = False,index= False)    

            if st.checkbox("Generate Pie Plot"):

                df1 = pd.read_csv("C:\ML\Preprocessing\Personal_Loan_db.csv")
                st.write(df1)
                st.subheader("Pie chart of prediction ")
                type_of_plot = st.selectbox("Select Type of Plot",['Age', 'Experience','Income','Family','Education','Mortgage'])

                st.success("Generating A Pie Plot")
                if type_of_plot == 'Age':
                    st.write(df1.iloc[:,0].value_counts().plot.pie(autopct="%1.1f%%"))
                    st.pyplot()
                elif type_of_plot == 'Experience':
                    st.write(df1.iloc[:,1].value_counts().plot.pie(autopct="%1.1f%%"))
                    st.pyplot()
                elif type_of_plot == 'Income':
                    st.write(df1.iloc[:,2].value_counts().plot.pie(autopct="%1.1f%%"))
                    st.pyplot()
                elif type_of_plot == 'Family':
                    st.write(df1.iloc[:,3].value_counts().plot.pie(autopct="%1.1f%%"))
                    st.pyplot()
                elif type_of_plot == 'Education':
                    st.write(df1.iloc[:,4].value_counts().plot.pie(autopct="%1.1f%%"))
                    st.pyplot()
                elif type_of_plot == "Mortgage":
                    st.write(df1.iloc[:,5].value_counts().plot.pie(autopct="%1.1f%%"))
                    st.pyplot()

            

    def About_us(self):
        st.title("Loan Prediction")
        st.title("About us")
        html_temp = """ 
        <hr style = "border: 1px solid black;">
        """
        # display the front end aspect
        st.markdown(html_temp, unsafe_allow_html = True) 
        st.subheader("The project automate the loan eligibility process (real time) based on customer detail provided while filling online application form.  It's a classification problem , given information about the application we predict whether the they'll be to pay the loan or not.")
        html_temp = """ 
        <br><br>
        """# display the front end aspect
        st.markdown(html_temp, unsafe_allow_html = True) 
        if st.checkbox("Developer information"):
            st.subheader("Sadik Tamboli")
            if st.checkbox("More info"):
                st.write("tambolisadik16@gmail.com")
            st.subheader("Sahil Mujhavar")
            #if st.checkbox("More  info"):
            #    st.write("")
            st.subheader("Aashutosh Maurya")
            #if st.checkbox("More   info"):
            #    st.write("")
        html_temp = """ 
        <br><br><br><br>
        """# display the front end aspect
        st.markdown(html_temp, unsafe_allow_html = True) 
        C=st.text_input('Comments:') 
        if st.button("Submit"):
            if C == "":
                st.warning("Required")
            else:
                to_add = {"Comment":[C]}
                to_add = pd.DataFrame(to_add)
                to_add.to_csv("C:\ML\Preprocessing\Comments.csv",mode='a',header = False,index= False)                  
                st.success("Success")
    
class Personal_L:
    def __init__(self) -> None:
      
        pass
    def Home(self,df1):

        #st.header("Loan Prediction ML")
        # front end elements of the web page 
        html_temp = """ 
        <div style ="background-color:yellow;padding:13px;border-radius:50px;"> 
        <h1 style ="color:black;text-align:center;"> Loan Prediction ML App</h1> 
        </div> <br><hr style = "border: 1px solid red;">
        <b style ="font-size: 30px;"><b style ="font-size: 50px;">W</b>elcome to Personal Loan Prediction</b>
            """

        # display the front end aspect
        st.markdown(html_temp, unsafe_allow_html = True) 

        if st.checkbox("Show Dataset"):
            st.write("### Enter the number of rows to view")
            rows = st.number_input("", min_value=0,value=5)
            if rows > 0:
                st.dataframe(df1.head(rows))



        st.subheader("Data Visualization")
    

        if st.checkbox("Plots"):
            st.subheader("Pie chart of Dataset ")
            type_of_plot = st.selectbox("Select Type of Plot",['Family','Education'])
            
            st.success("Generating A Pie Plot")
            
            if type_of_plot == 'Family':
                st.write(df1.iloc[:,5].value_counts().plot.pie(autopct="%1.1f%%"))
                st.pyplot()
            elif type_of_plot == 'Education':
                st.write(df1.iloc[:,7].value_counts().plot.pie(autopct="%1.1f%%"))
                st.pyplot()
            
            
      
        if st.checkbox("Show Line Chart "):
            st.line_chart(df1)





class Databases:

    def __init__(self) -> None:
        
        conn = sqlite3.connect('data.db')
        c = conn.cursor()
        self.conn=conn
        self.c=c

    def getdata(self,new_user,new_passwd,value):
        to_add = {"new_user":[new_user],"new_passwd":[new_passwd],"Value":[value]}
        to_add = pd.DataFrame(to_add)
        add=to_add.to_csv("C:\ML\Preprocessing\HL_db.csv",mode='a',header = False,index= False)              
        st.success("Success")



    

    def create_usertable(self):
	    self.c.execute('CREATE TABLE IF NOT EXISTS Data(username TEXT,password TEXT,value TEXT)')

    def add_userdata(self,username,password,value):
	    self.c.execute('INSERT INTO Data(username,password,value) VALUES (?,?,?)',(username,password,value))
	    self.conn.commit()

    def login_user(self,username,password):
	    self.c.execute('SELECT * FROM Data WHERE username =? AND password = ? ',(username,password))
	    data = self.c.fetchall()
	    return data 

    def get_id(self,user,passwd):
        self.c.execute('SELECT value FROM Data WHERE username =? AND password = ? ',(user,passwd))
        data = self.c.fetchall()
        return data 
        

    
        
            

    
if __name__ == '__main__':
    a=Model()
    b=Model_2()
   # a.predict()
    st.sidebar.header("Select loan type")
    loanopt = st.sidebar.radio("",["Home Loan","Personal Loan"])

    st.sidebar.title("Navigation")
    nav = st.sidebar.radio("",["Home ","Prediction ","About us "])

    if loanopt == "Home Loan":
        H=Home_L()
        
        if nav == "Home ":
            H.Home(a.df)
        if nav == "Prediction ":
            st.title("Loan Prediction")
            opt = st.radio("",["Signup","Login"])
            DB=Databases()
            if opt=="Signup":
                st.subheader("Create an Account")
                new_user = st.text_input('Create Username')
                new_passwd = st.text_input('Create Password',type='password')
                if st.button("Signup"):

                    if new_user =="" or new_passwd == "":
                        st.warning("Invalid username or password")
                    else:
                        value = randint(10000, 99999)
                        DB.create_usertable()
                        DB.add_userdata(new_user,new_passwd,value)
                        DB.getdata(new_user,new_passwd,value)
                        st.info("Your generation ID")
                        st.write(value)
                        st.success("You have successfully created an account.Go to the Login Menu to login")
            d=0
            if opt == "Login":
                
                st.subheader("Login")
                user = st.text_input('Username')
                passwd = st.text_input('Password',type='password')
                if st.button("Login"):
                    if user =="" or passwd == "":
                        st.warning("Invalid username or password")
                    else:
                        result = DB.login_user(user,passwd)
                        a=st.success("Logged in as {}.Below is your registered loan id".format(user))
                        data=DB.get_id(user,passwd)
                        st.info(data)
                        d=1
            
                        st.title("Welcome  {}".format(user))            
                home="home"
                H.get_parameters(home)


                
        if nav == "About us ":
            H.About_us()
        #H.streamlit(a.df)



    if loanopt == "Personal Loan":
        P=Personal_L()
        H=Home_L()
        if nav == "Home ":
            P.Home(b.df)
        if nav == "Prediction ":
            st.title("Loan Prediction")
            opt = st.radio("",["Signup","Login"])
            DB=Databases()
            if opt=="Signup":
                st.subheader("Create an Account")
                new_user = st.text_input('Create Username')
                new_passwd = st.text_input('Create Password',type='password')
                if st.button("Signup"):

                    if new_user =="" or new_passwd == "":
                        st.warning("Invalid username or password")
                    else:
                        value = randint(10000, 99999)
                        DB.create_usertable()
                        DB.add_userdata(new_user,new_passwd,value)
                        DB.getdata(new_user,new_passwd,value)
                        st.info("Your generation ID")
                        st.write(value)
                        st.success("You have successfully created an account.Go to the Login Menu to login")
            if opt == "Login":
                st.subheader("Login")
                user = st.text_input('Username')
                passwd = st.text_input('Password',type='password')
                if st.button("Login"):
                    if user =="" or passwd == "":
                        st.warning("Invalid username or password")
                    else:
                        result = DB.login_user(user,passwd)
                        if result:
                        
                            a=st.success("Logged in as {}.Below is your registered loan id".format(user))
                            data=DB.get_id(user,passwd)
                            st.info(data)
                            st.title("Welcome  {}".format(user))            
                Personal="personal"
                H.get_parameters(Personal)
        if nav == "About us ":
            H=Home_L()
            H.About_us()