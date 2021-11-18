import streamlit as st
import Orange
import pickle
import pandas as pd
import numpy as np




#########################  CONTAINERS  ##############################


# Sidebar
sd_bar = st.sidebar

# Container for the Header
header_cont = st.container()

# Container for the Dataset
dataset_cont = st.container()

# Container for the Features
features_cont = st.container()

# Container for Model Prediction
modelPrediction_cont = st.container()



######################  Load the Stack model  ########################


# StackModel contains Pre-processing pipeline (made on the training dataset)
# + Model (with its parameters & hyperparameters)
with open("StackModel.pkcls","rb") as model:
    loaded_model = pickle.load(model)



########################  CACHE FUNCTION  #############################


@st.cache
def get_data():
    df = pd.read_csv("ChurnTrain.csv")
    return df



########################## SIDEBAR ###################################


with sd_bar:
    st.markdown("## User Input (Existing Customers)")



###################  Extract input from the user  #####################


def get_user_input():

    df = get_data()
    states = np.array(df["state"])
    states_sorted = np.unique(states)
    
    state_val = sd_bar.selectbox(label = "State", options = states_sorted, index = 0)
    intplan_val = sd_bar.selectbox(label = "International plan", options = ["no", "yes"], index = 0)
    voiceplan_val = sd_bar.selectbox(label = "Voice mail plan", options = ["no", "yes"], index = 0)

    numvmailmsgs_val = sd_bar.slider(label = "Number of voice mail messages", min_value = 0,
                                 max_value = 50, value = 0, step = 1)

    ttldaymnts_val = sd_bar.slider(label = "Total day minutes", min_value = 6,
                                 max_value = 352, value = 6, step = 1)

    ttlevemnts_val = sd_bar.slider(label = "Total evening minutes", min_value = 38,
                                 max_value = 360, value = 38, step = 1)

    ttlnghtmnts_val = sd_bar.slider(label = "Total night minutes", min_value = 51,
                                 max_value = 337, value = 51, step = 1)

    ttlintlmnts_val = sd_bar.slider(label = "Total international minutes", min_value = 0,
                                 max_value = 18, value = 0, step = 1)

    ttlintlcalls_val = sd_bar.slider(label = "Total international calls", min_value = 0,
                                 max_value = 16, value = 0, step = 1)

    numcustcalls_val = sd_bar.slider(label = "Number of customer service calls", min_value = 0,
                                 max_value = 7, value = 0, step = 1)


    # define Orange domain
    state = Orange.data.DiscreteVariable("state",[state_val])
    international_plan = Orange.data.DiscreteVariable("international_plan",[intplan_val])
    voice_mail_plan = Orange.data.DiscreteVariable("voice_mail_plan",[voiceplan_val])
    number_vmail_messages = Orange.data.ContinuousVariable("number_vmail_messages")
    total_day_minutes = Orange.data.ContinuousVariable("total_day_minutes")
    total_eve_minutes = Orange.data.ContinuousVariable("total_eve_minutes")
    total_night_minutes = Orange.data.ContinuousVariable("total_night_minutes")
    total_intl_minutes = Orange.data.ContinuousVariable("total_intl_minutes")
    total_intl_calls = Orange.data.ContinuousVariable("total_intl_calls")
    number_customer_service_calls = Orange.data.ContinuousVariable("number_customer_service_calls")

    domain = Orange.data.Domain([state, international_plan, voice_mail_plan, number_vmail_messages,
                                 total_day_minutes, total_eve_minutes, total_night_minutes,
                                 total_intl_minutes, total_intl_calls, number_customer_service_calls]) 

    # input values X
    X = np.array([[0,0,0,numvmailmsgs_val,ttldaymnts_val,ttlevemnts_val,ttlnghtmnts_val,
                   ttlintlmnts_val,ttlintlcalls_val,numcustcalls_val]])

    # in this format, the data is now ready to be fed to StackModel
    user_input = Orange.data.Table(domain, X)

    return user_input


df_userinput = get_user_input()



###########################  HEADER  ##################################

    
with header_cont:
    st.markdown("# Telecom Customer Churn")
    st.markdown("This is a web app for a Machine Learning (ML) model trained and tested in [Orange Data Mining software] (https://orangedatamining.com/). The model "
            "predicts customer churn in telecom industry.")



###########################   DATASET  ##################################

    
with dataset_cont:
    st.markdown("## Dataset")
    st.markdown("The ML model was trained on the dataset below (adopted from [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3)). "
             "The last column (churn) is the target showing whether the customer unsubscribed from the service (yes) or not (no).")
    df = get_data()
    df = df.drop("Unnamed: 0", axis=1)
    
    st.dataframe(df)



##########################   FEATURES  ##################################

    
with features_cont:
    st.markdown("## Features")
    st.markdown("There are 19 input features in this dataset: ")
    st.markdown("(1) state, (2) account_length, (3) area_code, (4) international_plan, (5) voice_mail_plan, "
                "(6) number_vmail_messages, (7) total_day_minutes, (8) total_day_calls, (9) total_day_charge, "
                "(10) total_eve_minutes, (11) total_eve_calls, (12) total_eve_charge, (13) total_night_minutes, "
                "(14) total_night_calls, (15) total_night_charge, (16) total_intl_minutes, (17) total_intl_calls, "
                "(18) total_intl_charge, and (19) number_customer_service_calls.")
    
    st.markdown("However, after training the ML model only the following 10 features were identified as "
                "relevant for making predictions:")
    st.markdown("(1) **state** - the US home state of the customer")
    st.markdown("(2) **international_plan** - indicates whether the customer has international plan or not")
    st.markdown("(3) **voice_mail_plan** - indicates whether the customer has voice mail plan or not")
    st.markdown("(4) **number_vmail_messages** - number of voice mail messages made by the customer")
    st.markdown("(5) **total_day_minutes** - total number of minutes made by the customer during day")
    st.markdown("(6) **total_eve_minutes** - total number of minutes made by the customer during evening")
    st.markdown("(7) **total_night_minutes** - total number of minutes made by the customer during night")
    st.markdown("(8) **total_intl_minutes** - total number of minutes made by the customer on international calls")
    st.markdown("(9) **total_intl_calls** - total number of international calls made by the customer")
    st.markdown("(10) **number_customer_service_calls** - number of calls to the customer service team made")




#############################   MODEL PREDICTION   #########################

    
with modelPrediction_cont:
    st.markdown("## Model Prediction")
    st.markdown("The ML model is a Stack consisting of Gradient Boosting and Random Forest "
            "classifiers with a Logistic Regression acting as an aggregate. To make a prediction "
            "with the model, you need to select values for the 10 features by using the "
            "options and sliders in the sidebar on the left.")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### Input")
        st.write("state:  ", df_userinput[0,0])
        st.write("international plan:  ", df_userinput[0,1])
        st.write("voice mail plan:  ", df_userinput[0,2])
        st.write("number vmail messages:  ", df_userinput[0,3])
        st.write("total day minutes:  ", df_userinput[0,4])
        st.write("total eve minutes:  ", df_userinput[0,5])
        st.write("total night minutes:  ", df_userinput[0,6])
        st.write("total intl minutes:  ", df_userinput[0,7])
        st.write("total intl calls:  ", df_userinput[0,8])
        st.write("number customer service calls:  ", df_userinput[0,9])

    
    probs = loaded_model(df_userinput[0], 1)
    prob_no = probs[0]
    prob_yes = probs[1]
        
    with right_col:
    
        st.markdown("### Prediction Probabilities")
        st.markdown("Given the data on the left, the "
                "probability this customer will not churn, i.e. churn, is:")
        st.write("churn (no): ", round(prob_no*100, 1), "%")
        st.write("churn (yes): ", round(prob_yes*100,1), "%")

        st.markdown("### Prediction")
        if prob_no>prob_yes:
            st.markdown("This customer will **_not_** churn.")
        else:
            st.markdown("This customer **_will_** churn.")
        
