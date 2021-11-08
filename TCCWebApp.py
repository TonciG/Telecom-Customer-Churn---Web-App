# Below is a web app for Telecom Customer Churn sample project from
# 7120DMKT module (2021/22).


import streamlit as st
import pandas as pd
import numpy as np
import Orange
import pickle




#########################  CONTAINERS  ##############################


# container = horizontal section (st.container())
# column = vertical section (st.columns())
# Use "with container_name/column_name:" to add the content to
# container/column

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


# @st.cache - Function decorator to memorize function executions.
# st.header() - Display text in header formatting.
# st.text() - Write fixed-width and preformatted text.
# st.write() - Write arguments to the app.
# st.subheader() - Display text in subheader formatting.
# st.markdown() - Display string formatted as Markdown.
# st.slider() - Display a slider widget.
# st.selectbox() - Display a select widget.
# st.table() - Display a static table.



########################  CACHE FUNCTIONS  ############################


@st.cache
def get_data():
    df = pd.read_csv("ChurnTrain.csv")
    return df

    
@st.cache
def get_scaling_values():

    df = get_data()

    numvmailmsgs_min = df["number_vmail_messages"].min(axis = 0)
    numvmailmsgs_max = df["number_vmail_messages"].max(axis = 0)

    ttldaymnts_min = df["total_day_minutes"].min(axis = 0)
    ttldaymnts_max = df["total_day_minutes"].max(axis = 0)

    ttlevemnts_min = df["total_eve_minutes"].min(axis = 0)
    ttlevemnts_max = df["total_eve_minutes"].max(axis = 0)

    ttlnghtmnts_min = df["total_night_minutes"].min(axis = 0)
    ttlnghtmnts_max = df["total_night_minutes"].max(axis = 0)

    ttlintlmnts_min = df["total_intl_minutes"].min(axis = 0)
    ttlintlmnts_max = df["total_intl_minutes"].max(axis = 0)

    ttlintlcalls_min = df["total_intl_calls"].min(axis = 0)
    ttlintlcalls_max = df["total_intl_calls"].max(axis = 0)

    nmcustcalls_min = df["number_customer_service_calls"].min(axis = 0)
    nmcustcalls_max = df["number_customer_service_calls"].max(axis = 0)

    scaling_values = {"number_vmail_messages_min":numvmailmsgs_min,
                      "number_vmail_messages_max":numvmailmsgs_max,
                      "total_day_minutes_min":ttldaymnts_min,
                      "total_day_minutes_max":ttldaymnts_max,
                      "total_eve_minutes_min":ttlevemnts_min,
                      "total_eve_minutes_max":ttlevemnts_max,
                      "total_night_minutes_min":ttlnghtmnts_min,
                      "total_night_minutes_max":ttlnghtmnts_max,
                      "total_intl_minutes_min":ttlintlmnts_min,
                      "total_intl_minutes_max":ttlintlmnts_max,
                      "total_intl_calls_min": ttlintlcalls_min,
                      "total_intl_calls_max": ttlintlcalls_max,
                      "number_customer_service_calls_min":nmcustcalls_min,
                      "number_customer_service_calls_max":nmcustcalls_max,}

    return pd.DataFrame(data=scaling_values, index=[0])



########################## SIDEBAR ###################################


with sd_bar:
    st.markdown("## User Input")

# Extracting input from the user
def get_user_input():

    # Get unique state names as a list
    df = get_data()
    states = np.array(df["state"])
    states_sorted = np.unique(states)
    
    state_val = sd_bar.selectbox(label = "State", options = states_sorted, index = 0)
    intplan_val = sd_bar.selectbox(label = "International plan", options = ["no", "yes"], index = 0)
    voiceplan_val = sd_bar.selectbox(label = "Voice mail plan", options = ["no", "yes"], index = 0)

    numvmailmsgs_val = sd_bar.slider(label = "Number of voice mail messages", min_value = 0,
                                 max_value = 51, value = 5, step = 1)

    ttldaymnts_val = sd_bar.slider(label = "Total day minutes", min_value = 0,
                                 max_value = 351, value = 5, step = 1)

    ttlevemnts_val = sd_bar.slider(label = "Total evening minutes", min_value = 0,
                                 max_value = 364, value = 5, step = 1)

    ttlnghtmnts_val = sd_bar.slider(label = "Total night minutes", min_value = 23,
                                 max_value = 395, value = 50, step = 1)

    ttlintlmnts_val = sd_bar.slider(label = "Total international minutes", min_value = 0,
                                 max_value = 20, value = 5, step = 1)

    ttlintlcalls_val = sd_bar.slider(label = "Total international calls", min_value = 0,
                                 max_value = 20, value = 5, step = 1)

    numcustcalls_val = sd_bar.slider(label = "Number of customer service calls", min_value = 0,
                                 max_value = 10, value = 1, step = 1)

    data = {"state":state_val,
            "international_plan":intplan_val,
            "voice_mail_plan":voiceplan_val,
            "number_vmail_messages":numvmailmsgs_val,
            "total_day_minutes":ttldaymnts_val,
            "total_eve_minutes":ttlevemnts_val,
            "total_night_minutes":ttlnghtmnts_val,
            "total_intl_minutes":ttlintlmnts_val,
            "total_intl_calls":ttlintlcalls_val,
            "number_customer_service_calls":numcustcalls_val}

    user_input = pd.DataFrame(data = data, index=[0])
    return user_input


df_userinput = get_user_input()


# Pre-processing the user input
def input_preprocess():

    scal_val = get_scaling_values()

    df_userinput["number_vmail_messages"] = (df_userinput["number_vmail_messages"]-scal_val["number_vmail_messages_min"])/(scal_val["number_vmail_messages_max"]-scal_val["number_vmail_messages_min"])
    df_userinput["total_day_minutes"] = (df_userinput["total_day_minutes"]-scal_val["total_day_minutes_min"])/(scal_val["total_day_minutes_max"]-scal_val["total_day_minutes_min"])
    df_userinput["total_eve_minutes"] = (df_userinput["total_eve_minutes"]-scal_val["total_eve_minutes_min"])/(scal_val["total_eve_minutes_max"]-scal_val["total_eve_minutes_min"])
    df_userinput["total_night_minutes"] = (df_userinput["total_night_minutes"]-scal_val["total_night_minutes_min"])/(scal_val["total_night_minutes_max"]-scal_val["total_night_minutes_min"])
    df_userinput["total_intl_minutes"] = (df_userinput["total_intl_minutes"]-scal_val["total_intl_minutes_min"])/(scal_val["total_intl_minutes_max"]-scal_val["total_intl_minutes_min"])
    df_userinput["total_intl_calls"] = (df_userinput["total_intl_calls"]-scal_val["total_intl_calls_min"])/(scal_val["total_intl_calls_max"]-scal_val["total_intl_calls_min"])
    df_userinput["number_customer_service_calls"] = (df_userinput["number_customer_service_calls"]-scal_val["number_customer_service_calls_min"])/(scal_val["number_customer_service_calls_max"]-scal_val["number_customer_service_calls_min"])

    df = get_data()
    df = df.drop(["Unnamed: 0", "churn", "account_length", "area_code", "total_day_calls", "total_eve_calls", "total_night_calls", "total_day_charge", "total_eve_charge", "total_night_charge", "total_intl_charge"], axis=1)
    df_userinput_ppc = pd.concat([df_userinput, df], axis = 0)

    encode = ["state", "international_plan", "voice_mail_plan"]
    for col in encode:
        dummy = pd.get_dummies(df_userinput_ppc[col], prefix = col, prefix_sep = "=", drop_first = "False")
        df_userinput_ppc = pd.concat([df_userinput_ppc,dummy], axis = 1)
        del df_userinput_ppc[col]

    return df_userinput_ppc[:1]



###########################  HEADER  ##################################

    
with header_cont:
    st.markdown("# Telecom Customer Churn")
    st.markdown("This is a web app for the ML model built in [Orange Data Mining software] (https://orangedatamining.com/), which allows "
            "predicting customer churn in telecom industry.")


###########################   DATASET  ##################################

    
with dataset_cont:
    st.markdown("## Dataset")
    st.markdown("The ML model was trained on the dataset below that is adopted from [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3). "
             "The last column is the target denoting whether the customer unsubscribed from the service (churn = yes) or not (churn = no).")
    df = get_data()
    df = df.drop("Unnamed: 0", axis=1)

    st.dataframe(df) # same as st.write(df)!


##########################   FEATURES  ##################################

    
with features_cont:
    st.markdown("## Features")
    df = get_data()
    features = np.array(df.columns)
    st.markdown("There are 19 input features in this dataset: ")
    st.markdown("(1) state, (2) account_length, (3) area_code, (4) international_plan, (5) voice_mail_plan, "
                "(6) number_vmail_messages, (7) total_day_minutes, (8) total_day_calls, (9) total_day_charge, "
                "(10) total_eve_minutes, (11) total_eve_calls, (12) total_eve_charge, (13) total_night_minutes, "
                "(14) total_night_calls, (15) total_night_charge, (16) total_intl_minutes, (17) total_intl_calls, "
                "(18) total_intl_charge, and (19) number_customer_service_calls.")
    
    st.markdown("However, after training the ML model only the following 10 features were identified as "
                "relevant for making the predictions:")
    st.markdown("(1) **state** - the home (US) state of the customer (categorical feature with 51 values)")
    st.markdown("(2) **international_plan** - indicates whether the customer has international plan or not "
                "(categorical feature with 2 values)")
    st.markdown("(3) **voice_mail_plan** - indicates whether the customer has voice mail plan or not "
                "(categorical feature with 2 values)")
    st.markdown("(4) **number_vmail_messages** - number of voice mail messages made by the customer "
                "(numeric feature)")
    st.markdown("(5) **total_day_minutes** - total number of minutes made by the customer during day "
                "(numeric feature)")
    st.markdown("(6) **total_eve_minutes** - total number of minutes made by the customer during evening "
                "(numeric feature)")
    st.markdown("(7) **total_night_minutes** - total number of minutes made by the customer during night "
                "(numeric feature)")
    st.markdown("(8) **total_intl_minutes** - total number of minutes made by the customer on international "
                "calls (numeric feature)")
    st.markdown("(9) **total_intl_calls** - total number of international calls made by the customer (numeric feature)")
    st.markdown("(10) **number_customer_service_calls** - number of calls to the customer service team made "
                "by the customer (numeric feature)")


#############################   MODEL PREDICTION   #########################

    
with modelPrediction_cont:
    st.markdown("## Model Prediction")
    st.markdown("The final ML model is a Stack consisting of Gradient Boosting and Random Forest "
            "classifiers with a Logistic Regression as an aggregate. To make a prediction "
            "with the model, you first need to select values for the 10 features by using the "
            "sliders and options in the sidebar on the left.")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### User Input")
        st.write("state:  ", df_userinput.at[0,"state"])
        st.write("international plan:  ", df_userinput.at[0,"international_plan"])
        st.write("voice mail plan:  ", df_userinput.at[0,"voice_mail_plan"])
        st.write("number vmail messages:  ", df_userinput.at[0,"number_vmail_messages"])
        st.write("total day minutes:  ", df_userinput.at[0,"total_day_minutes"])
        st.write("total eve minutes:  ", df_userinput.at[0,"total_eve_minutes"])
        st.write("total night minutes:  ", df_userinput.at[0,"total_night_minutes"])
        st.write("total intl minutes:  ", df_userinput.at[0,"total_intl_minutes"])
        st.write("total intl calls:  ", df_userinput.at[0,"total_intl_calls"])
        st.write("num cust service calls:  ", df_userinput.at[0,"number_customer_service_calls"])

    df_input_ppc = input_preprocess()

    # load Stack model
    with open("Stack Model.pkcls","rb") as model:
        loaded_model = pickle.load(model)
    
    # make prediction:
    prediction = loaded_model.predict(df_input_ppc)

    # prediction probabilities:
    probs = prediction[1][0] # probs[0] - no; probs[1] - yes

    with right_col:
    
        st.markdown("### Prediction Probabilities")
        st.markdown("Given the user input on the left, the "
                "probability this customer will churn, i.e. not churn, is:")
        st.write("churn(no): ", round(probs[0]*100,1), "%")
        st.write("churn(yes): ", round(probs[1]*100,1), "%")

        st.markdown("### Prediction")

        if probs[0]>probs[1]:
            right_col.markdown("This customer is more likely **not** to churn.")
        else:
            right_col.markdown("This customer is more likely to **churn**.")
