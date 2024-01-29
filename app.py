import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import load_model
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from helpers_train import load_train_dataset

from helpers_train import load_train_dataset
from helpers_test import process_raw_csv_data, test_data_preprocessing, make_prediction, reshape_features, ensemble_voting


def main():
    st.sidebar.title("Dog Activity Recognizer")

    main_menu = st.sidebar.radio(
        "Select an option",
        ("Paper", "Data Analysis", "Evaluation", "APP")
    )

    if main_menu == "Paper":
        st.title("Paper")
        st.title("Paper")

        # pdf_url = "https://github.com/ghanshyampendawala/Dog-Activity-Tracker/blob/main/utils/taxpnl-FY2023-2024.pdf"

        # response = requests.get(pdf_url)
        # pdf_data = response.content

        # st.pdf_viewer(BytesIO(pdf_data))

    elif main_menu == "Data Analysis":
        st.title("Data Analysis")

        dataframe = {
            'ax': [-1.2190, -0.8440, 0.0580, 0.8145, -0.1509, -2.2520, -0.0933, -0.7860, -0.7340, 0.1392],
            'ay': [-0.1490, 0.1020, 0.1830, 0.7466, -0.7368, 0.1710, 0.8467, 0.1250, 0.0650, -0.8584],
            'az': [-2.5200, -0.5250, -0.9870, 0.7510, -0.6685, -2.0440, 0.4863, -0.6090, -0.7080, -0.2539],
            'wx': [-0.6710, -1.5870, -0.9160, -73.0591, 25.7568, 0.6710, -0.1221, -2.6250, -0.7930, -114.8071],
            'wy': [116.0280, -7.5070, 0.4270, 45.7153, -21.4844, 7.9960, 0.5493, -7.9960, -1.0990, -11.1694],
            'wz': [71.1670, -0.3050, 0.3050, 108.4595, -35.1563, -29.7850, 2.8687, 2.7470, -3.4180, 126.8921],
            'angleX': [167.1510, 168.7660, 168.8320, 65.2203, -109.9567, -167.4370, 59.6503, 165.3440, 173.1880, -137.5598],
            'angleY': [16.2430, 57.1730, -3.6860, -30.9540, 3.5925, 43.7530, 5.4108, 52.3830, 45.1210, 17.4078],
            'angleZ': [-130.0120, 60.1890, -20.5660, -89.7473, 138.5706, 82.5730, 8.2452, -84.7430, 20.3850, 32.6294],
            'label': ['running', 'sitting', 'lying', 'walking', 'climbing', 'walking', 'sitting', 'sitting', 'sitting', 'climbing']
        }

        # folder_path = "processed_data/train"
        # data_df = load_train_dataset(folder_path)
        data_df = pd.DataFrame(dataframe)

        # Print out the first 10 rows of the data
        # Give a title to the table with big font size
        st.write("Sample Data")
        st.write(data_df.sample(10))

        st.write("Data Shape")
        st.write(data_df.shape)

        # st.write("Label Distribution")
        label_Distribution = plt.imread("utils/Images/label_Distributions.png")
        st.image(label_Distribution, caption="")

        # Line chart of all actions
        lineplot_walking = plt.imread("utils/Images/LinePlot_walking.png")
        st.image(lineplot_walking, caption="")

        lineplot_running = plt.imread("utils/Images/LinePlot_running.png")
        st.image(lineplot_running, caption="")

        lineplot_sitting = plt.imread("utils/Images/LinePlot_sitting.png")
        st.image(lineplot_sitting, caption="")

        lineplot_lying = plt.imread("utils/Images/LinePlot_lying.png")
        st.image(lineplot_lying, caption="")

        lineplot_climbing = plt.imread("utils/Images/LinePlot_climbing.png")
        st.image(lineplot_climbing, caption="")

        scatter_plot = plt.imread("utils/Images/scatter_plot.png")
        st.image(scatter_plot, caption="")

        pairwise_plot = plt.imread("utils/Images/Pairwise_scatter.png")
        st.image(pairwise_plot, caption="")

        correlation_matrix = plt.imread("utils/Images/Correlation_matrix.png")
        st.image(correlation_matrix, caption="")

    elif main_menu == "Evaluation":
        st.title("Evaluation")

        # Classification Report make image bigger
        classification_report = plt.imread(
            "utils/Images/classification_report.png")
        st.image(classification_report, caption="",
                 width=800, use_column_width=False)

        # Make some line as above width
        st.write("")

        # Confusion Matrix
        confusion_matrix = plt.imread("utils/Images/confusion_matrix.png")
        st.image(confusion_matrix, caption="",
                 width=800, use_column_width=False)

    elif main_menu == "APP":

        # model_77 = keras.models.load_model('utils/model77.h5')
        # model_78 = keras.models.load_model('utils/model78.h5')
        # model_78_2 = keras.models.load_model('utils/model78_2.h5')
        # model_79 = keras.models.load_model('utils/model79.h5')
        model_80 = keras.models.load_model('utils/model80.h5')

        # models = [model_79, model_80]

        uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:
            # Create an empty container to hold the results
            result_container = st.empty()

            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()

            # To convert to a string based IO:
            data = bytes_data.decode("utf-8")

            # Processing and loading spinner
            with st.spinner("Processing data and making predictions..."):
                processed_df = process_raw_csv_data(data)
                processed_data = test_data_preprocessing(processed_df)
                features_reshaped = reshape_features(processed_data)
                # all_actions, most_occurred_pred = ensemble_voting(
                #     models, features_reshaped)
                all_actions, most_occurred_pred = make_prediction(
                    model_80, features_reshaped)

            # Remove the loading spinner and update the results container
            st.success("Prediction ready!")

            # Print out prediction results
            st.write(
                f"From my observation, your Pino is currently {most_occurred_pred}.")

            result_container.write("All actions")

            fix, ax = plt.subplots()
            plt.plot(all_actions, color='blue')
            plt.title('All actions')
            plt.xlabel('Time')
            plt.ylabel('Actions')

            # Update the results container with the new graph
            result_container.pyplot(fix)


#
if __name__ == "__main__":
    main()
